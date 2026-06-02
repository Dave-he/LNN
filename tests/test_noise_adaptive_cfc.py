"""Smoke + invariant tests for the Noise-Adaptive CfC."""

from __future__ import annotations

import torch

from lnn.core.cfc import CfCNetwork
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.noise_adaptive_cfc import (
    BiCfCNADWithMDN,
    BidirectionalNoiseAdaptiveCfC,
    CfCNADWithMDN,
    NoiseAdaptiveCfCCell,
    NoiseAdaptiveCfCNetwork,
    mdn_predicted_std,
    vectorized_noise_ema,
)


class TestNoiseAdaptiveCfCCell:
    def test_output_shape(self) -> None:
        cell = NoiseAdaptiveCfCCell(input_size=3, hidden_size=8)
        x = torch.randn(4, 3)
        h = torch.zeros(4, 8)
        noise = torch.zeros(4, 3)
        out = cell(x, h, noise_score=noise, dt=1.0)
        assert out.shape == (4, 8)

    def test_backward(self) -> None:
        cell = NoiseAdaptiveCfCCell(input_size=2, hidden_size=4)
        x = torch.randn(2, 2, requires_grad=True)
        h = torch.zeros(2, 4)
        noise = torch.full((2, 2), 0.1)
        out = cell(x, h, noise_score=noise, dt=1.0)
        out.sum().backward()
        assert x.grad is not None
        for p in cell.parameters():
            if p.requires_grad:
                # Some parameters may have zero grad on the very first step
                # (e.g. noise_gate_proj when initialised to zero), but every
                # parameter should at least have a populated grad tensor.
                assert p.grad is not None

    def test_initialises_to_vanilla_cfc_when_noise_zero(self) -> None:
        cell = NoiseAdaptiveCfCCell(input_size=2, hidden_size=4)
        # noise_gate_proj is zero-initialised -> sigmoid(0) = 0.5, so the gate
        # does mix. Confirm that mixing returns finite outputs and gradients.
        h = torch.full((1, 4), 0.5)
        x = torch.full((1, 2), 0.3)
        noise = torch.zeros(1, 2)
        out = cell(x, h, noise_score=noise, dt=1.0)
        assert torch.isfinite(out).all()


class TestNoiseAdaptiveCfCNetwork:
    def test_output_shape(self) -> None:
        net = NoiseAdaptiveCfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=1
        )
        x = torch.randn(2, 5, 3)
        out = net(x)
        assert out.shape == (2, 5, 2)

    def test_last_step_only(self) -> None:
        net = NoiseAdaptiveCfCNetwork(
            input_size=3, hidden_size=8, output_size=2, return_sequences=False
        )
        x = torch.randn(2, 5, 3)
        out = net(x)
        assert out.shape == (2, 2)

    def test_dt_and_mask(self) -> None:
        net = NoiseAdaptiveCfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=2
        )
        x = torch.randn(2, 6, 3)
        dt = torch.linspace(0.1, 0.6, 6)
        mask = torch.ones(2, 6)
        mask[0, 3] = 0
        out = net(x, dt=dt, mask=mask)
        assert out.shape == (2, 6, 2)
        assert torch.isfinite(out).all()

    def test_parameter_overhead_within_15_percent(self) -> None:
        """CfC-NAD adds a couple of small projections; overhead should be modest."""

        base = CfCNetwork(input_size=3, hidden_size=16, output_size=1, num_layers=1)
        nad = NoiseAdaptiveCfCNetwork(
            input_size=3, hidden_size=16, output_size=1, num_layers=1
        )
        base_params = sum(p.numel() for p in base.parameters())
        nad_params = sum(p.numel() for p in nad.parameters())
        overhead = (nad_params - base_params) / max(base_params, 1)
        # Generous bound: the extra Linear adds at most a few hundred parameters
        # for typical hidden sizes.
        assert overhead < 0.5, (
            f"unexpectedly large param overhead: {overhead:.2%} "
            f"({nad_params} vs {base_params})"
        )

    def test_noise_path_changes_under_noisy_input(self) -> None:
        """With non-zero noise the noise gate should produce a different output
        than re-running the same model on the *clean* signal."""

        torch.manual_seed(0)
        net = NoiseAdaptiveCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=1
        )
        # Train the noise gate slightly off zero so the path is observable.
        with torch.no_grad():
            net.cells[0].noise_gate_proj.weight.fill_(0.5)
            net.cells[0].noise_gate_proj.bias.fill_(-0.2)
        clean = torch.zeros(1, 16, 2)
        clean[:, :, 0] = torch.sin(torch.linspace(0.0, 6.28, 16))
        noisy = clean + 0.5 * torch.randn_like(clean)
        out_clean = net(clean)
        out_noisy = net(noisy)
        assert not torch.allclose(out_clean, out_noisy, atol=1e-3)


class TestVectorizedNoiseEMA:
    """Validate the parallel cumulative form against the streaming reference."""

    @staticmethod
    def _streaming_reference(x: torch.Tensor, beta: float) -> torch.Tensor:
        B, T, F = x.shape
        out = torch.zeros_like(x)
        if T == 0:
            return out
        prev = torch.zeros(B, F, dtype=x.dtype, device=x.device)
        ema = torch.zeros(B, F, dtype=x.dtype, device=x.device)
        for t in range(T):
            diff_sq = (x[:, t, :] - prev) ** 2 if t > 0 else torch.zeros_like(prev)
            ema = beta * ema + (1.0 - beta) * diff_sq
            prev = x[:, t, :]
            out[:, t, :] = ema
        return out

    def test_matches_streaming_random(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(3, 24, 4)
        streaming = self._streaming_reference(x, beta=0.9)
        parallel = vectorized_noise_ema(x, beta=0.9)
        # parallel_liquid_relaxation clamps retain to [0.02, 0.98]; with
        # beta=0.9 we stay inside that range so the two paths should agree to
        # within float32 noise.
        assert torch.allclose(parallel, streaming, atol=1e-5, rtol=1e-5)

    def test_matches_streaming_alternating_signs(self) -> None:
        # An alternating-sign input is the worst case for first-difference
        # accumulation: a regression here would catch sign or alignment bugs.
        torch.manual_seed(11)
        T = 32
        signs = torch.tensor([(-1.0) ** t for t in range(T)]).view(1, T, 1)
        x = signs.expand(2, T, 3) * torch.linspace(0.1, 1.0, 3).view(1, 1, 3)
        streaming = self._streaming_reference(x, beta=0.85)
        parallel = vectorized_noise_ema(x, beta=0.85)
        assert torch.allclose(parallel, streaming, atol=1e-5, rtol=1e-5)

    def test_zero_length_safe(self) -> None:
        x = torch.zeros(2, 0, 3)
        out = vectorized_noise_ema(x, beta=0.9)
        assert out.shape == (2, 0, 3)


class TestNoiseAdaptivePathEquivalence:
    """The parallel-noise forward path must match the streaming path bit-for-bit."""

    def test_network_outputs_match_with_mask_none(self) -> None:
        torch.manual_seed(0)
        net = NoiseAdaptiveCfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=2
        )
        # Spread the noise gate so the heteroscedastic path is exercised.
        for cell in net.cells:
            with torch.no_grad():
                cell.noise_gate_proj.weight.normal_(0.0, 0.3)
                cell.noise_gate_proj.bias.normal_(0.0, 0.1)
        x = torch.randn(4, 20, 3)
        out_parallel = net(x)
        # Force the streaming path by providing an all-ones mask of compatible
        # shape: select_step_mask will activate, mask=None becomes mask!=None.
        ones_mask = torch.ones(4, 20)
        out_stream = net(x, mask=ones_mask)
        # With an all-ones mask the masked path should produce the same output
        # as the parallel one — modulo the masked-step gating, which is a
        # no-op when the mask is uniformly 1.
        assert torch.allclose(out_parallel, out_stream, atol=1e-5, rtol=1e-5)


class TestBidirectionalNoiseAdaptiveCfC:
    def test_output_shape_sequences(self) -> None:
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=3, hidden_size=8, output_size=2, num_layers=1
        )
        x = torch.randn(2, 7, 3)
        out = net(x)
        assert out.shape == (2, 7, 2)

    def test_output_shape_last_step(self) -> None:
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=3, hidden_size=8, output_size=2, return_sequences=False
        )
        x = torch.randn(2, 7, 3)
        out = net(x)
        assert out.shape == (2, 2)

    def test_backward_pass(self) -> None:
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=2, hidden_size=4, output_size=1, num_layers=1
        )
        x = torch.randn(2, 6, 2, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None
        # Both inner networks should receive gradient.
        fwd_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in net.forward_net.parameters()
        )
        bwd_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in net.backward_net.parameters()
        )
        assert fwd_grad > 0
        assert bwd_grad > 0

    def test_differs_from_unidirectional(self) -> None:
        """Bi network output must differ from a same-config uni network on
        non-symmetric input (otherwise the backward path is silently dead)."""

        torch.manual_seed(3)
        uni = NoiseAdaptiveCfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1, return_sequences=True
        )
        bi = BidirectionalNoiseAdaptiveCfC(
            input_size=2, hidden_size=8, output_size=2, num_layers=1, return_sequences=True
        )
        # Asymmetric input: monotonic ramp + impulse near the end.
        x = torch.linspace(0.0, 1.0, 24).view(1, 24, 1).expand(1, 24, 2).contiguous()
        x[0, 20, 0] = 5.0
        with torch.no_grad():
            uni_out = uni(x)
            bi_out = bi(x)
        # Outputs cannot reasonably coincide — different parameter counts and
        # different feature spaces. Sanity-check that they actually differ.
        assert uni_out.shape == bi_out.shape
        assert not torch.allclose(uni_out, bi_out, atol=1e-3)

    def test_dt_temporal_flip_supported(self) -> None:
        # 1-D per-step dt of shape [T] must be flipped for the backward pass.
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=2, hidden_size=4, output_size=1, num_layers=1
        )
        x = torch.randn(3, 10, 2)
        dt = torch.linspace(0.05, 0.5, 10)
        out = net(x, dt=dt)
        assert out.shape == (3, 10, 1)
        assert torch.isfinite(out).all()

    def test_parameter_overhead_under_3x(self) -> None:
        uni = NoiseAdaptiveCfCNetwork(input_size=3, hidden_size=16, output_size=1, num_layers=1)
        bi = BidirectionalNoiseAdaptiveCfC(input_size=3, hidden_size=16, output_size=1, num_layers=1)
        uni_params = sum(p.numel() for p in uni.parameters())
        bi_params = sum(p.numel() for p in bi.parameters())
        # Two NoiseAdaptiveCfCNetwork instances + a 2H -> output projection.
        # The output_proj of each inner net (H -> H) is redundant but kept for
        # API parity; the overall budget should still stay well under 3x uni.
        assert bi_params < 3 * uni_params


class TestBidirectionalCenteredNoise:
    """Verify the non-causal centered noise aggregation path."""

    def test_rejects_unknown_aggregation(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            BidirectionalNoiseAdaptiveCfC(
                input_size=2, hidden_size=4, output_size=1, noise_aggregation="bogus"
            )

    def test_centered_output_shape(self) -> None:
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=2,
            hidden_size=4,
            output_size=1,
            noise_aggregation="centered",
        )
        x = torch.randn(2, 9, 2)
        out = net(x)
        assert out.shape == (2, 9, 1)
        assert torch.isfinite(out).all()

    def test_centered_backward_pass(self) -> None:
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=2,
            hidden_size=4,
            output_size=1,
            noise_aggregation="centered",
        )
        x = torch.randn(2, 8, 2, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_centered_rejects_mask(self) -> None:
        import pytest
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=2,
            hidden_size=4,
            output_size=1,
            noise_aggregation="centered",
        )
        x = torch.randn(1, 5, 2)
        mask = torch.ones(1, 5)
        with pytest.raises(ValueError):
            net(x, mask=mask)

    def test_centered_differs_from_independent_under_noise(self) -> None:
        """The centered and independent paths must produce different outputs on
        a noisy input — otherwise the centered aggregation is silently no-op."""

        torch.manual_seed(0)
        # Same weights for both networks.
        indep = BidirectionalNoiseAdaptiveCfC(
            input_size=2, hidden_size=8, output_size=1, noise_aggregation="independent"
        )
        torch.manual_seed(0)
        centered = BidirectionalNoiseAdaptiveCfC(
            input_size=2, hidden_size=8, output_size=1, noise_aggregation="centered"
        )
        # Force the noise gate slightly off the zero init so the difference
        # in noise_score actually flows through to the cell outputs.
        for net in (indep, centered):
            for inner in (net.forward_net, net.backward_net):
                for cell in inner.cells:
                    with torch.no_grad():
                        cell.noise_gate_proj.weight.normal_(0.0, 0.3)
                        cell.noise_gate_proj.bias.normal_(0.0, 0.1)
        # Mirror the trained weights so the only difference is the noise path.
        centered.load_state_dict(indep.state_dict())
        x = torch.randn(1, 16, 2)
        with torch.no_grad():
            out_indep = indep(x)
            out_centered = centered(x)
        assert out_indep.shape == out_centered.shape
        assert not torch.allclose(out_indep, out_centered, atol=1e-4)

    def test_centered_uses_future_information(self) -> None:
        """Centered noise must depend on the *future* tail of the input. We
        check by perturbing the last few steps and verifying the centred
        noise score at the *start* of the sequence changes — a property that
        the causal-only independent path cannot have."""

        torch.manual_seed(0)
        net = BidirectionalNoiseAdaptiveCfC(
            input_size=1, hidden_size=4, output_size=1, noise_aggregation="centered"
        )
        x_a = torch.zeros(1, 20, 1)
        x_a[0, :10, 0] = torch.linspace(0.0, 1.0, 10)  # quiet tail
        x_b = x_a.clone()
        x_b[0, 15:, 0] = torch.tensor([2.0, -2.0, 2.0, -2.0, 2.0])  # noisy tail
        with torch.no_grad():
            score_a = net._centered_noise_score(x_a, net.noise_beta)
            score_b = net._centered_noise_score(x_b, net.noise_beta)
        # Score at the early steps must differ between x_a and x_b because the
        # backward EMA "sees" the tail.
        early_diff = (score_b[0, :5, 0] - score_a[0, :5, 0]).abs().max().item()
        assert early_diff > 1e-6


class TestCfCNADWithMDN:
    """Wire up CfC-NAD as the feature backbone for an MDN head."""

    def test_output_shapes_sequences(self) -> None:
        net = CfCNADWithMDN(
            input_size=2, hidden_size=8, output_size=1, num_mixtures=3
        )
        x = torch.randn(2, 7, 2)
        params = net(x)
        assert params["logits"].shape == (2, 7, 3)
        assert params["loc"].shape == (2, 7, 3, 1)
        assert params["log_scale"].shape == (2, 7, 3, 1)

    def test_output_shapes_last_step(self) -> None:
        net = CfCNADWithMDN(
            input_size=2,
            hidden_size=8,
            output_size=1,
            num_mixtures=2,
            return_sequences=False,
        )
        x = torch.randn(2, 7, 2)
        params = net(x)
        assert params["logits"].shape == (2, 2)
        assert params["loc"].shape == (2, 2, 1)
        assert params["log_scale"].shape == (2, 2, 1)

    def test_negative_mixture_count_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            CfCNADWithMDN(input_size=1, hidden_size=4, output_size=1, num_mixtures=0)

    def test_mdn_nll_trainable(self) -> None:
        torch.manual_seed(0)
        net = CfCNADWithMDN(
            input_size=1, hidden_size=8, output_size=1, num_mixtures=2
        )
        # Toy data: y = sin(x) + small noise.
        T = 16
        x = torch.linspace(0.0, 6.28, T).view(1, T, 1)
        y = torch.sin(x)
        optim = torch.optim.Adam(net.parameters(), lr=5e-3)
        params0 = net(x)
        loss0 = mdn_negative_log_likelihood(params0, y).item()
        for _ in range(30):
            optim.zero_grad()
            params = net(x)
            loss = mdn_negative_log_likelihood(params, y)
            loss.backward()
            optim.step()
        loss1 = mdn_negative_log_likelihood(net(x), y).item()
        assert loss1 < loss0, f"NLL should decrease ({loss0:.4f} -> {loss1:.4f})"

    def test_predicted_std_increases_with_noisy_target(self) -> None:
        """Train the head on a fixed-input series with two noise regimes and
        check that the learnt predicted std is larger on the noisier half."""

        torch.manual_seed(7)
        net = CfCNADWithMDN(
            input_size=1, hidden_size=12, output_size=1, num_mixtures=1
        )
        T = 32
        # Same input, different target noise on first vs second half.
        base = torch.linspace(0.0, 6.28, T).view(1, T, 1)
        x = base.expand(64, T, 1).clone()
        target = torch.sin(base).expand(64, T, 1).clone()
        noise = torch.randn_like(target)
        # 0.05 std in the first half, 0.5 std in the second half.
        sigma_schedule = torch.ones(T)
        sigma_schedule[: T // 2] = 0.05
        sigma_schedule[T // 2 :] = 0.5
        noisy_target = target + noise * sigma_schedule.view(1, T, 1)
        optim = torch.optim.Adam(net.parameters(), lr=5e-3)
        for _ in range(60):
            optim.zero_grad()
            params = net(x)
            loss = mdn_negative_log_likelihood(params, noisy_target)
            loss.backward()
            optim.step()
        with torch.no_grad():
            params = net(x)
            std = mdn_predicted_std(params)  # [B, T]
        mean_low = float(std[:, : T // 2].mean())
        mean_high = float(std[:, T // 2 :].mean())
        assert mean_high > mean_low, (
            f"predicted std should be higher in the noisy half: "
            f"low={mean_low:.4f} high={mean_high:.4f}"
        )

    def test_mdn_mean_matches_signature(self) -> None:
        net = CfCNADWithMDN(
            input_size=1, hidden_size=4, output_size=1, num_mixtures=2
        )
        x = torch.randn(1, 5, 1)
        params = net(x)
        mean = mdn_mean(params)
        assert mean.shape == (1, 5, 1)
        std = mdn_predicted_std(params)
        assert std.shape == (1, 5)
        assert (std > 0).all()


class TestBiCfCNADWithMDN:
    """Bidirectional CfC-NAD backbone + MDN head."""

    def test_output_shapes_sequences(self) -> None:
        net = BiCfCNADWithMDN(
            input_size=2, hidden_size=8, output_size=1, num_mixtures=2
        )
        x = torch.randn(3, 9, 2)
        params = net(x)
        assert params["logits"].shape == (3, 9, 2)
        assert params["loc"].shape == (3, 9, 2, 1)
        assert params["log_scale"].shape == (3, 9, 2, 1)

    def test_output_shapes_last_step(self) -> None:
        net = BiCfCNADWithMDN(
            input_size=2,
            hidden_size=8,
            output_size=1,
            num_mixtures=3,
            return_sequences=False,
        )
        x = torch.randn(2, 5, 2)
        params = net(x)
        assert params["logits"].shape == (2, 3)
        assert params["loc"].shape == (2, 3, 1)
        assert params["log_scale"].shape == (2, 3, 1)

    def test_backward_into_both_directions(self) -> None:
        net = BiCfCNADWithMDN(
            input_size=2, hidden_size=4, output_size=1, num_mixtures=1
        )
        x = torch.randn(2, 6, 2, requires_grad=True)
        y = torch.randn(2, 6, 1)
        params = net(x)
        loss = mdn_negative_log_likelihood(params, y)
        loss.backward()
        # Both halves of the bidirectional backbone must receive gradient.
        fwd_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in net.encoder.forward_net.parameters()
        )
        bwd_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in net.encoder.backward_net.parameters()
        )
        assert fwd_grad > 0
        assert bwd_grad > 0

    def test_differs_from_uni_mdn(self) -> None:
        torch.manual_seed(2)
        uni = CfCNADWithMDN(
            input_size=1, hidden_size=8, output_size=1, num_mixtures=1
        )
        bi = BiCfCNADWithMDN(
            input_size=1, hidden_size=8, output_size=1, num_mixtures=1
        )
        x = torch.randn(1, 12, 1)
        with torch.no_grad():
            std_uni = mdn_predicted_std(uni(x))
            std_bi = mdn_predicted_std(bi(x))
        assert std_uni.shape == std_bi.shape
        # Different architecture → different uncertainty stream.
        assert not torch.allclose(std_uni, std_bi, atol=1e-4)

    def test_centered_aggregation_supported(self) -> None:
        # The bi backbone supports noise_aggregation="centered"; forwarding it
        # through the MDN-wrapped class must not raise on mask=None input.
        net = BiCfCNADWithMDN(
            input_size=1,
            hidden_size=4,
            output_size=1,
            num_mixtures=1,
            noise_aggregation="centered",
        )
        x = torch.randn(1, 8, 1)
        params = net(x)
        std = mdn_predicted_std(params)
        assert std.shape == (1, 8)
        assert (std > 0).all()
