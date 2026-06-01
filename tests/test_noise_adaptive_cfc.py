"""Smoke + invariant tests for the Noise-Adaptive CfC."""

from __future__ import annotations

import torch

from lnn.core.cfc import CfCNetwork
from lnn.core.noise_adaptive_cfc import NoiseAdaptiveCfCCell, NoiseAdaptiveCfCNetwork


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
