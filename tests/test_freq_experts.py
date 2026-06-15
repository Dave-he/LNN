"""Tests for freq_experts module (Round 110, PRD #10-72)."""
from __future__ import annotations

import math

import torch

from lnn.core.freq_experts import (
    FrequencyExpert,
    FrequencyExpertConfig,
    FrequencyMoEConfig,
    FrequencyRouter,
    TimeFreqMoECfCCell,
    TimeFreqMoECfCNetwork,
)


# ---------------------------------------------------------------------------
# FrequencyExpert tests
# ---------------------------------------------------------------------------


class TestFrequencyExpert:
    def test_forward_shape_complex(self):
        cfg = FrequencyExpertConfig(
            input_size=2, hidden_size=8, n_freqs=4, use_complex_basis=True,
        )
        e = FrequencyExpert(cfg)
        x = torch.randn(2, 16, 2)
        y = e(x)
        assert y.shape == (2, 16, 8)

    def test_forward_shape_real(self):
        cfg = FrequencyExpertConfig(
            input_size=2, hidden_size=8, n_freqs=4, use_complex_basis=False,
        )
        e = FrequencyExpert(cfg)
        x = torch.randn(2, 16, 2)
        y = e(x)
        assert y.shape == (2, 16, 8)

    def test_nan_safe(self):
        cfg = FrequencyExpertConfig(input_size=2, hidden_size=8, n_freqs=4)
        e = FrequencyExpert(cfg)
        x = torch.tensor([[[1.0, 2.0], [float("nan"), 3.0], [4.0, 5.0]]])
        y = e(x)
        assert torch.isfinite(y).all()

    def test_omega_bounded(self):
        cfg = FrequencyExpertConfig(input_size=2, hidden_size=8, n_freqs=4, max_omega=math.pi)
        e = FrequencyExpert(cfg)
        omega = e._get_omega()
        assert (omega >= 0).all()
        assert (omega <= math.pi + 1e-6).all()

    def test_different_omega_gives_different_output(self):
        cfg = FrequencyExpertConfig(input_size=2, hidden_size=8, n_freqs=4)
        e1 = FrequencyExpert(cfg)
        e2 = FrequencyExpert(cfg)
        # Change e2's frequencies
        with torch.no_grad():
            e2.omega_raw.data += 0.5
        x = torch.randn(1, 8, 2)
        y1 = e1(x)
        y2 = e2(x)
        assert not torch.allclose(y1, y2, atol=1e-3)

    def test_gradient_flows(self):
        cfg = FrequencyExpertConfig(input_size=2, hidden_size=8, n_freqs=4)
        e = FrequencyExpert(cfg)
        x = torch.randn(2, 8, 2)
        y = e(x)
        loss = y.sum()
        loss.backward()
        # Check gradients
        assert e.omega_raw.grad is not None
        assert e.omega_raw.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# FrequencyRouter tests
# ---------------------------------------------------------------------------


class TestFrequencyRouter:
    def test_forward_shape(self):
        r = FrequencyRouter(input_size=4, n_experts=4, top_k=2)
        x = torch.randn(2, 16, 4)
        full_w, top_idx, top_w, aux = r(x)
        # full_w: (B*T, K) = (32, 4)
        assert full_w.shape == (32, 4)
        # top_idx: (32, 2)
        assert top_idx.shape == (32, 2)
        # top_w: (32, 2)
        assert top_w.shape == (32, 2)
        # aux: scalar
        assert aux.dim() == 0

    def test_topk_in_range(self):
        r = FrequencyRouter(input_size=4, n_experts=4, top_k=2)
        x = torch.randn(2, 16, 4)
        _, top_idx, _, _ = r(x)
        assert (top_idx >= 0).all()
        assert (top_idx < 4).all()

    def test_full_weights_sum_to_one(self):
        r = FrequencyRouter(input_size=4, n_experts=4, top_k=2)
        x = torch.randn(2, 16, 4)
        full_w, _, _, _ = r(x)
        sums = full_w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_aux_loss_non_negative(self):
        r = FrequencyRouter(input_size=4, n_experts=4, top_k=2)
        x = torch.randn(2, 16, 4)
        _, _, _, aux = r(x)
        assert aux.item() >= 0


# ---------------------------------------------------------------------------
# TimeFreqMoECfCCell tests
# ---------------------------------------------------------------------------


class TestTimeFreqMoECfCCell:
    def test_init(self):
        cfg = FrequencyMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = TimeFreqMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        assert len(cell.experts) == 4

    def test_forward_shape(self):
        cfg = FrequencyMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = TimeFreqMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        x = torch.randn(2, 16, 2)
        out, aux, info = cell(x)
        assert out.shape == (2, 16, 1)
        assert aux.dim() == 0

    def test_nan_safe(self):
        cfg = FrequencyMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = TimeFreqMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        x = torch.tensor([[[1.0, 2.0], [float("nan"), 3.0], [4.0, 5.0]]])
        out, aux, _ = cell(x)
        assert torch.isfinite(out).all()

    def test_no_time_branch(self):
        cfg = FrequencyMoEConfig(
            input_size=2, hidden_size=8, output_size=1, use_time_branch=False,
        )
        cell = TimeFreqMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        x = torch.randn(2, 16, 2)
        out, _, _ = cell(x)
        assert out.shape == (2, 16, 1)


# ---------------------------------------------------------------------------
# TimeFreqMoECfCNetwork tests
# ---------------------------------------------------------------------------


class TestTimeFreqMoECfCNetwork:
    def test_init(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        assert net.input_size == 2

    def test_forward_shape(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 16, 2)
        out, aux, info = net(x)
        assert out.shape == (2, 16, 1)
        assert aux.dim() == 0

    def test_nan_safe(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 16, 2)
        x[0, 3, 0] = float("nan")
        out, _, _ = net(x)
        assert torch.isfinite(out).all()

    def test_get_utilization(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 16, 2)
        net(x)
        util = net.get_utilization()
        assert "routing_H" in util
        assert "max_min" in util
        assert "active_fraction" in util
        assert "utilization" in util

    def test_get_omegas(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        omegas = net.get_omegas()
        # Should be (K, n_freqs) = (4, 4)
        assert omegas.shape == (4, 4)
        assert (omegas >= 0).all()
        assert (omegas <= 2 * math.pi + 1e-6).all()

    def test_gradient_flows(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 8, 2)
        y_target = torch.randn(2, 8, 1)
        out, aux, _ = net(x)
        loss = ((out - y_target) ** 2).mean() + 0.01 * aux
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFreqExpertIntegration:
    def test_omegas_learn_during_training(self):
        """Test that the learned frequencies change during training."""
        net = TimeFreqMoECfCNetwork(
            input_size=1, hidden_size=8, output_size=1,
        )
        omega_before = net.get_omegas().clone()
        # Train a few steps
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(5):
            x = torch.randn(2, 16, 1)
            y = torch.randn(2, 16, 1)
            out, aux, _ = net(x)
            loss = ((out - y) ** 2).mean() + 0.01 * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
        omega_after = net.get_omegas()
        # At least some frequencies should have changed
        diff = (omega_after - omega_before).abs().sum()
        assert diff > 1e-3, f"Omegas did not change: {diff}"

    def test_freq_expert_captures_periodic_signal(self):
        """A frequency expert should be able to learn a sin wave."""
        torch.manual_seed(0)
        net = TimeFreqMoECfCNetwork(
            input_size=1, hidden_size=16, output_size=1,
            config=FrequencyMoEConfig(
                n_experts=4, top_k=2, n_freqs=4, use_time_branch=False,
            ),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        # Generate sin wave
        t = torch.linspace(0, 4 * math.pi, 32).unsqueeze(0).unsqueeze(-1)
        for _ in range(30):
            x = t  # (1, 32, 1)
            y = torch.sin(t)  # (1, 32, 1)
            out, aux, _ = net(x)
            loss = ((out - y) ** 2).mean() + 0.01 * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Check final loss
        with torch.no_grad():
            out, _, _ = net(t)
            mse = ((out - torch.sin(t)) ** 2).mean().item()
        # Should learn the sin wave to some degree (we don't expect perfect)
        # but loss should be less than 0.5
        assert mse < 0.5, f"Could not learn sin: mse={mse}"

    def test_outputs_depend_on_input(self):
        net = TimeFreqMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x1 = torch.randn(1, 8, 2)
        x2 = torch.randn(1, 8, 2)
        out1, _, _ = net(x1)
        out2, _, _ = net(x2)
        assert not torch.allclose(out1, out2, atol=1e-3)
