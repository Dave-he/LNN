"""Tests for round 99 segment reliability gate (PRD #10-61)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.reliability_gate import (
    apply_reliability_gate,
    segment_reliability,
)


# ---------------------------------------------------------------------------
# segment_reliability
# ---------------------------------------------------------------------------

class TestSegmentReliability:
    def test_unit_for_constant_input(self) -> None:
        """Constant input has zero local std → r=1 (highly reliable)."""
        x = torch.ones(64) * 0.5
        r = segment_reliability(x, sigma_min=0.01)
        assert r.item() == pytest.approx(1.0, abs=1e-6)

    def test_high_for_smooth_input(self) -> None:
        """Smooth low-amplitude input has low local std → r > 0.5.

        Note: a full-amplitude sin(2πt) has std ≈ 0.707, which is noisy
        relative to sigma_min=0.1. We use a low-amplitude signal here
        so the "smooth" regime is well-defined.
        """
        t = torch.linspace(0, 1, 64)
        x = torch.sin(2 * math.pi * t) * 0.05  # std ≈ 0.035
        r = segment_reliability(x, sigma_min=0.1)
        assert r.item() > 0.5

    def test_low_for_noisy_input(self) -> None:
        """Noisy input (high std) → r < 0.5."""
        torch.manual_seed(0)
        x = torch.randn(64) * 2.0  # std ~2.0
        r = segment_reliability(x, sigma_min=0.01)
        assert r.item() < 0.5

    def test_in_unit_interval(self) -> None:
        """Reliability is always in [0, 1]."""
        for _ in range(10):
            x = torch.randn(64)
            r = segment_reliability(x, sigma_min=0.01)
            assert 0.0 <= r.item() <= 1.0

    def test_sigma_min_affects_threshold(self) -> None:
        """Larger sigma_min makes more inputs reliable."""
        torch.manual_seed(0)
        x = torch.randn(64) * 0.5  # std ~0.5
        r_strict = segment_reliability(x, sigma_min=0.01)
        r_permissive = segment_reliability(x, sigma_min=10.0)
        assert r_permissive.item() > r_strict.item()

    def test_rejects_nonpositive_sigma_min(self) -> None:
        """sigma_min <= 0 is rejected."""
        x = torch.ones(64)
        with pytest.raises(ValueError):
            segment_reliability(x, sigma_min=0.0)
        with pytest.raises(ValueError):
            segment_reliability(x, sigma_min=-0.1)


# ---------------------------------------------------------------------------
# apply_reliability_gate
# ---------------------------------------------------------------------------

class TestApplyReliabilityGate:
    def test_gate_dampens_noisy_output(self) -> None:
        """Noisy input: gated output is closer to zero than ungated."""
        torch.manual_seed(0)
        x = torch.randn(64) * 2.0  # noisy
        y_pred = torch.ones(64) * 1.0
        y_gated, r = apply_reliability_gate(y_pred, x, sigma_min=0.01, mix=1.0)
        # y_gated = r * y_pred, r < 0.5 for noisy input
        assert y_gated.norm().item() < y_pred.norm().item()
        assert r.item() < 0.5

    def test_gate_preserves_clean_output(self) -> None:
        """Clean (constant) input: gated output ≈ ungated output."""
        x = torch.ones(64) * 0.5
        y_pred = torch.ones(64) * 0.7
        y_gated, r = apply_reliability_gate(y_pred, x, sigma_min=0.01, mix=1.0)
        # r=1.0 for constant input, so y_gated = y_pred
        assert torch.allclose(y_gated, y_pred, atol=1e-5)

    def test_mix_zero_disables_gate(self) -> None:
        """mix=0 → y_gated == y_pred regardless of reliability."""
        torch.manual_seed(0)
        x = torch.randn(64) * 2.0
        y_pred = torch.ones(64) * 1.0
        y_gated, r = apply_reliability_gate(y_pred, x, sigma_min=0.01, mix=0.0)
        assert torch.allclose(y_gated, y_pred, atol=1e-5)
        # r is still computed for diagnostic purposes
        assert 0.0 <= r.item() <= 1.0

    def test_mix_half_blends(self) -> None:
        """mix=0.5 → y_gated = 0.5 * y_pred + 0.5 * r * y_pred."""
        torch.manual_seed(0)
        x = torch.randn(64) * 0.5
        y_pred = torch.ones(64) * 1.0
        y_gated, r = apply_reliability_gate(y_pred, x, sigma_min=0.1, mix=0.5)
        expected = 0.5 * y_pred + 0.5 * r * y_pred
        assert torch.allclose(y_gated, expected, atol=1e-5)

    def test_rejects_mix_out_of_range(self) -> None:
        """mix outside [0, 1] is rejected."""
        x = torch.ones(64)
        y_pred = torch.ones(64)
        with pytest.raises(ValueError):
            apply_reliability_gate(y_pred, x, mix=-0.1)
        with pytest.raises(ValueError):
            apply_reliability_gate(y_pred, x, mix=1.5)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestReliabilityGateExports:
    def test_segment_reliability_exported(self) -> None:
        from lnn.core import segment_reliability as sr
        assert sr is segment_reliability

    def test_apply_reliability_gate_exported(self) -> None:
        from lnn.core import apply_reliability_gate as arg
        assert arg is apply_reliability_gate

    def test_in_all_list(self) -> None:
        import lnn.core as core
        assert "segment_reliability" in core.__all__
        assert "apply_reliability_gate" in core.__all__
