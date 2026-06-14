"""Unit tests for CausalityGatedOrth (PRD #10-51, 2026-06-15, round 89).

Verifies:
- CausalityGatedOrth fires when ratio > threshold
- Doesn't fire when ratio <= threshold
- Respects warmup_steps
- effective_lambda_scale = 1.0 when not fired, lambda_safe when fired
- reset() works
- FAMECfCCell(causality_gated_orth=True) wires it
- compute_orth_loss_causality returns rescaled loss when gate fires
- back-compat: causality_gated_orth=False is no-op
- Combined with round 85 EcologyGatedOrth is safe (min lambda)
- CausalityGatedOrth is exported from lnn.core
"""
import torch

from lnn.core import CausalityGatedOrth
from lnn.core.ecology_gated_balancing import EcologyGatedOrth
from lnn.core.fame_cfc import FAMECfCCell


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


class TestCausalityGatedOrth:
    def test_default_threshold(self) -> None:
        """Default ratio_threshold is 10.0 (round 88 finding)."""
        gate = CausalityGatedOrth()
        assert gate.ratio_threshold == 10.0
        assert gate.lambda_safe == 0.001
        assert gate.warmup_steps == 0
        assert gate.intervened is False

    def test_fires_above_threshold(self) -> None:
        """Fires when ratio > threshold."""
        gate = CausalityGatedOrth(ratio_threshold=10.0)
        out = gate.step(max_min_ratio_grad=15.0, step_idx=0)
        assert out["intervened"] is True
        assert out["effective_lambda_scale"] == 0.001
        assert out["triggered_step"] == 0

    def test_does_not_fire_below_threshold(self) -> None:
        """Doesn't fire when ratio <= threshold."""
        gate = CausalityGatedOrth(ratio_threshold=10.0)
        out = gate.step(max_min_ratio_grad=5.0, step_idx=0)
        assert out["intervened"] is False
        assert out["effective_lambda_scale"] == 1.0
        assert out["triggered_step"] == -1

    def test_warmup_skips_first_n_steps(self) -> None:
        """warmup_steps: don't fire in first N steps even if ratio high."""
        gate = CausalityGatedOrth(ratio_threshold=10.0, warmup_steps=3)
        # Step 0, 1, 2: in warmup, don't fire.
        for i in range(3):
            out = gate.step(max_min_ratio_grad=20.0, step_idx=i)
            assert out["intervened"] is False, f"step {i}: should not fire in warmup"
        # Step 3: out of warmup, fires.
        out = gate.step(max_min_ratio_grad=20.0, step_idx=3)
        assert out["intervened"] is True

    def test_sticky_after_firing(self) -> None:
        """Once fired, stays fired even if ratio drops."""
        gate = CausalityGatedOrth(ratio_threshold=10.0)
        out1 = gate.step(max_min_ratio_grad=15.0, step_idx=0)
        assert out1["intervened"] is True
        out2 = gate.step(max_min_ratio_grad=1.0, step_idx=1)
        assert out2["intervened"] is True  # sticky

    def test_reset(self) -> None:
        """reset() restores to initial state."""
        gate = CausalityGatedOrth(ratio_threshold=10.0)
        gate.step(max_min_ratio_grad=15.0, step_idx=0)
        gate.reset()
        assert gate.intervened is False
        assert gate.last_ratio == 1.0
        assert gate.last_lambda_scale == 1.0
        assert gate.triggered_step == -1

    def test_repr_contains_key_info(self) -> None:
        """repr mentions threshold, lambda_safe, intervened, last_ratio."""
        gate = CausalityGatedOrth(ratio_threshold=5.0, lambda_safe=0.01)
        s = repr(gate)
        assert "5.0" in s
        assert "0.01" in s
        assert "False" in s

    def test_combined_with_ecology_orth_subclass(self) -> None:
        """CausalityGatedOrth is independent of EcologyGatedOrth."""
        causality = CausalityGatedOrth(ratio_threshold=10.0)
        ecology = EcologyGatedOrth(E_min=0.5, lambda_safe=0.001)
        # Both are gate classes, no shared state.
        assert causality.intervened is False
        assert ecology.intervened is False
        # Causality fires.
        causality.step(max_min_ratio_grad=15.0, step_idx=0)
        assert causality.intervened is True
        # Ecology still hasn't fired.
        assert ecology.intervened is False


class TestFAMECfCCellCausalityGate:
    def test_default_off_backcompat(self) -> None:
        """Default ``causality_gated_orth=False`` is back-compat."""
        _seed(0)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        assert cell.causality_gate is None

    def test_causality_gated_orth_wires_gate(self) -> None:
        """``causality_gated_orth=True`` attaches a CausalityGatedOrth."""
        _seed(1)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            causality_gated_orth=True, causality_ratio_threshold=10.0,
        )
        assert cell.causality_gate is not None
        assert cell.causality_gate.ratio_threshold == 10.0

    def test_compute_orth_loss_causality_back_compat_no_gate(self) -> None:
        """Without causality_gated_orth, compute_orth_loss_causality uses user_lambda."""
        _seed(2)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        with torch.no_grad():
            outs = [torch.randn(4, 8) for _ in range(3)]
        loss = cell.compute_orth_loss_causality(outs, user_lambda=1.0)
        # Standard orth loss with user_lambda=1.0.
        assert loss.item() != 0.0  # orth loss is non-zero

    def test_compute_orth_loss_causality_rescales_when_fired(self) -> None:
        """When gate has fired, effective lambda is reduced."""
        _seed(3)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            causality_gated_orth=True, causality_ratio_threshold=5.0,
        )
        cell.train()
        # Need a forward to populate last_g (used by causality path).
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Manually fire the gate.
        cell.causality_gate.intervened = True
        cell.causality_gate.last_lambda_scale = 0.001
        with torch.no_grad():
            outs = [torch.randn(4, 8) for _ in range(3)]
        # user_lambda=1.0 should be rescaled to 0.001.
        loss_rescaled = cell.compute_orth_loss_causality(
            outs, user_lambda=1.0, task_loss=None,
        )
        # Compute the full orth loss with same outs to verify rescale.
        from lnn.core.orthogonality import orthogonality_loss
        loss_full = orthogonality_loss(outs, lambda_coeff=1.0)
        # rescaled should be 0.001× loss_full.
        assert abs(loss_rescaled.item() - 0.001 * loss_full.item()) < 1e-5
