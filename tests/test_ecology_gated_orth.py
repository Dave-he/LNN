"""Unit tests for Ecology-Gated Orth Rescaling (PRD #10-44, 2026-06-15).

Verifies the ``EcologyGatedOrth`` semantics and the
``FAMECfCCell.compute_orth_loss()`` API:

- ``EcologyGatedOrth`` returns scale=1.0 when E > threshold.
- Returns ``lambda_safe / lambda_coeff`` when E < threshold.
- Latched (no hysteresis).
- Respects ``warmup_steps``.
- ``FAMECfCCell.compute_orth_loss(outs, user_lambda)`` returns the
  rescaled loss when the gate has fired.
- Default ``ecology_gated_orth=False`` is back-compat.
- ``moe_ecology_diagnostic`` includes ``ecology_gate_orth`` key.
"""
import numpy as np
import torch

from lnn.core.ecology_gated_balancing import EcologyGatedOrth
from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.orthogonality import orthogonality_loss


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestEcologyGatedOrth:
    def test_initial_state(self) -> None:
        """Initial state: not intervened, no rescaling, no steps."""
        g = EcologyGatedOrth()
        assert g.intervened is False
        assert g.triggered_step == -1
        assert g.last_lambda_scale == 1.0
        assert g.n_steps == 0
        assert g.n_rescaled == 0

    def test_never_rescales_above_threshold(self) -> None:
        """E above E_min for 100 steps → never rescales."""
        g = EcologyGatedOrth(E_min=0.5)
        for i in range(100):
            info = g.step(E=10.0, lambda_coeff=1.0, step_idx=i)
            assert info["lambda_scale"] == 1.0
            assert info["effective_lambda"] == 1.0
            assert info["intervened"] is False
        assert g.n_rescaled == 0

    def test_rescales_on_drop(self) -> None:
        """E drops below E_min at step 5 → rescale kicks in."""
        g = EcologyGatedOrth(E_min=0.5, lambda_safe=0.001)
        for i in range(5):
            info = g.step(E=10.0, lambda_coeff=1.0, step_idx=i)
            assert info["lambda_scale"] == 1.0
        # E drops at step 5.
        info = g.step(E=0.3, lambda_coeff=1.0, step_idx=5)
        assert info["intervened"] is True
        assert info["triggered_step"] == 5
        # Effective lambda = 1.0 * (0.001/1.0) = 0.001.
        assert abs(info["lambda_scale"] - 0.001) < 1e-9
        assert abs(info["effective_lambda"] - 0.001) < 1e-9
        # Subsequent steps: stays fired.
        for i in range(6, 20):
            info = g.step(E=10.0, lambda_coeff=1.0, step_idx=i)
            assert info["intervened"] is True
            assert info["lambda_scale"] == 0.001
        # 1 step below threshold (step 5) — others above.
        assert g.n_below_threshold == 1

    def test_rescale_factor_proportional_to_lambda(self) -> None:
        """Higher user λ → smaller scale factor → same effective λ."""
        g = EcologyGatedOrth(E_min=0.5, lambda_safe=0.001)
        # Force intervention by setting E<threshold on first step.
        for lam, expected_scale in [(1.0, 0.001), (0.1, 0.01), (10.0, 0.0001)]:
            g.reset()
            g.step(E=0.1, lambda_coeff=lam, step_idx=0)
            assert abs(g.last_lambda_scale - expected_scale) < 1e-9

    def test_warmup_steps(self) -> None:
        """During warmup, E below threshold does NOT rescale."""
        g = EcologyGatedOrth(E_min=0.5, warmup_steps=10)
        for i in range(10):
            info = g.step(E=0.1, lambda_coeff=1.0, step_idx=i)
            assert info["intervened"] is False
            assert info["in_warmup"] is True
            assert info["lambda_scale"] == 1.0
        # After warmup, E below threshold should rescale.
        info = g.step(E=0.1, lambda_coeff=1.0, step_idx=10)
        assert info["intervened"] is True
        assert info["in_warmup"] is False
        assert info["lambda_scale"] == 0.001

    def test_reset_clears_state(self) -> None:
        """``reset`` clears intervention and counters."""
        g = EcologyGatedOrth(E_min=0.5)
        g.step(E=0.1, lambda_coeff=1.0, step_idx=0)
        g.reset()
        assert g.intervened is False
        assert g.triggered_step == -1
        assert g.last_lambda_scale == 1.0
        assert g.n_steps == 0
        assert g.n_rescaled == 0

    def test_state_snapshot(self) -> None:
        """``state()`` returns EcologyOrthGateState snapshot."""
        g = EcologyGatedOrth(E_min=0.5, lambda_safe=0.001)
        g.step(E=0.1, lambda_coeff=1.0, step_idx=3)
        st = g.state()
        assert st.intervened is True
        assert st.triggered_step == 3
        assert st.E == 0.1
        assert st.lambda_coeff == 1.0
        assert st.lambda_scale == 0.001
        assert st.effective_lambda == 0.001

    def test_repr_contains_key_state(self) -> None:
        """``__repr__`` mentions intervention state."""
        g = EcologyGatedOrth(E_min=0.5, lambda_safe=0.001)
        rep = repr(g)
        assert "E_min=0.5" in rep
        assert "lambda_safe=0.001" in rep
        assert "intervened=False" in rep
        g.step(E=0.1, lambda_coeff=1.0, step_idx=0)
        assert "intervened=True" in repr(g)


class TestFAMECfCCellEcologyGatedOrth:
    def test_default_backcompat(self) -> None:
        """Default ``ecology_gated_orth=False`` is fully back-compat."""
        _seed(0)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        assert cell.orth_gate is None
        # compute_orth_loss still works (no rescaling).
        outs = [torch.randn(4, 8) for _ in range(3)]
        loss = cell.compute_orth_loss(outs, user_lambda=1.0)
        expected = orthogonality_loss(outs, lambda_coeff=1.0)
        assert abs(float(loss.item()) - float(expected.item())) < 1e-5

    def test_gated_constructor_creates_gate(self) -> None:
        """``ecology_gated_orth=True`` attaches a gate."""
        _seed(1)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_orth=True, ecology_orth_lambda_safe=0.001,
        )
        assert cell.orth_gate is not None
        assert cell.orth_gate.lambda_safe == 0.001

    def test_gated_diagnostic_includes_orth_gate(self) -> None:
        """Diagnostic includes ``ecology_gate_orth`` key when gate is on."""
        _seed(2)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_orth=True,
        )
        cell.eval()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        diag = cell.moe_ecology_diagnostic(B=1.0)
        assert "ecology_gate_orth" in diag
        # E = 1/(1+eps) ≈ 1, well above threshold.
        assert diag["E"] > 0.5
        assert diag["ecology_gate_orth"]["intervened"] is False
        assert diag["ecology_gate_orth"]["lambda_scale"] == 1.0

    def test_gated_rescales_lambda_in_train_mode(self) -> None:
        """In training mode with low E, compute_orth_loss returns scaled loss."""
        _seed(3)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_orth=True, ecology_orth_lambda_safe=0.001,
        )
        cell.train()
        # Forward to populate last_g.
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Diagnostic with high B → low E → fires.
        diag = cell.moe_ecology_diagnostic(B=10.0)
        assert diag["E"] < 0.5
        assert diag["ecology_gate_orth"]["intervened"] is True
        # compute_orth_loss with user_lambda=1.0 → effective 0.001.
        outs = [torch.randn(4, 8) for _ in range(3)]
        loss_gated = cell.compute_orth_loss(outs, user_lambda=1.0)
        loss_unscaled = orthogonality_loss(outs, lambda_coeff=1.0)
        loss_safe = orthogonality_loss(outs, lambda_coeff=0.001)
        # Gated should match safe.
        assert abs(float(loss_gated.item()) - float(loss_safe.item())) < 1e-5
        # And be much smaller than unscaled.
        assert float(loss_gated.item()) < float(loss_unscaled.item()) * 0.01

    def test_gated_does_not_rescale_when_healthy(self) -> None:
        """When E > threshold, compute_orth_loss returns unscaled loss."""
        _seed(4)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_orth=True,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Diagnostic with B=0 → E huge, above threshold.
        diag = cell.moe_ecology_diagnostic(B=0.0)
        assert diag["E"] > 0.5
        assert diag["ecology_gate_orth"]["intervened"] is False
        # compute_orth_loss unchanged.
        outs = [torch.randn(4, 8) for _ in range(3)]
        loss_gated = cell.compute_orth_loss(outs, user_lambda=1.0)
        loss_unscaled = orthogonality_loss(outs, lambda_coeff=1.0)
        assert abs(float(loss_gated.item()) - float(loss_unscaled.item())) < 1e-5

    def test_gated_rescales_to_different_safe_value(self) -> None:
        """User can override ``ecology_orth_lambda_safe``."""
        _seed(5)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_orth=True, ecology_orth_lambda_safe=0.01,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        cell.moe_ecology_diagnostic(B=10.0)  # force gate fire
        outs = [torch.randn(4, 8) for _ in range(3)]
        loss_gated = cell.compute_orth_loss(outs, user_lambda=1.0)
        # Should rescale to 0.01, not 0.001.
        loss_safe = orthogonality_loss(outs, lambda_coeff=0.01)
        assert abs(float(loss_gated.item()) - float(loss_safe.item())) < 1e-5

    def test_compute_orth_loss_zero_lambda(self) -> None:
        """``user_lambda=0`` returns 0 loss (no orth applied)."""
        _seed(6)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_orth=True,
        )
        outs = [torch.randn(4, 8) for _ in range(3)]
        loss = cell.compute_orth_loss(outs, user_lambda=0.0)
        assert float(loss.item()) == 0.0
