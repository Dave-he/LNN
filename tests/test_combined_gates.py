"""Unit tests for CombinedEcologyGate (PRD #10-48, 2026-06-15, round 86).

Verifies the 2-axis adaptive policy that combines:
- EcologyGatedBalancer (round 84, soft φ intervention)
- EcologyGatedOrth (round 85, strong orth intervention)

- CombinedEcologyGate composes both sub-gates; both fire when E<E_min.
- Neither fires when E>threshold (no false positive).
- step() returns phi_intervened, orth_intervened, effective_lambda, etc.
- FAMECfCCell(ecology_combined=True) attaches both sub-gates AND
  the orchestrator, populates the diagnostic with all 3 keys.
- Back-compat: default ecology_combined=False leaves individual
  flags independently controlled.
"""
import numpy as np
import torch

from lnn.core.ecology_gated_balancing import (
    CombinedEcologyGate,
    EcologyGatedBalancer,
    EcologyGatedOrth,
)
from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.orthogonality import orthogonality_loss


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestCombinedEcologyGate:
    def test_initial_state(self) -> None:
        """Initial: neither sub-gate has fired, no rescale."""
        g = CombinedEcologyGate()
        assert g.phi_gate.intervened is False
        assert g.orth_subgate.intervened is False

    def test_never_fires_above_threshold(self) -> None:
        """E above E_min for 50 steps → neither gate fires."""
        g = CombinedEcologyGate(E_min=0.5)
        for i in range(50):
            info = g.step(E=10.0, lambda_coeff=1.0, step_idx=i)
            assert info["phi_intervened"] is False
            assert info["orth_intervened"] is False
            assert info["effective_lambda"] == 1.0
            assert info["lambda_scale"] == 1.0

    def test_both_gates_fire_on_drop(self) -> None:
        """E drops below threshold → both sub-gates fire."""
        g = CombinedEcologyGate(E_min=0.5, lambda_safe=0.001)
        for i in range(5):
            info = g.step(E=10.0, lambda_coeff=1.0, step_idx=i)
            assert info["phi_intervened"] is False
        # E drops at step 5.
        info = g.step(E=0.1, lambda_coeff=1.0, step_idx=5)
        assert info["phi_intervened"] is True
        assert info["orth_intervened"] is True
        # Orth: effective_lambda = 1.0 * 0.001 = 0.001
        assert abs(info["effective_lambda"] - 0.001) < 1e-9
        assert abs(info["lambda_scale"] - 0.001) < 1e-9
        # phi_enabled mirrors phi_intervened.
        assert info["phi_enabled"] is True

    def test_latched(self) -> None:
        """Once fired, both gates stay fired (no hysteresis)."""
        g = CombinedEcologyGate(E_min=0.5)
        g.step(E=0.1, lambda_coeff=1.0, step_idx=0)
        for i in range(1, 30):
            info = g.step(E=10.0, lambda_coeff=1.0, step_idx=i)
            assert info["phi_intervened"] is True
            assert info["orth_intervened"] is True

    def test_warmup(self) -> None:
        """During warmup, both gates stay silent even at low E."""
        g = CombinedEcologyGate(E_min=0.5, warmup_steps=10)
        for i in range(10):
            info = g.step(E=0.1, lambda_coeff=1.0, step_idx=i)
            assert info["phi_intervened"] is False
            assert info["orth_intervened"] is False
            assert info["in_warmup"] is True
        info = g.step(E=0.1, lambda_coeff=1.0, step_idx=10)
        assert info["phi_intervened"] is True
        assert info["orth_intervened"] is True
        assert info["in_warmup"] is False

    def test_triggered_step_is_earliest(self) -> None:
        """``triggered_step`` is the earliest of φ or orth."""
        g = CombinedEcologyGate(E_min=0.5)
        # Force fire on step 0.
        info = g.step(E=0.1, lambda_coeff=1.0, step_idx=0)
        # Both fire on the same step → triggered = 0.
        assert info["triggered_step"] == 0

    def test_state_snapshot(self) -> None:
        """``state()`` returns CombinedEcologyGateState dataclass."""
        g = CombinedEcologyGate(E_min=0.5, lambda_safe=0.001)
        g.step(E=0.1, lambda_coeff=1.0, step_idx=3)
        st = g.state()
        assert st.phi_intervened is True
        assert st.orth_intervened is True
        assert st.triggered_step == 3
        assert abs(st.effective_lambda - 0.001) < 1e-9
        assert abs(st.lambda_scale - 0.001) < 1e-9

    def test_reset(self) -> None:
        """``reset()`` clears both sub-gates."""
        g = CombinedEcologyGate(E_min=0.5)
        g.step(E=0.1, lambda_coeff=1.0, step_idx=0)
        g.reset()
        assert g.phi_gate.intervened is False
        assert g.orth_subgate.intervened is False

    def test_repr(self) -> None:
        """``__repr__`` shows both gate states."""
        g = CombinedEcologyGate(E_min=0.5)
        rep = repr(g)
        assert "phi=False" in rep
        assert "orth=False" in rep
        g.step(E=0.1, lambda_coeff=1.0, step_idx=0)
        rep2 = repr(g)
        assert "phi=True" in rep2
        assert "orth=True" in rep2

    def test_composes_sub_gates(self) -> None:
        """CombinedEcologyGate owns the sub-gates (not just runs them ad-hoc)."""
        g = CombinedEcologyGate()
        assert isinstance(g.phi_gate, EcologyGatedBalancer)
        assert isinstance(g.orth_subgate, EcologyGatedOrth)


class TestFAMECfCCellCombinedGate:
    def test_default_backcompat(self) -> None:
        """Default ``ecology_combined=False``: all sub-gates are None."""
        _seed(0)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        assert cell.combined_gate is None
        assert cell.ecology_gate is None
        assert cell.orth_gate is None

    def test_combined_attaches_all_three(self) -> None:
        """``ecology_combined=True`` attaches orchestrator + both sub-gates."""
        _seed(1)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_combined=True,
        )
        assert cell.combined_gate is not None
        assert cell.ecology_gate is not None
        assert cell.orth_gate is not None
        # The orchestrator's sub-gates are the SAME instances attached
        # to the cell (so the diagnostic sees the same fire state).
        assert cell.combined_gate.phi_gate is cell.ecology_gate
        assert cell.combined_gate.orth_subgate is cell.orth_gate

    def test_combined_diagnostic_includes_all_keys(self) -> None:
        """Diagnostic populates ``ecology_gate``, ``ecology_gate_orth``,
        AND ``ecology_gate_combined``."""
        _seed(2)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_combined=True,
        )
        cell.eval()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        diag = cell.moe_ecology_diagnostic(B=1.0)
        assert "ecology_gate" in diag
        assert "ecology_gate_orth" in diag
        assert "ecology_gate_combined" in diag
        # E is huge (1/(1+eps)) → no fire.
        assert diag["ecology_gate"]["intervened"] is False
        assert diag["ecology_gate_orth"]["intervened"] is False
        assert diag["ecology_gate_combined"]["phi_intervened"] is False
        assert diag["ecology_gate_combined"]["orth_intervened"] is False

    def test_combined_does_not_fire_when_healthy(self) -> None:
        """When E > threshold, the combined gate does not fire."""
        _seed(3)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_combined=True,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # B=0 → E huge → no fire.
        diag = cell.moe_ecology_diagnostic(B=0.0)
        assert diag["ecology_gate_combined"]["phi_intervened"] is False
        assert diag["ecology_gate_combined"]["orth_intervened"] is False
        assert diag["ecology_gate_combined"]["effective_lambda"] == 0.0

    def test_combined_fires_coactively_in_train(self) -> None:
        """In training mode with low E, both sub-gates fire coactively."""
        _seed(4)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_combined=True, ecology_orth_lambda_safe=0.001,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # B=10 → E tiny → both fire.
        diag = cell.moe_ecology_diagnostic(B=10.0)
        assert diag["E"] < 0.5
        assert diag["ecology_gate_combined"]["phi_intervened"] is True
        assert diag["ecology_gate_combined"]["orth_intervened"] is True
        # Orth scales user λ=10.0 down to 0.001.
        assert abs(diag["ecology_gate_combined"]["effective_lambda"] - 0.001) < 1e-9

    def test_combined_compute_orth_loss_uses_rescaled_lambda(self) -> None:
        """``compute_orth_loss`` picks up the orth gate's rescaling."""
        _seed(5)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_combined=True, ecology_orth_lambda_safe=0.001,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Force orth fire with high B.
        cell.moe_ecology_diagnostic(B=10.0)
        outs = [torch.randn(4, 8) for _ in range(3)]
        loss = cell.compute_orth_loss(outs, user_lambda=10.0)
        # Should be rescaled to 0.001.
        loss_safe = orthogonality_loss(outs, lambda_coeff=0.001)
        assert abs(float(loss.item()) - float(loss_safe.item())) < 1e-5

    def test_combined_individual_flags_dont_fight(self) -> None:
        """Setting ``ecology_combined=True`` does NOT double-fire."""
        _seed(6)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_combined=True,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Single diagnostic call should fire each sub-gate exactly once.
        diag = cell.moe_ecology_diagnostic(B=10.0)
        # The orchestrator's sub-gates share state with the cell's gates.
        assert diag["ecology_gate"]["intervened"] is True
        assert diag["ecology_gate_orth"]["intervened"] is True
        # No double-counting.
        assert diag["ecology_gate_combined"]["phi_intervened"] is True
        assert diag["ecology_gate_combined"]["orth_intervened"] is True
