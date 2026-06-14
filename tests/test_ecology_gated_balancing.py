"""Unit tests for Ecology-Gated φ-Balancing (PRD #10-43, 2026-06-14).

Verifies the ``EcologyGatedBalancer`` semantics and the
``FAMECfCCell(ecology_gated_balancing=True)`` auto-attach behaviour:

- Gate never fires when E > threshold.
- Gate fires exactly once when E first drops below threshold.
- Stays fired (hysteresis-free) after that.
- Respects ``warmup_steps``.
- Auto-attaches a ``PhiBalancer`` to the cell on first fire.
- Default ``ecology_gated_balancing=False`` is fully back-compat
  (no ``ecology_gate`` attribute, no auto-attach).
"""
import numpy as np
import torch

from lnn.core.ecology_gated_balancing import EcologyGatedBalancer
from lnn.core.fame_cfc import FAMECfCCell


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestEcologyGatedBalancer:
    def test_initial_state(self) -> None:
        """Initial state: not intervened, no trigger, no steps."""
        g = EcologyGatedBalancer()
        assert g.intervened is False
        assert g.triggered_step == -1
        assert g.n_steps == 0
        assert g.n_below_threshold == 0

    def test_never_fires_above_threshold(self) -> None:
        """E above E_min for 100 steps → never intervenes."""
        g = EcologyGatedBalancer(E_min=0.5)
        for i in range(100):
            info = g.step(E=10.0, B_active=0.0, step_idx=i)
            assert info["intervened"] is False
            assert info["triggered_step"] == -1
            assert info["below_threshold"] is False
        assert g.n_below_threshold == 0

    def test_fires_once_on_drop(self) -> None:
        """E drops below E_min at step 5 → fires at step 5, stays fired."""
        g = EcologyGatedBalancer(E_min=0.5)
        for i in range(5):
            info = g.step(E=10.0, B_active=0.0, step_idx=i)
            assert info["intervened"] is False
        # E drops to 0.3 at step 5.
        info = g.step(E=0.3, B_active=1.0, step_idx=5)
        assert info["intervened"] is True
        assert info["triggered_step"] == 5
        assert info["below_threshold"] is True
        # Subsequent steps: stays fired.
        for i in range(6, 20):
            info = g.step(E=10.0, B_active=0.0, step_idx=i)
            assert info["intervened"] is True
            assert info["triggered_step"] == 5
        assert g.n_below_threshold == 1  # only one step below threshold

    def test_warmup_steps(self) -> None:
        """During warmup, E below threshold does NOT fire."""
        g = EcologyGatedBalancer(E_min=0.5, warmup_steps=10)
        for i in range(10):
            info = g.step(E=0.1, B_active=1.0, step_idx=i)
            assert info["intervened"] is False
            assert info["in_warmup"] is True
            assert info["below_threshold"] is False
        # After warmup, E below threshold should now fire.
        info = g.step(E=0.1, B_active=1.0, step_idx=10)
        assert info["intervened"] is True
        assert info["triggered_step"] == 10
        assert info["in_warmup"] is False

    def test_reset_clears_state(self) -> None:
        """``reset`` clears intervention state and counters."""
        g = EcologyGatedBalancer(E_min=0.5)
        g.step(E=0.1, B_active=1.0, step_idx=0)
        g.reset()
        assert g.intervened is False
        assert g.triggered_step == -1
        assert g.n_steps == 0
        assert g.n_below_threshold == 0

    def test_state_snapshot(self) -> None:
        """``state()`` returns EcologyGateState snapshot."""
        g = EcologyGatedBalancer(E_min=0.5)
        g.step(E=0.1, B_active=1.0, step_idx=3)
        st = g.state()
        assert st.intervened is True
        assert st.triggered_step == 3
        assert st.E == 0.1
        assert st.B_active == 1.0

    def test_repr_contains_key_state(self) -> None:
        """``__repr__`` mentions intervention state."""
        g = EcologyGatedBalancer(E_min=0.5)
        rep = repr(g)
        assert "E_min=0.5" in rep
        assert "intervened=False" in rep
        g.step(E=0.1, B_active=1.0, step_idx=0)
        assert "intervened=True" in repr(g)


class TestFAMECfCCellEcologyGatedBalancing:
    def test_default_backcompat(self) -> None:
        """Default ``ecology_gated_balancing=False`` is fully back-compat."""
        _seed(0)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        # No ecology_gate attribute (or it's None).
        assert cell.ecology_gate is None
        diag = cell.moe_ecology_diagnostic(B=0.001)
        # Diagnostic dict has no "ecology_gate" key.
        assert "ecology_gate" not in diag

    def test_gated_constructor_creates_gate(self) -> None:
        """``ecology_gated_balancing=True`` attaches a gate."""
        _seed(1)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_balancing=True, ecology_E_min=0.5,
        )
        assert cell.ecology_gate is not None
        assert cell.ecology_gate.E_min == 0.5

    def test_gated_fires_after_forward(self) -> None:
        """After forward + diagnostic with low E, gate fires."""
        _seed(2)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_balancing=True, ecology_E_min=0.5,
            ecology_warmup_steps=0,
        )
        cell.eval()  # so we can simulate without autograd issues
        # No balancer yet.
        assert cell.balancer is None
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Diagnostic with high B → low E.
        diag = cell.moe_ecology_diagnostic(B=10.0)
        # E should be small, gate should fire.
        assert diag["E"] < 0.5
        assert "ecology_gate" in diag
        assert diag["ecology_gate"]["intervened"] is True
        # In eval mode, balancer is NOT auto-attached (only in training).
        assert cell.balancer is None

    def test_gated_auto_attaches_balancer_in_train_mode(self) -> None:
        """In training mode, gate auto-attaches a PhiBalancer."""
        _seed(3)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_balancing=True, ecology_E_min=0.5,
        )
        cell.train()
        assert cell.balancer is None
        # Forward first to populate last_g.
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Diagnostic with high B → low E → fires.
        diag = cell.moe_ecology_diagnostic(B=10.0)
        assert diag["ecology_gate"]["intervened"] is True
        # In training mode, balancer should now be auto-attached.
        assert cell.balancer is not None
        from lnn.core.phi_balancing import PhiBalancer
        assert isinstance(cell.balancer, PhiBalancer)

    def test_gated_does_not_attach_when_healthy(self) -> None:
        """When E > threshold, gate does not fire and balancer is None."""
        _seed(4)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_balancing=True, ecology_E_min=0.5,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Diagnostic with B=0 → E ~ 1/eps (huge), above threshold.
        diag = cell.moe_ecology_diagnostic(B=0.0)
        assert diag["E"] > 0.5
        assert diag["ecology_gate"]["intervened"] is False
        assert cell.balancer is None

    def test_warmup_delays_intervention(self) -> None:
        """``warmup_steps`` delays auto-attach even when E < threshold."""
        _seed(5)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_gated_balancing=True, ecology_E_min=0.5,
            ecology_warmup_steps=2,
        )
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # First diagnostic at step 1 (in warmup): no intervention.
        diag = cell.moe_ecology_diagnostic(B=10.0)
        assert diag["ecology_gate"]["intervened"] is False
        assert diag["ecology_gate"]["in_warmup"] is True
        # Step bumps to 2; still in warmup.
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        diag = cell.moe_ecology_diagnostic(B=10.0)
        # Actually step_idx goes 1 (first fwd) → 2 (second fwd).  Warmup
        # is 0..1 inclusive (warmup_steps=2 means steps 0,1 are warmup).
        # After 2 forwards, step_idx == 2 == warmup_steps, so not in warmup.
        assert diag["ecology_gate"]["in_warmup"] is False
        assert diag["ecology_gate"]["intervened"] is True
        assert cell.balancer is not None
