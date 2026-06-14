"""Unit tests for MoE Ecology diagnostic (PRD #10-42, 2026-06-14).

Verifies the ``moe_ecology_number`` formula and the
``MoEEcologyMonitor`` semantics from arXiv:2605.06415 (Zhang 2026):

- E = T·H/(O+B) is computed correctly on synthetic logits.
- Uniform softmax → max entropy → E high.
- Argmax-only → zero entropy → E near 0.
- Increasing B (balance weight) decreases E.
- ``MoEEcologyMonitor`` tracks per-expert utilization EMA, dead-expert
  count, and E trajectory.
- ``FAMECfCCell.moe_ecology_diagnostic()`` returns a valid dict.
"""
import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import MoEEcologyMonitor, moe_ecology_number


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestMoEEcologyNumberFormula:
    def test_uniform_softmax_max_entropy(self) -> None:
        """Uniform g → H = 1 (normalised) → E = T/(O+B)."""
        _seed(0)
        # K=4, uniform g = 0.25 per element.
        g = torch.full((8, 4), 0.25)
        # T=1, O=0, B=0 → E should blow up to 1/eps ≈ 1e8 (no denominator).
        E = moe_ecology_number(g, g, T=1.0, O=0.0, B=0.0)
        # With B=0 and O=0, denom = eps = 1e-8, so E is huge.
        assert float(E.item()) > 1e6

    def test_argmax_zero_entropy(self) -> None:
        """Argmax g (one expert = 1, rest = 0) → H = 0 → E = 0."""
        _seed(1)
        g = torch.zeros(8, 4)
        g[:, 0] = 1.0  # all routed to expert 0
        E = moe_ecology_number(g, g, T=1.0, O=0.0, B=1.0)
        # H = 0, so E = 0 regardless of T, O, B.
        assert abs(float(E.item())) < 1e-5

    def test_balance_weight_denominator(self) -> None:
        """Increasing B decreases E (for fixed H)."""
        _seed(2)
        g = torch.full((8, 4), 0.25)  # uniform
        E_b0 = moe_ecology_number(g, g, T=1.0, B=0.0)
        E_b1 = moe_ecology_number(g, g, T=1.0, B=1.0)
        E_b10 = moe_ecology_number(g, g, T=1.0, B=10.0)
        # All large (B=0 → eps denominator).  Compare B=1 vs B=10.
        assert float(E_b1.item()) > float(E_b10.item())

    def test_temperature_scaling(self) -> None:
        """Doubling T doubles E (for fixed H, O, B)."""
        _seed(3)
        g = torch.full((8, 4), 0.25)
        E_t1 = moe_ecology_number(g, g, T=1.0, O=0.0, B=1.0)
        E_t2 = moe_ecology_number(g, g, T=2.0, O=0.0, B=1.0)
        assert abs(float(E_t2.item()) - 2 * float(E_t1.item())) < 1e-3

    def test_paper_threshold_E_ge_0_5(self) -> None:
        """Demonstrate the paper's E ≥ 0.5 threshold on a healthy config.

        T=1, B=0.001 (our orthogonality λ), uniform routing (H=1) →
        E = 1 / 0.001 = 1000, way above 0.5.  Imbalanced routing (H=0.3)
        with same B → E = 0.3/0.001 = 300, also above 0.5.  Only when
        H drops to ~0.0005 with B=0.001 do we cross below 0.5.
        """
        _seed(4)
        # Healthy: uniform routing.
        g_uniform = torch.full((8, 4), 0.25)
        E_healthy = moe_ecology_number(g_uniform, g_uniform, T=1.0, B=0.001)
        assert float(E_healthy.item()) > 0.5
        # Slightly imbalanced: 60/20/10/10.
        g_skewed = torch.tensor([[0.6, 0.2, 0.1, 0.1]] * 8)
        E_skewed = moe_ecology_number(g_skewed, g_skewed, T=1.0, B=0.001)
        # H for 0.6/0.2/0.1/0.1 ≈ 0.79, so E ≈ 790.  Still above 0.5.
        assert float(E_skewed.item()) > 0.5


class TestMoEEcologyMonitor:
    def test_initial_state(self) -> None:
        """Initial util is uniform 1/K; no E history."""
        _seed(5)
        m = MoEEcologyMonitor(n_experts=4)
        assert torch.allclose(m.util_ema, torch.full((4,), 0.25))
        assert m.E_history == []
        assert m.dead_history == []

    def test_step_updates_ema(self) -> None:
        """``step`` updates utilization EMA in-place, no_grad."""
        _seed(6)
        m = MoEEcologyMonitor(n_experts=3, ema_alpha=1.0)  # alpha=1 = no smoothing
        g = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        info = m.step(g, T=1.0, O=0.0, B=0.0)
        # After 1 step with alpha=1, util_ema = g_mean.
        assert torch.allclose(m.util_ema, torch.tensor([1/3, 1/3, 1/3]), atol=1e-5)
        assert "E" in info and "dead_experts" in info and "utilization" in info
        assert len(m.E_history) == 1

    def test_dead_expert_detection(self) -> None:
        """An expert never routed to in the EMA is flagged as dead."""
        _seed(7)
        m = MoEEcologyMonitor(n_experts=3, ema_alpha=1.0, dead_threshold=0.01)
        g = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        m.step(g, T=1.0)
        # Experts 1 and 2 have 0 utilization.
        assert m.summary()["dead_experts"] == 2

    def test_history_capped_at_1000(self) -> None:
        """``E_history`` and ``dead_history`` cap at 1000 entries."""
        _seed(8)
        m = MoEEcologyMonitor(n_experts=3, ema_alpha=0.1)
        g = torch.full((4, 3), 1/3)
        for _ in range(1500):
            m.step(g, T=1.0, B=0.001)
        assert len(m.E_history) == 1000
        assert len(m.dead_history) == 1000

    def test_reset(self) -> None:
        """``reset`` clears EMA and history."""
        _seed(9)
        m = MoEEcologyMonitor(n_experts=3)
        m.step(torch.full((4, 3), 1/3), T=1.0, B=0.001)
        m.reset()
        assert torch.allclose(m.util_ema, torch.full((3,), 1/3))
        assert m.E_history == []
        assert m.dead_history == []

    def test_buffer_device_propagation(self) -> None:
        """Buffers move with ``.to(device)``."""
        _seed(10)
        m = MoEEcologyMonitor(n_experts=3)
        m.to("cpu")
        assert m.util_ema.device.type == "cpu"


class TestFAMECfCCellEcologyDiagnostic:
    def test_diagnostic_returns_valid_dict(self) -> None:
        """``moe_ecology_diagnostic`` returns E, dead, util."""
        _seed(11)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        cell.eval()  # so update is a no-op
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        # Need a forward first to populate last_g.
        with torch.no_grad():
            cell.forward(x_t, h, dt=1.0)
        diag = cell.moe_ecology_diagnostic(B=0.001)
        assert "E" in diag
        assert "dead_experts" in diag
        assert "utilization" in diag
        assert isinstance(diag["E"], float)
        assert isinstance(diag["dead_experts"], int)
        assert len(diag["utilization"]) == 3

    def test_diagnostic_before_forward_returns_nan(self) -> None:
        """``moe_ecology_diagnostic`` before any forward returns NaN/-1."""
        _seed(12)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        diag = cell.moe_ecology_diagnostic(B=0.001)
        # No last_g set → returns NaN E, dead=-1, empty util.
        assert diag["E"] != diag["E"]  # NaN check
        assert diag["dead_experts"] == -1
        assert diag["utilization"] == []

    def test_diagnostic_with_different_B(self) -> None:
        """Increasing B decreases E (everything else equal)."""
        _seed(13)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        cell.eval()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        d_b001 = cell.moe_ecology_diagnostic(B=0.001)
        d_b1 = cell.moe_ecology_diagnostic(B=1.0)
        # Same last_g, different B → smaller B gives larger E.
        assert d_b001["E"] > d_b1["E"]
