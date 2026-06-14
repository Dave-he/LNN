"""Unit tests for round 94 effective rank helper (PRD #10-56, response
to arXiv:2606.00243, Williams/Payeur/Lajoie, ICML 2026).

Verifies:
- rank-1 matrix gives eff_rank = 1
- identity matrix gives eff_rank = min(m, n)
- full-rank random matrix gives eff_rank ≈ min(m, n)
- 2D-only (raises on 1D/3D)
- mean_effective_rank aggregates correctly
- effective_rank_trajectory works on hidden state matrices
- rank_summary combines both
- zero matrix returns 0
"""
import torch

from lnn.core.effective_rank import (
    effective_rank,
    effective_rank_trajectory,
    mean_effective_rank,
    rank_summary,
)


class TestEffectiveRank:
    def test_rank1_matrix_gives_1(self) -> None:
        """A rank-1 matrix has only one nonzero singular value, so eff_rank = 1."""
        u = torch.randn(5, 1)
        v = torch.randn(1, 4)
        W = u @ v
        assert abs(effective_rank(W) - 1.0) < 1e-4

    def test_identity_matrix_gives_full_rank(self) -> None:
        """Identity matrix is full rank: eff_rank = min(m, n)."""
        W = torch.eye(6)
        er = effective_rank(W)
        assert abs(er - 6.0) < 1e-4

    def test_full_rank_random_gives_full_rank(self) -> None:
        """Random full-rank matrix: eff_rank is high (but bounded by Marčenko-Pastur)."""
        torch.manual_seed(0)
        W = torch.randn(10, 8)
        er = effective_rank(W)
        # Random rectangular Gaussian has eff_rank close to min(m,n) but
        # bounded by the Marčenko-Pastur distribution. For 10x8, the
        # expected eff_rank is around 5-7. We just check it's substantially
        # greater than 1.
        assert er > 4.0, f"expected eff_rank > 4, got {er}"

    def test_zero_matrix_gives_zero(self) -> None:
        W = torch.zeros(4, 4)
        assert effective_rank(W) == 0.0

    def test_diagonal_with_one_dominant_singular(self) -> None:
        """Diagonal with one large and rest tiny → eff_rank ≈ 1."""
        W = torch.diag(torch.tensor([10.0, 0.001, 0.001, 0.001]))
        er = effective_rank(W)
        assert er < 1.1, f"expected eff_rank ≈ 1, got {er}"

    def test_uniform_diagonal_full_rank(self) -> None:
        """All singular values equal → eff_rank = min(m, n)."""
        W = torch.diag(torch.ones(5))
        er = effective_rank(W)
        assert abs(er - 5.0) < 1e-4

    def test_rejects_1d(self) -> None:
        try:
            effective_rank(torch.randn(10))
            assert False, "should have raised"
        except ValueError:
            pass

    def test_rejects_3d(self) -> None:
        try:
            effective_rank(torch.randn(2, 3, 4))
            assert False, "should have raised"
        except ValueError:
            pass

    def test_rank_lower_than_algebraic(self) -> None:
        """A matrix that's algebraically rank-2 has eff_rank between 1 and 2."""
        u = torch.randn(6, 2)
        v = torch.randn(2, 5)
        W = u @ v
        er = effective_rank(W)
        assert 1.0 < er < 2.05, f"expected eff_rank in (1, 2), got {er}"


class TestMeanEffectiveRank:
    def test_empty_list_returns_zero(self) -> None:
        assert mean_effective_rank([]) == 0.0

    def test_single_matrix(self) -> None:
        W = torch.eye(4)
        assert abs(mean_effective_rank([W]) - 4.0) < 1e-4

    def test_mean_of_two(self) -> None:
        W1 = torch.eye(4)  # eff_rank = 4
        W2 = torch.zeros(3, 3)  # eff_rank = 0
        er = mean_effective_rank([W1, W2])
        assert abs(er - 2.0) < 1e-4


class TestEffectiveRankTrajectory:
    def test_constant_trajectory_is_rank1(self) -> None:
        """A trajectory that's constant has eff_rank = 1."""
        states = torch.ones(50, 8)
        er = effective_rank_trajectory(states)
        assert abs(er - 1.0) < 1e-3

    def test_diverse_trajectory_is_high_rank(self) -> None:
        """A trajectory that visits many distinct states has high eff_rank."""
        torch.manual_seed(0)
        states = torch.randn(100, 8)
        er = effective_rank_trajectory(states)
        # Marčenko-Pastur: random (T, d) trajectory with T >> d has
        # eff_rank ≈ d. For 100x8, expect eff_rank around 4-7.
        assert er > 3.0, f"expected eff_rank > 3, got {er}"

    def test_accepts_1d(self) -> None:
        """A 1D state is treated as (1, d) trajectory."""
        states = torch.randn(8)
        er = effective_rank_trajectory(states)
        # (1, 8) has eff_rank = 1 (single row).
        assert er < 1.5

    def test_rejects_3d(self) -> None:
        try:
            effective_rank_trajectory(torch.randn(2, 3, 4))
            assert False, "should have raised"
        except ValueError:
            pass


class TestRankSummary:
    def test_combines_weights_and_states(self) -> None:
        weights = [torch.eye(4), torch.eye(3)]
        states = torch.randn(50, 4)
        summary = rank_summary(weights, states)
        assert "mean_weight_eff_rank" in summary
        assert "per_weight_eff_rank" in summary
        assert "hidden_eff_rank" in summary
        assert len(summary["per_weight_eff_rank"]) == 2

    def test_states_optional(self) -> None:
        weights = [torch.eye(4)]
        summary = rank_summary(weights, states=None)
        assert summary["hidden_eff_rank"] is None

    def test_empty_weights_raises(self) -> None:
        try:
            rank_summary([])
            assert False, "should have raised"
        except ValueError:
            pass


class TestExports:
    def test_helpers_exported(self) -> None:
        from lnn.core import (  # noqa: F401
            effective_rank as er,
            effective_rank_trajectory as ert,
            mean_effective_rank as mer,
            rank_summary as rs,
        )
        assert callable(er)
        assert callable(ert)
        assert callable(mer)
        assert callable(rs)
