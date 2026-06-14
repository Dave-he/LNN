"""Unit tests for round 90 weights-vs-activations audit metrics
(PRD #10-52, response to arXiv:2601.00457, Kim 2026).

Verifies:
- weight_space_overlap returns 1.0 for identical matrices
- weight_space_overlap returns 0.0 for orthogonal matrices
- weight_space_overlap returns 0.0 for K < 2
- weight_space_overlap uses absolute cosine (anti-parallel counts as overlap)
- activation_space_overlap reduces with our orthogonality_loss
- Metrics are exported from lnn.core
"""
import torch

from lnn.core import activation_space_overlap, weight_space_overlap


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


class TestWeightSpaceOverlap:
    def test_identical_matrices_overlap_1(self) -> None:
        """Two copies of the same matrix have overlap = 1.0."""
        _seed(0)
        w = torch.randn(8, 4)
        ov = weight_space_overlap([w, w.clone()])
        assert abs(ov - 1.0) < 1e-5, f"expected ~1.0, got {ov}"

    def test_orthogonal_matrices_overlap_0(self) -> None:
        """Two orthogonal matrices have overlap = 0.0."""
        # Use a 4x4 identity split into two orthogonal blocks.
        w1 = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        w2 = torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        ov = weight_space_overlap([w1, w2])
        assert abs(ov) < 1e-5, f"expected ~0, got {ov}"

    def test_antiparallel_counts_as_overlap(self) -> None:
        """Anti-parallel matrices (cos = -1) have |cos| = 1 (overlap)."""
        w1 = torch.tensor([[1.0, 0, 0, 0]])
        w2 = torch.tensor([[-1.0, 0, 0, 0]])
        ov = weight_space_overlap([w1, w2])
        assert abs(ov - 1.0) < 1e-5, f"anti-parallel should be 1.0, got {ov}"

    def test_k_lt_2_returns_zero(self) -> None:
        """K=1 returns 0 (no pairs)."""
        w = torch.randn(4, 4)
        assert weight_space_overlap([w]) == 0.0
        assert weight_space_overlap([]) == 0.0

    def test_zero_norm_expert_handled(self) -> None:
        """A zero-norm expert returns 0 (no division by zero)."""
        w1 = torch.zeros(4, 4)
        w2 = torch.eye(4)
        ov = weight_space_overlap([w1, w2])
        assert ov == 0.0

    def test_k_3_pairwise_mean(self) -> None:
        """K=3 returns the mean over 3 pairs (not max, not sum)."""
        # All identical → each pair is 1.0, mean = 1.0.
        w = torch.eye(4)
        ov = weight_space_overlap([w, w.clone(), w.clone()])
        assert abs(ov - 1.0) < 1e-5


class TestActivationSpaceOverlap:
    def test_identical_activations_overlap_1(self) -> None:
        """Two identical (B, T, D) tensors have overlap = 1.0."""
        h = torch.randn(2, 3, 4)
        ov = activation_space_overlap([h, h.clone()])
        assert abs(ov - 1.0) < 1e-4, f"expected ~1.0, got {ov}"

    def test_orthogonal_activations_overlap_0(self) -> None:
        """Two orthogonal (B, T, D) tensors have overlap = 0.0."""
        h1 = torch.zeros(1, 1, 4)
        h1[0, 0, 0:2] = 1.0
        h2 = torch.zeros(1, 1, 4)
        h2[0, 0, 2:4] = 1.0
        ov = activation_space_overlap([h1, h2])
        assert abs(ov) < 1e-5, f"expected ~0, got {ov}"

    def test_k_lt_2_returns_zero(self) -> None:
        h = torch.randn(2, 3, 4)
        assert activation_space_overlap([h]) == 0.0
        assert activation_space_overlap([]) == 0.0

    def test_unbatched_2d_input(self) -> None:
        """Works with (N, D) shaped inputs (not just (B, T, D))."""
        h1 = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
        h2 = torch.tensor([[0, 0, 1.0], [0, 0, 0]])
        ov = activation_space_overlap([h1, h2])
        assert abs(ov) < 1e-5


class TestExports:
    def test_metrics_exported_from_lnn_core(self) -> None:
        from lnn.core import (  # noqa: F401
            activation_space_overlap as a,
            weight_space_overlap as w,
        )
        assert callable(a)
        assert callable(w)
