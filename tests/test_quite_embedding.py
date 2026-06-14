"""Round 102 — Tests for QuITE Query-Based Irregular TS Embedding (PRD #10-64)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.quite_embedding import (
    QueryIrregularEmbedding,
    apply_quite_embedding,
    quite_baseline_modes,
)


class TestQueryIrregularEmbedding:
    """Tests for the QueryIrregularEmbedding module."""

    def test_module_initialization(self):
        """Module stores n_queries, d_model, n_heads."""
        m = QueryIrregularEmbedding(d_input=1, n_queries=4, d_model=16, n_heads=4)
        assert m.n_queries == 4
        assert m.d_model == 16
        assert m.n_heads == 4

    def test_invalid_n_queries_raises(self):
        """n_queries < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_queries"):
            QueryIrregularEmbedding(n_queries=0, d_model=16)

    def test_invalid_d_model_n_heads_raises(self):
        """d_model not divisible by n_heads should raise ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=15, n_heads=4)

    def test_output_shape_correct(self):
        """Output shape is (B, n_queries, d_model)."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=3, n_queries=8, d_model=16)
        obs = torch.randn(2, 20, 3)
        times = torch.linspace(0, 1, 20).unsqueeze(0).expand(2, -1)
        out = m(obs, times)
        assert out.shape == (2, 8, 16)

    def test_handles_variable_length_no_mask(self):
        """Without mask, all positions are valid (uniform attention)."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=8)
        obs = torch.randn(3, 15, 2)
        times = torch.linspace(0, 1, 15).unsqueeze(0).expand(3, -1)
        out = m(obs, times, mask=None)
        assert out.shape == (3, 4, 8)
        # Output should not be all-zero
        assert out.abs().sum() > 0

    def test_handles_mask(self):
        """Mask=0 positions are ignored (should not affect output)."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=8)
        obs = torch.randn(1, 10, 2)
        times = torch.linspace(0, 1, 10).unsqueeze(0)
        # All-ones mask
        mask_full = torch.ones(1, 10, dtype=torch.bool)
        out_full = m(obs, times, mask=mask_full)
        # Mask with first 5 positions zeroed
        mask_half = torch.zeros(1, 10, dtype=torch.bool)
        mask_half[:, 5:] = True
        out_half = m(obs, times, mask=mask_half)
        # Outputs should differ (masking changes the aggregation)
        assert not torch.allclose(out_full, out_half, atol=1e-4)

    def test_handles_nan_values(self):
        """NaN observations should be treated as missing."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=8)
        obs_clean = torch.randn(1, 10, 2)
        obs_nan = obs_clean.clone()
        obs_nan[0, :5] = float("nan")  # First half is NaN
        times = torch.linspace(0, 1, 10).unsqueeze(0)
        out_clean = m(obs_clean, times)
        out_nan = m(obs_nan, times)
        # Outputs should differ (NaN masking takes effect)
        assert not torch.allclose(out_clean, out_nan, atol=1e-4)

    def test_gradient_flows(self):
        """Backward pass through module works (gradient non-zero)."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=8)
        obs = torch.randn(2, 10, 2, requires_grad=True)
        times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
        out = m(obs, times)
        loss = out.sum()
        loss.backward()
        # At least one parameter has non-zero gradient
        has_grad = False
        for p in m.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_time_embedding_separate(self):
        """Different time stamps produce different outputs (all else equal)."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=8)
        obs = torch.randn(1, 10, 2)
        times_a = torch.linspace(0, 1, 10).unsqueeze(0)
        times_b = torch.linspace(0, 2, 10).unsqueeze(0)  # different range
        out_a = m(obs, times_a)
        out_b = m(obs, times_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_query_diversity(self):
        """Different queries should produce different outputs (not collapsed)."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=3, n_queries=8, d_model=16)
        obs = torch.randn(1, 20, 3)
        times = torch.linspace(0, 1, 20).unsqueeze(0)
        out = m(obs, times)  # (1, 8, 16)
        queries_out = out.squeeze(0)  # (8, 16)
        # Pairwise distances between queries should be > 0
        diffs = queries_out.unsqueeze(0) - queries_out.unsqueeze(1)
        dists = (diffs ** 2).sum(dim=-1)
        # Off-diagonal distances should be > 0
        mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
        off_diag_dists = dists[mask]
        assert off_diag_dists.min() > 0

    def test_apply_quite_embedding_wrapper(self):
        """apply_quite_embedding is a thin wrapper around module."""
        torch.manual_seed(0)
        m = QueryIrregularEmbedding(d_input=2, n_queries=4, d_model=8)
        obs = torch.randn(1, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0)
        out_module = m(obs, times, None)
        out_wrapper = apply_quite_embedding(obs, times, None, m)
        assert torch.allclose(out_module, out_wrapper)


class TestQuiteBaselineModes:
    """Tests for quite_baseline_modes (mean/concat/add)."""

    def test_mean_mode_shape(self):
        """Mean mode outputs (B, D)."""
        obs = torch.randn(2, 10, 3)
        times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
        out = quite_baseline_modes(obs, times, None, mode="mean")
        assert out.shape == (2, 3)

    def test_concat_mode_shape(self):
        """Concat mode outputs (B, D+1)."""
        obs = torch.randn(2, 10, 3)
        times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
        out = quite_baseline_modes(obs, times, None, mode="concat")
        assert out.shape == (2, 4)  # D+1

    def test_add_mode_shape(self):
        """Add mode outputs (B, D)."""
        obs = torch.randn(2, 10, 3)
        times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
        out = quite_baseline_modes(obs, times, None, mode="add")
        assert out.shape == (2, 3)

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        obs = torch.randn(1, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0)
        with pytest.raises(ValueError, match="mean/concat/add"):
            quite_baseline_modes(obs, times, None, mode="invalid")

    def test_mean_ignores_nan(self):
        """Mean mode should ignore NaN observations."""
        obs_clean = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        obs_nan = obs_clean.clone()
        obs_nan[0, 0] = float("nan")
        times = torch.tensor([[0.0, 0.5, 1.0]])
        out_clean = quite_baseline_modes(obs_clean, times, None, mode="mean")
        out_nan = quite_baseline_modes(obs_nan, times, None, mode="mean")
        # NaN-aware mean should give (3+5)/2=4, (4+6)/2=5
        expected = torch.tensor([[4.0, 5.0]])
        assert torch.allclose(out_nan, expected, atol=1e-5)
        assert not torch.allclose(out_clean, out_nan, atol=1e-5)

    def test_concat_uses_last_valid(self):
        """Concat mode should use the last valid observation."""
        obs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        times = torch.tensor([[0.1, 0.5, 0.9]])
        out = quite_baseline_modes(obs, times, None, mode="concat")
        # Last valid is [5, 6] at time 0.9
        expected = torch.tensor([[5.0, 6.0, 0.9]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_add_combines_value_and_time(self):
        """Add mode should combine value with time embedding (mean)."""
        obs = torch.ones(1, 5, 2)  # all-1s
        times = torch.linspace(0, 1, 5).unsqueeze(0)
        out = quite_baseline_modes(obs, times, None, mode="add")
        # The output should be different from a pure mean (which would be 1.0)
        # because the time embedding contributes
        assert out.shape == (1, 2)
        # At least one component should differ from 1.0
        assert (out != 1.0).any()


class TestQuiteEmbeddingExports:
    """Verify exports are correct."""

    def test_exports(self):
        from lnn.core import (
            QueryIrregularEmbedding,
            apply_quite_embedding,
            quite_baseline_modes,
        )
        assert callable(QueryIrregularEmbedding)
        assert callable(apply_quite_embedding)
        assert callable(quite_baseline_modes)
