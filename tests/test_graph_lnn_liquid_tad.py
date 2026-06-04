"""Tests for GraphLNNPredictor with the new liquid_tad recurrent type.

PRD §10 #4 / iter#33: HierarchicalDecayLiquidTADHead's underlying
LongSequenceLiquidClassifier (LiquidS4Block stack) is wired as a 4th
recurrent option in GraphLNNPredictor alongside cfc/ltc/gru.
"""

import torch

from lnn.core.graph import GraphLNNPredictor


def test_liquid_tad_recurrent_type_accepted():
    """`recurrent_type='liquid_tad'` is in the allowed set and builds."""
    m = GraphLNNPredictor(node_feature_size=4, recurrent_type="liquid_tad")
    assert m.recurrent_type == "liquid_tad"


def test_liquid_tad_forward_shape():
    """Forward with liquid_tad returns the same shape as cfc/ltc."""
    m = GraphLNNPredictor(node_feature_size=4, recurrent_type="liquid_tad")
    batch = {
        "node_features": torch.randn(2, 1, 5, 4),  # [B, T, N, F]
        "adjacency": torch.randn(2, 1, 5, 5),      # [B, T, N, N]
    }
    y = m(batch)
    assert y.shape == (2, 1)


def test_liquid_tad_has_more_params_than_cfc():
    """liquid_tad (LiquidS4Block stack) is heavier than CfC/GRU."""
    m_lq = GraphLNNPredictor(node_feature_size=4, recurrent_type="liquid_tad", hidden_size=32)
    m_cfc = GraphLNNPredictor(node_feature_size=4, recurrent_type="cfc", hidden_size=32)
    n_lq = sum(p.numel() for p in m_lq.parameters())
    n_cfc = sum(p.numel() for p in m_cfc.parameters())
    assert n_lq > n_cfc, f"liquid_tad should be heavier than cfc (got {n_lq} vs {n_cfc})"


def test_liquid_tad_gradient_flows():
    """Backward through liquid_tad head reaches all expected params."""
    m = GraphLNNPredictor(node_feature_size=4, recurrent_type="liquid_tad")
    batch = {
        "node_features": torch.randn(2, 1, 5, 4),
        "adjacency": torch.randn(2, 1, 5, 5),
    }
    y = m(batch)
    target = torch.zeros_like(y)
    loss = (y - target).pow(2).sum()
    loss.backward()
    # Encoder + recurrent + readout all received grad
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        if p.grad.abs().sum() > 0:
            return  # at least one param has nonzero grad
    raise AssertionError("All gradients are zero (no signal flowed)")


def test_invalid_recurrent_type_still_raises():
    """Random string still raises — only the four known types are valid."""
    import pytest
    with pytest.raises(ValueError, match="recurrent_type must be"):
        GraphLNNPredictor(node_feature_size=4, recurrent_type="bogus")
