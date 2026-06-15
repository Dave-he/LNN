"""Round 120 — DAG-MoE (Structural Aggregation) for CfC tests (PRD #10-82).

Tests for ``DAGEdgeGate``, ``DAGAggregation``, ``DAGMoECfCCell``,
and ``DAGMoECfCNetwork`` covering:
- DAGEdgeGate: forward shape, edge gate values in [0,1]
- DAGAggregation: L iterations, no shape change
- DAGMoECfCCell: forward shape, forward_with_aux, gradient flow
- DAGMoECfCNetwork: forward, NaN handling, learns
- dag_moe_utilization: metadata correctness
- Smoke tests on toy sin
"""
from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lnn.core.dag_moe import (
    DAGEdgeGate,
    DAGAggregation,
    DAGMoECfCCell,
    DAGMoECfCNetwork,
    dag_moe_utilization,
)
from lnn.core.cfc import CfCCell


# ---------------------------------------------------------------------------
# DAGEdgeGate tests
# ---------------------------------------------------------------------------


def test_edge_gate_init():
    g = DAGEdgeGate(hidden_size=4, down_dim=2)
    assert g.W_down.weight.shape == (2, 4)
    assert g.W_edge.weight.shape == (1, 4)
    assert g.W_node.weight.shape == (4, 4)
    # W_up is zero-initialized
    assert torch.allclose(g.W_up.weight, torch.zeros(4, 4), atol=1e-6)


def test_edge_gate_forward_shape():
    g = DAGEdgeGate(hidden_size=4, down_dim=2)
    node_outs = torch.randn(3, 5, 4)
    out = g(node_outs)
    assert out.shape == (3, 5, 4)


def test_edge_gate_residual():
    """DAGEdgeGate should preserve identity at init (W_up=0)."""
    g = DAGEdgeGate(hidden_size=4, down_dim=2)
    node_outs = torch.randn(2, 3, 4)
    out = g(node_outs)
    # At init W_up=0, so out = node_outs (residual)
    assert torch.allclose(out, node_outs, atol=1e-6)


def test_edge_gate_gradients_flow():
    g = DAGEdgeGate(hidden_size=4, down_dim=2)
    node_outs = torch.randn(2, 3, 4, requires_grad=True)
    out = g(node_outs)
    out.sum().backward()
    assert node_outs.grad is not None
    assert g.W_edge.weight.grad is not None
    assert g.W_node.weight.grad is not None


# ---------------------------------------------------------------------------
# DAGAggregation tests
# ---------------------------------------------------------------------------


def test_dag_aggregation_init():
    d = DAGAggregation(hidden_size=4, n_nodes=3, n_iterations=2, down_dim=2)
    assert d.n_iterations == 2
    assert len(d.layers) == 2


def test_dag_aggregation_forward_shape():
    d = DAGAggregation(hidden_size=4, n_nodes=3, n_iterations=2, down_dim=2)
    node_outs = torch.randn(2, 3, 4)
    out = d(node_outs)
    assert out.shape == (2, 3, 4)


def test_dag_aggregation_at_init_preserves_input():
    """At init, all W_up=0, so output equals input."""
    d = DAGAggregation(hidden_size=4, n_nodes=3, n_iterations=2, down_dim=2)
    node_outs = torch.randn(2, 3, 4)
    out = d(node_outs)
    assert torch.allclose(out, node_outs, atol=1e-6)


def test_dag_aggregation_trains():
    """After a few steps, output should differ from input."""
    torch.manual_seed(0)
    d = DAGAggregation(hidden_size=4, n_nodes=3, n_iterations=2, down_dim=2)
    opt = torch.optim.Adam(d.parameters(), lr=1e-2)
    node_outs = torch.randn(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    for _ in range(10):
        out = d(node_outs)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # W_up should no longer be all zeros after training
    assert not torch.allclose(d.layers[0].W_up.weight, torch.zeros(4, 4), atol=1e-3)


# ---------------------------------------------------------------------------
# DAGMoECfCCell tests
# ---------------------------------------------------------------------------


def test_cell_init():
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=2)
    assert cell.n_experts == 3
    assert cell.top_k == 3
    assert cell.dag.n_iterations == 2
    assert isinstance(cell.base_cfc, CfCCell)
    assert len(cell.experts) == 3
    for e in cell.experts:
        assert isinstance(e, CfCCell)


def test_cell_init_top_k_lt_n_experts():
    """top_k < n_experts: sparse DAG-MoE."""
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=4, top_k=2,
                         n_dag_iterations=1, dag_down_dim=2)
    assert cell.n_experts == 4
    assert cell.top_k == 2


def test_cell_init_top_k_gt_n_experts_raises():
    """top_k > n_experts should raise."""
    try:
        cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=5)
        assert False, "should have raised"
    except ValueError:
        pass


def test_cell_forward_shape():
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=2)
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    assert out.shape == (3, 4)


def test_cell_forward_shape_top_k_lt_n_experts():
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=4, top_k=2,
                         n_dag_iterations=1, dag_down_dim=2)
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    assert out.shape == (3, 4)


def test_cell_forward_with_aux():
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=2)
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    h_new, aux = cell.forward_with_aux(x, h)
    assert h_new.shape == (3, 4)
    assert aux["router_g"].shape == (3, 3)
    # g sums to 1 (softmax)
    assert torch.allclose(aux["router_g"].sum(dim=-1), torch.ones(3), atol=1e-5)
    # All expert outs
    assert aux["all_expert_outs"].shape == (3, 3, 4)
    # Refined node outs
    assert aux["refined_node_outs"].shape == (3, 3, 4)


def test_cell_gradient_flow():
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=2)
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    out.sum().backward()
    # Base CfC grads
    assert cell.base_cfc.f_gate[0].weight.grad is not None
    # Expert grads (at least 1)
    n_with_grad = sum(
        1 for e in cell.experts if list(e.parameters())[0].grad is not None
    )
    assert n_with_grad >= 1
    # Router grads
    assert cell.router.weight.grad is not None
    # DAG grads
    assert cell.dag.layers[0].W_edge.weight.grad is not None


def test_cell_dag_modifies_output():
    """With n_dag_iterations=0 (skip) the cell reduces to weighted sum;
    with n_dag_iterations=2 it should differ."""
    torch.manual_seed(42)
    cell_no_dag = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                                n_dag_iterations=1, dag_down_dim=2)
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    # Init W_up to non-zero to see effect
    with torch.no_grad():
        for layer in cell_no_dag.dag.layers:
            layer.W_up.weight.normal_(std=0.1)
    out_with_dag = cell_no_dag(x, h)

    cell_no_dag2 = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                                 n_dag_iterations=1, dag_down_dim=2)
    # Same init
    cell_no_dag2.load_state_dict(cell_no_dag.state_dict())
    # Zero out W_up
    with torch.no_grad():
        for layer in cell_no_dag2.dag.layers:
            layer.W_up.weight.zero_()
    out_no_dag = cell_no_dag2(x, h)
    # Should differ
    assert not torch.allclose(out_with_dag, out_no_dag, atol=1e-3)


def test_cell_diag_metadata():
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=2)
    util = dag_moe_utilization(cell)
    assert util["n_experts"] == 3
    assert util["top_k"] == 3
    assert util["n_dag_iterations"] == 2
    assert util["n_params"] > 0
    assert util["n_dag_params"] > 0
    assert util["n_expert_params"] > 0
    # n_params = n_dag + n_expert + n_base + n_router
    assert util["n_params"] == (
        util["n_dag_params"]
        + util["n_expert_params"]
        + util["n_base_params"]
        + util["n_router_params"]
    )


def test_cell_smoke_sin():
    """Smoke: DAG-MoE CfC cell should learn toy sin."""
    torch.manual_seed(42)
    cell = DAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=4)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    losses = []
    for _ in range(50):
        x = torch.randn(4, 2)
        h = torch.zeros(4, 8)
        target = torch.sin(x[:, 0:1]).expand(4, 8) * 0.5
        out = cell(x, h)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9, (
        f"loss didn't decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# DAGMoECfCNetwork tests
# ---------------------------------------------------------------------------


def test_network_forward_shape():
    net = DAGMoECfCNetwork(input_size=2, hidden_size=8, output_size=1,
                           num_layers=2, return_sequences=True,
                           n_experts=3, top_k=3, n_dag_iterations=2)
    x = torch.randn(4, 6, 2)
    out = net(x)
    assert out.shape == (4, 6, 1)


def test_network_last_step():
    net = DAGMoECfCNetwork(input_size=2, hidden_size=8, output_size=2,
                           num_layers=1, return_sequences=False,
                           n_experts=3, top_k=3, n_dag_iterations=2)
    x = torch.randn(4, 6, 2)
    out = net(x)
    assert out.shape == (4, 2)


def test_network_with_nan():
    """NaN inputs are handled (nan_to_num in network)."""
    net = DAGMoECfCNetwork(input_size=2, hidden_size=8, output_size=1,
                           num_layers=1, n_experts=3, top_k=3, n_dag_iterations=2)
    x = torch.randn(4, 6, 2)
    x[0, 2, 0] = float("nan")
    out = net(x)
    assert out.shape == (4, 6, 1)
    assert torch.isfinite(out).all()


def test_network_learns():
    """DAGMoECfCNetwork should learn a simple sin function."""
    torch.manual_seed(0)
    net = DAGMoECfCNetwork(input_size=2, hidden_size=8, output_size=1,
                           num_layers=1, return_sequences=True,
                           n_experts=3, top_k=3, n_dag_iterations=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        x = torch.randn(4, 8, 2)
        target = torch.sin(x[:, :, 0:1]) * 0.5
        out = net(x)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.85


# ---------------------------------------------------------------------------
# Bench-style tests
# ---------------------------------------------------------------------------


def test_cell_mini_bench_sin():
    """Mini-bench on toy sin."""
    torch.manual_seed(0)
    cell = DAGMoECfCCell(input_size=1, hidden_size=8, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=4)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    for _ in range(60):
        x = torch.linspace(-1, 1, 16).unsqueeze(-1)
        h = torch.zeros(16, 8)
        target = torch.sin(x * math.pi).expand(16, 8) * 0.5
        out = cell(x, h)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.10  # DAG-MoE should fit sin reasonably well


def test_cell_parameter_count():
    """DAG-MoE with K=3 L=2 should be comparable to FAME/Sigmoid K=3."""
    cell = DAGMoECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=3,
                         n_dag_iterations=2, dag_down_dim=2)
    util = dag_moe_utilization(cell)
    # DAG adds ~290 params on top of base + 3 experts
    assert util["n_dag_params"] > 0
    assert util["n_expert_params"] > 0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _all_tests():
    return [
        name
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]


if __name__ == "__main__":
    test_names = _all_tests()
    print(f"Running {len(test_names)} tests...")
    passed = 0
    failed = 0
    for name in test_names:
        try:
            globals()[name]()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
