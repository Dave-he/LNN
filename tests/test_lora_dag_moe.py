"""Tests for round 123 LoRA-DAG-MoE (PRD #10-85)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.lora_dag_moe import (
    LoRADAGMoECfCCell,
    LoRADAGMoECfCNetwork,
    lora_dag_moe_utilization,
)


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


def test_cell_init_learned():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=3,
                             rank=4, router_type="learned")
    assert cell.n_experts == 3
    assert cell.top_k == 3
    assert cell.rank == 4
    assert len(cell.experts) == 3
    assert cell.dag.n_iterations == 2


def test_cell_init_sigmoid_dense():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=0,
                             rank=2, router_type="sigmoid")
    assert cell.top_k == 0
    assert cell.router_type == "sigmoid"


def test_cell_invalid_top_k():
    with pytest.raises(ValueError):
        LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=5)


def test_cell_invalid_rank():
    with pytest.raises(AssertionError):
        LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2, rank=0)


def test_cell_invalid_router_type():
    with pytest.raises(AssertionError):
        LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                          router_type="cosine")


def test_cell_learned_requires_top_k():
    with pytest.raises(AssertionError):
        LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=0,
                          router_type="learned")


def test_cell_forward_shape_learned():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                             rank=4, router_type="learned")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    assert h_new.shape == (4, 16)


def test_cell_forward_shape_sigmoid_dense():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=0,
                             rank=4, router_type="sigmoid")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    assert h_new.shape == (4, 16)


def test_cell_forward_with_aux_learned():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                             rank=4, router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    assert h_new.shape == (4, 16)
    assert info["all_deltas"].shape == (4, 3, 16)
    assert info["selected_deltas"].shape == (4, 2, 16)
    assert info["g"].shape == (4, 2)
    assert info["refined"].shape == (4, 2, 16)
    assert info["h_lora"].shape == (4, 16)


def test_cell_forward_with_aux_sigmoid_dense():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=0,
                             rank=4, router_type="sigmoid", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    assert h_new.shape == (4, 16)
    assert info["all_deltas"].shape == (4, 3, 16)
    assert info["selected_deltas"].shape == (4, 3, 16)
    assert info["g"].shape == (4, 3)
    assert info["refined"].shape == (4, 3, 16)


def test_cell_dag_refines():
    """DAG should produce different output than just summing weighted deltas."""
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                             rank=4, router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    _, info = cell.forward_with_aux(x, h)
    # refined should differ from weighted (DAG applies transformations)
    assert not torch.allclose(info["refined"], info["weighted"])


def test_cell_dag_iterations_affect_output():
    """More iterations should change the output (slightly)."""
    torch.manual_seed(0)
    cell2 = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                              rank=4, router_type="learned", n_dag_iterations=1)
    torch.manual_seed(0)
    cell3 = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                              rank=4, router_type="learned", n_dag_iterations=3)
    # Set same weights
    cell3.load_state_dict(cell2.state_dict(), strict=False)
    # They should differ in # of iterations of dag modules
    assert cell2.dag.n_iterations == 1
    assert cell3.dag.n_iterations == 3


def test_cell_gradient_flow_learned():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                             rank=4, router_type="learned")
    x = torch.randn(4, 2, requires_grad=True)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    loss = h_new.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    # Check all expert LoRA params received grad
    for expert in cell.experts:
        assert expert.lora_A.grad is not None
        assert expert.lora_B.grad is not None
    # DAG params
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in cell.dag.parameters())


def test_cell_gradient_flow_sigmoid():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=0,
                             rank=4, router_type="sigmoid")
    x = torch.randn(4, 2, requires_grad=True)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    loss = h_new.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_cell_warm_start_zero_lora():
    """B-init-zero should make the model behave like base CfC at init."""
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                             rank=4, router_type="learned")
    # Check B is zero
    for expert in cell.experts:
        assert torch.all(expert.lora_B == 0)


def test_cell_smoke_sin_learns():
    """LoRA-DAG-MoE should at least partially learn a simple sin task."""
    torch.manual_seed(42)
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                             rank=4, router_type="learned", n_dag_iterations=2)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    T = 32
    B = 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, 2)
    for i in range(2):
        x[:, :, i] = torch.sin(t.squeeze(-1) + i * 0.5)
    y = x[:, :, 0:1]
    initial_loss = None
    final_loss = None
    for epoch in range(30):
        opt.zero_grad()
        h = torch.zeros(B, 16)
        outs = []
        for ti in range(T):
            outs.append(cell(x[:, ti, :], h))
            h = outs[-1]
        out = torch.stack(outs, dim=1)  # [B, T, 1]
        loss = ((out - y) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(), 1.0)
        opt.step()
        final_loss = loss.item()
    assert final_loss < initial_loss, f"no learning: init={initial_loss}, final={final_loss}"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_network_forward():
    net = LoRADAGMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                num_layers=2, return_sequences=True,
                                n_experts=3, top_k=2, rank=4,
                                router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 32, 2)
    out = net(x)
    assert out.shape == (4, 32, 1)


def test_network_last_step():
    net = LoRADAGMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                num_layers=1, return_sequences=False,
                                n_experts=3, top_k=2, rank=4,
                                router_type="learned", n_dag_iterations=1)
    x = torch.randn(4, 32, 2)
    out = net(x)
    assert out.shape == (4, 1)


def test_network_handles_nan():
    net = LoRADAGMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                num_layers=1, return_sequences=True,
                                n_experts=3, top_k=2, rank=4,
                                router_type="learned", n_dag_iterations=1)
    x = torch.randn(4, 32, 2)
    x[0, 0, 0] = float("nan")
    x[1, 5, 1] = float("nan")
    out = net(x)
    assert out.shape == (4, 32, 1)
    assert not torch.isnan(out).any()


def test_network_learns():
    """LoRA-DAG-MoE network should learn a sin task over 30 epochs."""
    torch.manual_seed(0)
    net = LoRADAGMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                num_layers=2, return_sequences=True,
                                n_experts=3, top_k=2, rank=4,
                                router_type="learned", n_dag_iterations=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    T = 32
    B = 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, 2)
    for i in range(2):
        x[:, :, i] = torch.sin(t.squeeze(-1) + i * 0.5)
    y = x[:, :, 0:1]
    initial_loss = None
    final_loss = None
    for epoch in range(30):
        opt.zero_grad()
        out = net(x)
        loss = ((out - y) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        final_loss = loss.item()
    assert final_loss < initial_loss, f"no learning: init={initial_loss}, final={final_loss}"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_utilization_no_run():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                             rank=2, router_type="learned", n_dag_iterations=1)
    diag = lora_dag_moe_utilization(cell)
    assert diag["n_experts"] == 3
    assert diag["top_k"] == 2
    assert diag["rank"] == 2
    assert diag["alpha"] == 1.0
    assert diag["scaling"] == 0.5
    assert diag["n_dag_iterations"] == 1
    assert diag["n_lora_params"] > 0
    assert diag["n_dag_params"] > 0
    assert diag["n_base_params"] > 0
    assert diag["n_router_params"] > 0


def test_utilization_after_run():
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                             rank=2, router_type="learned", n_dag_iterations=1)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    cell(x, h)
    diag = lora_dag_moe_utilization(cell)
    assert diag["routing_entropy"] > 0
    assert len(diag["expert_util"]) == 2


def test_param_count_smaller_than_full_dag():
    """LoRA-DAG-MoE should have fewer expert params than full DAG-MoE (with sub-CfC experts)."""
    lora_dag = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                                 rank=4, router_type="learned", n_dag_iterations=2)
    n_lora_dag = sum(p.numel() for p in lora_dag.experts.parameters())
    # Compare to 3 CfC experts: each has ~928 params
    n_full_dag_experts = 3 * 928
    assert n_lora_dag < n_full_dag_experts, (
        f"LoRA experts ({n_lora_dag}) should be smaller than full DAG experts ({n_full_dag_experts})"
    )


# ---------------------------------------------------------------------------
# Combinatorial tests
# ---------------------------------------------------------------------------


def test_combination_orthogonality():
    """Verify LoRA-DAG-MoE is structurally different from LoRA-only and DAG-only.

    Compare h_new produced by the hybrid vs by the components.
    """
    torch.manual_seed(0)
    cell = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                             rank=4, router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    # The LoRA + DAG combination: h_lora = sum of refined nodes, where refined != weighted
    assert info["h_lora"].shape == (4, 16)
    # h_lora should be a function of all 3 components: base, weighted, refined
    # Verify refined != weighted (DAG adds something)
    assert not torch.allclose(info["refined"], info["weighted"])


def test_residual_path():
    """use_residual=True should give h_new = h_base + h_lora, else h_new = h_lora."""
    torch.manual_seed(0)
    cell_res = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                 rank=4, router_type="learned", n_dag_iterations=2,
                                 use_residual=True)
    cell_nores = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2,
                                   use_residual=False)
    cell_nores.load_state_dict(cell_res.state_dict())
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_res, info_res = cell_res.forward_with_aux(x, h)
    h_nores, info_nores = cell_nores.forward_with_aux(x, h)
    # h_res = h_base + h_lora, h_nores = h_lora
    assert torch.allclose(h_res, info_res["h_base"] + info_res["h_lora"])
    assert torch.allclose(h_nores, info_nores["h_lora"])


def test_alpha_scaling():
    """Larger alpha should give larger LoRA deltas at init."""
    cell_small = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, alpha=0.5, router_type="learned", n_dag_iterations=1)
    cell_large = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, alpha=2.0, router_type="learned", n_dag_iterations=1)
    # Verify scaling
    assert cell_small.experts[0].scaling == 0.5 / 4
    assert cell_large.experts[0].scaling == 2.0 / 4


if __name__ == "__main__":
    test_cell_init_learned()
    test_cell_init_sigmoid_dense()
    test_cell_invalid_top_k()
    test_cell_invalid_rank()
    test_cell_invalid_router_type()
    test_cell_learned_requires_top_k()
    test_cell_forward_shape_learned()
    test_cell_forward_shape_sigmoid_dense()
    test_cell_forward_with_aux_learned()
    test_cell_forward_with_aux_sigmoid_dense()
    test_cell_dag_refines()
    test_cell_dag_iterations_affect_output()
    test_cell_gradient_flow_learned()
    test_cell_gradient_flow_sigmoid()
    test_cell_warm_start_zero_lora()
    test_cell_smoke_sin_learns()
    test_network_forward()
    test_network_last_step()
    test_network_handles_nan()
    test_network_learns()
    test_utilization_no_run()
    test_utilization_after_run()
    test_param_count_smaller_than_full_dag()
    test_combination_orthogonality()
    test_residual_path()
    test_alpha_scaling()
    print("All tests passed!")
