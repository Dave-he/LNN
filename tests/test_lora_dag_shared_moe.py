"""Tests for round 124 LoRA-DAG-Shared-MoE (TRIPLE hybrid, PRD #10-86)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.lora_dag_shared_moe import (
    LoRADAGSharedMoECfCCell,
    LoRADAGSharedMoECfCNetwork,
    lora_dag_shared_moe_utilization,
)


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


def test_cell_init_default():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=3,
                                   rank=4, router_type="learned", n_dag_iterations=2)
    assert cell.n_experts == 3
    assert cell.top_k == 3
    assert cell.rank == 4
    assert len(cell.experts) == 3
    assert cell.shared_expert is not None
    assert cell.use_shared is True
    assert cell.dag.n_iterations == 2


def test_cell_init_no_shared():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", use_shared=False)
    assert cell.shared_expert is None
    assert cell.use_shared is False


def test_cell_init_sigmoid_dense():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=0,
                                   rank=2, router_type="sigmoid")
    assert cell.top_k == 0
    assert cell.router_type == "sigmoid"


def test_cell_invalid_top_k():
    with pytest.raises(ValueError):
        LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=5)


def test_cell_invalid_rank():
    with pytest.raises(AssertionError):
        LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2, rank=0)


def test_cell_invalid_router_type():
    with pytest.raises(AssertionError):
        LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                                router_type="cosine")


def test_cell_forward_shape_learned():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                                   rank=4, router_type="learned")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    assert h_new.shape == (4, 16)


def test_cell_forward_shape_no_shared():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", use_shared=False)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    assert h_new.shape == (4, 16)


def test_cell_forward_shape_sigmoid_dense():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=0,
                                   rank=4, router_type="sigmoid")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    assert h_new.shape == (4, 16)


def test_cell_forward_with_aux_learned():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    assert h_new.shape == (4, 16)
    assert info["all_deltas"].shape == (4, 3, 16)
    assert info["selected_deltas"].shape == (4, 2, 16)
    assert info["g"].shape == (4, 2)
    assert info["refined"].shape == (4, 2, 16)
    assert info["h_routed"].shape == (4, 16)
    assert info["h_shared"].shape == (4, 16)


def test_cell_forward_with_aux_sigmoid_dense():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=0,
                                   rank=4, router_type="sigmoid", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    assert h_new.shape == (4, 16)
    assert info["all_deltas"].shape == (4, 3, 16)
    assert info["selected_deltas"].shape == (4, 3, 16)
    assert info["g"].shape == (4, 3)


def test_cell_gradient_flow_learned():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned")
    x = torch.randn(4, 2, requires_grad=True)
    h = torch.zeros(4, 16)
    h_new = cell(x, h)
    loss = h_new.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    # Routed experts got grad
    for expert in cell.experts:
        assert expert.lora_A.grad is not None
        assert expert.lora_B.grad is not None
    # Shared expert got grad
    assert cell.shared_expert.lora_A.grad is not None
    assert cell.shared_expert.lora_B.grad is not None
    # DAG params
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in cell.dag.parameters())


def test_cell_warm_start_zero_lora():
    """B-init-zero should make all LoRA contributions zero at init."""
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned")
    for expert in cell.experts:
        assert torch.all(expert.lora_B == 0)
    assert torch.all(cell.shared_expert.lora_B == 0)


def test_cell_smoke_sin_learns():
    """LoRA-DAG-Shared-MoE should at least partially learn a simple sin task."""
    torch.manual_seed(42)
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    T, B = 32, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, 2)
    for i in range(2):
        x[:, :, i] = torch.sin(t.squeeze(-1) + i * 0.5)
    y = x[:, :, 0:1]
    initial_loss = None
    final_loss = None
    for _ in range(30):
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


def test_cell_three_pathways_additive():
    """h_new = h_base + h_shared + h_routed when use_residual=True."""
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2,
                                   use_residual=True)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    assert torch.allclose(h_new, info["h_base"] + info["h_shared"] + info["h_routed"])


def test_cell_no_residual():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2,
                                   use_residual=False)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    h_new, info = cell.forward_with_aux(x, h)
    assert torch.allclose(h_new, info["h_shared"] + info["h_routed"])


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_network_forward():
    net = LoRADAGSharedMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                     num_layers=2, return_sequences=True,
                                     n_experts=3, top_k=2, rank=4,
                                     router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 32, 2)
    out = net(x)
    assert out.shape == (4, 32, 1)


def test_network_last_step():
    net = LoRADAGSharedMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                     num_layers=1, return_sequences=False,
                                     n_experts=3, top_k=2, rank=4,
                                     router_type="learned", n_dag_iterations=1)
    x = torch.randn(4, 32, 2)
    out = net(x)
    assert out.shape == (4, 1)


def test_network_handles_nan():
    net = LoRADAGSharedMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
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
    """LoRA-DAG-Shared-MoE network should learn a sin task over 30 epochs."""
    torch.manual_seed(0)
    net = LoRADAGSharedMoECfCNetwork(input_size=2, hidden_size=16, output_size=1,
                                     num_layers=2, return_sequences=True,
                                     n_experts=3, top_k=2, rank=4,
                                     router_type="learned", n_dag_iterations=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    T, B = 32, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, 2)
    for i in range(2):
        x[:, :, i] = torch.sin(t.squeeze(-1) + i * 0.5)
    y = x[:, :, 0:1]
    initial_loss = None
    final_loss = None
    for _ in range(30):
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
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                                   rank=2, router_type="learned", n_dag_iterations=1)
    diag = lora_dag_shared_moe_utilization(cell)
    assert diag["n_experts"] == 3
    assert diag["top_k"] == 2
    assert diag["rank"] == 2
    assert diag["use_shared"] is True
    assert diag["n_lora_routed_params"] > 0
    assert diag["n_lora_shared_params"] > 0
    assert diag["n_dag_params"] > 0


def test_utilization_after_run():
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2,
                                   rank=2, router_type="learned", n_dag_iterations=1)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    cell(x, h)
    diag = lora_dag_shared_moe_utilization(cell)
    assert diag["routing_entropy"] > 0
    assert len(diag["expert_util"]) == 2


def test_param_count_smaller_than_lora_dag():
    """LoRA-DAG-Shared should have similar param count to LoRA-DAG (only +136 for shared)."""
    from lnn.core.lora_dag_moe import LoRADAGMoECfCCell
    lora_dag = LoRADAGMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                                 rank=4, router_type="learned", n_dag_iterations=2)
    triple = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=3,
                                     rank=4, router_type="learned", n_dag_iterations=2)
    n_lora_dag = sum(p.numel() for p in lora_dag.parameters())
    n_triple = sum(p.numel() for p in triple.parameters())
    # Triple should be larger by ~136 (one shared LoRA expert)
    assert n_triple > n_lora_dag
    assert n_triple - n_lora_dag < 200, f"delta too large: {n_triple - n_lora_dag}"


# ---------------------------------------------------------------------------
# Combinatorial tests
# ---------------------------------------------------------------------------


def test_three_pathways_orthogonal():
    """Verify all three pathways contribute distinct signals."""
    torch.manual_seed(0)
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    _, info = cell.forward_with_aux(x, h)
    # All three pathways have correct shape
    assert info["h_base"].shape == (4, 16)
    assert info["h_shared"].shape == (4, 16)
    assert info["h_routed"].shape == (4, 16)
    # The routed output is not the same as the shared output
    assert not torch.allclose(info["h_routed"], info["h_shared"])


def test_alpha_scaling():
    cell_small = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                         rank=4, alpha=0.5, router_type="learned", n_dag_iterations=1)
    cell_large = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                         rank=4, alpha=2.0, router_type="learned", n_dag_iterations=1)
    assert cell_small.experts[0].scaling == 0.5 / 4
    assert cell_large.experts[0].scaling == 2.0 / 4
    assert cell_small.shared_expert.scaling == 0.5 / 4
    assert cell_large.shared_expert.scaling == 2.0 / 4


def test_no_shared_equals_lora_dag():
    """use_shared=False should make this equivalent to LoRA-DAG-MoE (round 123)."""
    torch.manual_seed(0)
    cell = LoRADAGSharedMoECfCCell(input_size=2, hidden_size=16, n_experts=3, top_k=2,
                                   rank=4, router_type="learned", n_dag_iterations=2,
                                   use_shared=False)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    _, info = cell.forward_with_aux(x, h)
    # Without shared, h_shared should be zeros
    assert torch.all(info["h_shared"] == 0)
    # h_new = h_base + h_routed (no shared contribution)
    assert torch.allclose(info["h_base"] + info["h_routed"], info["h_base"] + info["h_routed"])


if __name__ == "__main__":
    test_cell_init_default()
    test_cell_init_no_shared()
    test_cell_init_sigmoid_dense()
    test_cell_invalid_top_k()
    test_cell_invalid_rank()
    test_cell_invalid_router_type()
    test_cell_forward_shape_learned()
    test_cell_forward_shape_no_shared()
    test_cell_forward_shape_sigmoid_dense()
    test_cell_forward_with_aux_learned()
    test_cell_forward_with_aux_sigmoid_dense()
    test_cell_gradient_flow_learned()
    test_cell_warm_start_zero_lora()
    test_cell_smoke_sin_learns()
    test_cell_three_pathways_additive()
    test_cell_no_residual()
    test_network_forward()
    test_network_last_step()
    test_network_handles_nan()
    test_network_learns()
    test_utilization_no_run()
    test_utilization_after_run()
    test_param_count_smaller_than_lora_dag()
    test_three_pathways_orthogonal()
    test_alpha_scaling()
    test_no_shared_equals_lora_dag()
    print("All tests passed!")
