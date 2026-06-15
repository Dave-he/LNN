"""Round 122 — ProbLoRA-MoE (Probabilistic Routing + LoRA-rank-r) tests (PRD #10-84).

Tests for ``ProbLoRAExpert``, ``ProbLoRACfCCell``, and
``ProbLoRACfCNetwork`` covering:
- ProbLoRAExpert: init, forward shape, B-initialized-to-zero (warm start)
- ProbLoRACfCCell: init in 3 modes, forward shape, forward_with_aux
- Gradient flow through base + router + experts
- ProbLoRACfCNetwork: forward, last_step, NaN, learns
- Mini-bench on toy sin, parameter count
"""
from __future__ import annotations

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.problora_moe import (
    ProbLoRACfCCell,
    ProbLoRACfCNetwork,
    ProbLoRAExpert,
    problora_moe_utilization,
)


# ---------------------------------------------------------------------------
# ProbLoRAExpert tests
# ---------------------------------------------------------------------------


def test_expert_init():
    e = ProbLoRAExpert(in_features=6, out_features=4, rank=2, alpha=1.0)
    assert e.A.weight.shape == (2, 6)
    assert e.B.weight.shape == (4, 2)
    assert e.scale == 0.5  # alpha / rank = 1.0 / 2


def test_expert_b_zero_at_init():
    """B must be zero at init (canonical LoRA warm-start)."""
    e = ProbLoRAExpert(in_features=6, out_features=4, rank=2, alpha=1.0, small_init=True)
    assert torch.allclose(e.B.weight, torch.zeros(4, 2))


def test_expert_forward_shape():
    e = ProbLoRAExpert(in_features=6, out_features=4, rank=2, alpha=1.0)
    x = torch.randn(3, 6)
    delta = e(x)
    assert delta.shape == (3, 4)


def test_expert_forward_zero_at_init():
    """At init with small_init, the delta is identically zero."""
    e = ProbLoRAExpert(in_features=6, out_features=4, rank=2, alpha=1.0, small_init=True)
    x = torch.randn(3, 6)
    delta = e(x)
    assert torch.allclose(delta, torch.zeros(3, 4), atol=1e-7)


def test_expert_forward_with_dropout():
    e = ProbLoRAExpert(in_features=6, out_features=4, rank=2, alpha=1.0, dropout=0.5)
    x = torch.randn(3, 6)
    delta = e(x)
    assert delta.shape == (3, 4)


# ---------------------------------------------------------------------------
# ProbLoRACfCCell tests
# ---------------------------------------------------------------------------


def test_cell_init_exact_k():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
    assert cell.n_experts == 3
    assert cell.top_k == 2
    assert cell.rank == 2
    assert cell.mode == "exact_k"
    assert isinstance(cell.base_cfc, CfCCell)
    assert len(cell.experts) == 3
    assert cell.adapter_dim == 2 + 4  # I + H


def test_cell_init_sample():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="sample",
    )
    assert cell.mode == "sample"


def test_cell_init_dynamic_k():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="dynamic_k",
    )
    assert cell.mode == "dynamic_k"


def test_cell_forward_shape_exact_k():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    assert out.shape == (3, 4)


def test_cell_forward_shape_sample():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="sample",
    )
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    assert out.shape == (3, 4)


def test_cell_forward_shape_dynamic_k():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="dynamic_k",
    )
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    assert out.shape == (3, 4)


def test_cell_forward_with_aux():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    h_new, aux = cell.forward_with_aux(x, h)
    assert h_new.shape == (3, 4)
    assert aux["router_probs"].shape == (3, 3)
    assert aux["router_g"].shape == (3, 2)
    # probs sum to 1
    assert torch.allclose(aux["router_probs"].sum(dim=-1), torch.ones(3), atol=1e-5)
    # g sums to 1
    assert torch.allclose(aux["router_g"].sum(dim=-1), torch.ones(3), atol=1e-5)


def test_cell_gradient_flow_exact_k():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
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
    assert cell.router.proj.weight.grad is not None


def test_cell_gradient_flow_sample():
    """Sample mode: gradient flows through marginal probabilities."""
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="sample",
    )
    x = torch.randn(3, 2)
    h = torch.zeros(3, 4)
    out = cell(x, h)
    out.sum().backward()
    # Router grads
    assert cell.router.proj.weight.grad is not None


def test_cell_diag_metadata():
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
    util = problora_moe_utilization(cell)
    assert util["n_experts"] == 3
    assert util["top_k"] == 2
    assert util["rank"] == 2
    assert util["mode"] == "exact_k"
    assert util["n_params"] > 0
    assert util["n_router_params"] > 0
    assert util["n_expert_params"] > 0
    assert util["n_base_params"] > 0


def test_cell_smoke_sin_exact_k():
    """Smoke: ProbLoRA-MoE CfC cell should learn toy sin."""
    torch.manual_seed(42)
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=8, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
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


def test_cell_smoke_sin_sample():
    """Sample mode should also learn."""
    torch.manual_seed(42)
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=8, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="sample",
    )
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
    assert losses[-1] < losses[0] * 0.9


def test_cell_smoke_sin_dynamic_k():
    """Dynamic_k mode should also learn."""
    torch.manual_seed(42)
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=8, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="dynamic_k",
    )
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
    assert losses[-1] < losses[0] * 0.9


# ---------------------------------------------------------------------------
# ProbLoRACfCNetwork tests
# ---------------------------------------------------------------------------


def test_network_forward_shape():
    net = ProbLoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=2, return_sequences=True,
        n_experts=3, top_k=2, rank=2, alpha=1.0, mode="exact_k",
    )
    x = torch.randn(4, 6, 2)
    out = net(x)
    assert out.shape == (4, 6, 1)


def test_network_last_step():
    net = ProbLoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=2,
        num_layers=1, return_sequences=False,
        n_experts=3, top_k=2, rank=2, alpha=1.0, mode="exact_k",
    )
    x = torch.randn(4, 6, 2)
    out = net(x)
    assert out.shape == (4, 2)


def test_network_with_nan():
    net = ProbLoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=1, n_experts=3, top_k=2, rank=2, alpha=1.0, mode="exact_k",
    )
    x = torch.randn(4, 6, 2)
    x[0, 2, 0] = float("nan")
    out = net(x)
    assert out.shape == (4, 6, 1)
    assert torch.isfinite(out).all()


def test_network_learns():
    """ProbLoRACfCNetwork should learn a simple sin function."""
    torch.manual_seed(0)
    net = ProbLoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=1, return_sequences=True,
        n_experts=3, top_k=2, rank=2, alpha=1.0, mode="exact_k",
    )
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
    cell = ProbLoRACfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
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
    assert loss.item() < 0.10


def test_cell_parameter_count():
    """ProbLoRA with K=3, rank=2 should be smaller than dense sub-MLP K=3."""
    cell = ProbLoRACfCCell(
        input_size=2, hidden_size=4, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
    util = problora_moe_utilization(cell)
    # Total = base + 3 experts (rank=2) + router (3*(I+H))
    # LoRA: 3 * 2 * (I+H + H) = 3 * 2 * 10 = 60
    # base CfC: 2 -> 4 with H=4, plus f_gate, etc.
    # router: 3 * 6 = 18
    assert util["n_params"] > 0


def test_cell_parameter_efficiency():
    """ProbLoRA with K=3, rank=2 should be smaller than ProbMoE with K=3 sub-MLP."""
    from lnn.core.prob_moe import ProbMoECfCCell
    problora = ProbLoRACfCCell(
        input_size=2, hidden_size=8, n_experts=3, top_k=2,
        rank=2, alpha=1.0, mode="exact_k",
    )
    probmoe = ProbMoECfCCell(
        input_size=2, hidden_size=8, n_experts=3, top_k=2,
        mode="exact_k",
    )
    n_problora = sum(p.numel() for p in problora.parameters())
    n_probmoe = sum(p.numel() for p in probmoe.parameters())
    # ProbLoRA should be smaller (rank-2 LoRA < full sub-MLP)
    assert n_problora < n_probmoe, (
        f"ProbLoRA={n_problora} should be smaller than ProbMoE={n_probmoe}"
    )


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
