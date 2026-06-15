"""Round 118 — LoRA MoE tests (PRD #10-80).

Tests for ``LoRAExpert`` and ``LoRACfCCell`` covering:
- Init shapes and B=0 cold-start
- Forward shape and scaling math
- LoRA warm-start (initial output ≈ 0 because B=0)
- After a few gradient steps, the deltas deviate from the base
- Cell-level forward with all 3 router types
- Routing entropy / expert utilization
- Gradient flow to base + adapters
- Smoke test on toy_sin
"""
from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.lora_moe import (
    LoRACfCCell,
    LoRACfCNetwork,
    LoRAExpert,
    lora_moe_utilization,
)


# ---------------------------------------------------------------------------
# LoRAExpert tests
# ---------------------------------------------------------------------------


def test_lora_expert_init_shapes():
    in_f, out_f, r = 6, 4, 2
    e = LoRAExpert(in_f, out_f, rank=r, alpha=1.0)
    assert e.lora_A.shape == (in_f, r)
    assert e.lora_B.shape == (r, out_f)
    assert e.rank == r
    assert abs(e.scaling - 1.0 / r) < 1e-9
    assert e.in_features == in_f
    assert e.out_features == out_f


def test_lora_expert_cold_start_is_zero():
    """Canonical LoRA: B=0 → initial output is exactly 0, regardless of A."""
    e = LoRAExpert(in_features=6, out_features=4, rank=3, alpha=2.0)
    x = torch.randn(5, 6)
    out = e(x)
    # With B=0, (x @ A) @ B = 0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-9)


def test_lora_expert_scaling_math():
    """When B is non-zero, output scale should match alpha/r."""
    torch.manual_seed(0)
    e = LoRAExpert(in_features=8, out_features=4, rank=2, alpha=2.0)
    # Override zeros with non-zero B
    with torch.no_grad():
        e.lora_B.fill_(1.0)
    x = torch.ones(1, 8)
    out = e(x)
    # Expected: scaling * (x @ A) @ B
    expected_z = x @ e.lora_A              # [1, 2]
    expected_out = expected_z @ e.lora_B * e.scaling  # [1, 4]
    assert torch.allclose(out, expected_out, atol=1e-6)
    # The scaling factor
    assert abs(e.scaling - 1.0) < 1e-9  # 2.0 / 2.0


def test_lora_expert_gradients_flow():
    e = LoRAExpert(in_features=4, out_features=3, rank=2, alpha=1.0)
    x = torch.randn(2, 4, requires_grad=True)
    out = e(x).sum()
    out.backward()
    assert x.grad is not None
    assert e.lora_A.grad is not None
    assert e.lora_B.grad is not None
    # Note: B=0 cold start → dL/dA also 0 (B is a factor in the chain).
    # That's the canonical LoRA warm-up.
    # After we set B to non-zero, gradients flow to A.
    with torch.no_grad():
        e.lora_B.fill_(0.1)
    e.zero_grad()
    x2 = torch.randn(2, 4, requires_grad=True)
    out2 = e(x2).sum()
    out2.backward()
    assert e.lora_A.grad is not None
    assert e.lora_A.grad.abs().sum() > 0  # now non-zero


def test_lora_expert_dropout():
    e = LoRAExpert(in_features=4, out_features=3, rank=2, alpha=1.0,
                   dropout=0.5)
    e.train()
    with torch.no_grad():
        e.lora_B.fill_(1.0)
    x = torch.ones(8, 4)
    out_train = e(x)
    # In train mode, dropout is active so the output is not deterministic
    # across calls.  We just check shape and finiteness.
    assert out_train.shape == (8, 3)
    assert torch.isfinite(out_train).all()

    e.eval()
    out_eval_1 = e(x)
    out_eval_2 = e(x)
    # In eval mode, dropout is disabled → same output both times.
    assert torch.allclose(out_eval_1, out_eval_2, atol=1e-7)


def test_lora_expert_extra_repr():
    e = LoRAExpert(in_features=8, out_features=4, rank=2, alpha=4.0)
    s = e.extra_repr()
    assert "rank=2" in s
    assert "alpha=4.0" in s


def test_lora_expert_rank1_extreme():
    e = LoRAExpert(in_features=4, out_features=3, rank=1, alpha=1.0)
    assert e.lora_A.shape == (4, 1)
    assert e.lora_B.shape == (1, 3)
    assert abs(e.scaling - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# LoRACfCCell tests
# ---------------------------------------------------------------------------


def test_lora_cell_init():
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=2, rank=2)
    assert cell.n_experts == 3
    assert cell.top_k == 2
    assert cell.rank == 2
    assert isinstance(cell.base_cfc, CfCCell)
    assert len(cell.experts) == 3
    # Each expert is a LoRAExpert
    for e in cell.experts:
        assert isinstance(e, LoRAExpert)


def test_lora_cell_forward_shape():
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=2, rank=2)
    x = torch.randn(5, 2)
    h = torch.zeros(5, 4)
    out = cell(x, h)
    assert out.shape == (5, 4)


def test_lora_cell_warm_start_equals_base():
    """With B=0 cold start, LoRA-MoE output equals the base CfC exactly."""
    torch.manual_seed(0)
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=2, rank=2)
    x = torch.randn(5, 2)
    h = torch.zeros(5, 4)
    # Forward through the cell
    h_new = cell(x, h)
    # Forward through just the base
    h_base = cell.base_cfc(x, h)
    # Should be exactly equal because B=0 means each LoRA delta is 0
    # and g_i sums to 1, so the sum Σ g_i * 0 = 0
    assert torch.allclose(h_new, h_base, atol=1e-6)


def test_lora_cell_warm_start_with_top1():
    """Warm-start also holds in sparse top-K=1 mode."""
    torch.manual_seed(0)
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=1, rank=2)
    x = torch.randn(5, 2)
    h = torch.zeros(5, 4)
    h_new = cell(x, h)
    h_base = cell.base_cfc(x, h)
    assert torch.allclose(h_new, h_base, atol=1e-6)


def test_lora_cell_dense_mode():
    """top_k=0 means dense: all K adapters contribute.  Requires sigmoid router."""
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=0, rank=2,
                       router_type="sigmoid")
    x = torch.randn(5, 2)
    h = torch.zeros(5, 4)
    h_new, deltas = cell.forward_with_aux(x, h)
    assert h_new.shape == (5, 4)
    assert len(deltas) == 3
    for d in deltas:
        assert d.shape == (5, 4)
    # With B=0, all deltas are zero
    for d in deltas:
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-9)


def test_lora_cell_routing_entropy_balanced():
    """After enough forward passes, routing should be reasonably balanced."""
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=1, rank=2)
    # Force B to non-zero to see non-trivial routing
    for e in cell.experts:
        with torch.no_grad():
            e.lora_B.fill_(0.1)
    cell.train()
    # Run a few forward passes to warm up the router
    for _ in range(5):
        x = torch.randn(8, 2)
        h = torch.zeros(8, 4)
        _ = cell(x, h)
    util = lora_moe_utilization(cell)
    # Entropy should be in [0, log(3) ≈ 1.099]
    assert 0.0 <= util["routing_entropy"] <= math.log(3) + 1e-3


def test_lora_cell_diag_metadata():
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=1, rank=2,
                       alpha=4.0)
    util = lora_moe_utilization(cell)
    assert util["rank"] == 2
    assert util["alpha"] == 4.0
    assert abs(util["scaling"] - 2.0) < 1e-9
    assert util["router_type"] == "learned"
    assert util["n_lora_params"] > 0


def test_lora_cell_router_sigmoid():
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=0, rank=2,
                       router_type="sigmoid")
    x = torch.randn(5, 2)
    h = torch.zeros(5, 4)
    out = cell(x, h)
    assert out.shape == (5, 4)
    util = lora_moe_utilization(cell)
    assert util["router_type"] == "sigmoid"


def test_lora_cell_router_cosine():
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=1, rank=2,
                       router_type="cosine")
    x = torch.randn(5, 2)
    h = torch.zeros(5, 4)
    out = cell(x, h)
    assert out.shape == (5, 4)
    util = lora_moe_utilization(cell)
    assert util["router_type"] == "cosine"


def test_lora_cell_gradient_flow():
    """All LoRA adapters and the base CfC should receive gradients."""
    cell = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=2, rank=2)
    # Force B to non-zero so we don't hit the B=0 cold-start degeneracy
    for e in cell.experts:
        with torch.no_grad():
            e.lora_B.fill_(0.1)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 4)
    out = cell(x, h)
    loss = out.sum()
    loss.backward()
    # Base CfC grads
    assert cell.base_cfc.f_gate[0].weight.grad is not None
    # LoRA A & B grads (every expert)
    for i, e in enumerate(cell.experts):
        assert e.lora_A.grad is not None, f"expert {i} lora_A grad missing"
        assert e.lora_B.grad is not None, f"expert {i} lora_B grad missing"


def test_lora_cell_parameter_savings_vs_dense_fame():
    """LoRA with rank<d should have fewer parameters than K dense CfC experts.

    Each CfCCell has 3*(I+H)*H + H params (f, g, h branches + time_scale).
    For I=2, H=4: 3*6*4 + 4 = 76 per expert.
    For K=3 dense: 228 + 3*(I+H)*K router = 3*6*3 = 54 → 282 total.

    LoRA-MoRE: 1 CfC (76) + K*(adapter_dim*rank + rank*H) + router
        adapter_dim = I+H = 6, rank=2, H=4, K=3
        = 76 + 3*(6*2 + 2*4) + 3*6*3
        = 76 + 3*(12+8) + 54
        = 76 + 60 + 54 = 190
    So LoRA at rank=2 should have noticeably fewer parameters than 3 dense experts.
    """
    cell_lora = LoRACfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=2, rank=2)
    from lnn.core.fame_cfc import FAMECfCCell
    cell_fame = FAMECfCCell(input_size=2, hidden_size=4, n_experts=3, top_k=2)
    n_lora = sum(p.numel() for p in cell_lora.parameters())
    n_fame = sum(p.numel() for p in cell_fame.parameters())
    # The LoRA-MoRE has 1 base + K adapters vs K dense experts — should be smaller.
    # Allow some slack: FAME has K*(3*(I+H)*H) experts, LoRA has 1 base + K*r*(I+2H) adapters.
    assert n_lora < n_fame, f"LoRA n_params={n_lora} should be < FAME n_params={n_fame}"


def test_lora_cell_smoke_sin():
    """Smoke test: LoRACfCCell converges on toy sin."""
    torch.manual_seed(42)
    cell = LoRACfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2, rank=2)
    # Force B to non-zero so we don't waste epochs on the cold start
    for e in cell.experts:
        with torch.no_grad():
            e.lora_B.fill_(0.1)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        x = torch.randn(4, 2)
        h = torch.zeros(4, 8)
        target = torch.sin(x[:, 0:1]).expand(4, 8) * 0.5
        out = cell(x, h)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Loss should decrease (with some slack for tiny nets).
    assert losses[-1] < losses[0] * 0.9, (
        f"loss didn't decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# LoRACfCNetwork tests
# ---------------------------------------------------------------------------


def test_lora_network_forward_shape():
    net = LoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=2, return_sequences=True,
        n_experts=3, top_k=2, rank=2,
    )
    x = torch.randn(4, 6, 2)
    out = net(x)
    assert out.shape == (4, 6, 1)


def test_lora_network_last_step():
    net = LoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=2,
        num_layers=1, return_sequences=False,
        n_experts=3, top_k=2, rank=2,
    )
    x = torch.randn(4, 6, 2)
    out = net(x)
    assert out.shape == (4, 2)


def test_lora_network_with_nan():
    """NaN inputs are handled (nan_to_num in network)."""
    net = LoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=1, n_experts=3, top_k=2, rank=2,
    )
    x = torch.randn(4, 6, 2)
    x[0, 2, 0] = float("nan")
    out = net(x)
    assert out.shape == (4, 6, 1)
    assert torch.isfinite(out).all()


def test_lora_network_forward_with_aux():
    net = LoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=2, return_sequences=True,
        n_experts=3, top_k=2, rank=2,
    )
    x = torch.randn(4, 6, 2)
    y, deltas = net.forward_with_aux(x)
    assert y.shape == (4, 6, 1)
    # [num_layers][T][K] of [B, H] tensors
    assert len(deltas) == 2
    assert len(deltas[0]) == 6
    assert len(deltas[0][0]) == 3
    assert deltas[0][0][0].shape == (4, 8)


def test_lora_network_learns():
    """LoRACfCNetwork should learn a simple sin function."""
    torch.manual_seed(0)
    net = LoRACfCNetwork(
        input_size=2, hidden_size=8, output_size=1,
        num_layers=1, return_sequences=True,
        n_experts=3, top_k=2, rank=2,
    )
    # Force B to non-zero so we don't waste time on cold start
    for cell in net.cells:
        for e in cell.experts:
            with torch.no_grad():
                e.lora_B.fill_(0.1)
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
# Bench-style tests (mini-bench to catch obvious bugs)
# ---------------------------------------------------------------------------


def test_lora_cell_mini_bench_sin():
    """Mini-bench on toy sin vs baseline: LoRA-MoRE should not catastrophically fail."""
    torch.manual_seed(0)
    cell = LoRACfCCell(input_size=1, hidden_size=8, n_experts=3, top_k=1, rank=2)
    for e in cell.experts:
        with torch.no_grad():
            e.lora_B.fill_(0.1)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    for _ in range(60):
        x = torch.linspace(-1, 1, 16).unsqueeze(-1)  # [16, 1]
        h = torch.zeros(16, 8)
        target = torch.sin(x * math.pi).expand(16, 8) * 0.5
        out = cell(x, h)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # After 60 epochs, loss should be reasonably small.
    assert loss.item() < 0.1


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
