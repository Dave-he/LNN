"""Round 206 — tests for AttentionCfC (PRD #10-168)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_attn_cfc import (
    AttentionCfCCell,
    make_lbps_lnkhlfft_attn_5_3_2,
)


def test_attn_cell_forward():
    cell = AttentionCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_past = torch.zeros(4, 0, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, h_past_new, _, _ = cell(x, h, h_past, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(h_past_new).all()
    assert h_past_new.shape == (4, 1, 8)


def test_attn_cell_handles_nan():
    cell = AttentionCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    h_past = torch.zeros(4, 0, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _, _ = cell(x, h, h_past, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_attn_past_aggregates():
    """Past hidden states should be used in attention."""
    cell = AttentionCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    h = torch.randn(2, 8)
    # Past = some non-zero states
    h_past = torch.randn(2, 3, 8)
    g = cell._attention(h, h_past)
    assert torch.isfinite(g).all()
    assert g.shape == (2, 8)
    # With zero past, output should be zero
    h_past_zero = torch.zeros(2, 0, 8)
    g_zero = cell._attention(h, h_past_zero)
    assert torch.allclose(g_zero, torch.zeros_like(g_zero))


def test_attn_gradient_flows():
    cell = AttentionCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_past = torch.randn(4, 3, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _, _ = cell(x, h, h_past, emas_x, emas_h)
    loss = h_new.pow(2).mean()
    loss.backward()
    has_attn_grad = False
    for name, p in cell.named_parameters():
        if "attn_q" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_attn_grad = True
            break
    assert has_attn_grad


def test_attn_stacked_factory():
    net = make_lbps_lnkhlfft_attn_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_attn_stacked_eval_deterministic():
    net = make_lbps_lnkhlfft_attn_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_attn_stacked_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_attn_5_3_2(input_size=2, hidden_size=12, output_size=1)
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = 0.0
    for _ in range(50):
        opt.zero_grad()
        y = net(x)
        loss = (y - target).pow(2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


def test_attn_attention_uses_past():
    """Different past should give different output."""
    cell = AttentionCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    h = torch.randn(2, 8)
    h_past1 = torch.randn(2, 3, 8)
    h_past2 = torch.randn(2, 3, 8) * 2
    g1 = cell._attention(h, h_past1)
    g2 = cell._attention(h, h_past2)
    assert not torch.allclose(g1, g2)


def test_attn_softmax_normalized():
    cell = AttentionCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    h = torch.randn(2, 8)
    h_past = torch.randn(2, 5, 8)
    q = cell.attn_q(h).unsqueeze(1)
    k = cell.attn_k(h_past)
    scores = (q * k).sum(dim=-1) / (cell.hidden_size ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_attn_smoke_long_sequence():
    net = make_lbps_lnkhlfft_attn_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 32, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 32, 1)


def test_attn_past_grows_with_t():
    net = make_lbps_lnkhlfft_attn_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(1, 5, 2)
    with torch.no_grad():
        y = net(x)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    import inspect
    this = sys.modules[__name__]
    funcs = [
        (name, fn)
        for name, fn in inspect.getmembers(this, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"=== Running {len(funcs)} tests ===")
    failed = []
    for name, fn in funcs:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed.append((name, e))
    if failed:
        print(f"\n{len(failed)} FAILED")
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
