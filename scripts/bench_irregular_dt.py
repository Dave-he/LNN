"""Validate the *theoretical claim* of TFP retention on irregular-Δt sequences.

Hypothesis:
    CfC's retention ``σ(-f·τ·dt)`` wraps the elapsed-time dependency inside a
    sigmoid whose scale is fixed at init (learnable τ). When the actual Δt
    between forward calls *jitters* (irregular sampling), the *learned* τ is
    trained against a particular dt distribution and may not generalise when
    the dt distribution shifts at test time.

    TFP's retention ``exp(-dt/τ)`` makes dt appear *explicitly and symmetrically*
    in the gate, so the same τ generalises across dt scales. The paper
    (arXiv 2607.08283 §IV) shows this in VLA belief filtering; we test it on
    a simple AR(2) regime-change task.

Methodology:
    1. Build AR(2) + 3-regime synthetic dataset with timestamps.
    2. Train each model on *regular dt* (Δt = 1.0 always).
    3. Test on *irregular dt* (Δt ~ LogNormal(0, 0.5)) — different distribution
       from training.
    4. Compare degradation: MSE_irregular / MSE_regular ratio.
       Lower ratio = more dt-robust.

Outputs:
    analysis/jetson/<date>_irregular_dt_benchmark.{md,json}
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.core.cfc import CfCCell
from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell


# ---------------------------------------------------------------------------
# Data: AR(2) + 3-regime with explicit timestamps
# ---------------------------------------------------------------------------


def make_ar2_with_timestamps(
    n_samples: int,
    seq_len: int,
    n_feat: int = 4,
    seed: int = 0,
    dt_log_sigma: float = 0.0,  # 0 = regular; >0 = log-normal jitter
):
    """Return ``(x, target, dt_seq)`` where ``dt_seq[t]`` is the elapsed time
    from step t-1 to step t. ``dt_seq[0] = 1.0`` always."""
    torch.manual_seed(seed)
    x = torch.zeros(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, seq_len, 1)
    dt = torch.ones(n_samples, seq_len)
    if dt_log_sigma > 0:
        # LogNormal(0, σ) centred so E[dt]=1.
        dt = torch.exp(torch.randn(n_samples, seq_len) * dt_log_sigma - 0.5 * dt_log_sigma ** 2)
        dt[:, 0] = 1.0  # anchor first step
    for s in range(n_samples):
        regime = torch.randint(0, 3, (1,)).item()
        ar1, ar2 = ((0.6, 0.2), (-0.3, 0.5), (0.4, -0.4))[regime]
        noise = torch.randn(seq_len, n_feat) * 0.1
        for t in range(1, seq_len):
            x[s, t] = ar1 * x[s, t - 1] + ar2 * x[s, max(t - 2, 0)] + noise[t]
        y[s, :-1, 0] = x[s, 1:, :].sum(-1)
    return x, y, dt


# ---------------------------------------------------------------------------
# Sequence wrapper that supports per-step dt
# ---------------------------------------------------------------------------


class _SeqWrap(nn.Module):
    def __init__(self, cell: nn.Module, out_dim: int):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor | float = 1.0) -> torch.Tensor:
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for i in range(t):
            dt_i = dt[:, i] if isinstance(dt, torch.Tensor) else dt
            # Per-sample dt: shape [B] -> [B, 1] so cells can broadcast with hidden.
            if isinstance(dt_i, torch.Tensor) and dt_i.dim() == 1:
                dt_i = dt_i.unsqueeze(-1)
            h = self.cell(x[:, i, :], h, dt=dt_i)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


def make_model(name: str, in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    if name == "cfc":
        return _SeqWrap(CfCCell(in_dim, hidden), out_dim)
    if name == "mfc-cfc":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="cfc"), out_dim)
    if name == "mfc-tfp":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="tfp"), out_dim)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Train/eval with dt support
# ---------------------------------------------------------------------------


def train_eval(
    model: nn.Module,
    x_tr, y_tr, dt_tr,
    x_te_reg, y_te_reg,
    x_te_irreg, y_te_irreg, dt_te_irreg,
    epochs: int = 3, batch: int = 8, lr: float = 1e-2,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        for b in range(0, n, batch):
            xb = x_tr[b:b + batch]
            yb = y_tr[b:b + batch]
            dtb = dt_tr[b:b + batch]
            opt.zero_grad()
            pred = model(xb, dt=dtb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        test_mse_reg = nn.functional.mse_loss(model(x_te_reg, dt=1.0), y_te_reg).item()
        test_mse_irreg = nn.functional.mse_loss(model(x_te_irreg, dt=dt_te_irreg), y_te_irreg).item()
    return {
        "test_mse_regular": test_mse_reg,
        "test_mse_irregular": test_mse_irreg,
        "train_seconds": train_s,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=384)
    parser.add_argument("--dt-sigma", type=float, default=0.5)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    torch.manual_seed(0)
    x_reg, y_reg, dt_reg = make_ar2_with_timestamps(
        args.n_samples, args.seq_len, seed=0, dt_log_sigma=0.0,
    )
    n_tr = int(0.8 * x_reg.shape[0])
    x_tr, y_tr, dt_tr = x_reg[:n_tr], y_reg[:n_tr], dt_reg[:n_tr]
    x_te_reg, y_te_reg = x_reg[n_tr:], y_reg[n_tr:]

    # Use the *same* underlying AR(2) signals but with jittered timestamps
    x_te_irreg = x_te_reg.clone()
    y_te_irreg = y_te_reg.clone()
    torch.manual_seed(1)
    dt_te_irreg = torch.exp(torch.randn(*x_te_reg.shape[:2]) * args.dt_sigma - 0.5 * args.dt_sigma ** 2)
    dt_te_irreg[:, 0] = 1.0

    print(f"Train dt: all 1.0 (regular)")
    print(f"Test regular dt: all 1.0")
    print(f"Test irregular dt: LogNormal(0, {args.dt_sigma})")
    print(f"  effective dt range: [{dt_te_irreg.min().item():.3f}, {dt_te_irreg.max().item():.3f}]")
    print(f"  mean dt: {dt_te_irreg.mean().item():.3f}")

    rows = []
    for name in ("cfc", "mfc-cfc", "mfc-tfp"):
        msess_reg, msess_irreg, trains = [], [], []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            model = make_model(name, in_dim=4, hidden=args.hidden, out_dim=1)
            res = train_eval(model, x_tr, y_tr, dt_tr,
                             x_te_reg, y_te_reg,
                             x_te_irreg, y_te_irreg, dt_te_irreg,
                             epochs=args.epochs)
            msess_reg.append(res["test_mse_regular"])
            msess_irreg.append(res["test_mse_irregular"])
            trains.append(res["train_seconds"])
        rows.append({
            "model": name,
            "test_mse_regular_mean": statistics.mean(msess_reg),
            "test_mse_regular_std": statistics.stdev(msess_reg) if len(msess_reg) > 1 else 0.0,
            "test_mse_irregular_mean": statistics.mean(msess_irreg),
            "test_mse_irregular_std": statistics.stdev(msess_irreg) if len(msess_irreg) > 1 else 0.0,
            "train_seconds_mean": statistics.mean(trains),
        })
        # Compute degradation ratio
        rows[-1]["degradation_ratio"] = (
            rows[-1]["test_mse_irregular_mean"] / max(rows[-1]["test_mse_regular_mean"], 1e-9)
        )

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_irregular_dt_benchmark.json"
    md_path = out_dir / f"{args.date}_irregular_dt_benchmark.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "ar2_regime_irregular_dt",
        "config": vars(args),
        "rows": rows,
    }, indent=2))

    md = [f"""---
title: TFP retention vs CfC σ-decay on irregular Δt — {args.date}
date: {args.date}
tags: [LNN, CfC, TFP, retention, irregular-dt, robustness, dt-explicit]
---

# TFP retention vs CfC σ-decay on irregular Δt — {args.date}

## 任务
合成 **非平稳 AR(2) + 3-regime** 时间序列（与上轮 benchmark 同 task）。
**关键差异**：训练 dt = 1.0（恒定），测试 dt ~ LogNormal(0, {args.dt_sigma})（jittered）。
验证 TFP 论文 (arXiv 2607.08283) 的核心 claim："retention 显式依赖 dt → 对 dt 分布变化更鲁棒"。

## 结果（{args.repeats} 次重复 mean±std）

| 模型 | 测试 MSE (regular dt) | 测试 MSE (irregular dt) | **degradation ratio** | 训练秒 |
|---|---:|---:|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['model']} | {r['test_mse_regular_mean']:.4f} ± {r['test_mse_regular_std']:.4f} | "
                  f"{r['test_mse_irregular_mean']:.4f} ± {r['test_mse_irregular_std']:.4f} | "
                  f"**{r['degradation_ratio']:.2f}×** | "
                  f"{r['train_seconds_mean']:.2f} |\n")
    md.append("\n## 解读\n")
    md.append("- **degradation ratio < 1** = 不规则 dt 下 MSE 比 regular 更低（噪声帮助泛化）\n")
    md.append("- **degradation ratio ≈ 1** = 不规则 dt 下 MSE 几乎不变（理想鲁棒）\n")
    md.append("- **degradation ratio >> 1** = 不规则 dt 下 MSE 显著上升（dt 分布依赖）\n")
    md.append("- TFP 的 retention 显式依赖 dt ⇒ 在 irregular dt 下应当比 CfC σ-decay **更鲁棒**（ratio 更接近 1）。\n")
    md.append("## Verdict\nTBD — see report.\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")
    print("\nFinal rows:")
    for r in rows:
        print(f"  {r['model']}: regular={r['test_mse_regular_mean']:.4f}, "
              f"irregular={r['test_mse_irregular_mean']:.4f}, ratio={r['degradation_ratio']:.2f}×")


if __name__ == "__main__":
    main()
