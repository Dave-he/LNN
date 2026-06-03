#!/usr/bin/env python3
"""GRU recovery capacity scan.

Round 25 (cron, §28 §25) found LSTM works (+36.1%) but GRU fails
(+3.9%) at h=16, ep=20. This script tests whether GRU recovers at
larger capacity / longer training - i.e. is the GRU failure
*regime-specific* (a quirk of small h+short training) or
*architecture-inherent* (RNN family quirk).

Hypothesis (falsifiable):
  * If GRU recovers to >=+20% at h=32/ep=40 or larger, the
    failure is regime-specific and GRU is a 'small-budget
    underfitter'.
  * If GRU stays at +5% or worse across all regimes tested,
    the failure is architecture-inherent.

6 runs total: 2 (audio modes) x 3 (hidden sizes) at fixed
ep=20 (matches round 25 standard small-budget benchmark).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import (
    CrossModalAttnBiCfCNADWithMDN,
    UniVideoSelfXAttnWithMDN,
)
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class _GRUEncoder(nn.Module):
    """Bidirectional GRU second-encoder (matches LSTM round 28 in capacity)."""
    def __init__(self, video_dim: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=video_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * hidden_size, hidden_size)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(out)


class GRUEncoderXAttnWithMDN(nn.Module):
    """Same harness as LSTMEncoderXAttnWithMDN but with GRU instead of LSTM."""
    def __init__(self, video_dim: int, hidden_size: int, output_size: int, num_mixtures: int):
        super().__init__()
        self._inner = CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim, audio_dim=video_dim, hidden_size=hidden_size,
            output_size=output_size, num_mixtures=num_mixtures,
        )
        self._inner.audio_encoder = _GRUEncoder(video_dim, hidden_size)

    def forward(self, video, audio=None, dt=None, mask=None, return_attention=False):
        v_feat = self._inner.video_encoder(video)
        a_feat = self._inner.audio_encoder(video)
        v_from_a, _ = self._inner._attend(self._inner.q_v(v_feat), self._inner.k_a(a_feat), self._inner.v_a(a_feat))
        a_from_v, _ = self._inner._attend(self._inner.q_a(a_feat), self._inner.k_v(v_feat), self._inner.v_v(v_feat))
        v_refined = v_feat + v_from_a
        a_refined = a_feat + a_from_v
        fused = self._inner.fuse_proj(torch.cat([v_refined, a_refined], dim=-1))
        return self._inner.mdn(fused)


def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}


def _is_gru_model(model):
    return type(model).__name__ == "GRUEncoderXAttnWithMDN"


def _train(model, loader, opt, device):
    model.train()
    losses = []
    for batch, target in loader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        opt.zero_grad(set_to_none=True)
        # video_only (BiCfCNADWithMDN) expects 4-channel fused input;
        # the GRU model expects (video, audio) separately. Dispatch on
        # model type to avoid the wrong input reaching the wrong model.
        if _is_gru_model(model):
            out = model(batch["video"], batch["audio"])
        else:
            fused = torch.cat([batch["video"], batch["audio"]], dim=-1)
            out = model(fused)
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    sq = []
    for batch, target in loader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        if _is_gru_model(model):
            out = model(batch["video"], batch["audio"])
        else:
            fused = torch.cat([batch["video"], batch["audio"]], dim=-1)
            out = model(fused)
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq)
    return float(sq.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--feature-noise-std", type=float, default=0.02)
    p.add_argument("--num-mixtures", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()
    device = torch.device("cpu")
    print(f"=== GRU Recovery Capacity Scan (ep={args.epochs}) ===")
    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples, window=args.window,
        feature_noise_std=args.feature_noise_std, seed=args.seed,
    )
    train_loader, _, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )

    # Reference: video_only at each hidden
    v_mses = {}
    print("\n--- Reference: video_only at each hidden_size ---")
    for h in args.hidden_sizes:
        torch.manual_seed(args.seed)
        model = BiCfCNADWithMDN(input_size=4, hidden_size=h, output_size=5, num_mixtures=args.num_mixtures).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            _train(model, train_loader, opt, device)
        v_mses[h] = _eval(model, test_loader, device)
        print(f"  video_only h={h:>3d} | test MSE = {v_mses[h]:.4f}")

    # GRU at each hidden
    gru_mses = {}
    print("\n--- GRU bidirectional at each hidden_size ---")
    for h in args.hidden_sizes:
        torch.manual_seed(args.seed)
        model = GRUEncoderXAttnWithMDN(video_dim=3, hidden_size=h, output_size=5, num_mixtures=args.num_mixtures).to(device)
        # Debug: print f_gate shape
        try:
            f_gate = model._inner.video_encoder.encoder.forward_net.cells[0].f_gate[0]
            print(f"  [debug] GRU h={h} f_gate weight: {f_gate.weight.shape}")
        except Exception as e:
            print(f"  [debug] f_gate lookup failed: {e}")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            _train(model, train_loader, opt, device)
        gru_mses[h] = _eval(model, test_loader, device)
        print(f"  GRU h={h:>3d}        | test MSE = {gru_mses[h]:.4f}")

    # Summary
    print(f"\n=== Summary (test MSE) ===")
    print(f"{'hidden':>6s} | {'video_only':>11s} | {'GRU':>11s} | {'GRU gain':>10s} | {'verdict':<25s}")
    recovered = False
    for h in args.hidden_sizes:
        v = v_mses[h]
        g = gru_mses[h]
        gain = (v - g) / v if v > 0 else 0
        if gain >= 0.20:
            verdict = "GRU RECOVERS (>=+20%)"
            recovered = True
        elif gain >= 0.05:
            verdict = "GRU partial recovery"
        else:
            verdict = "GRU STAYS CATASTROPHIC"
        print(f"{h:>6d} | {v:11.4f} | {g:11.4f} | {gain*100:+9.1f}% | {verdict}")
    print()
    print("=> GRU at hidden=16 (round 25 +3.9%) was", "regime-specific" if recovered else "architecture-inherent")
    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_gru_capacity_scan.json"
    payload = {
        "run_id": run_id, "config": vars(args),
        "video_only_mse": v_mses, "gru_mse": gru_mses,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
