#!/usr/bin/env python3
"""Hidden-size capacity scan on the SYNTHETIC heterogeneous burst data.

W+1 from §20.8 (round 15): does the hidden=8 anomaly on real EMMA
rover (where uni_video_xattn unexpectedly beats cross_attn) replicate
on the synthetic burst data?  If yes -> LNN-universal phenomenon; if
no -> EMMA-specific.

3 model kinds x 4 hidden_sizes = 12 runs, on the
HeterogeneousForcedDataset (burst, n=800, ep=20, K=2, seed=42) that
established the §11 v6 PASS at +27.6%.

Reports the gain-vs-video_only curve at each hidden_size, plus the
video_only / uni_video_xattn / cross_attn absolute MSEs.
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
from lnn.data.multimodal_physreg import (
    HeterogeneousForcedDataset,
    create_heterogeneous_forced_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, model_kind, batch):
    if model_kind == "video_only":
        fused = torch.cat([batch["video"], batch["audio"]], dim=-1)
        return model(fused)
    return model(batch["video"], batch["audio"])


def _train_one_epoch(model, model_kind, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        optimizer.zero_grad(set_to_none=True)
        mdn_params = _forward(model, model_kind, batch)
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _evaluate(model, model_kind, dataloader, device):
    model.eval()
    sq_errs = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        mdn_params = _forward(model, model_kind, batch)
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


def _build_model(model_kind, hidden_size, num_mixtures):
    if model_kind == "video_only":
        return BiCfCNADWithMDN(input_size=2, hidden_size=hidden_size, output_size=2, num_mixtures=num_mixtures)
    if model_kind == "cross_attn":
        return CrossModalAttnBiCfCNADWithMDN(
            video_dim=1, audio_dim=1, hidden_size=hidden_size, output_size=2, num_mixtures=num_mixtures,
        )
    if model_kind == "uni_video_xattn":
        return UniVideoSelfXAttnWithMDN(
            video_dim=1, audio_dim=1, hidden_size=hidden_size, output_size=2, num_mixtures=num_mixtures,
        )
    raise ValueError(model_kind)


def _run(model_kind, hidden_size, args, device, dataset):
    torch.manual_seed(args.seed)
    train_loader, _, test_loader = create_heterogeneous_forced_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )
    model = _build_model(model_kind, hidden_size, args.num_mixtures).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    for _ in range(args.epochs):
        _train_one_epoch(model, model_kind, train_loader, optimizer, device)
    elapsed = time.perf_counter() - start
    test = _evaluate(model, model_kind, test_loader, device)
    return {
        "model_kind": model_kind,
        "hidden_size": hidden_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": elapsed,
        "test": test,
    }


def main():
    parser = argparse.ArgumentParser(description="hidden_size scan on synthetic burst data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-mixtures", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--output-dir", default="analysis/multimodal_physreg")
    args = parser.parse_args()
    device = torch.device("cpu")
    print("=== Synthetic Burst hidden_size Scan ===")
    print(f"hidden_sizes: {args.hidden_sizes} | epochs={args.epochs} | n={args.num_samples}")
    dataset = HeterogeneousForcedDataset(
        num_samples=args.num_samples, seq_len=args.seq_len, force_kind="burst", seed=args.seed,
    )
    results = []
    for h in args.hidden_sizes:
        for kind in ("video_only", "uni_video_xattn", "cross_attn"):
            r = _run(kind, h, args, device, dataset)
            print(f"  hidden={h:>3d} | {kind:18s} | params={r['parameters']:>5d} | test MSE = {r['test']['param_mse']:.4f}")
            results.append(r)
    by_h = {}
    for r in results:
        by_h.setdefault(r["hidden_size"], {})[r["model_kind"]] = r["test"]["param_mse"]
    print("\n=== Test param MSE by hidden_size (synthetic burst) ===")
    print(f"{'hidden':>6s} | {'video_only':>11s} | {'uni_video_xattn':>15s} | {'cross_attn':>11s} | {'xattn_gain':>11s} | {'ca_gain':>9s}")
    ca_gains = []
    uv_gains = []
    for h in sorted(by_h):
        v = by_h[h]["video_only"]
        uv = by_h[h]["uni_video_xattn"]
        ca = by_h[h]["cross_attn"]
        uv_gain = (v - uv) / v if v > 0 else 0
        ca_gain = (v - ca) / v if v > 0 else 0
        ca_gains.append(ca_gain)
        uv_gains.append(uv_gain)
        print(f"{h:>6d} | {v:11.4f} | {uv:15.4f} | {ca:11.4f} | {uv_gain * 100:+10.1f}% | {ca_gain * 100:+8.1f}%")
    if len(ca_gains) >= 2:
        ca_range = max(ca_gains) - min(ca_gains)
        print(f"\ncross_attn gain range (synth burst): {ca_range * 100:.1f}pp")
        # Cross-check: did the hidden=8 anomaly replicate?
        h8 = next((h for h in by_h if h == 8), None)
        if h8 is not None:
            uv_at_8 = by_h[h8]["uni_video_xattn"]
            ca_at_8 = by_h[h8]["cross_attn"]
            v_at_8 = by_h[h8]["video_only"]
            xattn_8 = (v_at_8 - uv_at_8) / v_at_8 if v_at_8 > 0 else 0
            ca_8 = (v_at_8 - ca_at_8) / v_at_8 if v_at_8 > 0 else 0
            print(f"\nhidden=8 anomaly check (synth): uni_video_xattn +{xattn_8 * 100:.1f}% vs cross_attn +{ca_8 * 100:.1f}%")
            if xattn_8 > ca_8:
                print("=> hidden=8 anomaly REPLICATES on synthetic data -> LNN-universal")
            else:
                print("=> hidden=8 anomaly DOES NOT REPLICATE -> EMMA-specific")
    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_synth_burst_hidden_size_scan.json"
    payload = {"run_id": run_id, "config": vars(args), "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
