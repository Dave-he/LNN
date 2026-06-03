#!/usr/bin/env python3
"""Round-27 generalisation test: adaptive freeze on synthetic burst data.

Tests whether the round-26 SOTA recipe (warmup K=40, freeze
audio_encoder, continue training) reproduces on the synthetic
``HeterogeneousForcedDataset(force_kind='burst')`` benchmark that
established cross_attn's +27.6% gain in round 8.

If adaptive freeze beats BOTH pure cross_attn AND pure video_only at
some K on burst, the recipe is task-agnostic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN
from lnn.data.multimodal_physreg import (
    HeterogeneousForcedDataset,
    create_heterogeneous_forced_dataloaders,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _train_xattn_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


def _train_vo_epoch(model, dataloader, optimizer, device):
    """Pure video_only on burst: concat video+audio as single 2-ch input."""
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        optimizer.zero_grad(set_to_none=True)
        fused = torch.cat([batch["video"], batch["audio"]], dim=-1)
        out = model(fused)
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _eval_xattn(model, dataloader, device):
    model.eval()
    sq_errs = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - params).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


@torch.no_grad()
def _eval_vo(model, dataloader, device):
    model.eval()
    sq_errs = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        fused = torch.cat([batch["video"], batch["audio"]], dim=-1)
        out = model(fused)
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - params).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


def _freeze_audio_encoder(model):
    n = 0
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
        n += 1
    return n


def _run_adaptive_freeze(args, device, dataset, K):
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = create_heterogeneous_forced_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1,
        hidden_size=args.hidden_size,
        output_size=2,
        num_mixtures=args.num_mixtures,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    for ep in range(1, K + 1):
        loss = _train_xattn_epoch(model, train_loader, optim, device)
        if ep == K or ep % max(args.epochs // 5, 1) == 0:
            val = _eval_xattn(model, val_loader, device)["param_mse"]
            print(f"[warmup       ] epoch {ep}/{K}  train NLL {loss:.4f}  val MSE {val:.4f}")
    nz = _freeze_audio_encoder(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[freeze       ] {nz} tensors frozen (audio_encoder); "
          f"{sum(p.numel() for p in trainable)} params still trainable")
    optim = torch.optim.Adam(trainable, lr=args.lr)
    remaining = args.epochs - K
    for ep in range(1, remaining + 1):
        loss = _train_xattn_epoch(model, train_loader, optim, device)
        if ep == remaining or (K + ep) % max(args.epochs // 5, 1) == 0:
            val = _eval_xattn(model, val_loader, device)["param_mse"]
            print(f"[frozen-audio ] epoch {ep}/{remaining}  train NLL {loss:.4f}  val MSE {val:.4f}")
    elapsed = time.perf_counter() - start
    return {
        "test_mse": _eval_xattn(model, test_loader, device)["param_mse"],
        "elapsed_s": elapsed,
    }


def _run_pure_xattn(args, device, dataset):
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = create_heterogeneous_forced_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=args.hidden_size,
        output_size=2, num_mixtures=args.num_mixtures,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for ep in range(1, args.epochs + 1):
        loss = _train_xattn_epoch(model, train_loader, optim, device)
        if ep == args.epochs or ep % max(args.epochs // 5, 1) == 0:
            val = _eval_xattn(model, val_loader, device)["param_mse"]
            print(f"[pure xattn   ] epoch {ep}/{args.epochs}  train NLL {loss:.4f}  val MSE {val:.4f}")
    return {"test_mse": _eval_xattn(model, test_loader, device)["param_mse"]}


def _run_pure_vo(args, device, dataset):
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = create_heterogeneous_forced_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )
    # video_only with input_size=2 (video=1 + audio=1 concat)
    model = BiCfCNADWithMDN(
        input_size=2, hidden_size=args.hidden_size,
        output_size=2, num_mixtures=args.num_mixtures,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for ep in range(1, args.epochs + 1):
        loss = _train_vo_epoch(model, train_loader, optim, device)
        if ep == args.epochs or ep % max(args.epochs // 5, 1) == 0:
            val = _eval_vo(model, val_loader, device)["param_mse"]
            print(f"[pure vo      ] epoch {ep}/{args.epochs}  train NLL {loss:.4f}  val MSE {val:.4f}")
    return {"test_mse": _eval_vo(model, test_loader, device)["param_mse"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-mixtures", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-kind", default="burst")
    parser.add_argument("--warmup-epochs", nargs="+", type=int, default=[20, 40, 60])
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = HeterogeneousForcedDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        force_kind=args.force_kind,
        seed=args.seed,
    )

    print(f"=== Adaptive freeze on BURST (h={args.hidden_size}, ep={args.epochs}) ===")
    print(f"  num_samples={args.num_samples}, seq_len={args.seq_len}, K_mix={args.num_mixtures}")

    print("\n----- Pure cross_attn endpoint -----")
    pure_xattn = _run_pure_xattn(args, device, dataset)
    print(f"  pure xattn test MSE: {pure_xattn['test_mse']:.4f}")

    print("\n----- Pure video_only endpoint -----")
    pure_vo = _run_pure_vo(args, device, dataset)
    print(f"  pure vo    test MSE: {pure_vo['test_mse']:.4f}")

    adaptive = {}
    for K in args.warmup_epochs:
        print(f"\n----- adaptive freeze K={K} -----")
        adaptive[f"K{K}"] = _run_adaptive_freeze(args, device, dataset, K)

    # Summary
    print("\n===== SUMMARY (BURST) =====")
    print(f"{'config':30s}{'test MSE':>14s}{'vs vo':>14s}")
    print(f"{'pure cross_attn':30s}{pure_xattn['test_mse']:>14.4f}"
          f"{(pure_vo['test_mse'] - pure_xattn['test_mse']) / pure_vo['test_mse'] * 100:>13.1f}%")
    print(f"{'pure video_only':30s}{pure_vo['test_mse']:>14.4f}{'baseline':>14s}")
    best = pure_vo['test_mse']
    best_label = "pure video_only"
    for label, r in adaptive.items():
        gain = (pure_vo['test_mse'] - r['test_mse']) / pure_vo['test_mse'] * 100
        print(f"{'adaptive ' + label:30s}{r['test_mse']:>14.4f}{gain:>13.1f}%")
        if r['test_mse'] < best:
            best = r['test_mse']
            best_label = f"adaptive {label}"
    print(f"\nBEST: {best_label} test MSE {best:.4f}")
    print(f"Claim (any K < min(pure_xattn, pure_vo)): "
          f"{'PASS ✅' if best < min(pure_xattn['test_mse'], pure_vo['test_mse']) else 'FAIL'}")

    out_dir = ROOT / "analysis/multimodal_physreg"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "pure_xattn_test_mse": pure_xattn["test_mse"],
        "pure_vo_test_mse": pure_vo["test_mse"],
        "adaptive": adaptive,
        "best_label": best_label,
        "best_test_mse": best,
        "claim_passed": best < min(pure_xattn["test_mse"], pure_vo["test_mse"]),
    }
    out_path = out_dir / f"2026-06-03_r27_adaptive_freeze_burst_h{args.hidden_size}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults saved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
