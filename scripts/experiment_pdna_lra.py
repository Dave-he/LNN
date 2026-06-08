#!/usr/bin/env python3
"""PDNA stage C smoke ablation (PRD §10 #10 stage C — Long Range Arena).

Tests whether the PDNA LNN augmentation signal from sMNIST Gapped
(iter#20, +2.53 pp on multi-gap) **generalises** to a standard LRA-style
long-range task — synthetic Pathfinder.

Synthetic Pathfinder (no download):
- 32x32 grayscale image, row-major flattened to [1024]-length sequence
- 2 endpoint markers + 0/1 random piecewise-linear path
- Binary classification: are the two endpoints connected?
- Class balance 50/50; endpoints placed at least 16 cells apart so the
  task genuinely requires long-range integration.

Variants (mirrors iter#20 subset — drops cfc_selfattend to keep 3 arms for time):
  A. baseline_cfc   : CfC only
  B. cfc_pulse      : CfC + PDNAPulseHead(use_self_attend=False)
  C. full_pdna      : CfC + PDNAPulseHead(use_self_attend=True)

Usage::

    python scripts/experiment_pdna_lra.py \\
        --seeds 3 --hidden-size 64 --epochs 5 --train-samples 2000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.core.cfc import CfCNetwork  # noqa: E402
from lnn.data.pathfinder_synth import PathfinderConfig, generate_pathfinder  # noqa: E402

# Reuse the iter#20 model class to avoid duplication.
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_pdna_smoke import PDNAClassifier  # noqa: E402

ANALYSIS_DIR = ROOT / "analysis" / "pdna_lra"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------- train/eval
def _train_one(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> dict:
    device = torch.device("cpu")  # CPU-only smoke
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.to(device)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    train_seconds = time.time() - t0

    # Evaluate
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total += y.numel()
    acc = 100.0 * n_correct / n_total
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"train_seconds": train_seconds, "n_params": n_params, "test_acc": acc}


# --------------------------------------------------------- one full run
def _run_variant(name: str, args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = PathfinderConfig()
    train_seqs, train_labels = generate_pathfinder(args.train_samples, cfg=cfg, seed=seed)
    test_seqs, test_labels = generate_pathfinder(args.test_samples, cfg=cfg, seed=seed + 99999)
    # PathfinderConfig has grid_size=32, so seq_len = 32*32 = 1024.
    seq_len = cfg.grid_size * cfg.grid_size
    # Reshape to [B, T, F] where T=seq_len, F=1 (grayscale per pixel).
    train_loader = DataLoader(
        TensorDataset(train_seqs.view(-1, seq_len, 1), train_labels),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_seqs.view(-1, seq_len, 1), test_labels),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    if name == "baseline_cfc":
        model = PDNAClassifier(input_size=1, hidden_size=args.hidden_size, num_classes=2, use_pulse=False, use_attend=False)
    elif name == "cfc_pulse":
        model = PDNAClassifier(input_size=1, hidden_size=args.hidden_size, num_classes=2, use_pulse=True, use_attend=True, attend_beta=0.0)
    elif name == "full_pdna":
        model = PDNAClassifier(input_size=1, hidden_size=args.hidden_size, num_classes=2, use_pulse=True, use_attend=True)
    else:
        raise ValueError(f"Unknown variant: {name}")

    payload = _train_one(model, train_loader, test_loader, args)
    payload["variant"] = name
    payload["seed"] = seed
    payload["seq_len"] = seq_len
    payload["grid_size"] = cfg.grid_size
    return payload


# -------------------------------------------------------- aggregation
def _aggregate(per_run: list[dict]) -> dict:
    accs = [r["test_acc"] for r in per_run]
    return {
        "n_seeds": len(per_run),
        "test_acc_mean": statistics.mean(accs),
        "test_acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "train_seconds_mean": statistics.mean([r["train_seconds"] for r in per_run]),
        "n_params_mean": int(statistics.mean([r["n_params"] for r in per_run])),
    }


def _small_n_flag(n: int) -> str:
    if n < 3:
        return f" ⚠️N<3 (n={n})"
    if n < 5:
        return f" ⚠N<5 (n={n})"
    return f" n={n}"


def _format_markdown(payload: dict) -> str:
    md = []
    md.append("# PDNA stage C — synthetic Pathfinder (LRA-style long-range) ablation\n")
    md.append(f"_Generated: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}_\n")
    md.append(f"_Variants × seeds: {len(payload['variants'])} × {payload['n_seeds']}_\n")
    md.append(f"_Hidden size: {payload['hidden_size']}, Epochs: {payload['epochs']}, "
              f"Train samples: {payload['train_samples']}, Test samples: {payload['test_samples']}_\n")
    md.append(f"_Seq len: {payload['seq_len']} (= 32x32 pixel sequence), Grid: {payload['grid_size']}x{payload['grid_size']}_\n")
    md.append("\n## Per-variant test accuracy\n")
    md.append("| Variant | n_params | test_acc (mean ± std) | train_s |")
    md.append("|---|---:|---:|---:|")
    for v in payload["variants"]:
        a = v["aggregate"]
        flag = _small_n_flag(a["n_seeds"])
        md.append(
            f"| {v['name']}{flag} | {a['n_params_mean']} | "
            f"{a['test_acc_mean']:.2f}±{a['test_acc_std']:.2f} | "
            f"{a['train_seconds_mean']:.1f} |"
        )
    md.append("\n## Key deltas (vs baseline_cfc)\n")
    base = next(v for v in payload["variants"] if v["name"] == "baseline_cfc")["aggregate"]
    md.append("| Comparison | Δtest_acc (pp) | Verdict |")
    md.append("|---|---:|---|")
    for v in payload["variants"]:
        if v["name"] == "baseline_cfc":
            continue
        a = v["aggregate"]
        d = a["test_acc_mean"] - base["test_acc_mean"]
        verdict = "✅ better" if d > 1.0 else ("❌ worse" if d < -1.0 else "🟰 mixed")
        md.append(f"| {v['name']} | {d:+.2f} | {verdict} |")
    md.append("\n## Per-seed raw accuracies\n")
    for v in payload["variants"]:
        md.append(f"\n### {v['name']}\n")
        md.append("| seed | test_acc |")
        md.append("|---:|---:|")
        for r in v["per_seed"]:
            md.append(f"| {r['seed']} | {r['test_acc']:.2f} |")
    return "\n".join(md) + "\n"


# ----------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument(
        "--variants", nargs="+",
        default=["baseline_cfc", "cfc_pulse", "full_pdna"],
    )
    parser.add_argument("--out-prefix", type=str, default=dt.date.today().isoformat())
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="If a per-seed JSON for (variant, seed) already exists, skip it.",
    )
    args = parser.parse_args()

    print(f"[pdna-lra] device=cpu, seeds={args.seeds}, "
          f"hidden={args.hidden_size}, epochs={args.epochs}, variants={args.variants}", flush=True)

    variants_payload = []
    for v_name in args.variants:
        per_seed = []
        for s in range(args.seeds):
            seed = 42 + s * 1111  # 42, 1153, 2264
            seed_json = ANALYSIS_DIR / f"{args.out_prefix}_pdna_lra_{v_name}_seed{seed}.json"
            if args.skip_existing and seed_json.exists():
                print(f"[pdna-lra] skip existing {seed_json.name}", flush=True)
                with open(seed_json) as fh:
                    res = json.load(fh)
            else:
                print(f"[pdna-lra] running variant={v_name} seed={seed} ...", flush=True)
                res = _run_variant(v_name, args, seed)
                with open(seed_json, "w") as fh:
                    json.dump(res, fh, indent=2)
            per_seed.append(res)
        agg = _aggregate(per_seed)
        variants_payload.append({"name": v_name, "per_seed": per_seed, "aggregate": agg})

    payload = {
        "variants": variants_payload,
        "n_seeds": args.seeds,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "seq_len": 32 * 32,
        "grid_size": 32,
    }
    md = _format_markdown(payload)
    md_path = ANALYSIS_DIR / f"{args.out_prefix}_pdna_lra_summary.md"
    json_path = ANALYSIS_DIR / f"{args.out_prefix}_pdna_lra_summary.json"
    with open(md_path, "w") as fh:
        fh.write(md)
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[pdna-lra] wrote {md_path} + {json_path}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
