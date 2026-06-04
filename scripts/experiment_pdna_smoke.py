#!/usr/bin/env python3
"""PDNA stage B smoke ablation (PRD §10 #10 stage B).

Replicates the 4-variant × N-seed ablation from arXiv 2603.00153v1 §3.5
(Paras Sharma, 2026) on Sequential MNIST with the paper's Gapped evaluation
protocol (§4 of the paper).

Variants (mirrors the paper's Table 1):
  A. baseline_cfc       : CfCNetwork only
  B. cfc_noise          : CfCNetwork + matched-magnitude random noise
                          (the paper's critical control ruling out the
                           "any non-zero dynamics suffices" explanation)
  C. cfc_pulse          : CfCNetwork + PDNAPulseHead(use_self_attend=False)
  D. cfc_selfattend     : CfCNetwork + PDNAPulseHead(pulse=disabled, attend only)
                          (NOTE: our head combines both; we model this as
                           use_self_attend=True but with alpha pinned to 0
                           to keep a fair 1:1 with paper's "attend-only" arm)
  E. full_pdna          : CfCNetwork + PDNAPulseHead(use_self_attend=True)

Gapped protocol: contiguous gaps at 0%/5%/15%/30% (centered at T/2) and
multi-gap (4 evenly spaced gaps totalling 20%).

This is the CPU-friendly smoke version of the paper's RTX A4000 5×5 run;
we use 3 seeds and hidden=64 (paper uses 5 seeds, hidden=128) to keep
Jetson CPU wall-clock under 10 minutes per variant.

Usage::

    python scripts/experiment_pdna_smoke.py \
        --seeds 3 --hidden-size 64 --epochs 5 --train-subset 10000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork, PDNAPulseHead

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "pdna"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- dataset
def _load_smnist(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Sequential MNIST: 28 timesteps × 28 features, label = digit class."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)

    # Optional subset for smoke speed
    if args.train_subset and args.train_subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.train_subset)))
    if args.test_subset and args.test_subset < len(test_ds):
        test_ds = Subset(test_ds, list(range(args.test_subset)))

    # Unroll images: [B, 1, 28, 28] -> [B, 28, 28] (time-major, each row is a step)
    def _to_seq(ds):
        xs, ys = [], []
        for img, label in ds:
            xs.append(img.view(28, 28))  # [28, 28]
            ys.append(label)
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    train_seq = _to_seq(train_ds)
    test_seq = _to_seq(test_ds)
    train_loader = DataLoader(train_seq, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_seq, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ------------------------------------------------------------------ models
class PDNAClassifier(nn.Module):
    """CfC backbone + optional PDNAPulseHead + linear classifier on last step.

    The paper feeds `h[:, -1, :]` (last timestep hidden state) to the
    classifier. Our CfCNetwork applies output_proj to the full sequence;
    for the post-hoc head we use the backbone's *pre-output-proj* hidden
    state sequence so the head augments the representation directly.
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 64,
        num_classes: int = 10,
        use_pulse: bool = True,
        use_attend: bool = True,
        pulse_alpha: float | None = None,
        attend_beta: float | None = None,
    ):
        super().__init__()
        self.backbone = CfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=hidden_size, num_layers=1
        )
        # The CfCNetwork always applies output_proj at the end. We don't want that
        # for the head path; we expose the raw hidden states.
        self.backbone.output_proj = nn.Identity()
        self.use_pulse = use_pulse
        self.use_attend = use_attend
        if use_pulse or use_attend:
            self.head = PDNAPulseHead(
                hidden_size=hidden_size, use_self_attend=use_attend
            )
            if pulse_alpha is not None:
                with torch.no_grad():
                    self.head.alpha.fill_(pulse_alpha)
            if attend_beta is not None:
                with torch.no_grad():
                    self.head.beta.fill_(attend_beta)
        else:
            self.head = None
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h = self.backbone(x)              # [B, T, hidden] (output_proj is Identity)
        if self.head is not None:
            h = self.head(h)               # [B, T, hidden]
        return self.classifier(h[:, -1, :])  # [B, num_classes]


class NoiseHead(nn.Module):
    """The paper's Variant B critical control: matched-magnitude random noise."""

    def __init__(self, hidden_size: int, alpha_init: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Per-timestep per-dim Gaussian noise, gated by α
        noise = torch.randn_like(h)
        return h + self.alpha * noise


class NoisePDNAClassifier(nn.Module):
    """CfC + matched-magnitude noise head (the paper's Variant B)."""

    def __init__(self, input_size: int = 28, hidden_size: int = 64, num_classes: int = 10):
        super().__init__()
        self.backbone = CfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=hidden_size, num_layers=1
        )
        self.backbone.output_proj = nn.Identity()
        self.head = NoiseHead(hidden_size=hidden_size, alpha_init=0.01)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.head(h)
        return self.classifier(h[:, -1, :])


# ------------------------------------------------------------ gapped eval
def _apply_gap(x: torch.Tensor, gap_pct: float, multi: bool = False) -> torch.Tensor:
    """Zero out gap_pct of the timestep dimension. Used at test time only.

    Single contiguous gap is centered at T/2; multi-gap distributes 4 gaps
    evenly across the sequence (matching the paper's multi-gap protocol).
    """
    B, T, F = x.shape
    x_g = x.clone()
    if multi:
        n_gaps = 4
        gap_size = max(1, int(T * gap_pct / n_gaps))
        # 4 evenly spaced gap centers across [0, T-1]
        centers = [int((i + 0.5) * T / n_gaps) for i in range(n_gaps)]
        for c in centers:
            s = max(0, c - gap_size // 2)
            e = min(T, s + gap_size)
            x_g[:, s:e, :] = 0.0
    else:
        gap_size = int(T * gap_pct)
        if gap_size > 0:
            s = T // 2 - gap_size // 2
            e = s + gap_size
            x_g[:, s:e, :] = 0.0
    return x_g


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate on (gap 0% / 5% / 15% / 30% / multi-gap) and return dict of accuracies."""
    model.eval()
    gap_levels = {"0%": 0.0, "5%": 0.05, "15%": 0.15, "30%": 0.30}
    results = {**{k: 0.0 for k in gap_levels}, "multi20": 0.0}
    counts = {k: 0 for k in results}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        for label, pct in gap_levels.items():
            x_g = _apply_gap(x, pct, multi=False)
            logits = model(x_g)
            preds = logits.argmax(dim=-1)
            results[label] += (preds == y).sum().item()
            counts[label] += y.numel()
        x_m = _apply_gap(x, 0.20, multi=True)
        preds = model(x_m).argmax(dim=-1)
        results["multi20"] += (preds == y).sum().item()
        counts["multi20"] += y.numel()
    return {k: 100.0 * results[k] / counts[k] for k in results}


# ----------------------------------------------------------- train/eval
def _train_one(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
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

    # Gapped evaluation
    gap_results = _evaluate(model, test_loader, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"train_seconds": train_seconds, "n_params": n_params, **gap_results}


# --------------------------------------------------------- one full run
def _run_variant(name: str, args: argparse.Namespace, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    if name == "baseline_cfc":
        model = PDNAClassifier(use_pulse=False, use_attend=False)
    elif name == "cfc_noise":
        model = NoisePDNAClassifier()
    elif name == "cfc_pulse":
        # Pulse only: disable self-attend gate so attend contributes nothing
        model = PDNAClassifier(use_pulse=True, use_attend=True, attend_beta=0.0)
    elif name == "cfc_selfattend":
        # Self-attend only: pin pulse alpha to 0
        model = PDNAClassifier(use_pulse=True, use_attend=True, pulse_alpha=0.0)
    elif name == "full_pdna":
        model = PDNAClassifier(use_pulse=True, use_attend=True)
    else:
        raise ValueError(f"Unknown variant: {name}")

    payload = _train_one(model, *_load_smnist(args), args, device)
    payload["variant"] = name
    payload["seed"] = seed
    return payload


# -------------------------------------------------------- aggregation
def _aggregate(per_run: list[dict]) -> dict:
    """Mean ± std across seeds, per gap level. Mirrors the paper's Table 4."""
    keys = ["0%", "5%", "15%", "30%", "multi20"]
    agg: dict = {"n_seeds": len(per_run)}
    for k in keys:
        vals = [r[k] for r in per_run]
        agg[f"{k}_mean"] = statistics.mean(vals)
        agg[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    agg["train_seconds_mean"] = statistics.mean([r["train_seconds"] for r in per_run])
    agg["n_params_mean"] = int(statistics.mean([r["n_params"] for r in per_run]))
    return agg


def _small_n_flag(n: int) -> str:
    if n < 3:
        return f" ⚠️N<3 (n={n})"
    if n < 5:
        return f" ⚠N<5 (n={n})"
    return f" n={n}"


def _format_markdown(payload: dict) -> str:
    md = []
    md.append("# PDNA stage B ablation — sMNIST Gapped protocol (iter#20)\n")
    md.append(f"_Generated: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}_\n")
    md.append(f"_Variants × seeds: {len(payload['variants'])} × {payload['n_seeds']}_\n")
    md.append(f"_Hidden size: {payload['hidden_size']}, Epochs: {payload['epochs']}, "
              f"Train subset: {payload['train_subset']}_\n")
    md.append("\n## Per-variant accuracy across gap levels (mean ± std)\n")
    md.append("| Variant | n_params | Gap 0% | Gap 5% | Gap 15% | Gap 30% | Multi-gap |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for v in payload["variants"]:
        a = v["aggregate"]
        flag = _small_n_flag(a["n_seeds"])
        md.append(
            f"| {v['name']}{flag} | {a['n_params_mean']} | "
            f"{a['0%_mean']:.2f}±{a['0%_std']:.2f} | "
            f"{a['5%_mean']:.2f}±{a['5%_std']:.2f} | "
            f"{a['15%_mean']:.2f}±{a['15%_std']:.2f} | "
            f"{a['30%_mean']:.2f}±{a['30%_std']:.2f} | "
            f"{a['multi20_mean']:.2f}±{a['multi20_std']:.2f} |"
        )
    md.append("\n## Key deltas (vs baseline_cfc)\n")
    base = next(v for v in payload["variants"] if v["name"] == "baseline_cfc")["aggregate"]
    md.append("| Comparison | ΔGap 5% (pp) | ΔMulti-gap (pp) | Verdict |")
    md.append("|---|---:|---:|---|")
    for v in payload["variants"]:
        if v["name"] == "baseline_cfc":
            continue
        a = v["aggregate"]
        d5 = a["5%_mean"] - base["5%_mean"]
        dm = a["multi20_mean"] - base["multi20_mean"]
        verdict = "✅ better" if (d5 > 0 and dm > 0) else ("❌ worse" if (d5 < 0 and dm < 0) else "🟰 mixed")
        md.append(f"| {v['name']} | {d5:+.2f} | {dm:+.2f} | {verdict} |")
    md.append("\n## Per-seed raw accuracies\n")
    for v in payload["variants"]:
        md.append(f"\n### {v['name']} (n_params_mean={v['aggregate']['n_params_mean']})\n")
        md.append("| seed | Gap 0% | Gap 5% | Gap 15% | Gap 30% | Multi |")
        md.append("|---:|---:|---:|---:|---:|---:|")
        for r in v["per_seed"]:
            md.append(
                f"| {r['seed']} | {r['0%']:.2f} | {r['5%']:.2f} | "
                f"{r['15%']:.2f} | {r['30%']:.2f} | {r['multi20']:.2f} |"
            )
    return "\n".join(md) + "\n"


# ----------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--data-root", type=str, default=str(ROOT / "papers" / "data" / "mnist"))
    parser.add_argument(
        "--variants", nargs="+",
        default=["baseline_cfc", "cfc_noise", "cfc_pulse", "cfc_selfattend", "full_pdna"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-prefix", type=str, default=dt.date.today().isoformat())
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="If a per-seed JSON for (variant, seed) already exists in analysis/pdna/, skip it."
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[pdna-stage-b] device={device}, seeds={args.seeds}, "
          f"hidden={args.hidden_size}, epochs={args.epochs}, variants={args.variants}")

    variants_payload = []
    for v_name in args.variants:
        per_seed = []
        for s in range(args.seeds):
            seed = 42 + s * 1111  # 42, 1153, 2264
            seed_json = ANALYSIS_DIR / f"{args.out_prefix}_pdna_{v_name}_seed{seed}.json"
            if args.skip_existing and seed_json.exists():
                print(f"[pdna-stage-b] skip existing {seed_json.name}", flush=True)
                with open(seed_json) as f:
                    res = json.load(f)
            else:
                print(f"[pdna-stage-b] running variant={v_name} seed={seed} ...", flush=True)
                res = _run_variant(v_name, args, seed, device)
                with open(seed_json, "w") as f:
                    json.dump(res, f, indent=2)
            per_seed.append(res)
        agg = _aggregate(per_seed)
        variants_payload.append({"name": v_name, "per_seed": per_seed, "aggregate": agg})

    payload = {
        "variants": variants_payload,
        "n_seeds": args.seeds,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "train_subset": args.train_subset,
    }
    md = _format_markdown(payload)
    md_path = ANALYSIS_DIR / f"{args.out_prefix}_pdna_stage_b_summary.md"
    json_path = ANALYSIS_DIR / f"{args.out_prefix}_pdna_stage_b_summary.json"
    with open(md_path, "w") as f:
        f.write(md)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[pdna-stage-b] wrote {md_path} + {json_path}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
