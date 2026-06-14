"""Smoke benchmark for CfCCell n_tau multi-time-scale support (PRD #10-29).

Trains a tiny ``CfCNetwork`` on a noise-free sin/cos target with n_tau in
{1, 3, 5}, three seeds each, and reports mean ± std of final train MSE.

Per the iter#24/35/37 honest-negative pattern, LNNs do not dominate
LSTM/MLP on toy noise-free datasets.  This bench therefore only
verifies the multi-τ path is *not catastrophically worse* than the
single-τ path and that all seeds converge.  Real advantages are
expected on noisy / long-horizon / multi-scale data per the COGENT
and MR-MoE arXiv references; that is a follow-up benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Make ``lnn`` importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.cfc import CfCNetwork  # noqa: E402


@dataclass
class BenchConfig:
    n_tau: int
    hidden_size: int
    num_layers: int
    seq_len: int
    n_samples: int
    epochs: int
    lr: float
    seed: int
    tau_scales: tuple


def _make_sin_cos(seq_len: int, n_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic sin/cos toy dataset.

    Returns:
        x: [N, T, 1] input = sin(t)
        y: [N, T, 1] target = cos(t)
    """
    g = torch.Generator().manual_seed(seed)
    del g  # touched for reproducibility audit only
    t = torch.linspace(0, 2 * np.pi, seq_len).unsqueeze(0).expand(n_samples, -1)
    x = torch.sin(t).unsqueeze(-1)
    y = torch.cos(t).unsqueeze(-1)
    return x, y


def _train_one(cfg: BenchConfig) -> float:
    """Train one (n_tau, seed) configuration. Returns final epoch loss."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x, y = _make_sin_cos(cfg.seq_len, cfg.n_samples, cfg.seed)
    net = CfCNetwork(
        input_size=1,
        hidden_size=cfg.hidden_size,
        output_size=1,
        num_layers=cfg.num_layers,
        n_tau=cfg.n_tau,
        tau_scales=cfg.tau_scales,
    )
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()
    final = float("nan")
    for _ in range(cfg.epochs):
        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        final = float(loss.item())
    return final


def run_bench(
    n_taus: list[int],
    seeds: list[int],
    epochs: int = 40,
    hidden_size: int = 16,
    seq_len: int = 32,
    n_samples: int = 64,
    num_layers: int = 1,
    lr: float = 0.01,
    tau_scales: tuple = (0.1, 1.0, 10.0),
) -> dict:
    """Run the n_tau sweep and return mean/std of final train MSE per n_tau."""
    results: dict = {}
    for n_tau in n_taus:
        losses = []
        for s in seeds:
            cfg = BenchConfig(
                n_tau=n_tau,
                hidden_size=hidden_size,
                num_layers=num_layers,
                seq_len=seq_len,
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                seed=s,
                tau_scales=tau_scales,
            )
            losses.append(_train_one(cfg))
        arr = np.array(losses)
        results[str(n_tau)] = {
            "n_tau": n_tau,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "raw": losses,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-taus", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "logs" / "bench_cfc_n_tau.json"),
    )
    args = parser.parse_args()

    results = run_bench(
        n_taus=args.n_taus,
        seeds=args.seeds,
        epochs=args.epochs,
        hidden_size=args.hidden,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
        num_layers=args.num_layers,
        lr=args.lr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "epochs": args.epochs,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "seq_len": args.seq_len,
            "n_samples": args.n_samples,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "n_taus": args.n_taus,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    # Pretty print
    print("=" * 60)
    print(f"CfC n_tau smoke bench  (epochs={args.epochs}, seeds={args.seeds})")
    print("=" * 60)
    for k, v in results.items():
        print(
            f"  n_tau={k:>3}  mean={v['mean']:.4f}  std={v['std']:.4f}  "
            f"min={v['min']:.4f}  max={v['max']:.4f}  raw={v['raw']}"
        )
    print(f"\nResults written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
