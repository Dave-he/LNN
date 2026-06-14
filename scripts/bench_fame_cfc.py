"""Smoke benchmark for FAMECfCCell / FAMECfCNetwork (PRD #10-36, 2026-06-14).

Trains a tiny ``FAMECfCNetwork`` on a noise-free sin/cos target with
n_experts K=3 fixed and top_k in {1, 2, 3} (top_k=3 = dense softmax
equivalence to round 77 MR-MoE), three seeds each, and reports mean ±
std of final train MSE plus average activated-experts-per-step
(a proxy for sparsity: 1.0 = pure argmax, 3.0 = dense).

Per the iter#24/35/37 honest-negative pattern, LNNs do not dominate
LSTM/MLP on toy noise-free datasets.  This bench therefore only
verifies the FAME sparse routing path is *not catastrophically worse*
than dense softmax and that sparsity is honoured (top_k=1 < top_k=2
< top_k=3 in mean activated experts).

Real advantages of FAME-style sparse routing are expected on
*heterogeneous* time series (paper §4: 5000+ vending machines with
diverse lifecycle / seasonality / volatility), not on a single
homogeneous sin curve.  The heterogeneous benchmark is a follow-up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make ``lnn`` importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.fame_cfc import FAMECfCNetwork  # noqa: E402


def _make_sin_cos(seq_len: int, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic sin/cos toy dataset.

    Returns:
        x: [N, T, 1] input = sin(t)
        y: [N, T, 1] target = cos(t)
    """
    t = torch.linspace(0, 2 * np.pi, seq_len).unsqueeze(0).expand(n_samples, -1)
    x = torch.sin(t).unsqueeze(-1)
    y = torch.cos(t).unsqueeze(-1)
    return x, y


def _train_one(
    top_k: int,
    n_experts: int,
    hidden_size: int,
    seq_len: int,
    n_samples: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """Train one (top_k, seed) configuration. Returns metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = _make_sin_cos(seq_len, n_samples)
    net = FAMECfCNetwork(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        num_layers=1,
        n_experts=n_experts,
        top_k=top_k,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    final = float("nan")
    activated_per_step = 0.0
    n_steps_logged = 0
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        final = float(loss.item())
        # Sample activated experts on a fresh batch for diagnostics.
        with torch.no_grad():
            _ = net.cells[0](x[0:1, 0, :], torch.zeros(1, hidden_size), dt=1.0)
            g = net.cells[0].last_g  # [1, K]
            activated = (g > 0).sum(dim=-1).item()  # K' (top_k)
            activated_per_step += activated
            n_steps_logged += 1
    return {
        "loss": final,
        "activated_per_step": activated_per_step / max(1, n_steps_logged),
    }


def run_bench(
    top_k_list: list[int],
    seeds: list[int],
    n_experts: int = 3,
    epochs: int = 30,
    hidden_size: int = 16,
    seq_len: int = 32,
    n_samples: int = 64,
    lr: float = 0.01,
) -> dict:
    results: dict = {}
    for tk in top_k_list:
        rows = []
        for s in seeds:
            rows.append(
                _train_one(
                    top_k=tk,
                    n_experts=n_experts,
                    hidden_size=hidden_size,
                    seq_len=seq_len,
                    n_samples=n_samples,
                    epochs=epochs,
                    lr=lr,
                    seed=s,
                )
            )
        losses = np.array([r["loss"] for r in rows])
        acts = np.array([r["activated_per_step"] for r in rows])
        results[str(tk)] = {
            "top_k": tk,
            "loss_mean": float(losses.mean()),
            "loss_std": float(losses.std()),
            "loss_min": float(losses.min()),
            "loss_max": float(losses.max()),
            "activated_mean": float(acts.mean()),
            "activated_std": float(acts.std()),
            "raw_losses": [float(x) for x in losses],
            "raw_activated": [float(x) for x in acts],
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--top-k", dest="top_k_list", type=int, nargs="+", default=[1, 2, 3],
    )
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "logs" / "bench_fame_cfc.json"),
    )
    args = parser.parse_args()

    results = run_bench(
        top_k_list=args.top_k_list,
        seeds=args.seeds,
        n_experts=args.n_experts,
        epochs=args.epochs,
        hidden_size=args.hidden,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
        lr=args.lr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "epochs": args.epochs,
            "seeds": args.seeds,
            "top_k_list": args.top_k_list,
            "n_experts": args.n_experts,
            "hidden": args.hidden,
            "seq_len": args.seq_len,
            "n_samples": args.n_samples,
            "lr": args.lr,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("=" * 78)
    print(
        f"FAME CfC smoke bench  (epochs={args.epochs}, seeds={args.seeds}, "
        f"n_experts={args.n_experts})"
    )
    print("=" * 78)
    for k, v in results.items():
        print(
            f"  top_k={k:>3}  loss={v['loss_mean']:.4f}±{v['loss_std']:.4f}  "
            f"activated={v['activated_mean']:.2f}±{v['activated_std']:.2f}/step  "
            f"raw_loss={v['raw_losses']}  raw_act={v['raw_activated']}"
        )
    print(f"\nResults written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
