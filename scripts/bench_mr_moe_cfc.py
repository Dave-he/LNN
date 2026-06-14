"""Smoke benchmark for MRMoECfCCell (PRD #10-24, 2026-06-14).

Trains a tiny ``MRMoECfCNetwork`` on a noise-free sin/cos target with
n_experts K in {1, 3, 5}, three seeds each, and reports mean ± std of
final train MSE plus the average router entropy (a proxy for expert
specialisation: low entropy ≈ router collapse; high entropy ≈ uniform).

Per the iter#24/35/37 honest-negative pattern, LNNs do not dominate
LSTM/MLP on toy noise-free datasets.  This bench therefore only
verifies the K-expert path is *not catastrophically worse* than the
single-expert path and that the router does not collapse to a one-hot
during 30 epochs of training on toy data.  Real advantages are
expected on noisy / long-horizon / multi-scale data per the MR-MoE
arXiv reference; that is a follow-up benchmark.
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

from lnn.core.mr_moe_cfc import MRMoECfCNetwork  # noqa: E402


def _make_sin_cos(seq_len: int, n_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic sin/cos toy dataset.

    Returns:
        x: [N, T, 1] input = sin(t)
        y: [N, T, 1] target = cos(t)
    """
    del seed  # kept in the signature for caller-side seed documentation
    t = torch.linspace(0, 2 * np.pi, seq_len).unsqueeze(0).expand(n_samples, -1)
    x = torch.sin(t).unsqueeze(-1)
    y = torch.cos(t).unsqueeze(-1)
    return x, y


def _train_one(
    n_experts: int,
    n_tau_per_expert: int,
    hidden_size: int,
    seq_len: int,
    n_samples: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """Train one (n_experts, seed) configuration. Returns metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = _make_sin_cos(seq_len, n_samples, seed)
    net = MRMoECfCNetwork(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        num_layers=1,
        n_experts=n_experts,
        n_tau_per_expert=n_tau_per_expert,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    final = float("nan")
    final_entropy = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        final = float(loss.item())
        # Sample router entropy on a fresh batch for diagnostics.
        with torch.no_grad():
            _ = net.cells[0](x[0:1, 0, :], torch.zeros(1, hidden_size), dt=1.0)
            g = net.cells[0].last_g  # [1, K]
            ent = -(g * g.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
            final_entropy = float(ent)
    return {"loss": final, "router_entropy": final_entropy}


def run_bench(
    n_experts_list: list[int],
    seeds: list[int],
    n_tau_per_expert: int = 1,
    epochs: int = 30,
    hidden_size: int = 16,
    seq_len: int = 32,
    n_samples: int = 64,
    lr: float = 0.01,
) -> dict:
    results: dict = {}
    for k in n_experts_list:
        rows = []
        for s in seeds:
            rows.append(
                _train_one(
                    n_experts=k,
                    n_tau_per_expert=n_tau_per_expert,
                    hidden_size=hidden_size,
                    seq_len=seq_len,
                    n_samples=n_samples,
                    epochs=epochs,
                    lr=lr,
                    seed=s,
                )
            )
        losses = np.array([r["loss"] for r in rows])
        ents = np.array([r["router_entropy"] for r in rows])
        results[str(k)] = {
            "n_experts": k,
            "loss_mean": float(losses.mean()),
            "loss_std": float(losses.std()),
            "loss_min": float(losses.min()),
            "loss_max": float(losses.max()),
            "entropy_mean": float(ents.mean()),
            "entropy_std": float(ents.std()),
            "raw_losses": [float(x) for x in losses],
            "raw_entropies": [float(x) for x in ents],
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--n-experts", dest="n_experts_list", type=int, nargs="+", default=[1, 3, 5],
    )
    parser.add_argument("--n-tau-per-expert", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "logs" / "bench_mr_moe_cfc.json"),
    )
    args = parser.parse_args()

    results = run_bench(
        n_experts_list=args.n_experts_list,
        seeds=args.seeds,
        n_tau_per_expert=args.n_tau_per_expert,
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
            "n_experts_list": args.n_experts_list,
            "n_tau_per_expert": args.n_tau_per_expert,
            "hidden": args.hidden,
            "seq_len": args.seq_len,
            "n_samples": args.n_samples,
            "lr": args.lr,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("=" * 70)
    print(
        f"MR-MoE CfC smoke bench  (epochs={args.epochs}, seeds={args.seeds}, "
        f"n_tau_per_expert={args.n_tau_per_expert})"
    )
    print("=" * 70)
    for k, v in results.items():
        print(
            f"  K={k:>3}  loss={v['loss_mean']:.4f}±{v['loss_std']:.4f}  "
            f"entropy={v['entropy_mean']:.4f}±{v['entropy_std']:.4f}  "
            f"raw_loss={v['raw_losses']}  raw_ent={v['raw_entropies']}"
        )
    print(f"\nResults written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
