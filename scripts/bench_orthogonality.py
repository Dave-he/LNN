"""Smoke benchmark for orthogonality constraint (PRD #10-37, 2026-06-14).

Trains a tiny ``FAMECfCNetwork`` (K=3, top_k=1 — the unstable cell
that round 79 sweep identified) on toy sin/cos with
``lambda_coeff ∈ {0.0, 0.001, 0.01, 0.1, 1.0}`` for the geometric
orthogonality constraint.  Reports mean ± std of final task loss
across 3 seeds, plus a "diverged?" flag (final loss > 0.5 = unstable).

The motivation is the round 79 sweep finding: K=3 top_k=1 produced
mean loss 0.7595 ± 0.7906 (with one seed diverging to 1.86) because
the router-argmax single-expert mode is unstable when experts can
collapse to similar representations.  Orthogonality is the
defensive counter-measure.
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
from lnn.core.orthogonality import orthogonality_loss  # noqa: E402


def _make_sin_cos(seq_len: int, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 2 * np.pi, seq_len).unsqueeze(0).expand(n_samples, -1)
    x = torch.sin(t).unsqueeze(-1)
    y = torch.cos(t).unsqueeze(-1)
    return x, y


def _train_one(
    lambda_coeff: float,
    K: int,
    top_k: int,
    hidden_size: int,
    seq_len: int,
    n_samples: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = _make_sin_cos(seq_len, n_samples)
    net = FAMECfCNetwork(
        input_size=1, hidden_size=hidden_size, output_size=1,
        num_layers=1, n_experts=K, top_k=top_k,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    final = float("nan")
    final_orth = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        y_pred, expert_outs = net.forward_with_aux(x)
        task_loss = loss_fn(y_pred, y)
        # Use only the last step's expert outputs from the first (only) layer.
        last_outs = expert_outs[0][-1]  # K × [B, H]
        aux = orthogonality_loss(last_outs, lambda_coeff=lambda_coeff)
        total = task_loss + aux
        total.backward()
        opt.step()
        final = float(task_loss.item())
        final_orth = float(aux.item())
    return {"task_loss": final, "aux_loss": final_orth}


def run_bench(
    lambda_coeffs: list[float],
    seeds: list[int],
    K: int = 3,
    top_k: int = 1,
    epochs: int = 25,
    hidden_size: int = 16,
    seq_len: int = 32,
    n_samples: int = 64,
    lr: float = 0.01,
) -> dict:
    results: dict = {}
    for lam in lambda_coeffs:
        rows = []
        for s in seeds:
            rows.append(
                _train_one(
                    lambda_coeff=lam, K=K, top_k=top_k,
                    hidden_size=hidden_size, seq_len=seq_len, n_samples=n_samples,
                    epochs=epochs, lr=lr, seed=s,
                )
            )
        losses = np.array([r["task_loss"] for r in rows])
        auxes = np.array([r["aux_loss"] for r in rows])
        results[str(lam)] = {
            "lambda": lam,
            "task_loss_mean": float(losses.mean()),
            "task_loss_std": float(losses.std()),
            "task_loss_min": float(losses.min()),
            "task_loss_max": float(losses.max()),
            "aux_loss_mean": float(auxes.mean()),
            "aux_loss_std": float(auxes.std()),
            "raw_task_losses": [float(x) for x in losses],
            "raw_aux_losses": [float(x) for x in auxes],
            "diverged_seeds": int((losses > 0.5).sum()),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--lambda-coeffs", dest="lambda_coeffs", type=float, nargs="+",
        default=[0.0, 0.001, 0.01, 0.1, 1.0],
    )
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--top-k", dest="top_k", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "logs" / "bench_orthogonality.json"),
    )
    args = parser.parse_args()

    results = run_bench(
        lambda_coeffs=args.lambda_coeffs,
        seeds=args.seeds,
        K=args.K, top_k=args.top_k,
        epochs=args.epochs, hidden_size=args.hidden,
        seq_len=args.seq_len, n_samples=args.n_samples, lr=args.lr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "epochs": args.epochs, "seeds": args.seeds,
            "lambda_coeffs": args.lambda_coeffs, "K": args.K, "top_k": args.top_k,
            "hidden": args.hidden, "seq_len": args.seq_len,
            "n_samples": args.n_samples, "lr": args.lr,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("=" * 78)
    print(
        f"Orthogonality bench  (K={args.K}, top_k={args.top_k}, "
        f"epochs={args.epochs}, seeds={args.seeds})"
    )
    print("=" * 78)
    for k, v in results.items():
        print(
            f"  λ={k:>6}  task={v['task_loss_mean']:.4f}±{v['task_loss_std']:.4f}  "
            f"aux={v['aux_loss_mean']:.4f}  diverged_seeds={v['diverged_seeds']}  "
            f"raw_task={v['raw_task_losses']}"
        )
    print(f"\nResults written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
