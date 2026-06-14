"""Smoke benchmark for CosineRouter (PRD #10-41, 2026-06-14).

Compares the parameter-free CosineRouter (arXiv:2605.12476) against
the round 80-81 stack on the K=3 top_k=1 cell (round 79 hard-blocker):

1. **learned** (round 78 baseline, no φ, no orth)
2. **learned + orth (λ=0.001)** (round 80)
3. **learned + φ (η=0.05)** (round 81)
4. **cosine** (new, round 82, no learned router)
5. **cosine + orth (λ=0.001)** (new, round 82)

Reports task loss mean ± std across 3 seeds.
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
    router_type: str,
    lambda_coeff: float,
    phi_step_size: float,
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
    use_phi = phi_step_size > 0.0
    net = FAMECfCNetwork(
        input_size=1, hidden_size=hidden_size, output_size=1,
        num_layers=1, n_experts=K, top_k=top_k,
        router_type=router_type,
        phi_balance=use_phi, ema_alpha=0.05, phi_step_size=phi_step_size,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    final = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        y_pred, expert_outs = net.forward_with_aux(x)
        task_loss = loss_fn(y_pred, y)
        last_outs = expert_outs[0][-1]  # K × [B, H]
        if lambda_coeff > 0.0:
            aux = orthogonality_loss(last_outs, lambda_coeff=lambda_coeff)
            total = task_loss + aux
        else:
            total = task_loss
        total.backward()
        opt.step()
        final = float(task_loss.item())
    return {"task_loss": final}


def run_bench(
    conditions: list[dict],
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
    for cond in conditions:
        rows = [
            _train_one(
                router_type=cond["router"],
                lambda_coeff=cond["lambda"],
                phi_step_size=cond["phi"],
                K=K, top_k=top_k,
                hidden_size=hidden_size, seq_len=seq_len, n_samples=n_samples,
                epochs=epochs, lr=lr, seed=s,
            )
            for s in seeds
        ]
        losses = np.array([r["task_loss"] for r in rows])
        key = f"{cond['router']}_lam={cond['lambda']}_eta={cond['phi']}"
        results[key] = {
            "router": cond["router"],
            "lambda": cond["lambda"],
            "phi_step_size": cond["phi"],
            "task_loss_mean": float(losses.mean()),
            "task_loss_std": float(losses.std()),
            "task_loss_min": float(losses.min()),
            "task_loss_max": float(losses.max()),
            "raw_task_losses": [float(x) for x in losses],
            "diverged_seeds": int((losses > 0.5).sum()),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--top-k", dest="top_k", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=32)
    parser.add_argument("--n-samples", dest="n_samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "logs" / "bench_cosine_router.json"),
    )
    args = parser.parse_args()

    # 5 conditions: learned baseline, learned+orth, learned+phi, cosine, cosine+orth.
    conditions = [
        {"router": "learned", "lambda": 0.0,   "phi": 0.0},   # round 78 raw
        {"router": "learned", "lambda": 0.001, "phi": 0.0},   # round 80 orth
        {"router": "learned", "lambda": 0.0,   "phi": 0.05},  # round 81 phi
        {"router": "cosine",  "lambda": 0.0,   "phi": 0.0},   # round 82 cosine
        {"router": "cosine",  "lambda": 0.001, "phi": 0.0},   # round 82 cosine+orth
    ]
    results = run_bench(
        conditions=conditions, seeds=args.seeds,
        K=args.K, top_k=args.top_k,
        epochs=args.epochs, hidden_size=args.hidden,
        seq_len=args.seq_len, n_samples=args.n_samples, lr=args.lr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "epochs": args.epochs, "seeds": args.seeds,
            "K": args.K, "top_k": args.top_k,
            "hidden": args.hidden, "seq_len": args.seq_len,
            "n_samples": args.n_samples, "lr": args.lr,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("=" * 78)
    print(
        f"CosineRouter bench  (K={args.K}, top_k={args.top_k}, "
        f"epochs={args.epochs}, seeds={args.seeds})"
    )
    print("=" * 78)
    for k, v in results.items():
        print(
            f"  {k:<35}  task={v['task_loss_mean']:.4f}±{v['task_loss_std']:.4f}  "
            f"diverged={v['diverged_seeds']}  raw={v['raw_task_losses']}"
        )
    print(f"\nResults written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
