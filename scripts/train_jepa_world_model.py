#!/usr/bin/env python3
"""Train a JEPA world model on (obs, action, next_obs) transitions.

Phase C of PRD-MSD-LNN. Uses transitions from
``scripts/bench_pid_expert_demo.py`` (PD-controller rollouts) to fit
a :class:`lnn.core.jepa_world_model.JEPAPolicy.forward_train` model
that predicts next-latent given current-latent + action.

Outputs:
    - checkpoints/jepa_<ts>.pt       — full model state_dict
    - analysis/decisions/<ts>_jepa_world_model.json  — train stats
    - analysis/decisions/<ts>_jepa_world_model.md    — human summary

Typical usage:
    python scripts/train_jepa_world_model.py --epochs 30 --latent-dim 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.jepa_world_model import JEPAPolicy  # noqa: E402


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def find_latest_demos() -> tuple[Path, Path, Path]:
    """Find the most recent PD demos JSON under analysis/decisions/."""
    out_dir = REPO_ROOT / "analysis" / "decisions"
    candidates = sorted(out_dir.glob("*_pointmass_pid_demos_n0.json"))
    if not candidates:
        raise FileNotFoundError(f"no *_pointmass_pid_demos_n0.json under {out_dir}")
    latest = candidates[-1]
    payload = json.loads(latest.read_text())
    return (
        torch.tensor(payload["obs"], dtype=torch.float32),
        torch.tensor(payload["actions"], dtype=torch.float32),
        torch.tensor(payload["next_obs"], dtype=torch.float32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--demos", type=Path, default=None,
                        help="path to *_pointmass_pid_demos_n0.json (default: latest)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=1)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    # Load demos.
    if args.demos is not None:
        payload = json.loads(args.demos.read_text())
        obs = torch.tensor(payload["obs"], dtype=torch.float32)
        actions = torch.tensor(payload["actions"], dtype=torch.float32)
        next_obs = torch.tensor(payload["next_obs"], dtype=torch.float32)
    else:
        obs, actions, next_obs = find_latest_demos()
    N = obs.shape[0]
    obs_dim = obs.shape[1]
    action_dim = actions.shape[1]
    print(
        f"[train-jepa] N={N} obs_dim={obs_dim} action_dim={action_dim} "
        f"latent_dim={args.latent_dim} hidden={args.hidden_size} "
        f"n_tau={args.n_tau} head={args.head_type}",
        flush=True,
    )

    # Treat each transition as a length-1 sequence for now.
    # (Full multi-step BPTT could be a future enhancement.)
    train_obs = obs.unsqueeze(1)        # [N, 1, obs_dim]
    train_next = next_obs.unsqueeze(1)  # [N, 1, obs_dim]
    train_act = actions.unsqueeze(1)    # [N, 1, action_dim]

    policy = JEPAPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        head_type=args.head_type,
        n_tau=args.n_tau,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    history: list[dict] = []
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        running_loss = 0.0
        n_batches = 0
        for start in range(0, N, args.batch_size):
            idx = perm[start:start + args.batch_size]
            out = policy.forward_train(
                train_obs[idx], train_next[idx], train_act[idx],
            )
            mse = out["mse"]
            opt.zero_grad()
            mse.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running_loss += float(mse.item())
            n_batches += 1
        avg = running_loss / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "mse": avg})
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"[train-jepa] epoch {epoch+1}/{args.epochs} mse={avg:.5f}", flush=True)

    final_mse = history[-1]["mse"]
    initial_mse = history[0]["mse"]
    print(f"[train-jepa] final_mse={final_mse:.5f} initial_mse={initial_mse:.5f}", flush=True)

    # Persist outputs.
    stamp = _ts()
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}_jepa_world_model.json"
    md_path = out_dir / f"{stamp}_jepa_world_model.md"

    json_payload = {
        "config": vars(args),
        "n_transitions": N,
        "history": history,
        "final_mse": final_mse,
        "initial_mse": initial_mse,
        "improvement_ratio": initial_mse / max(final_mse, 1e-9),
    }
    json_path.write_text(json.dumps(json_payload, indent=2))

    md = [
        f"# JEPA world-model training ({stamp})",
        "",
        f"PD demos: ``{N}`` transitions, obs_dim={obs_dim}, action_dim={action_dim}.",
        f"Latent dim: {args.latent_dim}, hidden: {args.hidden_size}, n_tau: {args.n_tau}.",
        f"head_type: {args.head_type}, epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}.",
        "",
        f"Initial MSE: **{initial_mse:.5f}**",
        f"Final MSE:   **{final_mse:.5f}**",
        f"Improvement: **{initial_mse / max(final_mse, 1e-9):.2f}x**",
        "",
        "Per-epoch MSE:",
        "",
        "| epoch | mse |",
        "|---:|---:|",
    ]
    md.extend(f"| {h['epoch']} | {h['mse']:.5f} |" for h in history)
    md_path.write_text("\n".join(md) + "\n")

    print(f"[train-jepa] wrote {json_path}")
    print(f"[train-jepa] wrote {md_path}")

    if args.save_checkpoint:
        ckpt_dir = REPO_ROOT / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / f"jepa_{stamp}.pt"
        torch.save(policy.state_dict(), ckpt)
        print(f"[train-jepa] checkpoint {ckpt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())