#!/usr/bin/env python3
"""Quantization-aware training for distilled JEPA policy.

Phase C.4 of PRD-MSD-LNN. Addresses the Phase C.3 honest negative:

  * INT8 weight-only PTQ drops reach_rate from 1.000 (FP32) to 0.000.
  * Root cause: BC distillation reaches MSE 0.00001 — *below* the
    int8 weight-rounding noise floor. Weights are brittle.

Fix: during distillation, periodically quantize-and-dequantize the
policy weights in the forward pass (simulating the PTQ noise the
deployment-time INT8 model will see). This is "fake-quantization
ste" (QAT-lite) — not a true QAT implementation (we don't propagate
gradients through the fake-quant op), but it's enough to push the
distilled weights away from the rounding-noise sensitivity.

Expected: FP32 reach_rate stays at 1.000; INT8 PTQ reach_rate improves
from 0.000 toward 0.500+ (qualitatively robust). Numbers will be
re-measured by ``scripts/bench_jepa_quantization.py``.

Typical usage:
    python scripts/train_jepa_quant_aware.py --epochs 100 \
        --quantize-every 5 --quantize-bits 8
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


def fake_quantize_inplace(model: torch.nn.Module, num_bits: int = 8, per_channel: bool = True) -> None:
    """Round each weight to a fake-quantized grid then restore (QAT-lite).

    After this call, ``model.parameters()`` hold the *quantized* weights.
    Gradients flow through the rounded values (which detach from the
    original full-precision weights via the straight-through estimator
    pattern). We use a simple detach pattern: store the original
    weights, do ``w.data = quantize(w).data``, restore on the next
    forward. For QAT-lite we just keep weights quantized after training.
    """
    levels = 2 ** num_bits - 1
    for p in model.parameters():
        if p.dim() < 2:
            continue  # skip biases / 1-d params
        if per_channel:
            # per-channel: compute scale per output dim (dim 0)
            w = p.data
            abs_max = w.abs().reshape(w.shape[0], -1).max(dim=1).values
            abs_max = abs_max.clamp(min=1e-8).reshape(-1, *([1] * (w.dim() - 1)))
            scale = abs_max / levels
            q = torch.round(w / scale).clamp(-levels // 2 - 1, levels // 2)
            p.data = (q * scale).to(p.dtype)


def load_latest_demos() -> tuple[torch.Tensor, torch.Tensor]:
    out_dir = REPO_ROOT / "analysis" / "decisions"
    candidates = sorted(out_dir.glob("*_pointmass_pid_demos_n0.json"))
    payload = json.loads(candidates[-1].read_text())
    obs = torch.tensor(payload["obs"], dtype=torch.float32)
    actions = torch.tensor(payload["actions"], dtype=torch.float32)
    return obs, actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--quantize-every", type=int, default=5,
                        help="re-quantize weights every N epochs")
    parser.add_argument("--quantize-bits", type=int, default=8,
                        help="simulated weight precision (default 8)")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    obs, pd_actions = load_latest_demos()
    if args.max_states is not None:
        obs = obs[:args.max_states]
        pd_actions = pd_actions[:args.max_states]
    N = obs.shape[0]
    print(f"[qat] N={N} obs_dim={args.obs_dim} bits={args.quantize_bits} "
          f"every={args.quantize_every}", flush=True)

    policy = JEPAPolicy(
        obs_dim=args.obs_dim, action_dim=args.action_dim,
        latent_dim=args.latent_dim, hidden_size=args.hidden_size,
        head_type=args.head_type, n_tau=args.n_tau,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        running = 0.0
        n_batches = 0
        for s in range(0, N, args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = obs[idx]
            y = pd_actions[idx]
            z = policy.encoder.encode_step(x)
            pred = policy.policy_head(z, x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        # Fake-quantize weights after each epoch's update (QAT-lite).
        if (epoch + 1) % args.quantize_every == 0:
            fake_quantize_inplace(policy, num_bits=args.quantize_bits)
        avg = running / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "mse": avg})
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"[qat] epoch {epoch+1}/{args.epochs} mse={avg:.5f}", flush=True)

    initial_mse = history[0]["mse"]
    final_mse = history[-1]["mse"]
    print(f"[qat] final_mse={final_mse:.5f} initial_mse={initial_mse:.5f} "
          f"improvement={initial_mse / max(final_mse, 1e-9):.2f}x", flush=True)

    # Outputs.
    stamp = _ts()
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}_jepa_qat.json"
    md_path = out_dir / f"{stamp}_jepa_qat.md"
    payload = {
        "config": vars(args),
        "n_states": N,
        "history": history,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "improvement_ratio": initial_mse / max(final_mse, 1e-9),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md = [
        f"# JEPA QAT distillation — {stamp}",
        "",
        f"Target: PD demos ({N} states), obs_dim={args.obs_dim}, bits={args.quantize_bits}",
        f"Re-quantize every {args.quantize_every} epochs.",
        f"Initial MSE: **{initial_mse:.5f}**",
        f"Final MSE:   **{final_mse:.5f}**",
        f"Improvement: **{initial_mse / max(final_mse, 1e-9):.2f}x**",
        "",
        "Next: re-run ``bench_jepa_quantization.py`` to confirm INT8 "
        "reach_rate improved from 0.000 toward >= 0.500.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(f"[qat] wrote {json_path}")
    print(f"[qat] wrote {md_path}")

    if args.save_checkpoint:
        ckpt_dir = REPO_ROOT / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / f"jepa_qat_{args.quantize_bits}bit_{stamp}.pt"
        torch.save(policy.state_dict(), ckpt)
        print(f"[qat] checkpoint {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())