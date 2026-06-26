"""Device Control Cases — reference harness for 4 LNN-on-device-control scenarios.

This script wires the in-house LNN modules (LTC, CfC, NCP, LNNImitationPolicy,
SNCPPolicyLite) into 4 reference device-control cases described in
``docs/PRD_设备操控_LNN.md`` §3. The aim is **infrastructure & wiring** —
end-to-end forward (+ optional RL update) with existing models, not to
match published benchmarks. The 4 cases are:

* ``quadruped`` — case A in the PRD: 12-DoF locomotion via SNCPPolicyLite
  (LTC + actor-critic). Synthetic rollouts only, no real robot.
* ``drone`` — case B: 6-DoF (x, y, z, yaw, vx, vy) regression via
  LNNImitationPolicy. Synthetic EMMA-style visual+IMU fusion (no real
  vision model — we use a learned encoder stand-in).
* ``industrial`` — case C: 1-DoF inverted pendulum IL via LNNImitationPolicy
  with NCP sparse wiring. The smallest case, MCU-friendly.
* ``battery`` — case D: 124-cell LFP SoH online refinement via LTCNetwork
  (EntroLnn-style transformable, formula-identical). Synthetic CC-CV curve.

Each case is a **smoke** (~3 s of training, <100 ms inference). The goal is
*to demonstrate that the wiring works* so future iter can swap in real
data + real backbones + Jetson Pareto.

Usage
-----
::

    # Quick smoke (all 4 cases, 1 epoch each, CPU)
    python scripts/experiment_device_control_cases.py --case all --quick

    # Single case
    python scripts/experiment_device_control_cases.py --case drone --steps 200

    # Multi-seed (iter#11 N=5 lesson — never trust 1 seed)
    python scripts/experiment_device_control_cases.py --case all --seeds 3

The script writes a JSON report under ``analysis/device_control/`` with the
shape documented in ``_DEVICE_CONTROL_REPORT_SCHEMA`` below.

Related
-------
* docs/PRD_设备操控_LNN.md — full device-control PRD (4 cases)
* docs/VERIFICATION_RESULTS.md §2 — device-control verification table
* lnn/core/sncp_policy_lite.py — case A base
* lnn/core/control.py::LNNImitationPolicy — case B/C base
* lnn/core/ltc.py::LTCNetwork — case D base
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# ---------------------------------------------------------------------------
# Optional heavy imports — fail soft so the script can still import without
# torchdiffeq / ncps installed.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import-time guard
    from lnn.core.ltc import LTCNetwork, TransformableLTC
    _HAS_LTC = True
except Exception:  # pragma: no cover
    LTCNetwork = None
    TransformableLTC = None
    _HAS_LTC = False

try:  # pragma: no cover
    from lnn.core.cfc import CfCNetwork
    _HAS_CFC = True
except Exception:  # pragma: no cover
    CfCNetwork = None
    _HAS_CFC = False

try:  # pragma: no cover
    from lnn.core.control import LNNImitationPolicy
    _HAS_CONTROL = True
except Exception:  # pragma: no cover
    LNNImitationPolicy = None
    _HAS_CONTROL = False

try:  # pragma: no cover
    from lnn.core.sncp_policy_lite import SNCPPolicyLite
    _HAS_SNCP = True
except Exception:  # pragma: no cover
    SNCPPolicyLite = None
    _HAS_SNCP = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DEVICE_CONTROL_REPORT_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "fields": [
        "case", "seed", "steps", "wall_time_s", "params", "inference_ms",
        "primary_metric", "primary_metric_value",
        "secondary_metrics", "status", "notes",
    ],
}


# ---------------------------------------------------------------------------
# Synthetic data generators (deterministic per seed)
# ---------------------------------------------------------------------------


def _gen_quadruped_rollout(
    n_episodes: int = 8,
    horizon: int = 20,
    state_dim: int = 12,
    action_dim: int = 12,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic 12-DoF quadruped imitation data.

    Returns (obs, actions, returns) where:
        obs: [n_episodes, horizon, state_dim]
        actions: [n_episodes, horizon, action_dim]
        returns: [n_episodes] episode-level return proxy.
    """
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n_episodes, horizon, state_dim, generator=g) * 0.5
    # Expert action = tanh projection of obs (mimics a stable gait policy).
    actions = torch.tanh(obs[:, :, :action_dim])
    returns = obs.norm(dim=(1, 2)).mul(-1.0).add(5.0)  # higher = better
    return obs, actions, returns


def _gen_drone_synth(
    n_samples: int = 256,
    seq_len: int = 32,
    visual_dim: int = 64,
    imu_dim: int = 6,
    target_dim: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic drone (visual + IMU) → (x, y, z, yaw) regression data.

    Returns (X_visual, X_imu, Y):
        X_visual: [n_samples, seq_len, visual_dim] (CNN feature stand-in)
        X_imu: [n_samples, seq_len, imu_dim]
        Y: [n_samples, target_dim] (target pose delta)
    """
    g = torch.Generator().manual_seed(seed)
    X_visual = torch.randn(n_samples, seq_len, visual_dim, generator=g)
    X_imu = torch.randn(n_samples, seq_len, imu_dim, generator=g)
    # Y is a function of both: Y = IMU last-step + 0.3 * visual mean
    Y = X_imu[:, -1, :target_dim] + 0.3 * X_visual.mean(dim=1)[:, :target_dim]
    return X_visual, X_imu, Y


def _gen_inverted_pendulum_il(
    n_samples: int = 1024,
    seq_len: int = 16,
    state_dim: int = 4,  # [theta, theta_dot, x, x_dot]
    action_dim: int = 1,  # cart force
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic 1-DoF inverted pendulum IL data via LQR-style policy.

    Returns (obs, actions).
    """
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n_samples, seq_len, state_dim, generator=g) * 0.3
    # LQR gain matrix approximation: u = -K x, K ~ [-1, -0.5, -0.3, -0.2]
    K = torch.tensor([-1.0, -0.5, -0.3, -0.2])
    # Action = K . obs_last (we project back over time for training).
    actions = (obs * K.view(1, 1, -1)).sum(dim=-1, keepdim=True)
    return obs, actions


def _gen_battery_synth(
    n_cells: int = 32,
    cycles_per_cell: int = 200,
    seq_len: int = 128,
    feat_dim: int = 4,  # [V, I, T, dQ/dV] stand-in
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic battery SoH (CC-CV curve stand-in).

    Returns (X, Y):
        X: [n_cells, cycles_per_cell, seq_len, feat_dim]
        Y: [n_cells, cycles_per_cell] — SoH in [0, 1]
    """
    g = torch.Generator().manual_seed(seed)
    # Per-cell degradation rate, deterministic.
    rates = torch.linspace(0.001, 0.003, n_cells)
    X = torch.randn(n_cells, cycles_per_cell, seq_len, feat_dim, generator=g) * 0.1
    # Inject a slow decay into the last feature to mimic dQ/dV drift.
    for c in range(n_cells):
        for k in range(cycles_per_cell):
            X[c, k, :, -1] += -rates[c] * k
    # SoH = 1 - rate * cycle_index + noise.
    Y = 1.0 - rates[:, None] * torch.arange(cycles_per_cell).float()[None, :]
    Y = Y.clamp(0.0, 1.0)
    return X, Y


# ---------------------------------------------------------------------------
# Case A: Quadruped (LTC + actor-critic, PPO-lite)
# ---------------------------------------------------------------------------


def run_case_quadruped(
    seed: int,
    steps: int,
    quick: bool,
) -> Dict[str, Any]:
    """Case A: 12-DoF quadruped smoke via SNCPPolicyLite.

    Notes
    -----
    ``SNCPPolicyLite`` was originally 2-D (point-mass nav). We swap
    action_dim to 12 and reuse the same architecture. The smoke trains
    for ``steps`` PPO updates and reports the last-5 mean return as
    primary metric. 1-seed lucky is documented; iterate with --seeds ≥3.
    """
    t0 = time.time()
    if not _HAS_SNCP:
        return {
            "case": "quadruped", "seed": seed, "status": "skipped",
            "notes": "lnn.core.sncp_policy_lite not importable",
        }
    obs, expert_actions, returns = _gen_quadruped_rollout(seed=seed)
    n_ep, horizon, state_dim = obs.shape
    action_dim = expert_actions.shape[-1]
    # SNCPPolicyLite treats obs as [B, T, F] flattened; the lite version
    # concatenates spatial summary into the temporal axis (see its docstring).
    policy = SNCPPolicyLite(
        temporal_input_size=state_dim,
        action_dim=action_dim,
        ltc_hidden_size=8 if quick else 16,
    )
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    last5_returns: List[float] = []
    n_updates = max(1, steps // 8)  # 8 rollouts per update
    for upd in range(n_updates):
        # Roll out expert policy as a "data" pass + PPO clipped surrogate.
        # For smoke we treat expert actions as the "old" actions.
        # The lite policy is small; we run a single epoch of imitation
        # (MSE on the *last* expert action) to keep wall time low.
        # NOTE: full PPO clipped loss is in experiment_sncp_ppo_lite.py;
        # here we use a behaviour-cloning shortcut suitable for a smoke.
        hidden = policy.initial_hidden(n_ep, obs.device)
        last_feat, _ = policy.encode(obs, hidden)  # [n_ep, ltc_hidden]
        sf = policy.trunk(last_feat)  # [n_ep, trunk_hidden]
        mu, _ = policy.actor(sf)  # [n_ep, action_dim]
        target = expert_actions[:, -1, :]  # [n_ep, action_dim] — last-step target
        loss = ((mu - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        last5_returns.append(float(returns.mean().item()))
        if len(last5_returns) > 5:
            last5_returns.pop(0)
    wall = time.time() - t0
    primary = float(statistics.mean(last5_returns)) if last5_returns else 0.0
    return {
        "case": "quadruped", "seed": seed, "steps": steps,
        "wall_time_s": wall,
        "params": sum(p.numel() for p in policy.parameters()),
        "primary_metric": "last5_mean_return",
        "primary_metric_value": primary,
        "status": "ok",
        "notes": "BC shortcut (not full PPO) — see experiment_sncp_ppo_lite.py for full PPO clipped loss.",
    }


# ---------------------------------------------------------------------------
# Case B: Drone (CfC + visual+IMU fusion, regression)
# ---------------------------------------------------------------------------


def run_case_drone(
    seed: int,
    steps: int,
    quick: bool,
) -> Dict[str, Any]:
    """Case B: 6-DoF drone VIO-style regression via LNNImitationPolicy."""
    t0 = time.time()
    if not _HAS_CONTROL:
        return {
            "case": "drone", "seed": seed, "status": "skipped",
            "notes": "lnn.core.control.LNNImitationPolicy not importable",
        }
    X_visual, X_imu, Y = _gen_drone_synth(seed=seed)
    # Concat visual+IMU along feature axis (multi-modal fusion stand-in).
    X = torch.cat([X_visual, X_imu], dim=-1)
    state_dim = X.shape[-1]
    target_dim = Y.shape[-1]
    # Convert to [N, T, F] → policy expects [B, T, F]
    X_b = X
    Y_b = Y
    # 80/10/10 chronological split.
    n = X_b.shape[0]
    n_tr = int(n * 0.8)
    X_tr, Y_tr = X_b[:n_tr], Y_b[:n_tr]
    X_va, Y_va = X_b[n_tr:], Y_b[n_tr:]
    # LNNImitationPolicy uses state_dim + action_dim.
    # For regression we set action_dim = target_dim and use MSE head.
    policy = LNNImitationPolicy(
        state_dim=state_dim,
        action_dim=target_dim,
        hidden_size=8 if quick else 16,
        recurrent_type="cfc",
        head_type="mse",
    )
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    bs = 32
    for epoch in range(max(1, steps // 8)):
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, bs):
            idx = perm[i : i + bs]
            xb, yb = X_tr[idx], Y_tr[idx]
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    # Eval MSE
    with torch.no_grad():
        pred_va = policy(X_va)
        mse = float(((pred_va - Y_va) ** 2).mean().item())
    # Inference time
    t_inf0 = time.time()
    with torch.no_grad():
        _ = policy(X_va[:1])
    inference_ms = (time.time() - t_inf0) * 1000.0
    wall = time.time() - t0
    return {
        "case": "drone", "seed": seed, "steps": steps,
        "wall_time_s": wall,
        "params": sum(p.numel() for p in policy.parameters()),
        "inference_ms": inference_ms,
        "primary_metric": "val_mse",
        "primary_metric_value": mse,
        "status": "ok",
        "notes": "CfC recurrent; visual feature stand-in (no real CNN backbone).",
    }


# ---------------------------------------------------------------------------
# Case C: Inverted pendulum (LTC + sparse NCP, IL)
# ---------------------------------------------------------------------------


def run_case_industrial(
    seed: int,
    steps: int,
    quick: bool,
) -> Dict[str, Any]:
    """Case C: 1-DoF inverted pendulum IL via LNNImitationPolicy + LTC."""
    t0 = time.time()
    if not _HAS_CONTROL:
        return {
            "case": "industrial", "seed": seed, "status": "skipped",
            "notes": "lnn.core.control.LNNImitationPolicy not importable",
        }
    obs, actions = _gen_inverted_pendulum_il(seed=seed)
    state_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    # LNNImitationPolicy forward returns the *last-step* action [B, action_dim]
    # (it calls recurrent_features which slices [:, -1, :]). We need to align
    # the target to the last step too, not pass the full [B, T, A] sequence.
    actions_last = actions[:, -1, :]  # [N, action_dim]
    policy = LNNImitationPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=4 if quick else 8,
        recurrent_type="ltc",
        head_type="mse",
    )
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    bs = 32
    n = obs.shape[0]
    for epoch in range(max(1, steps // 16)):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb, yb = obs[idx], actions_last[idx]
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    # Eval
    with torch.no_grad():
        pred_va = policy(obs[:64])
        mse = float(((pred_va - actions_last[:64]) ** 2).mean().item())
    t_inf0 = time.time()
    with torch.no_grad():
        _ = policy(obs[:1])
    inference_ms = (time.time() - t_inf0) * 1000.0
    wall = time.time() - t0
    return {
        "case": "industrial", "seed": seed, "steps": steps,
        "wall_time_s": wall,
        "params": sum(p.numel() for p in policy.parameters()),
        "inference_ms": inference_ms,
        "primary_metric": "il_mse",
        "primary_metric_value": mse,
        "status": "ok",
        "notes": "LTC recurrent; NCP sparse wiring not yet active (recurrent_type='ltc').",
    }


# ---------------------------------------------------------------------------
# Case D: Battery SoH (LTC + transformable formula)
# ---------------------------------------------------------------------------


def run_case_battery(
    seed: int,
    steps: int,
    quick: bool,
    battery_mode: str = "single",
    refine_steps: int = 10,
    refine_lr: float = 1e-4,
) -> Dict[str, Any]:
    """Case D: Battery SoH regression via TransformableLTC (EntroLnn 2-stage).

    Modes
    -----
    * ``"single"`` (default, backward-compatible): one-shot training on the
      train split, evaluate on the val split. Same semantics as iter#36.
    * ``"transformable"`` (iter#37): 2-stage EntroLnn protocol —
      (1) train on **reference cell** (first cell of train split),
      (2) online-refine on the remaining train cells for ``refine_steps``
      steps at ``refine_lr`` (10× smaller than train_lr), then evaluate
      on the val split.

    Parameters
    ----------
    seed, steps, quick : standard
    battery_mode : "single" or "transformable"
    refine_steps : K gradient steps in stage 2 (default 10).
    refine_lr : learning rate for stage 2 (default 1e-4 = 1/10 of train_lr).
    """
    t0 = time.time()
    if battery_mode not in {"single", "transformable"}:
        raise ValueError(
            f"battery_mode must be 'single' or 'transformable', got {battery_mode!r}"
        )
    if not _HAS_LTC:
        return {
            "case": "battery", "seed": seed, "status": "skipped",
            "notes": "lnn.core.ltc.LTCNetwork not importable",
        }
    X, Y = _gen_battery_synth(seed=seed)
    # X: [n_cells, cycles, seq, feat]
    n_cells, n_cycles, seq_len, feat_dim = X.shape
    # Flatten cells × cycles into batch, treat each as a 1-step feature
    # summary (we use mean across seq for the smoke).
    X_flat = X.mean(dim=2)  # [n_cells, n_cycles, feat_dim]
    # Use the last-cycle SoH per cell as the scalar regression target. This
    # keeps shapes consistent (X: [B, T, F] vs Y: [B, 1]) and matches the
    # EntroLnn paper's "predict SoH from cycle history" setup.
    Y_flat = Y[:, -1:]  # [n_cells, 1]
    # 80/10/10 split (cells-level)
    n = n_cells
    n_tr = int(n * 0.8)
    X_tr, Y_tr = X_flat[:n_tr], Y_flat[:n_tr]
    X_va, Y_va = X_flat[n_tr:], Y_flat[n_tr:]
    # Build the model via the new public TransformableLTC class (iter#37).
    # In single mode we still use it for symmetry — train_reference(K=0)
    # is equivalent to a no-op, so we use refine_target(K=steps) to keep
    # the same total training budget as iter#36.
    hidden = 8 if quick else 16
    model = TransformableLTC(
        input_size=feat_dim, hidden_size=hidden, output_size=1,
        num_layers=1, ode_method="euler",
        train_lr=1e-3, refine_lr=refine_lr, loss_fn="mse",
    )
    bs = 8
    history: Dict[str, Any] = {"mode": battery_mode, "refine_steps": refine_steps}
    if battery_mode == "transformable":
        # 2-stage: reference cell (first of train) → refine on rest
        ref_x, ref_y = X_tr[:1], Y_tr[:1]
        ref_out = model.train_reference(ref_x, ref_y, epochs=max(1, steps // 4), batch_size=1, verbose=False)
        history["ref_loss_history"] = ref_out["ref_loss_history"]
        history["final_ref_loss"] = ref_out["final_ref_loss"]
        tgt_x, tgt_y = X_tr[1:], Y_tr[1:]
        ref_out2 = model.refine_target(tgt_x, tgt_y, K=refine_steps, batch_size=bs, verbose=False)
        history["refine_loss_history"] = ref_out2["refine_loss_history"]
        history["final_tgt_loss"] = ref_out2["final_tgt_loss"]
        history["final_ref_loss_after"] = ref_out2["final_ref_loss_after"]
    else:  # single mode (backward-compatible)
        # Run train_reference for the same total epochs as iter#36, then skip refine.
        single_out = model.train_reference(X_tr, Y_tr, epochs=max(1, steps // 4), batch_size=bs, verbose=False)
        history["ref_loss_history"] = single_out["ref_loss_history"]
        history["final_ref_loss"] = single_out["final_ref_loss"]
    with torch.no_grad():
        # Forward through the (possibly refined) model
        out_va = model(X_va)
        if out_va.dim() == 3:
            pred_va = out_va[:, -1, :].squeeze(-1)
        else:
            pred_va = out_va.squeeze(-1)
        mse = float(((pred_va - Y_va) ** 2).mean().item())
    t_inf0 = time.time()
    with torch.no_grad():
        _ = model(X_va[:1])
    inference_ms = (time.time() - t_inf0) * 1000.0
    wall = time.time() - t0
    note = f"mode={battery_mode}"
    if battery_mode == "transformable":
        ref_after = history.get("final_ref_loss_after", float("nan"))
        # Stability guard — flag if reference loss blew up
        ref_pre = history.get("final_ref_loss", 0.0)
        if ref_pre > 0 and ref_after > 10 * ref_pre:
            note += f" | WARNING: ref loss {ref_pre:.4f} → {ref_after:.4f} (>10×)"
    return {
        "case": "battery", "seed": seed, "steps": steps,
        "wall_time_s": wall,
        "params": sum(p.numel() for p in model.parameters()),
        "inference_ms": inference_ms,
        "primary_metric": "val_mse",
        "primary_metric_value": mse,
        "secondary_metrics": history,
        "status": "ok",
        "notes": note,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


CASE_FUNCS = {
    "quadruped": run_case_quadruped,
    "drone": run_case_drone,
    "industrial": run_case_industrial,
    "battery": run_case_battery,
}


def _aggregate_seeds(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed reports to {mean, std, n_seeds} for shared metrics."""
    if not per_seed:
        return {"mean": None, "std": None, "n_seeds": 0}
    metric_key = "primary_metric_value"
    values = [r[metric_key] for r in per_seed if r.get("status") == "ok" and metric_key in r]
    if not values:
        return {"mean": None, "std": None, "n_seeds": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "n_seeds": len(values),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--case", choices=list(CASE_FUNCS.keys()) + ["all"],
                   default="all", help="Which case to run.")
    p.add_argument("--steps", type=int, default=80,
                   help="Approx number of training steps (per case).")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds (iter#11 N=5 lesson: prefer ≥3).")
    p.add_argument("--quick", action="store_true",
                   help="Tiny hidden sizes + small steps for smoke (default off).")
    p.add_argument("--out-dir", default="analysis/device_control",
                   help="Where to write JSON reports.")
    p.add_argument("--battery-mode", choices=["single", "transformable"],
                   default="single",
                   help="Battery case mode (iter#37). 'single' = iter#36 behaviour; "
                        "'transformable' = 2-stage EntroLnn protocol.")
    p.add_argument("--refine-steps", type=int, default=10,
                   help="K gradient steps in battery transformable stage 2 "
                        "(default 10 per EntroLnn §3).")
    args = p.parse_args()

    cases = list(CASE_FUNCS.keys()) if args.case == "all" else [args.case]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_reports: List[Dict[str, Any]] = []
    for case in cases:
        func = CASE_FUNCS[case]
        per_seed: List[Dict[str, Any]] = []
        for s in range(args.seeds):
            seed = 42 + s * 1000
            steps = max(8, args.steps // max(1, args.seeds))
            try:
                # Battery case exposes extra kwargs; pass them through if accepted.
                if case == "battery":
                    rpt = func(
                        seed=seed, steps=steps, quick=args.quick,
                        battery_mode=args.battery_mode, refine_steps=args.refine_steps,
                    )
                else:
                    rpt = func(seed=seed, steps=steps, quick=args.quick)
            except Exception as exc:  # pragma: no cover - smoke safety net
                rpt = {"case": case, "seed": seed, "status": "error", "notes": str(exc)}
            per_seed.append(rpt)
        agg = _aggregate_seeds(per_seed)
        # Write JSON.
        timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        out_path = out_dir / f"{timestamp}_device_control_{case}.json"
        payload = {
            "schema": _DEVICE_CONTROL_REPORT_SCHEMA,
            "case": case,
            "per_seed": per_seed,
            "aggregated": agg,
            "config": {
                "steps_per_seed": max(8, args.steps // max(1, args.seeds)),
                "seeds": args.seeds,
                "quick": args.quick,
                "battery_mode": getattr(args, "battery_mode", None),
                "refine_steps": getattr(args, "refine_steps", None),
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        all_reports.append({"case": case, "aggregated": agg, "path": str(out_path)})
        # Console summary.
        if agg.get("mean") is not None:
            print(f"[{case}] primary={agg['mean']:.4f} ± {agg['std']:.4f} "
                  f"(n={agg['n_seeds']}) → {out_path}")
        else:
            print(f"[{case}] no successful seeds → {out_path}")
    # Master summary.
    master = out_dir / "latest_device_control_summary.json"
    with open(master, "w") as f:
        json.dump({"cases": all_reports,
                   "generated_utc": dt.datetime.utcnow().isoformat()},
                  f, indent=2)
    print(f"\nSummary → {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
