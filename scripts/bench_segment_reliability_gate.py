"""Round 99 bench (PRD #10-61) — Segment Reliability Gate.

Direct test of arXiv:2606.03631 (Xie et al., KDD 2026) —
*AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE*.

The paper's contribution is an uncertainty-aware reliability gate that
dampens expert contributions on noisy inputs. We test the same
mechanism on a 1D regression task with optional input noise.

For each of 2 models (CfC, LSTM) on 3 datasets (toy_sin, structured,
random), we compare:
- baseline (no gate)
- +segment_reliability gate (sigma_min=0.1, mix=1.0)

Both on clean and noisy (Gaussian noise sigma=0.1) inputs.

Cells: 2 models x 3 datasets x 2 conditions x 2 noise levels x 3 seeds = 72 cells

Metrics:
- task_loss: final MSE on the (noisy) input
- clean_consistency: mean |y_pred_noisy - y_pred_clean| (H1)
- gate_value: average r used during the test

Run:
    .venv312/bin/python scripts/bench_segment_reliability_gate.py --quick
    .venv312/bin/python scripts/bench_segment_reliability_gate.py        # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.reliability_gate import apply_reliability_gate


# ---------------------------------------------------------------------------
# Target function (same as rounds 91, 92, 93, 94, 98)
# ---------------------------------------------------------------------------

def target_fn(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_toy_sin(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = target_fn(t)
    return t, y


def make_structured(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.zeros_like(t)
    regime1 = t < 0.5
    y[regime1] = torch.sin(2 * np.pi * t[regime1])
    y[~regime1] = torch.sign(torch.sin(20 * np.pi * t[~regime1]))
    return t, y


def make_random(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.randn(T)
    return t, y


DATASETS = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CfCRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = CfCCell(input_size=1, hidden_size=16, n_tau=1)
        self.head = nn.Linear(16, 1)


class LSTMSeq2Seq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)


FACTORIES = {
    "CfC": CfCRegressor,
    "LSTM": LSTMSeq2Seq,
}


# ---------------------------------------------------------------------------
# Forward (non-detached states, single-pass)
# ---------------------------------------------------------------------------

def forward_with_states(
    model: nn.Module,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model, CfCRegressor):
        outs = []
        states = []
        h = torch.zeros(1, model.cell.hidden_size)
        for ti in t:
            x_t = ti.reshape(1, 1)
            h = model.cell(x_t, h, dt=1.0)
            outs.append(model.head(h))
            states.append(h.squeeze(0))
        y = torch.cat(outs, dim=-1).squeeze(0)
        return y, torch.stack(states, dim=0)
    if isinstance(model, LSTMSeq2Seq):
        x_seq = t.unsqueeze(0).unsqueeze(-1)
        out, _ = model.lstm(x_seq)
        y = model.head(out).squeeze(0).squeeze(-1)
        return y, out.squeeze(0)
    raise ValueError(f"unknown model type {type(model)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

SIGMA_MIN = 0.1
NOISE_SIGMA = 0.1


def add_noise(t: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return t + torch.randn(t.shape, generator=g) * sigma


def train_model(
    model: nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    use_gate: bool,
    noise_in_input: bool,
    seed: int,
    mix: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Train and return (final_train_loss, clean_consistency, gate_value, task_noisy, task_clean)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_task_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        # Use noisy input during training if noise_in_input
        if noise_in_input:
            t_in = add_noise(t, NOISE_SIGMA, seed)
        else:
            t_in = t
        y_pred, _ = forward_with_states(model, t_in)
        if use_gate:
            # Apply reliability gate on the (noisy) input
            y_pred_g, _ = apply_reliability_gate(y_pred, t_in, sigma_min=SIGMA_MIN, mix=mix)
        else:
            y_pred_g = y_pred
        task_loss = ((y_pred_g - y) ** 2).mean()
        task_loss.backward()
        opt.step()
        final_task_loss = float(task_loss.item())
    # Final test on NOISY input
    with torch.no_grad():
        t_noisy = add_noise(t, NOISE_SIGMA, seed + 1000)
        y_noisy, _ = forward_with_states(model, t_noisy)
        if use_gate:
            y_noisy_g, r_test = apply_reliability_gate(
                y_noisy, t_noisy, sigma_min=SIGMA_MIN, mix=mix,
            )
        else:
            y_noisy_g = y_noisy
            r_test = torch.tensor(1.0)
        task_loss_noisy = ((y_noisy_g - y) ** 2).mean().item()
        # Also test on CLEAN input
        y_clean, _ = forward_with_states(model, t)
        if use_gate:
            y_clean_g, _ = apply_reliability_gate(
                y_clean, t, sigma_min=SIGMA_MIN, mix=mix,
            )
        else:
            y_clean_g = y_clean
        # Clean consistency: how much does the prediction change due to noise?
        clean_consistency = (y_noisy_g - y_clean_g).abs().mean().item()
        # Recompute task loss on clean for H2 (no degradation on clean)
        task_loss_clean = ((y_clean_g - y) ** 2).mean().item()
    return final_task_loss, clean_consistency, float(r_test), task_loss_noisy, task_loss_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--mix", type=float, default=1.0,
                   help="Gating mix in [0, 1]. 1.0=full, 0.5=half. Use 0.5 to soften.")
    p.add_argument("--out", default="results/bench_segment_reliability_gate.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    mix = args.mix
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "sigma_min": SIGMA_MIN,
        "mix": mix,
        "noise_sigma": NOISE_SIGMA,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T = 64

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for model_name, factory in FACTORIES.items():
            for cond in ("baseline", "gate"):
                use_gate = cond == "gate"
                cond_out: list[dict] = []
                for seed in range(n_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    t, y = ds_fn(T, seed=seed)
                    model = factory()
                    final_task, clean_cons, gate_val, task_noisy, task_clean = train_model(
                        model, t, y, epochs=epochs, lr=1e-2,
                        use_gate=use_gate, noise_in_input=False, seed=seed, mix=mix,
                    )
                    cond_out.append({
                        "final_train_loss": final_task,
                        "task_loss_noisy": task_noisy,
                        "task_loss_clean": task_clean,
                        "clean_consistency": clean_cons,
                        "gate_value": gate_val,
                    })
                def agg(field: str) -> tuple[float, float]:
                    vals = [s[field] for s in cond_out if s[field] is not None]
                    if not vals:
                        return 0.0, 0.0
                    return float(np.mean(vals)), float(np.std(vals))
                ds_out[f"{model_name}_{cond}"] = {
                    "final_train_loss_mean_std": agg("final_train_loss"),
                    "task_loss_noisy_mean_std": agg("task_loss_noisy"),
                    "task_loss_clean_mean_std": agg("task_loss_clean"),
                    "clean_consistency_mean_std": agg("clean_consistency"),
                    "gate_value_mean_std": agg("gate_value"),
                    "per_seed": cond_out,
                }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Round 99 segment reliability gate bench (epochs={epochs}, seeds={n_seeds}, sigma_min={SIGMA_MIN}, mix={mix}) ===\n")
    print(f"{'dataset':12s} | {'model':5s} | {'cond':10s} | {'task_noisy':>10s} | {'task_clean':>10s} | {'clean_cons':>10s} | {'gate':>6s}")
    print("-" * 90)
    for ds_name in DATASETS:
        for model_name in FACTORIES:
            for cond in ("baseline", "gate"):
                key = f"{model_name}_{cond}"
                if key not in out["datasets"][ds_name]:
                    continue
                c = out["datasets"][ds_name][key]
                tn_m, _ = c["task_loss_noisy_mean_std"]
                tc_m, _ = c["task_loss_clean_mean_std"]
                cc_m, _ = c["clean_consistency_mean_std"]
                gv_m, _ = c["gate_value_mean_std"]
                print(f"{ds_name:12s} | {model_name:5s} | {cond:10s} | {tn_m:10.4f} | {tc_m:10.4f} | {cc_m:10.4f} | {gv_m:6.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
