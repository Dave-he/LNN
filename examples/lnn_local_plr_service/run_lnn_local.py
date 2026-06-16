"""Train a small PLR / PLR+CfC two-axis model on a synthetic sequence
task, then dump a checkpoint that ``server.py`` can load and serve
over HTTP.

This is the round 134 local-deployment smoke test (companion to
round 133's CfC local service).  We train **two** tiny models:

* ``plr_small.pt``: PLR-only encoder (PLRConfig in_channels=4,
  hidden_channels=8, n_layers=2, use_cfc_head=False).  Cheap,
  no nonlinear gating.
* ``plr_cfc_small.pt``: PLR + CfC two-axis cell (PLRCfCCell with
  out_channels=8, cfc_hidden=8).  Round 134's NEW BEST on
  ``structured_irr`` (see bench_liquid_tad_results.md).

Both are trained on a synthetic **regime-switch** signal: slow drift
interspersed with fast sinusoidal transients, plus light Gaussian
noise.  This is the same family as the round 134 benchmark, just
collapsed to a single next-step regression so a tiny model can fit
in a few hundred Adam steps on CPU.

Run::

    python examples/lnn_local_plr_service/run_lnn_local.py

Artifacts:
    examples/lnn_local_plr_service/artifacts/plr_small.pt
    examples/lnn_local_plr_service/artifacts/plr_cfc_small.pt
    examples/lnn_local_plr_service/artifacts/train_log.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from lnn.core.liquid_tad import PLRCell, PLRConfig, PLREncoder, PLRCfCCell


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = 4
SEQ_LEN = 48
HIDDEN_SIZE = 8


def make_dataset(
    n_samples: int = 256,
    seq_len: int = SEQ_LEN,
    input_size: int = INPUT_SIZE,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic regime-switch next-step regression.

    Each sample is a 1-D signal ``s_t`` constructed as:

        s_t = slow_drift_t + regime[t] * fast_oscillation_t + noise

    where ``slow_drift_t`` is a slow random walk, ``regime[t]`` is a
    binary mask that turns on a fast sinusoidal component over two
    disjoint windows, and ``noise ~ N(0, 0.05)``.

    We project ``s_t`` into ``input_size`` channels by stacking a
    delayed-copy feature bank ``[s_t, s_{t-1}, s_{t-2}, ..., s_{t-input_size+1}]``.

    Target: ``s_{t+1}`` (next-step value of the underlying scalar).
    """
    rng = torch.Generator().manual_seed(seed)
    total = n_samples + seq_len + 1
    # Per-sample slow drift (different random walk per channel).
    drift = torch.cumsum(0.02 * torch.randn(n_samples, seq_len + 1, generator=rng), dim=1)
    # Regime mask: 1 in two windows, 0 elsewhere.
    regime = torch.zeros(n_samples, seq_len + 1)
    regime[:, 12:22] = 1.0
    regime[:, 32:42] = 1.0
    # Fast oscillation with per-sample random phase.
    phase = 2 * math.pi * torch.rand(n_samples, 1, generator=rng)
    t_grid = torch.linspace(0, 6 * math.pi, seq_len + 1).unsqueeze(0)
    fast = 0.5 * torch.sin(8.0 * t_grid + phase)
    signal = drift + regime * fast + 0.05 * torch.randn(n_samples, seq_len + 1, generator=rng)

    # Build feature bank by broadcasting each scalar ``signal[n, t]``
    # into all ``input_size`` channels.  This keeps the dataset
    # self-contained (signal length = seq_len + 1 = 49) without
    # needing a lag-feature indexing scheme.  It's degenerate but
    # useful for a smoke test.
    X = signal[:, :seq_len].unsqueeze(-1).expand(n_samples, seq_len, input_size).contiguous()
    y = signal[:, 1 : seq_len + 1].unsqueeze(-1)        # (N, T, 1)
    return X, y


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PLRWrapper(nn.Module):
    """Wraps a PLREncoder + linear projection to ``output_size`` for
    next-step regression.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.cfg = PLRConfig(
            in_channels=input_size,
            hidden_channels=hidden_size,
            n_layers=2,
            use_cfc_head=False,
            share_alpha_across_layers=False,
            alpha_per_channel=False,
            tau_init=1.0,
        )
        self.encoder = PLREncoder(self.cfg)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


class PLRCfCWrapper(nn.Module):
    """Wraps a PLRCfCCell + linear projection to ``output_size``."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.cell = PLRCfCCell(
            in_channels=input_size,
            out_channels=hidden_size,
            cfc_hidden=hidden_size,
            return_sequences=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cell(x)
        return self.head(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 80,
    lr: float = 1e-2,
    label: str = "model",
) -> dict:
    """Train ``model`` for ``epochs`` Adam steps on (X, y)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    last_loss = float("nan")
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[{label}] epoch {epoch:3d} loss={last_loss:.5f}")
    elapsed = time.time() - t0
    return {"label": label, "epochs": epochs, "final_loss": last_loss, "wall_s": elapsed}


def main() -> None:
    torch.manual_seed(42)
    print(f"[plr_local] building dataset (n=256, T={SEQ_LEN}, F={INPUT_SIZE})")
    X, y = make_dataset()
    print(f"[plr_local] dataset shapes: X={tuple(X.shape)}, y={tuple(y.shape)}")

    log = {"input_size": INPUT_SIZE, "seq_len": SEQ_LEN, "hidden_size": HIDDEN_SIZE, "models": []}

    # ---- PLR-only -----------------------------------------------------------
    plr_model = PLRWrapper(INPUT_SIZE, HIDDEN_SIZE, output_size=1)
    plr_metrics = train_model(plr_model, X, y, label="plr")
    n_params_plr = sum(p.numel() for p in plr_model.parameters())
    plr_ckpt = {
        "kind": "plr",
        "config": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": 1,
            "plr_cfg": plr_model.cfg.__dict__,
        },
        "state_dict": plr_model.state_dict(),
        "metrics": plr_metrics,
    }
    plr_path = ARTIFACTS / "plr_small.pt"
    torch.save(plr_ckpt, plr_path)
    print(f"[plr_local] saved {plr_path} ({n_params_plr} params)")
    log["models"].append({"kind": "plr", "n_params": n_params_plr, **plr_metrics})

    # ---- PLR+CfC two-axis ---------------------------------------------------
    plr_cfc_model = PLRCfCWrapper(INPUT_SIZE, HIDDEN_SIZE, output_size=1)
    plr_cfc_metrics = train_model(plr_cfc_model, X, y, label="plr_cfc")
    n_params_plr_cfc = sum(p.numel() for p in plr_cfc_model.parameters())
    plr_cfc_ckpt = {
        "kind": "plr_cfc",
        "config": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": 1,
        },
        "state_dict": plr_cfc_model.state_dict(),
        "metrics": plr_cfc_metrics,
    }
    plr_cfc_path = ARTIFACTS / "plr_cfc_small.pt"
    torch.save(plr_cfc_ckpt, plr_cfc_path)
    print(f"[plr_local] saved {plr_cfc_path} ({n_params_plr_cfc} params)")
    log["models"].append({"kind": "plr_cfc", "n_params": n_params_plr_cfc, **plr_cfc_metrics})

    log_path = ARTIFACTS / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[plr_local] saved {log_path}")
    print(f"[plr_local] done in {plr_metrics['wall_s'] + plr_cfc_metrics['wall_s']:.2f} s total")


if __name__ == "__main__":
    main()
