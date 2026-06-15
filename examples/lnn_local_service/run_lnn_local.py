"""Train a small CfC network on a synthetic sequence task, then dump a
checkpoint that ``server.py`` can load and serve over HTTP.

Why this exists
---------------
The LNN repo is mostly a *research* codebase: round 1..132 of ablations,
papers, and a few thousand lines of cell variants.  Running it end-to-end
on a workstation is non-trivial because most scripts need the full
EMMA-rover or natural-gas dataset, a GPU with up-to-date drivers, and
20+ minutes of wall-clock per seed.

This script is the *minimum-viable* smoke test:

    * synthetic 1-D signal (sin + linear drift) → next-step regression
    * one CfC layer, hidden_size=8 (329 params total)
    * 80 steps of Adam, CPU, < 5 s wall-clock
    * saves a state-dict the FastAPI server can load

Run::

    python examples/lnn_local_service/run_lnn_local.py

Artifacts:
    examples/lnn_local_service/artifacts/cfc_small.pt
    examples/lnn_local_service/artifacts/train_log.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def make_dataset(n_samples: int = 256, seq_len: int = 32, input_size: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic regression: input is a noisy sinusoid projected into ``input_size``
    channels; target is the *next-step* value of the underlying scalar.

    We deliberately keep this self-contained — no external datasets, no HDF5,
    no internet downloads.  CfC's claim is that the closed-form solution to
    the LTC ODE generalises even on a tiny synthetic regression like this.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 6.0 * torch.pi, steps=seq_len + 1).unsqueeze(0).expand(n_samples, -1)
    # base signal + slow drift
    base = torch.sin(t) + 0.05 * t
    noise = 0.1 * torch.randn(n_samples, seq_len + 1, generator=g)
    sig = base + noise
    # project to ``input_size`` channels via fixed random linear map
    proj = torch.randn(input_size, 1, generator=g)
    x = sig[:, :-1].unsqueeze(-1) * proj.t()  # [N, T, F]
    y = sig[:, 1:]  # [N, T]  next-step value of the scalar
    return x, y


def build_model(input_size: int = 4, hidden_size: int = 8, output_size: int = 1) -> CfCNetwork:
    return CfCNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=1,
        return_sequences=True,  # predict one value per step
    )


def main(epochs: int = 20, batch_size: int = 32, hidden_size: int = 8) -> dict:
    torch.manual_seed(0)
    x_train, y_train = make_dataset(seed=1)
    x_val, y_val = make_dataset(seed=2)

    model = build_model(input_size=x_train.shape[-1], hidden_size=hidden_size, output_size=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            yb = y_train[idx].unsqueeze(-1)  # [B, T, 1]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            nb += 1
        train_mse = running / max(nb, 1)

        with torch.no_grad():
            val_mse = loss_fn(model(x_val), y_val.unsqueeze(-1)).item()
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"epoch {epoch:02d}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")

    elapsed = time.time() - t0
    ckpt = ARTIFACTS / "cfc_small.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"input_size": 4, "hidden_size": hidden_size, "output_size": 1, "num_layers": 1, "return_sequences": True},
            "final_train_mse": history[-1]["train_mse"],
            "final_val_mse": history[-1]["val_mse"],
            "wall_time_s": elapsed,
        },
        ckpt,
    )
    log = ARTIFACTS / "train_log.json"
    log.write_text(json.dumps({"history": history, "wall_time_s": elapsed}, indent=2))
    print(f"saved {ckpt}  (val_mse={history[-1]['val_mse']:.4f}, {elapsed:.1f}s)")
    return {"ckpt": str(ckpt), "log": str(log), "val_mse": history[-1]["val_mse"], "wall_time_s": elapsed}


if __name__ == "__main__":
    main()
