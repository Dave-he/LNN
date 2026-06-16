"""Tiny FastAPI service that serves the locally-trained PLR / PLR+CfC
models from round 134.

Endpoints
---------
GET  /health               → liveness + checkpoint paths + model kinds
GET  /info                 → per-model config + last training metrics
POST /predict              → body { "sequence": [[...], ...], "model": "plr" | "plr_cfc" }
                            → { "predictions": [[...]], "model": "plr", "params": N }
POST /predict_summary      → body as /predict
                            → { "summary": { mean, std, min, max, last, ... }, "model": "plr_cfc" }

Run::

    # 1) train + dump both checkpoints (one-time, ~10 s on CPU)
    python examples/lnn_local_plr_service/run_lnn_local.py

    # 2) launch the service
    uvicorn examples.lnn_local_plr_service.server:app \
        --host 127.0.0.1 --port 8766

    # 3) probe
    curl -s http://127.0.0.1:8766/health
    curl -s -X POST http://127.0.0.1:8766/predict \
         -H 'content-type: application/json' \
         -d '{"sequence": [[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5]], "model": "plr_cfc"}'
"""
from __future__ import annotations

import os
import statistics
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lnn.core.liquid_tad import PLRCfCCell, PLRConfig, PLREncoder


HERE = Path(__file__).resolve().parent
DEFAULT_PLR_CKPT = HERE / "artifacts" / "plr_small.pt"
DEFAULT_PLRCFC_CKPT = HERE / "artifacts" / "plr_cfc_small.pt"


# ----------------------------- model wrappers --------------------------------
#
# These mirror the wrappers in ``run_lnn_local.py`` so the state_dict
# keys match exactly.  We can't import ``run_lnn_local`` because it
# pulls in the dataset builder + training loop (not needed at
# inference time).


class _PLRWrapper(torch.nn.Module):
    """PLR encoder + linear head; identical to run_lnn_local.PLRWrapper."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, plr_cfg_dict: dict) -> None:
        super().__init__()
        cfg = PLRConfig(
            in_channels=input_size,
            hidden_channels=hidden_size,
            n_layers=plr_cfg_dict.get("n_layers", 2),
            use_cfc_head=plr_cfg_dict.get("use_cfc_head", False),
            share_alpha_across_layers=plr_cfg_dict.get("share_alpha_across_layers", False),
            alpha_per_channel=plr_cfg_dict.get("alpha_per_channel", False),
            tau_init=plr_cfg_dict.get("tau_init", 1.0),
        )
        self.encoder = PLREncoder(cfg)
        self.head = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _PLRCfCWrapper(torch.nn.Module):
    """PLRCfCCell + linear head; identical to run_lnn_local.PLRCfCWrapper."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.cell = PLRCfCCell(
            in_channels=input_size,
            out_channels=hidden_size,
            cfc_hidden=hidden_size,
            return_sequences=True,
        )
        self.head = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cell(x))


# ----------------------------- request schemas ------------------------------


class PredictRequest(BaseModel):
    sequence: list = Field(
        ...,
        min_length=1,
        description="Input sequence. Accepts [T, F] (preferred) or [T] (auto-broadcast).",
    )
    model: str = Field(
        default="plr_cfc",
        description="Which model to run. One of 'plr', 'plr_cfc'. Defaults to 'plr_cfc'.",
    )


# ------------------------------ model loaders -------------------------------


def _build_plr_from_cfg(cfg: dict) -> torch.nn.Module:
    """Rebuild a PLRWrapper (PLR encoder + linear head) from a saved cfg."""
    plr_cfg_dict = cfg.get("plr_cfg") or {}
    return _PLRWrapper(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],
        plr_cfg_dict=plr_cfg_dict,
    )


def _build_plr_cfc_from_cfg(cfg: dict) -> torch.nn.Module:
    return _PLRCfCWrapper(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],
    )


def _load_checkpoint(ckpt_path: Path, kind: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not ckpt_path.exists():
        raise RuntimeError(
            f"checkpoint for {kind!r} not found at {ckpt_path}. Run "
            f"`python examples/lnn_local_plr_service/run_lnn_local.py` first."
        )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    if kind == "plr":
        model = _build_plr_from_cfg(cfg)
    elif kind == "plr_cfc":
        model = _build_plr_cfc_from_cfg(cfg)
    else:
        raise ValueError(f"unknown model kind: {kind!r}")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


# ------------------------------ app state -----------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both checkpoints once at startup; expose them via app.state."""
    plr_path = Path(os.environ.get("PLR_CKPT", DEFAULT_PLR_CKPT))
    plr_cfc_path = Path(os.environ.get("PLRCFC_CKPT", DEFAULT_PLRCFC_CKPT))

    app.state.plr_model, app.state.plr_blob = _load_checkpoint(plr_path, "plr")
    app.state.plr_cfc_model, app.state.plr_cfc_blob = _load_checkpoint(plr_cfc_path, "plr_cfc")
    app.state.plr_ckpt_path = plr_path
    app.state.plr_cfc_ckpt_path = plr_cfc_path
    try:
        yield
    finally:
        # Nothing to release; we keep the models in memory.
        pass


app = FastAPI(
    title="LNN Local PLR Service",
    version="0.1.0",
    description="Round 134 local CPU service for PLR / PLR+CfC two-axis cells.",
    lifespan=lifespan,
)


# ------------------------------ helpers -------------------------------------


def _coerce_sequence(seq: list, input_size: int) -> torch.Tensor:
    """Coerce request payload to ``(1, T, input_size)`` float tensor.

    Accepts:
    - ``[T, F]`` — multi-channel, F must equal ``input_size``.
    - ``[T]`` — 1-D signal; broadcasts to ``input_size`` channels by
      replicating the same scalar (this is a degenerate but useful
      shortcut for smoke tests).
    """
    if not isinstance(seq, list) or len(seq) == 0:
        raise HTTPException(status_code=400, detail="sequence must be a non-empty list")

    first = seq[0]
    if isinstance(first, list):
        # [T, F]
        F = len(first)
        if F != input_size:
            raise HTTPException(
                status_code=400,
                detail=f"sequence width {F} does not match model input_size {input_size}",
            )
        T = len(seq)
        flat = [float(v) for step in seq for v in step]
        x = torch.tensor(flat, dtype=torch.float32).view(1, T, input_size)
        return x

    # 1-D: [T]; broadcast to (T, input_size) by repeating the scalar.
    try:
        arr = [float(v) for v in seq]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="sequence elements must be numeric (1-D) or lists of numerics (2-D)",
        )
    T = len(arr)
    x = torch.tensor(arr, dtype=torch.float32).view(1, T, 1).expand(1, T, input_size)
    return x


def _select_model(app: FastAPI, kind: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    if kind == "plr":
        return app.state.plr_model, app.state.plr_blob
    if kind == "plr_cfc":
        return app.state.plr_cfc_model, app.state.plr_cfc_blob
    raise HTTPException(
        status_code=400,
        detail=f"unknown model {kind!r}; expected one of 'plr', 'plr_cfc'",
    )


# ------------------------------ endpoints -----------------------------------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "checkpoints": {
            "plr": str(app.state.plr_ckpt_path),
            "plr_cfc": str(app.state.plr_cfc_ckpt_path),
        },
        "models_loaded": ["plr", "plr_cfc"],
        "device": "cpu",
    }


@app.get("/info")
def info() -> dict:
    plr_blob = app.state.plr_blob
    plr_cfc_blob = app.state.plr_cfc_blob
    return {
        "models": {
            "plr": {
                "config": plr_blob["config"],
                "n_params": sum(p.numel() for p in app.state.plr_model.parameters()),
                "metrics": plr_blob.get("metrics", {}),
            },
            "plr_cfc": {
                "config": plr_cfc_blob["config"],
                "n_params": sum(p.numel() for p in app.state.plr_cfc_model.parameters()),
                "metrics": plr_cfc_blob.get("metrics", {}),
            },
        }
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    model, blob = _select_model(app, req.model)
    input_size = blob["config"]["input_size"]
    x = _coerce_sequence(req.sequence, input_size)
    with torch.no_grad():
        y = model(x)
    return {
        "model": req.model,
        "predictions": y.squeeze(0).tolist(),
        "params": sum(p.numel() for p in model.parameters()),
        "input_shape": list(x.shape),
    }


@app.post("/predict_summary")
def predict_summary(req: PredictRequest) -> dict:
    model, blob = _select_model(app, req.model)
    input_size = blob["config"]["input_size"]
    x = _coerce_sequence(req.sequence, input_size)
    with torch.no_grad():
        y = model(x)
    flat = y.flatten().tolist()
    if not flat:
        summary = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "last": 0.0, "n": 0}
    else:
        summary = {
            "mean": statistics.fmean(flat),
            "std": statistics.pstdev(flat) if len(flat) > 1 else 0.0,
            "min": min(flat),
            "max": max(flat),
            "last": flat[-1],
            "n": len(flat),
        }
    return {"model": req.model, "summary": summary, "params": sum(p.numel() for p in model.parameters())}
