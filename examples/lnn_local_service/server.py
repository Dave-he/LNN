"""Tiny FastAPI service that serves the locally-trained CfC model.

Endpoints
---------
GET  /health           → liveness + model metadata
GET  /info             → model config + last training metrics
POST /predict          → body { "sequence": [[...], ...] } → { "predictions": [[...]] }
POST /predict_summary  → body { "sequence": [[...], ...] } → { "summary": {...} }

Run::

    # 1) train + dump the checkpoint (one-time, ~12 s on CPU)
    python examples/lnn_local_service/run_lnn_local.py

    # 2) launch the service
    uvicorn examples.lnn_local_service.server:app --host 127.0.0.1 --port 8765

    # 3) probe
    curl -s http://127.0.0.1:8765/health
    curl -s -X POST http://127.0.0.1:8765/predict \
         -H 'content-type: application/json' \
         -d '{"sequence": [[0.1,0.2,0.3,0.4], [0.2,0.3,0.4,0.5]]}'
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

from lnn.core.cfc import CfCNetwork


HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "artifacts" / "cfc_small.pt"


# ----------------------------- request schemas ------------------------------


class PredictRequest(BaseModel):
    # Accept either ``[T]`` (1-D) or ``[T, F]`` (2-D) inputs.  We don't
    # constrain the inner width here — the route handler validates it
    # against the model's configured ``input_size`` and returns 400 on
    # mismatch.  Pydantic-level min_length=1 is enforced below on the
    # top-level list to keep ``[]`` from hitting the handler at all.
    sequence: list = Field(
        ...,
        min_length=1,
        description="Input sequence.  Accepts [T] or [T, F]; the service "
        "broadcasts 1-D to the model's input_size, rejects other widths.",
    )


# ------------------------------ app state -----------------------------------


def _load_model(ckpt_path: Path) -> tuple[CfCNetwork, dict[str, Any]]:
    if not ckpt_path.exists():
        raise RuntimeError(
            f"checkpoint not found at {ckpt_path}.  Run "
            f"`python examples/lnn_local_service/run_lnn_local.py` first."
        )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = CfCNetwork(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],
        num_layers=cfg["num_layers"],
        return_sequences=cfg["return_sequences"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _project_input(sequence: list[list[float]], input_size: int) -> torch.Tensor:
    """Reshape the user-supplied sequence to [1, T, input_size].

    Accepts:
      * 2-D ``[T, F]`` where ``F == input_size`` (used directly)
      * 1-D ``[T]`` or ``[F]`` (broadcast to ``[T, input_size]`` by tiling)
      * 2-D ``[T, F']`` where ``F' != input_size`` → 400.
    """
    if not sequence:
        raise HTTPException(status_code=400, detail="sequence must be non-empty")
    arr0 = sequence[0]
    if isinstance(arr0, (int, float)):
        # 1-D input [T]: tile to [T, input_size]
        seq = [[float(v)] * input_size for v in sequence]
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    if not isinstance(arr0, list):
        raise HTTPException(status_code=400, detail="sequence must be list[list[float]] or list[float]")
    width = len(arr0)
    if width == input_size:
        return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    if width == 1:
        seq = [[float(row[0])] * input_size for row in sequence]
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    raise HTTPException(
        status_code=400,
        detail=f"sequence width {width} does not match model input_size {input_size}",
    )


# --------------------------- FastAPI app ------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    ckpt = Path(os.environ.get("LNN_CKPT", DEFAULT_CKPT))
    model, blob = _load_model(ckpt)
    app.state.model = model
    app.state.ckpt = blob
    app.state.ckpt_path = str(ckpt)
    try:
        yield
    finally:
        # nothing to tear down; the model is pure-CPU and stateless
        pass


app = FastAPI(
    title="LNN Local Service",
    version="0.1.0",
    description="Tiny CPU-only FastAPI wrapper around a CfC model trained by "
    "examples/lnn_local_service/run_lnn_local.py.",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ckpt": app.state.ckpt_path,
        "config": app.state.ckpt["config"],
    }


@app.get("/info")
def info() -> dict[str, Any]:
    blob = app.state.ckpt
    return {
        "config": blob["config"],
        "final_train_mse": blob.get("final_train_mse"),
        "final_val_mse": blob.get("final_val_mse"),
        "wall_time_s": blob.get("wall_time_s"),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    input_size = app.state.ckpt["config"]["input_size"]
    x = _project_input(req.sequence, input_size)
    with torch.no_grad():
        y = app.state.model(x)  # [1, T, 1] for return_sequences=True
    return {"predictions": y.squeeze(0).squeeze(-1).tolist()}


@app.post("/predict_summary")
def predict_summary(req: PredictRequest) -> dict[str, Any]:
    input_size = app.state.ckpt["config"]["input_size"]
    x = _project_input(req.sequence, input_size)
    with torch.no_grad():
        y = app.state.model(x).squeeze(0).squeeze(-1).tolist()
    return {
        "summary": {
            "n_steps": len(y),
            "first": y[0],
            "last": y[-1],
            "mean": statistics.fmean(y),
            "stdev": statistics.pstdev(y) if len(y) > 1 else 0.0,
            "min": min(y),
            "max": max(y),
        }
    }
