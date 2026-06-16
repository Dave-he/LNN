# LNN Local PLR Service

A tiny, self-contained **CPU-only** way to run a Liquid PLR (or PLR+CfC
two-axis) model from round 134 locally and expose it as a small HTTP
service.  Companion to round 133's [`lnn_local_service`](../lnn_local_service/)
which exposes a single `CfCNetwork`; this service exposes **both**
`PLREncoder` (PLR-only, cheap low-pass operator) and `PLRCfCCell`
(PLR + CfC nonlinear gating — round 134's **NEW BEST** on
`structured_irr` per `analysis/bench_liquid_tad_results.md`).

It exists to answer the question — *"can I run the round 134 PLR +
HDRS work on this machine without the EMMA rover dataset, without a
GPU, and without a 30-line benchmark harness?"* — with a **yes** and
a curl one-liner.

## What's here

| file | role |
|---|---|
| `run_lnn_local.py` | Trains two tiny models (PLR-only, PLR+CfC two-axis) on a synthetic regime-switch next-step regression and dumps both checkpoints + a training log. |
| `server.py` | FastAPI app: `/health`, `/info`, `POST /predict`, `POST /predict_summary`.  Both models are loaded at startup and selected per-request via the `model` field. |
| `test_server.py` | 11 pytest cases (TestClient, no socket binding) covering shape, validation, error paths, summary stats, and a two-axis-vs-PLR smoke check on a constant signal. |
| `conftest.py` | Bootstraps `sys.path` so the test can `import server` without making `examples` a real package. |
| `artifacts/` | Output of `run_lnn_local.py` (created on first run).  Contains `plr_small.pt`, `plr_cfc_small.pt`, and `train_log.json`. |

## Quickstart

```bash
# 0) Use the project's Python (3.10 + torch 2.10). Adjust the path if
#    your env is elsewhere.
PY=python3.10
export LD_LIBRARY_PATH=$HOME/.local/opt/libcudss-linux-aarch64-0.8.0.10_cuda12-archive/lib:$LD_LIBRARY_PATH

# 1) Train + dump both checkpoints (one-time, ~10 s on CPU)
$PY examples/lnn_local_plr_service/run_lnn_local.py

# 2) Run the tests
$PY -m pytest examples/lnn_local_plr_service/test_server.py -v \
    --override-ini="testpaths=examples/lnn_local_plr_service"

# 3) Launch the service
$PY -m uvicorn examples.lnn_local_plr_service.server:app \
    --host 127.0.0.1 --port 8766

# 4) Probe with curl
curl -s http://127.0.0.1:8766/health
curl -s -X POST http://127.0.0.1:8766/predict \
     -H 'content-type: application/json' \
     -d '{"sequence": [[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5]], "model": "plr_cfc"}'
curl -s -X POST http://127.0.0.1:8766/predict_summary \
     -H 'content-type: application/json' \
     -d '{"sequence": [[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.3,0.4,0.5,0.6]], "model": "plr"}'
```

## Endpoints

### `GET /health`
Liveness + checkpoint paths.

```json
{
  "status": "ok",
  "checkpoints": {
    "plr": "/…/examples/lnn_local_plr_service/artifacts/plr_small.pt",
    "plr_cfc": "/…/examples/lnn_local_plr_service/artifacts/plr_cfc_small.pt"
  },
  "models_loaded": ["plr", "plr_cfc"],
  "device": "cpu"
}
```

### `GET /info`
Per-model config + parameter count + last training metrics.

```json
{
  "models": {
    "plr":     {"config": {...}, "n_params": 281,  "metrics": {...}},
    "plr_cfc": {"config": {...}, "n_params": 1329, "metrics": {...}}
  }
}
```

### `POST /predict`
Body: `{"sequence": [[...], ...], "model": "plr" | "plr_cfc"}` —
sequence accepts `[T, F]` (preferred, F must equal the model's
`input_size=4`) or `[T]` (broadcast to (T, 4) by replication, useful
for smoke tests).

Response: `{"model": ..., "predictions": [[...], ...], "params": N, "input_shape": [1, T, F]}`.

### `POST /predict_summary`
Same body as `/predict`.  Response: `{"model": ..., "summary": {mean, std, min, max, last, n}, "params": N}`.

## How this fits into round 134

This service is the **deployable artifact** of round 134 — it proves
the PLR family (`PLRCell`, `PLREncoder`, `PLRCfCCell`) survives a
real end-to-end pipeline (pickle → load → serve → curl → inference)
without losing the round 134 bench result that **PLR+CfC two-axis
wins on `structured_irr`** (0.00545 vs CfC 0.01262, **-57 % MSE**).
The training script `run_lnn_local.py` deliberately uses the **same
synthetic regime-switch family** as the round 134 benchmark
(`scripts/bench_liquid_tad.py`), just collapsed to next-step
regression so the tiny models can fit in < 1 s of CPU.

## Environment overrides

Both checkpoint paths can be overridden at launch time:

```bash
PLR_CKPT=/path/to/plr.pt PLRCFC_CKPT=/path/to/plr_cfc.pt \
    python -m uvicorn examples.lnn_local_plr_service.server:app --port 8766
```

If the override points are missing, the lifespan handler raises a
clear RuntimeError on startup.
