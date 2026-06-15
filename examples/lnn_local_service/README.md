# LNN Local Service

A tiny, self-contained **CPU-only** way to run a Liquid CfC network locally
and expose it as a small HTTP service.  This is the smallest possible
end-to-end example in the LNN repo: a synthetic regression task, a
`CfCNetwork` trained for ~12 s on a single CPU core, and a 4-endpoint
FastAPI app that serves the trained model.

It exists to answer the recurring question — *"can I actually run an
LNN on this machine without the EMMA rover dataset, without a GPU, and
without reading 30 ablation scripts?"* — with a **yes**, and a curl
one-liner.

## What's here

| file | role |
|---|---|
| `run_lnn_local.py` | Trains a `CfCNetwork(input=4, hidden=8, out=1)` on a synthetic sin+drift next-step regression and dumps a checkpoint + training log. |
| `server.py` | FastAPI app: `/health`, `/info`, `POST /predict`, `POST /predict_summary`. |
| `test_server.py` | 7 pytest cases (TestClient, no socket binding) covering shape, validation, error paths, and aggregates. |
| `conftest.py` | Bootstraps `sys.path` so the test can `import server` without making `examples` a real package. |
| `artifacts/` | Output of `run_lnn_local.py` (created on first run).  Contains `cfc_small.pt` (~3 KB) and `train_log.json`. |

## Quickstart

```bash
# 0) Use the project's Python (3.14 + torch 2.11 + fastapi 0.135).  Adjust
#    the path if your env is elsewhere.
PY=/home/hyx/.pyenv/versions/3.14.4/bin/python3

# 1) Train + dump the checkpoint (one-time, ~12 s on CPU)
$PY examples/lnn_local_service/run_lnn_local.py

# 2) Run the tests
$PY -m pytest examples/lnn_local_service/test_server.py -v \
    --override-ini="testpaths=examples/lnn_local_service"

# 3) Launch the service
$PY -m uvicorn examples.lnn_local_service.server:app \
    --host 127.0.0.1 --port 8765

# 4) Probe with curl
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/predict \
     -H 'content-type: application/json' \
     -d '{"sequence": [[0.1,0.2,0.3,0.4], [0.2,0.3,0.4,0.5]]}'
```

## Endpoints

### `GET /health`
Returns the model config and the absolute path of the loaded checkpoint.

```json
{
  "status": "ok",
  "ckpt": "/…/examples/lnn_local_service/artifacts/cfc_small.pt",
  "config": {"input_size": 4, "hidden_size": 8, "output_size": 1,
             "num_layers": 1, "return_sequences": true}
}
```

### `GET /info`
Returns the final train/val MSE and wall-clock time of the training run
that produced the loaded checkpoint.

### `POST /predict`
Body: `{"sequence": [[…], …]}` — accepts either `[T]` (1-D, broadcast
to `[T, input_size]`) or `[T, F]` (2-D, must have `F == input_size`).

Returns: `{"predictions": [v0, v1, …, v_{T-1}]}` — one scalar per step.

### `POST /predict_summary`
Same input shape; returns a JSON object with `n_steps, first, last, mean,
stdev, min, max` for quick eyeballing without writing a client loop.

## Honest caveats

- The synthetic dataset is tiny (256 samples) and the model overfits
  immediately.  `final_train_mse ≈ 0.027`, `final_val_mse ≈ 0.53` is
  the expected regime — **this is a smoke test, not a benchmark.**  The
  goal is "CfC forward+backward works on this machine, model loads from
  disk, FastAPI routes correctly", not SOTA.
- CPU only.  No CUDA is required.  The CUDA driver warning from
  `torch` is harmless on this box (driver 12060 < the bundled CUDA
  13.0 build); torch silently falls back to CPU.
- The service holds the entire model in process memory.  It does **not**
  shard, batch concurrent requests, or use a request queue.  For real
  load, put it behind a `uvicorn --workers 2` and a reverse proxy.

## Why this lives in `examples/`

The rest of `examples/quickstart_bicfc_ensemble.py` is the *real*,
end-to-end BiCfC ensemble recipe (30 seeds, EMMA rover, ~minutes per
seed).  This service is the *minimum-viable* counterpoint: a few hundred
lines, a 3 KB checkpoint, and a curl probe.  Use it as the on-ramp
before you reach for the full ensemble script.
