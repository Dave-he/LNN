"""End-to-end test for the FastAPI LNN local service.

Strategy
--------
We don't want to actually spin up uvicorn in CI; instead we use
``fastapi.testclient.TestClient`` which runs the app in-process.
That means *real* request routing, *real* Pydantic validation, and
*real* model inference — but no socket binding.

Run::

    pytest examples/lnn_local_service/test_server.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


HERE = Path(__file__).resolve().parent
CKPT = HERE / "artifacts" / "cfc_small.pt"


def _build_client() -> TestClient:
    # import the server module lazily so the test fails clearly if the
    # checkpoint is missing
    from server import app
    return TestClient(app)


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not CKPT.exists():
        pytest.skip(
            f"checkpoint missing at {CKPT}; run "
            "`python examples/lnn_local_service/run_lnn_local.py` first."
        )
    # TestClient triggers FastAPI's lifespan only when used as a context
    # manager.  We return the entered client and the tests rely on it
    # staying open for the duration of the module.
    import contextlib

    cm = _build_client()
    entered = cm.__enter__()
    try:
        yield entered
    finally:
        cm.__exit__(None, None, None)


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["config"]["input_size"] == 4
    assert body["config"]["hidden_size"] == 8


def test_info_reports_final_metrics(client: TestClient) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert "final_train_mse" in body and "final_val_mse" in body
    assert body["final_train_mse"] >= 0
    assert body["final_val_mse"] >= 0
    assert body["wall_time_s"] > 0


def test_predict_with_2d_input_shape(client: TestClient) -> None:
    seq = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
    r = client.post("/predict", json={"sequence": seq})
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == len(seq)
    # sanity: predictions are finite floats
    for v in body["predictions"]:
        assert isinstance(v, float)
        assert math.isfinite(v)


def test_predict_with_1d_input_broadcasts(client: TestClient) -> None:
    seq = [0.0, 0.1, 0.2, 0.3, 0.4]
    r = client.post("/predict", json={"sequence": seq})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == len(seq)


def test_predict_rejects_wrong_width(client: TestClient) -> None:
    bad = [[0.1, 0.2, 0.3]]  # width=3, model expects 4
    r = client.post("/predict", json={"sequence": bad})
    assert r.status_code == 400
    assert "input_size" in r.json()["detail"]


def test_predict_rejects_empty_sequence(client: TestClient) -> None:
    r = client.post("/predict", json={"sequence": []})
    # Pydantic returns 422; FastAPI returns 400 when the request body
    # itself is well-formed but the handler rejects an empty payload.
    # Both are acceptable — we just need the empty case to be rejected.
    assert r.status_code in (400, 422)


def test_predict_summary_returns_aggregates(client: TestClient) -> None:
    seq = [[0.1, 0.2, 0.3, 0.4]] * 8
    r = client.post("/predict_summary", json={"sequence": seq})
    assert r.status_code == 200
    s = r.json()["summary"]
    for k in ("n_steps", "first", "last", "mean", "stdev", "min", "max"):
        assert k in s
    assert s["n_steps"] == 8
