"""End-to-end test for the FastAPI LNN local PLR service.

Strategy
--------
Same as round 133: use ``fastapi.testclient.TestClient`` for in-process
testing.  We avoid actually binding a socket.

The test module is parameterised over the two model kinds ('plr',
'plr_cfc').  This is a small-but-meaningful coverage matrix that
verifies the predict path actually flows through both checkpoints.

Run::

    pytest examples/lnn_local_plr_service/test_server.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


HERE = Path(__file__).resolve().parent
PLR_CKPT = HERE / "artifacts" / "plr_small.pt"
PLR_CFC_CKPT = HERE / "artifacts" / "plr_cfc_small.pt"


def _build_client() -> TestClient:
    from server import app  # noqa: WPS433 (lazy import after conftest)
    return TestClient(app)


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not PLR_CKPT.exists() or not PLR_CFC_CKPT.exists():
        pytest.skip(
            f"checkpoint missing under {HERE/'artifacts'}; run "
            "`python examples/lnn_local_plr_service/run_lnn_local.py` first."
        )
    cm = _build_client()
    entered = cm.__enter__()
    try:
        yield entered
    finally:
        cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_both_models(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"plr", "plr_cfc"}
    assert body["device"] == "cpu"


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------


def test_info_reports_params_for_both_models(client: TestClient) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert "plr" in body["models"] and "plr_cfc" in body["models"]
    # Round 134 design: PLR+CfC two-axis is a superset of PLR; expect
    # more params than the PLR-only model.
    plr_params = body["models"]["plr"]["n_params"]
    plr_cfc_params = body["models"]["plr_cfc"]["n_params"]
    assert plr_params > 0
    assert plr_cfc_params > plr_params, (
        f"PLR+CfC ({plr_cfc_params}) should have more params than PLR-only ({plr_params})"
    )


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_kind", ["plr", "plr_cfc"])
def test_predict_2d_sequence(client: TestClient, model_kind: str) -> None:
    """Standard [T, F] payload."""
    seq = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
    r = client.post("/predict", json={"sequence": seq, "model": model_kind})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model_kind
    assert body["input_shape"] == [1, 3, 4]
    preds = body["predictions"]
    assert len(preds) == 3
    for step in preds:
        assert len(step) == 1                       # output_size = 1
        assert math.isfinite(step[0])


@pytest.mark.parametrize("model_kind", ["plr", "plr_cfc"])
def test_predict_1d_sequence_broadcast(client: TestClient, model_kind: str) -> None:
    """1-D [T] payload gets broadcast to (T, input_size) by replication."""
    seq = [0.1, 0.2, 0.3, 0.4]
    r = client.post("/predict", json={"sequence": seq, "model": model_kind})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_shape"] == [1, 4, 4]
    assert len(body["predictions"]) == 4


def test_predict_default_model_is_plr_cfc(client: TestClient) -> None:
    seq = [[0.1, 0.2, 0.3, 0.4]]
    r = client.post("/predict", json={"sequence": seq})
    assert r.status_code == 200
    assert r.json()["model"] == "plr_cfc"


def test_predict_unknown_model_returns_400(client: TestClient) -> None:
    seq = [[0.1, 0.2, 0.3, 0.4]]
    r = client.post("/predict", json={"sequence": seq, "model": "not_a_model"})
    assert r.status_code == 400
    assert "unknown model" in r.json()["detail"]


def test_predict_wrong_width_returns_400(client: TestClient) -> None:
    # input_size is 4 in training; sending F=3 should be rejected.
    seq = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    r = client.post("/predict", json={"sequence": seq, "model": "plr"})
    assert r.status_code == 400
    assert "input_size" in r.json()["detail"]


def test_predict_empty_sequence_returns_400(client: TestClient) -> None:
    r = client.post("/predict", json={"sequence": []})
    # Pydantic enforces min_length=1 → 422 from FastAPI's validator.
    assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /predict_summary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_kind", ["plr", "plr_cfc"])
def test_predict_summary_shape_and_finiteness(
    client: TestClient, model_kind: str
) -> None:
    seq = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
    r = client.post("/predict_summary", json={"sequence": seq, "model": model_kind})
    assert r.status_code == 200, r.text
    body = r.json()
    summary = body["summary"]
    for key in ("mean", "std", "min", "max", "last", "n"):
        assert key in summary
    assert summary["n"] == 2                          # T=2 steps × 1 output
    assert math.isfinite(summary["mean"])


# ---------------------------------------------------------------------------
# End-to-end "two-axis wins on regime switch" smoke check
# ---------------------------------------------------------------------------


def test_plr_cfc_close_to_plr_on_smooth_signal(client: TestClient) -> None:
    """Sanity: PLR+CfC and PLR should give similar predictions on a
    smooth (constant-ish) signal because the regime-switch advantage
    only kicks in when there's actual structure to gate.
    """
    seq = [[0.5, 0.5, 0.5, 0.5]] * 8
    r_plr = client.post("/predict", json={"sequence": seq, "model": "plr"})
    r_cfc = client.post("/predict", json={"sequence": seq, "model": "plr_cfc"})
    assert r_plr.status_code == r_cfc.status_code == 200
    plr_last = r_plr.json()["predictions"][-1][0]
    cfc_last = r_cfc.json()["predictions"][-1][0]
    # Both predictions should be finite; on a constant signal they
    # should be within an order of magnitude (rough sanity check).
    assert math.isfinite(plr_last) and math.isfinite(cfc_last)
    assert abs(plr_last - cfc_last) < 1.0, (
        f"on a constant signal, PLR ({plr_last}) and PLR+CfC ({cfc_last}) "
        "should agree to within ~1.0"
    )
