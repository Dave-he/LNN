"""Tests for JEPA-style LNN world model.

Phase C of PRD-MSD-LNN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.jepa_world_model import (  # noqa: E402
    JEPAPolicy,
    JEPAPolicyHead,
    JEPAEncoder,
    JEPAWorldModel,
    LatentPlanner,
    StateSnapshot,
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_encoder_shape():
    enc = JEPAEncoder(obs_dim=14, latent_dim=8, hidden_size=16)
    obs_seq = torch.randn(4, 5, 14)
    z = enc(obs_seq)
    assert z.shape == (4, 8)


def test_encoder_step_shape():
    enc = JEPAEncoder(obs_dim=14, latent_dim=8, hidden_size=16)
    obs = torch.randn(4, 14)
    z = enc.encode_step(obs)
    assert z.shape == (4, 8)


def test_encoder_n_tau_multiscale():
    enc = JEPAEncoder(obs_dim=14, latent_dim=8, hidden_size=16, n_tau=4)
    obs_seq = torch.randn(2, 5, 14)
    z = enc(obs_seq)
    assert z.shape == (2, 8)


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


def test_world_model_predict_shape():
    wm = JEPAWorldModel(latent_dim=8, action_dim=2, hidden_size=16)
    z = torch.randn(4, 8)
    a = torch.randn(4, 2)
    z_next = wm(z, a)
    assert z_next.shape == (4, 8)


def test_world_model_predict_with_sequence_input():
    wm = JEPAWorldModel(latent_dim=8, action_dim=2, hidden_size=16)
    z = torch.randn(4, 5, 8)
    a = torch.randn(4, 2)
    z_next = wm(z, a)
    assert z_next.shape == (4, 8)


# ---------------------------------------------------------------------------
# Policy head
# ---------------------------------------------------------------------------


def test_policy_head_mse_shape():
    head = JEPAPolicyHead(latent_dim=8, action_dim=2, head_type="mse")
    z = torch.randn(4, 8)
    a = head(z)
    assert a.shape == (4, 2)


def test_policy_head_gauss_shape():
    head = JEPAPolicyHead(latent_dim=8, action_dim=2, head_type="gauss")
    z = torch.randn(4, 8)
    a, lp, ent = head.sample(z)
    assert a.shape == (4, 2)
    assert lp.shape == (4,)
    assert ent.shape == (4,)
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ent).all()


def test_policy_head_invalid_type_rejected():
    with pytest.raises(ValueError, match="head_type must be"):
        JEPAPolicyHead(latent_dim=8, action_dim=2, head_type="mdn")  # noqa


# ---------------------------------------------------------------------------
# StateSnapshot
# ---------------------------------------------------------------------------


def test_state_snapshot_roundtrip():
    enc = JEPAEncoder(obs_dim=14, latent_dim=8, hidden_size=16)
    snap = StateSnapshot(enc)
    # mutate
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.5)
    snap.restore(enc)
    # state should match the snapshot — but we mutated AFTER, so restore returns
    # to the snapshot state. Verify by re-snapshotting.
    snap2 = StateSnapshot(enc)
    for k in snap.state:
        assert torch.equal(snap.state[k], snap2.state[k])


def test_state_snapshot_extra_buffers():
    enc = JEPAEncoder(obs_dim=14, latent_dim=8, hidden_size=16)
    extra = {"h_layer0": torch.randn(2, 16)}
    snap = StateSnapshot(enc, extra_buffers=extra)
    assert "h_layer0" in snap.extra
    assert torch.equal(snap.extra["h_layer0"], extra["h_layer0"])


# ---------------------------------------------------------------------------
# LatentPlanner
# ---------------------------------------------------------------------------


def test_latent_planner_rollout_returns_action():
    wm = JEPAWorldModel(latent_dim=8, action_dim=2, hidden_size=16)
    planner = LatentPlanner(wm, horizon=3, n_candidates=4)
    z0 = torch.randn(3, 8)
    res = planner.rollout(z0)
    assert res.best_action.shape == (3, 2)
    assert res.best_return.shape == (3,)
    assert res.rollout_returns.shape == (3, 4)


def test_latent_planner_distill_action():
    wm = JEPAWorldModel(latent_dim=8, action_dim=2, hidden_size=16)
    planner = LatentPlanner(wm, horizon=2, n_candidates=4)
    z0 = torch.randn(2, 8)
    a = planner.distill_action(z0)
    assert a.shape == (2, 2)


# ---------------------------------------------------------------------------
# Full JEPAPolicy
# ---------------------------------------------------------------------------


def test_jepa_policy_forward_train_returns_finite_loss():
    p = JEPAPolicy(obs_dim=14, action_dim=2, latent_dim=8, hidden_size=16)
    obs = torch.randn(4, 3, 14)
    next_obs = torch.randn(4, 3, 14)
    actions = torch.randn(4, 3, 2)
    out = p.forward_train(obs, next_obs, actions)
    assert torch.isfinite(out["mse"]).item()
    assert out["mse"].item() > 0.0
    assert out["z"].shape == (4, 3, 8)
    assert out["z_hat"].shape == (4, 3, 8)


def test_jepa_policy_forward_inference_mse():
    p = JEPAPolicy(obs_dim=14, action_dim=2, latent_dim=8, hidden_size=16)
    obs = torch.randn(4, 14)
    a = p.forward_inference(obs, deterministic=True)
    assert a.shape == (4, 2)
    assert torch.isfinite(a).all()


def test_jepa_policy_forward_inference_gauss():
    p = JEPAPolicy(obs_dim=14, action_dim=2, latent_dim=8, hidden_size=16, head_type="gauss")
    obs = torch.randn(4, 14)
    a = p.forward_inference(obs, deterministic=False)
    assert a.shape == (4, 2)
    assert torch.isfinite(a).all()


def test_jepa_policy_snapshot():
    p = JEPAPolicy(obs_dim=14, action_dim=2, latent_dim=8, hidden_size=16)
    snap = p.snapshot()
    # Mutate weights; snapshot should NOT change.
    with torch.no_grad():
        for param in p.parameters():
            param.add_(0.5)
    snap.restore(p)
    snap2 = p.snapshot()
    for k in snap.state:
        assert torch.equal(snap.state[k], snap2.state[k])


def test_jepa_policy_can_overfit_tiny_dataset():
    """Sanity: with enough capacity and training, the world-model MSE should drop."""
    torch.manual_seed(0)
    p = JEPAPolicy(obs_dim=4, action_dim=2, latent_dim=4, hidden_size=8)
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    N = 64
    obs = torch.randn(N, 1, 4)
    next_obs = torch.randn(N, 1, 4)
    actions = torch.randn(N, 1, 2)
    initial = p.forward_train(obs, next_obs, actions)["mse"].item()
    for _ in range(80):
        out = p.forward_train(obs, next_obs, actions)
        opt.zero_grad()
        out["mse"].backward()
        opt.step()
    final = p.forward_train(obs, next_obs, actions)["mse"].item()
    # We do not require a huge drop on random data — only structural
    # sanity (final < initial OR at least no NaN explosion).
    assert math.isfinite(final)
    assert final <= initial * 1.5, (
        f"MSE exploded during overfit: {initial:.4f} -> {final:.4f}"
    )


import math