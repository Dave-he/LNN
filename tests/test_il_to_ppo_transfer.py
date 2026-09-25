"""Tests for the IL -> PPO recurrent weight transfer utility.

Phase B of PRD-MSD-LNN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.control import LNNImitationPolicy  # noqa: E402
from lnn.core.sncp_policy_lite import SNCPPolicyLite  # noqa: E402
from lnn.utils.lnn_il_to_ppo_transfer import (  # noqa: E402
    TransferReport,
    train_il_on_demos,
    transfer_il_to_ppo,
)


# ---------------------------------------------------------------------------
# Architectural compatibility guards
# ---------------------------------------------------------------------------


def _make_pair(hidden_size: int = 16, input_size: int = 14, action_dim: int = 2):
    # Important: encoder_size must equal the recurrent input size we want.
    # LNNImitationPolicy.encoder maps state_dim -> encoder_size; we set it to
    # `input_size` so the recurrent core's input_size == PPO's temporal_input_size.
    il = LNNImitationPolicy(
        state_dim=input_size, action_dim=action_dim,
        hidden_size=hidden_size, encoder_size=input_size,
        recurrent_type="ltc", head_type="mse",
    )
    ppo = SNCPPolicyLite(
        temporal_input_size=input_size, ltc_hidden_size=hidden_size,
        trunk_hidden_size=hidden_size, action_dim=action_dim, ode_method="euler",
    )
    return il, ppo


def test_il_ppo_pair_shapes_match():
    il, ppo = _make_pair(hidden_size=16, input_size=14)
    assert il.recurrent.hidden_size == ppo.ltc.hidden_size == 16
    assert il.recurrent.input_size == ppo.temporal_input_size == 14


def test_transfer_copies_recurrent_weights_exactly():
    il, ppo = _make_pair(hidden_size=16, input_size=14)
    # Mutate IL recurrent weights to known values.
    with torch.no_grad():
        for p in il.recurrent.parameters():
            p.add_(0.123)

    report = transfer_il_to_ppo(il, ppo)
    # All recurrent weights must be identical (zero tolerance).
    assert report.weight_match_tolerance < 1e-6, (
        f"recurrent output mismatch after copy: {report.weight_match_tolerance}"
    )
    # And the params must literally be the same Tensor identity (clones share data).
    src_params = list(il.recurrent.parameters())
    dst_params = list(ppo.ltc.parameters())
    for s, d in zip(src_params, dst_params):
        assert torch.equal(s, d)


def test_transfer_does_not_modify_actor_critic_trunk():
    """The actor / critic / trunk of PPO must be left at their random init."""
    il, ppo = _make_pair(hidden_size=16, input_size=14)
    actor_before = ppo.actor.mu.weight.detach().clone()
    critic_before = ppo.critic.weight.detach().clone()
    transfer_il_to_ppo(il, ppo)
    actor_after = ppo.actor.mu.weight.detach().clone()
    critic_after = ppo.critic.weight.detach().clone()
    assert torch.equal(actor_before, actor_after)
    assert torch.equal(critic_before, critic_after)


def test_transfer_rejects_hidden_size_mismatch():
    il = LNNImitationPolicy(state_dim=14, action_dim=2, hidden_size=16, recurrent_type="ltc")
    ppo = SNCPPolicyLite(temporal_input_size=14, ltc_hidden_size=32, ode_method="euler")
    with pytest.raises(ValueError, match="hidden_size mismatch"):
        transfer_il_to_ppo(il, ppo)


def test_transfer_rejects_input_size_mismatch():
    il = LNNImitationPolicy(state_dim=14, action_dim=2, hidden_size=16, recurrent_type="ltc")
    ppo = SNCPPolicyLite(temporal_input_size=4, ltc_hidden_size=16, ode_method="euler")
    with pytest.raises(ValueError, match="input_size mismatch"):
        transfer_il_to_ppo(il, ppo)


def test_transfer_report_has_useful_fields():
    il, ppo = _make_pair(hidden_size=16, input_size=14)
    rep = transfer_il_to_ppo(il, ppo)
    assert isinstance(rep, TransferReport)
    d = rep.to_dict()
    assert d["src_recurrent_type"] == "ltc"
    assert d["dst_recurrent_type"] == "ltc"
    assert d["src_hidden_size"] == 16
    assert d["dst_hidden_size"] == 16
    assert d["src_input_size"] == 14
    assert d["dst_input_size"] == 14
    assert d["n_params_copied"] > 0
    assert d["weight_match_tolerance"] < 1e-6


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def test_train_il_on_demos_reduces_loss():
    """A linear state->action mapping should be learnable from random data."""
    torch.manual_seed(0)
    N = 256
    state_dim = 14
    action_dim = 2
    # Synthetic expert: action = first 2 dims of state.
    obs = torch.randn(N, state_dim)
    actions = obs[:, :action_dim]
    _policy, history = train_il_on_demos(
        obs, actions,
        state_dim=state_dim, action_dim=action_dim,
        hidden_size=16, recurrent_type="ltc", head_type="mse",
        epochs=20, batch_size=32, lr=3e-3, log_every=0,
    )
    assert len(history) == 20
    assert history[-1] < history[0] * 0.5, (
        f"loss did not decrease enough: {history[0]:.4f} -> {history[-1]:.4f}"
    )


def test_train_il_then_transfer_gives_matching_outputs():
    """End-to-end: train IL, transfer to PPO, verify recurrent outputs match."""
    torch.manual_seed(0)
    state_dim = 14
    action_dim = 2
    N = 128
    obs = torch.randn(N, state_dim)
    actions = obs[:, :action_dim]
    il, _ = train_il_on_demos(
        obs, actions,
        state_dim=state_dim, action_dim=action_dim,
        hidden_size=16, encoder_size=state_dim,  # match PPO's temporal_input_size
        recurrent_type="ltc", head_type="mse",
        epochs=10, batch_size=32, lr=3e-3, log_every=0,
    )
    ppo = SNCPPolicyLite(
        temporal_input_size=state_dim, ltc_hidden_size=16,
        trunk_hidden_size=16, action_dim=action_dim, ode_method="euler",
    )
    rep = transfer_il_to_ppo(il, ppo)
    assert rep.weight_match_tolerance < 1e-6
    # Probe with the same input; outputs must match.
    x = torch.randn(1, 4, state_dim)
    with torch.no_grad():
        y_il = il.recurrent(x)
        y_ppo = ppo.ltc(x)
    if y_il.dim() == 3:
        y_il = y_il[:, -1, :]
        y_ppo = y_ppo[:, -1, :]
    assert torch.allclose(y_il, y_ppo, atol=1e-5)