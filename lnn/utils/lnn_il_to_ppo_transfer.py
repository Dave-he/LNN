"""Imitation-learning -> PPO weight transfer utility.

Phase B of PRD-MSD-LNN. When ``LNNImitationPolicy`` (BC policy) is trained
on PD-expert demos, its recurrent core encodes useful dynamics. We can
warm-start a ``SNCPPolicyLite`` actor-critic by copying the recurrent
weights — the actor / critic / trunk heads remain randomly initialised
because BC has no information about values or PPO-style stochasticity.

Currently supported:

* ``ltc -> ltc``: direct weight copy (both recurrent cores are
  ``LTCNetwork`` with the same ``cells[0]`` shape).
* ``cfc -> cfc``: identical structure copy.
* Mixed ``ltc <-> cfc``: rejected with a clear error message
  (architectural mismatch — use a CfC-based PPO policy instead).

The transfer is intentionally **partial**: only the temporal encoder is
copied. Trunk / actor / critic are left at their random init so that PPO
can still learn the value function and stochastic exploration. (See
``tests/test_il_to_ppo_transfer.py`` for invariants.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from lnn.core.control import LNNImitationPolicy
from lnn.core.ltc import LTCNetwork
from lnn.core.sncp_policy_lite import SNCPPolicyLite


@dataclass
class TransferReport:
    src_recurrent_type: str
    dst_recurrent_type: str
    src_hidden_size: int
    dst_hidden_size: int
    src_input_size: int
    dst_input_size: int
    n_params_copied: int
    weight_match_tolerance: float  # max abs diff after copy

    def to_dict(self) -> dict:
        return {
            "src_recurrent_type": self.src_recurrent_type,
            "dst_recurrent_type": self.dst_recurrent_type,
            "src_hidden_size": self.src_hidden_size,
            "dst_hidden_size": self.dst_hidden_size,
            "src_input_size": self.src_input_size,
            "dst_input_size": self.dst_input_size,
            "n_params_copied": self.n_params_copied,
            "weight_match_tolerance": self.weight_match_tolerance,
        }


def _validate_shapes(il_policy: LNNImitationPolicy, ppo_policy: SNCPPolicyLite) -> None:
    """Verify the recurrent cores have compatible architectures for transfer."""
    src = il_policy.recurrent
    dst = ppo_policy.ltc

    if not isinstance(src, type(dst)):
        raise ValueError(
            f"recurrent core mismatch: IL has {type(src).__name__}, "
            f"PPO has {type(dst).__name__}. "
            "Both must be LTCNetwork (or both CfCNetwork) for direct weight copy."
        )

    # hidden size
    src_h = getattr(src, "hidden_size", None)
    dst_h = getattr(dst, "hidden_size", None)
    if src_h is None or dst_h is None or src_h != dst_h:
        raise ValueError(
            f"hidden_size mismatch: IL={src_h}, PPO={dst_h}. "
            "Both must match exactly."
        )

    # input size (PPO's temporal_input_size may include spatial summary).
    src_in = getattr(src, "input_size", None)
    dst_in = getattr(ppo_policy, "temporal_input_size", None)
    if src_in is None or dst_in is None:
        raise ValueError("could not infer input sizes for transfer")
    if src_in != dst_in:
        raise ValueError(
            f"input_size mismatch: IL recurrent input={src_in}, "
            f"PPO temporal_input_size={dst_in}. "
            "Either re-train the IL policy with the same input size as PPO, "
            "or adapt the encoder."
        )


def _copy_recurrent_weights(src: nn.Module, dst: nn.Module) -> int:
    """Copy state_dict from src.cells[0] (and any siblings) to dst.cells[0].

    Returns the total number of parameters copied.
    """
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    if set(src_state.keys()) != set(dst_state.keys()):
        # fall back to filter: only copy common keys
        common = {k: v for k, v in src_state.items() if k in dst_state}
    else:
        common = dict(src_state)
    n_params = 0
    for k, v in common.items():
        dst_state[k] = v.detach().clone()
        n_params += v.numel()
    dst.load_state_dict(dst_state)
    return n_params


def transfer_il_to_ppo(
    il_policy: LNNImitationPolicy,
    ppo_policy: SNCPPolicyLite,
    atol: float = 1e-6,
) -> TransferReport:
    """Copy the recurrent core weights from ``il_policy`` into ``ppo_policy``.

    The actor / critic / trunk of ``ppo_policy`` are NOT modified.
    Returns a :class:`TransferReport` describing what was copied and how
    close the resulting weights are to the source (should be identical).
    """
    _validate_shapes(il_policy, ppo_policy)
    src = il_policy.recurrent
    dst = ppo_policy.ltc
    src_h = src.hidden_size
    src_in = src.input_size
    n_params = _copy_recurrent_weights(src, dst)

    # verify: forward through both, compare outputs (same input).
    src.eval()
    dst.eval()
    with torch.no_grad():
        x_probe = torch.randn(1, 4, src_in)
        y_src = src(x_probe)
        y_dst = dst(x_probe)
        # LTC may return sequences [B, T, H]; collapse if so.
        if y_src.dim() == 3:
            y_src = y_src[:, -1, :]
            y_dst = y_dst[:, -1, :]
        diff = float((y_src - y_dst).abs().max().item())

    return TransferReport(
        src_recurrent_type=il_policy.recurrent_type,
        dst_recurrent_type="ltc",  # SNCPPolicyLite is LTC-backed
        src_hidden_size=src_h,
        dst_hidden_size=ppo_policy.ltc_hidden_size,
        src_input_size=src_in,
        dst_input_size=ppo_policy.temporal_input_size,
        n_params_copied=n_params,
        weight_match_tolerance=diff,
    )


def train_il_on_demos(
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    state_dim: int,
    action_dim: int,
    hidden_size: int = 32,
    encoder_size: Optional[int] = None,
    recurrent_type: str = "ltc",
    head_type: str = "mse",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    log_every: int = 10,
) -> tuple[LNNImitationPolicy, list[float]]:
    """Train an ``LNNImitationPolicy`` on (obs, actions) demo transitions.

    This is the Phase B companion to :func:`transfer_il_to_ppo`: train BC
    on PD-expert demos, then transfer the recurrent weights into PPO.

    Args:
        obs: ``[N, state_dim]`` per-step observations (no time dim — each
            row is one env step, treated as a length-1 sequence).
        actions: ``[N, action_dim]`` corresponding expert actions.
        state_dim: input state dimensionality (e.g. 14 for PointMassNavLite).
        action_dim: output action dimensionality (2 for PointMassNavLite).
        hidden_size: recurrent hidden size (must match the target PPO's
            ``ltc_hidden_size`` for clean transfer).
        recurrent_type: ``ltc`` (default; matches SNCPPolicyLite) or ``cfc``.
        head_type: ``mse`` for deterministic targets (single-mode PD demos).
        epochs: number of full passes over the dataset.
        batch_size: minibatch size.
        lr: Adam learning rate.
        seed: torch.manual_seed for reproducibility.

    Returns:
        (trained_policy, loss_history)
    """
    torch.manual_seed(seed)
    policy = LNNImitationPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        encoder_size=encoder_size or hidden_size,
        recurrent_type=recurrent_type,
        head_type=head_type,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    N = obs.shape[0]
    history: list[float] = []

    for epoch in range(epochs):
        # shuffle once per epoch
        perm = torch.randperm(N)
        running_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            x = obs[idx].unsqueeze(1)  # [B, T=1, F]
            y = actions[idx]
            pred = policy(x)
            if isinstance(pred, dict):
                # MDN head — use mean squared error to the mode for now
                # (training MDN-NLL for a single-mode PD expert is overkill
                # and r307 already flagged MDN as honest-negative on
                # single-mode targets).
                pred = pred.get("mean", pred.get("logits"))
                if pred is None:
                    raise RuntimeError("MDN output dict missing expected keys")
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running_loss += float(loss.item())
            n_batches += 1
        avg_loss = running_loss / max(n_batches, 1)
        history.append(avg_loss)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[il] epoch {epoch+1}/{epochs} avg_loss={avg_loss:.5f}", flush=True)

    return policy, history