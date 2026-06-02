"""Noise-Adaptive Decay CfC (CfC-NAD).

Implements a CfC variant whose mixing dynamics are conditioned on a running
estimate of input noise. Motivation comes from the 2026-05-26 paper "Comparative
Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition:
Robustness, Efficiency, and Clinical Utility" which reports that vanilla CfC
already beats LSTM under noise, but uses a single time-scale gate that ignores
local heteroscedasticity.

The cell maintains a streaming EMA of the squared first-difference of the
input as a cheap online noise score. A learnable noise gate then biases the
fresh CfC candidate toward the previous hidden state when the local noise is
high — effectively a heteroscedastic low-pass on h.

Memory footprint stays O(1): the only added state per sample is one EMA
vector of shape [batch, input_size] (the noise score) and one previous-input
vector for the first-difference proxy.

When no input mask is supplied at forward time, the network pre-computes the
noise EMA for the whole sequence in parallel using
:func:`lnn.core.long_sequence.parallel_liquid_relaxation`. This is numerically
equivalent to the streaming form but removes the Python-level tensor ops per
step that dominated the original CPU latency overhead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.long_sequence import parallel_liquid_relaxation
from lnn.core.mdn import MDNHead
from lnn.core.sequence_utils import select_step_delta, select_step_mask


def vectorized_noise_ema(masked_input: torch.Tensor, beta: float) -> torch.Tensor:
    """Streaming EMA of squared first-differences, computed in parallel.

    Mirrors the recurrence used by the streaming path:
        noise_ema[:, 0, :] = 0
        noise_ema[:, t, :] = beta * noise_ema[:, t-1, :]
            + (1 - beta) * (masked_input[:, t, :] - masked_input[:, t-1, :]) ** 2

    Uses :func:`lnn.core.long_sequence.parallel_liquid_relaxation` so the
    cumulative recurrence runs in O(T) on the GPU without a Python-level loop.
    Returns a tensor with the same shape as ``masked_input``.
    """

    if masked_input.dim() != 3:
        raise ValueError(
            f"masked_input must have shape [batch, time, features], got {tuple(masked_input.shape)}"
        )
    if masked_input.shape[1] == 0:
        return torch.zeros_like(masked_input)
    diff = torch.zeros_like(masked_input)
    if masked_input.shape[1] > 1:
        diff[:, 1:, :] = masked_input[:, 1:, :] - masked_input[:, :-1, :]
    diff_sq = diff * diff
    retain = torch.full_like(diff_sq, float(beta))
    return parallel_liquid_relaxation(retain, diff_sq)


class NoiseAdaptiveCfCCell(nn.Module):
    """Closed-form Continuous-time cell with heteroscedastic gating.

    The base mixing follows the standard CfC formulation:
        h_cfc = decay * g(x, h) + (1 - decay) * h_branch(x, h)
        decay = sigmoid(-f_gate(x, h, noise) * time_scale * dt)

    A separate noise gate then interpolates between h_cfc and the previous
    hidden state:
        h_new = (1 - noise_gate) * h_cfc + noise_gate * h_prev

    where ``noise_gate = sigmoid(W_noise @ noise_score)`` reads off the streaming
    noise EMA of the input. Under low noise the cell reduces to vanilla CfC;
    under high noise it holds the prior state, suppressing spurious updates.
    """

    def __init__(self, input_size: int, hidden_size: int, noise_beta: float = 0.9) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # EMA smoothing factor for the noise score (closer to 1 = smoother).
        self.noise_beta = float(noise_beta)

        # f_gate receives [x, h, noise_score] so the gate can be modulated
        # directly by the per-feature noise estimate.
        gate_in = input_size + hidden_size + input_size
        self.f_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Per-channel learnable time scale (matches the baseline CfC layout).
        self.time_scale = nn.Parameter(torch.ones(hidden_size))
        # Maps per-feature noise scores to a per-channel retention gate.
        self.noise_gate_proj = nn.Linear(input_size, hidden_size)
        # Initialise noise gate near zero — start as vanilla CfC; let training
        # decide how much heteroscedastic retention to use.
        nn.init.zeros_(self.noise_gate_proj.weight)
        nn.init.zeros_(self.noise_gate_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        noise_score: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        gate_in = torch.cat([x_t, h, noise_score], dim=-1)
        f = self.f_gate(gate_in)
        combined = torch.cat([x_t, h], dim=-1)
        g = self.g_branch(combined)
        h_branch = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_cfc = decay * g + (1.0 - decay) * h_branch
        # Heteroscedastic blend: pull toward h under high noise.
        noise_gate = torch.sigmoid(self.noise_gate_proj(noise_score))
        h_new = (1.0 - noise_gate) * h_cfc + noise_gate * h
        return h_new


class NoiseAdaptiveCfCNetwork(nn.Module):
    """Stacked Noise-Adaptive CfC network with the standard (x, dt, mask) API.

    The forward signature matches :class:`lnn.core.cfc.CfCNetwork` so the network
    is a drop-in replacement in :class:`lnn.core.trainer.Trainer`.

    Args:
        input_size: Feature dimension of x.
        hidden_size: Hidden state width per layer.
        output_size: Output projection dimension.
        num_layers: Number of stacked cells.
        return_sequences: If False, returns only the last step.
        noise_beta: EMA smoothing factor for the noise score (default 0.9).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        noise_beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(NoiseAdaptiveCfCCell(in_dim, hidden_size, noise_beta=noise_beta))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        h = h0
        layer_input = x
        # When no mask is provided we can pre-compute the streaming noise EMA
        # in parallel for the whole sequence (Liquid-S4 style cumulative form).
        # This removes the per-step Python tensor ops without changing the
        # recurrence semantics.
        use_parallel_noise = mask is None
        for i, cell in enumerate(self.cells):
            outputs: list[torch.Tensor] = []
            h_i = h[i]
            beta = cell.noise_beta if hasattr(cell, "noise_beta") else 0.9
            layer_clean = torch.nan_to_num(layer_input)
            noise_ema_full: torch.Tensor | None = None
            if use_parallel_noise:
                noise_ema_full = vectorized_noise_ema(layer_clean, beta)
            # Streaming fallback state (only used when mask is not None).
            prev_x = torch.zeros_like(layer_clean[:, 0, :])
            noise_ema_stream = torch.zeros_like(layer_clean[:, 0, :])
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, cell.input_size, x.device, x.dtype
                )
                x_t = layer_clean[:, t, :]
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask

                if noise_ema_full is not None:
                    noise_score = noise_ema_full[:, t, :]
                else:
                    diff_sq = (x_t - prev_x) ** 2 if t > 0 else torch.zeros_like(x_t)
                    noise_ema_stream = beta * noise_ema_stream + (1.0 - beta) * diff_sq
                    prev_x = x_t
                    noise_score = noise_ema_stream

                h_candidate = cell(x_t, h_i, noise_score=noise_score, dt=dt_t)
                if update_mask is None:
                    h_i = h_candidate
                else:
                    h_i = update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [
                    h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0)
                    for j in range(self.num_layers)
                ],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])


__all__ = [
    "NoiseAdaptiveCfCCell",
    "NoiseAdaptiveCfCNetwork",
    "BidirectionalNoiseAdaptiveCfC",
    "CfCNADWithMDN",
    "BiCfCNADWithMDN",
    "mdn_predicted_std",
    "vectorized_noise_ema",
]


def _flip_temporal(
    t: torch.Tensor | float | int | None, seq_len: int
) -> torch.Tensor | float | int | None:
    """Flip the time axis of a dt/mask argument so it lines up with a reversed
    input sequence. Returns the input unchanged for shapes where the time axis
    cannot be identified (e.g. scalar, [B], [B, 1])."""

    if t is None or isinstance(t, int | float):
        return t
    if not torch.is_tensor(t):
        return t
    if t.dim() == 0:
        return t
    if t.dim() == 1:
        # Only flip if the first dim matches seq_len (i.e. it is a per-step
        # schedule). [B] arguments stay the same.
        if t.shape[0] == seq_len:
            return torch.flip(t, dims=[0])
        return t
    if t.dim() == 2:
        if t.shape[1] == seq_len:  # [B, T]
            return torch.flip(t, dims=[1])
        if t.shape[0] == seq_len:  # [T, F] or [T, 1]
            return torch.flip(t, dims=[0])
        return t
    if t.dim() == 3:
        if t.shape[1] == seq_len:  # [B, T, F]
            return torch.flip(t, dims=[1])
        return t
    return t


class BidirectionalNoiseAdaptiveCfC(nn.Module):
    """Bidirectional Noise-Adaptive CfC.

    Stacks two independent :class:`NoiseAdaptiveCfCNetwork` instances. The
    forward instance reads ``x`` left-to-right; the backward instance reads
    ``flip(x)`` and the resulting per-step features are flipped back so they
    line up with the original time axis. Per-step features are concatenated
    along the feature axis and fed through a single output projection.

    Motivated by sxlxbo/CTDFormer (2026-05-17) which replaces multi-head
    attention with a bidirectional CfC for bearing fault diagnosis. The
    falsifiable claim verified in :mod:`scripts.benchmark_bi_cfc_nad` is that
    bi-CfC-NAD should beat uni-CfC-NAD on a windowed-median regression task
    whose targets depend on both past and future context.

    Args:
        noise_aggregation: ``"independent"`` (default, each branch keeps its
            own causal/anti-causal noise EMA) or ``"centered"`` (both branches
            share a non-causal noise score computed as the average of the
            forward and backward EMAs). ``"centered"`` exploits the fact that
            a bidirectional model has already broken causality and can use
            future information to improve the local noise estimate.
    """

    _SUPPORTED_AGGREGATIONS = ("independent", "centered")

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
    ) -> None:
        super().__init__()
        if noise_aggregation not in self._SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"noise_aggregation must be one of {self._SUPPORTED_AGGREGATIONS}, "
                f"got {noise_aggregation!r}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.noise_beta = float(noise_beta)
        self.noise_aggregation = noise_aggregation

        # Inner networks emit per-step features of width ``hidden_size``.
        self.forward_net = NoiseAdaptiveCfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            noise_beta=noise_beta,
        )
        self.backward_net = NoiseAdaptiveCfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            noise_beta=noise_beta,
        )
        # Concatenate forward + backward features then project.
        self.output_proj = nn.Linear(2 * hidden_size, output_size)

    @staticmethod
    def _centered_noise_score(x: torch.Tensor, beta: float) -> torch.Tensor:
        """Non-causal noise score: the average of the forward and time-flipped
        backward EMAs. Both halves are computed in parallel via
        :func:`vectorized_noise_ema`."""

        forward_ema = vectorized_noise_ema(x, beta)
        backward_ema_rev = vectorized_noise_ema(torch.flip(x, dims=[1]), beta)
        backward_ema = torch.flip(backward_ema_rev, dims=[1])
        return 0.5 * (forward_ema + backward_ema)

    def _run_with_external_noise(
        self,
        net: NoiseAdaptiveCfCNetwork,
        x: torch.Tensor,
        noise: torch.Tensor,
        dt: float | torch.Tensor | None,
    ) -> torch.Tensor:
        """Run a NoiseAdaptiveCfCNetwork with an externally supplied per-step
        noise score (shape ``[B, T, F_in]``). Only used by the centered path;
        kept here so the data flow stays in one place and the inner network
        does not need to grow new public arguments."""

        batch_size, seq_len, _ = x.shape
        h_state = torch.zeros(
            net.num_layers, batch_size, net.hidden_size, device=x.device, dtype=x.dtype
        )
        layer_input = x
        for i, cell in enumerate(net.cells):
            outputs: list[torch.Tensor] = []
            h_i = h_state[i]
            # External noise is only meaningful for the first layer (input
            # space); deeper layers fall back to their own parallel EMA.
            if i == 0:
                noise_for_layer = noise
            else:
                beta = cell.noise_beta if hasattr(cell, "noise_beta") else 0.9
                noise_for_layer = vectorized_noise_ema(layer_input, beta)
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                x_t = layer_input[:, t, :]
                noise_t = noise_for_layer[:, t, :]
                h_i = cell(x_t, h_i, noise_score=noise_t, dt=dt_t)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h_state = torch.cat(
                [
                    h_i.unsqueeze(0) if j == i else h_state[j].unsqueeze(0)
                    for j in range(net.num_layers)
                ],
                dim=0,
            )
        return net.output_proj(layer_input)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,  # noqa: ARG002 — kept for API parity
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape [batch, time, features], got {tuple(x.shape)}")
        seq_len = x.shape[1]

        if self.noise_aggregation == "centered":
            if mask is not None:
                # Centered EMA assumes parallel computation; bail out gracefully
                # so masked inputs still produce a well-defined result.
                raise ValueError(
                    "noise_aggregation='centered' does not support a mask "
                    "argument; use 'independent' or supply mask=None."
                )
            x_clean = torch.nan_to_num(x)
            centered = self._centered_noise_score(x_clean, self.noise_beta)
            fwd_features = self._run_with_external_noise(
                self.forward_net, x_clean, centered, dt
            )
            dt_rev = _flip_temporal(dt, seq_len)
            bwd_features_rev = self._run_with_external_noise(
                self.backward_net,
                torch.flip(x_clean, dims=[1]),
                torch.flip(centered, dims=[1]),
                dt_rev,
            )
            bwd_features = torch.flip(bwd_features_rev, dims=[1])
        else:
            fwd_features = self.forward_net(x, dt=dt, mask=mask)
            x_rev = torch.flip(x, dims=[1])
            dt_rev = _flip_temporal(dt, seq_len)
            mask_rev = _flip_temporal(mask, seq_len)
            bwd_features_rev = self.backward_net(x_rev, dt=dt_rev, mask=mask_rev)
            bwd_features = torch.flip(bwd_features_rev, dims=[1])
        combined = torch.cat([fwd_features, bwd_features], dim=-1)
        if self.return_sequences:
            return self.output_proj(combined)
        return self.output_proj(combined[:, -1, :])


def mdn_predicted_std(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Total predicted standard deviation under a diagonal Gaussian mixture.

    Computes the mixture-variance closed form ``Var = Σ w_k (σ_k² + μ_k²) - μ²``
    then returns its square root. Works for any ``num_mixtures`` and any
    ``output_size`` — the per-output stds are aggregated into a scalar by
    averaging across the output dimension (handy for 1-D regression where the
    consumer just wants a single uncertainty per step).
    """

    logits = params["logits"]
    loc = params["loc"]  # [..., K, D]
    scale = torch.exp(params["log_scale"])  # [..., K, D]
    weights = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [..., K, 1]
    mean = (weights * loc).sum(dim=-2)  # [..., D]
    second_moment = (weights * (scale.pow(2) + loc.pow(2))).sum(dim=-2)  # [..., D]
    var = (second_moment - mean.pow(2)).clamp_min(0.0)
    return var.mean(dim=-1).sqrt()  # [...] scalar per step


class CfCNADWithMDN(nn.Module):
    """CfC-NAD feature extractor wired into an MDN head for heteroscedastic
    regression. Each time step produces a Gaussian-mixture distribution over
    the target, so the model emits both a point prediction (``mdn_mean``) and
    an aleatoric uncertainty estimate (:func:`mdn_predicted_std`).

    Falsifiable test in :mod:`scripts.benchmark_cfcnad_mdn_uncertainty`:
    when trained on a heteroscedastic per-sample SNR schedule, the predicted
    standard deviation should correlate (Pearson r) with the actual per-sample
    noise standard deviation on a held-out set (claim threshold r ≥ 0.5).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_mixtures: int = 1,
        num_layers: int = 1,
        return_sequences: bool = True,
        noise_beta: float = 0.9,
    ) -> None:
        super().__init__()
        if num_mixtures < 1:
            raise ValueError("num_mixtures must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.return_sequences = return_sequences

        self.encoder = NoiseAdaptiveCfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            noise_beta=noise_beta,
        )
        self.mdn = MDNHead(
            input_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(x, h0=h0, dt=dt, mask=mask)
        params = self.mdn(features)
        if not self.return_sequences:
            params = {
                "logits": params["logits"][:, -1],
                "loc": params["loc"][:, -1],
                "log_scale": params["log_scale"][:, -1],
            }
        return params


class BiCfCNADWithMDN(nn.Module):
    """Bidirectional CfC-NAD feature extractor wired into an MDN head.

    Same heteroscedastic-uncertainty interface as :class:`CfCNADWithMDN`, but
    the backbone is :class:`BidirectionalNoiseAdaptiveCfC` so each step sees
    both past and future context. The two-sided context should help the model
    calibrate the noise score more accurately than the unidirectional variant
    (round 4's hypothesis bumped to W+1).

    Concretely this is a drop-in for ``CfCNADWithMDN`` with one additional
    knob: ``noise_aggregation`` is forwarded to the bidirectional backbone.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_mixtures: int = 1,
        num_layers: int = 1,
        return_sequences: bool = True,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
    ) -> None:
        super().__init__()
        if num_mixtures < 1:
            raise ValueError("num_mixtures must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.return_sequences = return_sequences

        # Configure the bidirectional backbone so its output is a per-step
        # feature stream of width ``hidden_size``.
        self.encoder = BidirectionalNoiseAdaptiveCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
        )
        self.mdn = MDNHead(
            input_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(x, h0=h0, dt=dt, mask=mask)
        params = self.mdn(features)
        if not self.return_sequences:
            params = {
                "logits": params["logits"][:, -1],
                "loc": params["loc"][:, -1],
                "log_scale": params["log_scale"][:, -1],
            }
        return params
