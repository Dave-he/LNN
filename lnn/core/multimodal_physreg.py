"""Multimodal physics-parameter regression models (EMMA-inspired).

The flagship model is :class:`MultimodalBiCfCNADWithMDN` — a two-stream
encoder that feeds separate video and audio sequences through their own
``BidirectionalNoiseAdaptiveCfC`` backbones, concatenates the resulting
per-step feature streams, and projects the fused features through an
:class:`MDNHead` to produce a Gaussian-mixture distribution over the
continuous physical parameters ``theta in R^K`` (e.g. ``[k, c]`` for a
damped harmonic oscillator).

The design is inspired by EMMA (CVPR 2026, arXiv 2605.24047) which showed
that adding an audio stream to a video-only LTC parameter estimator
improves accuracy on a forced-dynamics rover task. Here we test whether
the same intuition carries over to the **regression** setting (rather than
EMMA's unsupervised inverse-modelling) on a much smaller synthetic
benchmark that fits within CPU-friendly training budgets.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.mdn import MDNHead
from lnn.core.noise_adaptive_cfc import BidirectionalNoiseAdaptiveCfC


class _SingleStreamEncoder(nn.Module):
    """Wrap a :class:`BidirectionalNoiseAdaptiveCfC` as a per-step feature stream.

    The wrapped backbone already returns per-step features of width
    ``hidden_size`` when ``return_sequences=True``.  This thin wrapper only
    re-shapes and re-validates the inputs so the parent module can treat
    the two streams symmetrically.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
    ) -> None:
        super().__init__()
        self.encoder = BidirectionalNoiseAdaptiveCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
        )

    def forward(
        self,
        x: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"expected input with shape [batch, time, features], got {tuple(x.shape)}"
            )
        return self.encoder(x, dt=dt, mask=mask)


class MultimodalBiCfCNADWithMDN(nn.Module):
    """Two-stream Bi-CfC-NAD + MDN head for multimodal physics parameter regression.

    Each stream (video, audio) is independently encoded by a
    :class:`BidirectionalNoiseAdaptiveCfC` backbone.  The two per-step
    feature streams are fused by concatenation followed by a small linear
    projection; the projected features feed an :class:`MDNHead` which
    emits a Gaussian mixture over the target parameters at every time
    step.  The loss (``mdn_negative_log_likelihood``) is taken at the
    final step, but downstream code can also use the per-step standard
    deviation via ``mdn_predicted_std`` for calibration studies.

    Parameters
    ----------
    video_dim, audio_dim:
        Channel sizes of the two modalities.
    hidden_size:
        Width of each per-stream encoder.
    output_size:
        Dimensionality of the physical-parameter vector (e.g. 2 for
        ``[k, c]``).
    num_mixtures:
        Number of Gaussian components in the MDN head.
    fusion:
        ``"concat"`` (default) or ``"mean"``.  Concat preserves per-stream
        information at the cost of extra parameters; mean forces the
        streams into a shared hidden width.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        hidden_size: int = 16,
        output_size: int = 2,
        num_mixtures: int = 1,
        num_layers: int = 1,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        if num_mixtures < 1:
            raise ValueError("num_mixtures must be >= 1")
        if fusion not in {"concat", "mean"}:
            raise ValueError("fusion must be 'concat' or 'mean'")

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.fusion = fusion

        self.video_encoder = _SingleStreamEncoder(
            input_size=video_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
        )
        self.audio_encoder = _SingleStreamEncoder(
            input_size=audio_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
        )

        if fusion == "concat":
            mdn_input_size = hidden_size * 2
        else:  # mean
            mdn_input_size = hidden_size
        self.mdn = MDNHead(
            input_size=mdn_input_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
        )

    def _fuse(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return torch.cat([video_feat, audio_feat], dim=-1)
        return 0.5 * (video_feat + audio_feat)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if video.shape[0] != audio.shape[0]:
            raise ValueError("video and audio must share the batch dimension")
        if video.shape[1] != audio.shape[1]:
            raise ValueError("video and audio must share the time dimension")
        video_feat = self.video_encoder(video, dt=dt, mask=mask)
        audio_feat = self.audio_encoder(audio, dt=dt, mask=mask)
        fused = self._fuse(video_feat, audio_feat)
        return self.mdn(fused)

    def encode_modality(
        self,
        modality: str,
        x: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expose a single encoder for ablations (e.g. video-only forward)."""
        if modality == "video":
            return self.video_encoder(x, dt=dt, mask=mask)
        if modality == "audio":
            return self.audio_encoder(x, dt=dt, mask=mask)
        raise ValueError(f"unknown modality {modality!r}; expected 'video' or 'audio'")
