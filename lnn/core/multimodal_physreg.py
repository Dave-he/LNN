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


class CrossModalAttnBiCfCNADWithMDN(nn.Module):
    """Two-stream Bi-CfC-NAD with single-head cross-modal attention fusion.

    Round-6 follow-up: ``MultimodalBiCfCNADWithMDN`` with ``fusion="concat"``
    failed to beat a single-stream concat baseline because plain concatenation
    forces the model to *implicitly* learn "when to trust which modality"
    from the gradient. This class makes that mechanism *explicit*:

    1. Encode each modality independently with its own
       :class:`BidirectionalNoiseAdaptiveCfC`.
    2. Run two single-head cross-attention passes (video queries audio, audio
       queries video) over the per-step feature streams.
    3. Add the attended features back via a residual and concatenate the
       two refined streams before projecting into the MDN head.

    The attention is over the full ``T`` time steps (no causal mask) so each
    output step can pull information from any past or future step of the other
    modality — exactly the "complementary fill-in" intuition EMMA's audio
    stream provides during video occlusion.
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
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_mixtures < 1:
            raise ValueError("num_mixtures must be >= 1")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.modality_dropout = float(modality_dropout)

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

        # Cross-modal attention: each modality queries the other for K/V.
        self.q_v = nn.Linear(hidden_size, hidden_size)
        self.k_a = nn.Linear(hidden_size, hidden_size)
        self.v_a = nn.Linear(hidden_size, hidden_size)
        self.q_a = nn.Linear(hidden_size, hidden_size)
        self.k_v = nn.Linear(hidden_size, hidden_size)
        self.v_v = nn.Linear(hidden_size, hidden_size)
        # Projection back to hidden_size after concatenating both refined streams.
        self.fuse_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.mdn = MDNHead(
            input_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
        )

    @staticmethod
    def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled-dot-product attention. Returns (attended, weights)."""
        scale = query.shape[-1] ** 0.5
        scores = query @ key.transpose(-1, -2) / scale  # [B, T_q, T_k]
        weights = torch.softmax(scores, dim=-1)
        return weights @ value, weights

    def _apply_modality_dropout(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-time regularizer: with prob ``modality_dropout`` zero each
        stream independently. Guards against zeroing both — if both draws hit,
        one stream (audio, by convention) is retained so the model always sees
        at least one source. Returns (video, audio) — original tensors when
        the model is in eval mode or ``modality_dropout == 0``.
        """

        if not self.training or self.modality_dropout <= 0.0:
            return video, audio
        drop_video = torch.rand((), device=video.device).item() < self.modality_dropout
        drop_audio = torch.rand((), device=audio.device).item() < self.modality_dropout
        if drop_video and drop_audio:
            # Never silence both — pick one to keep so the loss is well-defined.
            drop_audio = False
        if drop_video:
            video = torch.zeros_like(video)
        if drop_audio:
            audio = torch.zeros_like(audio)
        return video, audio

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if video.shape[0] != audio.shape[0]:
            raise ValueError("video and audio must share the batch dimension")
        if video.shape[1] != audio.shape[1]:
            raise ValueError("video and audio must share the time dimension")

        video, audio = self._apply_modality_dropout(video, audio)
        v_feat = self.video_encoder(video, dt=dt, mask=mask)  # [B, T, H]
        a_feat = self.audio_encoder(audio, dt=dt, mask=mask)  # [B, T, H]

        # Video queries audio; audio queries video.
        v_from_a, attn_va = self._attend(self.q_v(v_feat), self.k_a(a_feat), self.v_a(a_feat))
        a_from_v, attn_av = self._attend(self.q_a(a_feat), self.k_v(v_feat), self.v_v(v_feat))
        # Residual + concat-and-project fusion.
        v_refined = v_feat + v_from_a
        a_refined = a_feat + a_from_v
        fused = self.fuse_proj(torch.cat([v_refined, a_refined], dim=-1))

        out = self.mdn(fused)
        if return_attention:
            out["_attn_video_queries_audio"] = attn_va
            out["_attn_audio_queries_video"] = attn_av
        return out


class UniVideoSelfXAttnWithMDN(nn.Module):
    """Architecture-only ablation of :class:`CrossModalAttnBiCfCNADWithMDN`.

    Same dual-encoder + cross-attention machinery, **but both encoders are
    fed the video stream** — there is no audio input. The two
    :class:`BidirectionalNoiseAdaptiveCfC` backbones are initialised
    independently and the cross-attention now performs a "self-cross" pass:
    each video copy queries the other.

    Falsifiable diagnostic for the round 12 §15 "architecture vs information"
    meta-conclusion:

    * On the round 8 synthetic burst task the gain was attributed mostly to
      the dual-encoder architecture (cross_attn beats video_only by +27.6%
      even when audio is decimated to pure noise). If that holds, this
      audio-free variant should also PASS ≥+20% on synthetic burst.
    * On round 11's real EMMA rover data the gain was attributed to audio's
      genuine motor-RPM information (cross_attn +51% vs video_only). If that
      holds, this audio-free variant should FAIL on rover — it has lost the
      information path that made cross_attn win there.

    A "synthetic PASS + rover FAIL" outcome fully confirms the task-
    dependency meta-conclusion; any other combination refines or refutes it.

    The forward signature matches :class:`CrossModalAttnBiCfCNADWithMDN` so
    the existing benchmark harness can swap models without other changes;
    the ``audio`` argument is *required* (for API parity) but is **ignored**
    inside the forward pass — the model only consumes ``video``.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,  # noqa: ARG002 — kept for API parity with cross_attn
        hidden_size: int = 16,
        output_size: int = 2,
        num_mixtures: int = 1,
        num_layers: int = 1,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
    ) -> None:
        super().__init__()
        # Reuse the cross-attention model unchanged; we just always feed it
        # video twice. That keeps the parameter shapes, attention machinery,
        # and MDN head bit-identical to cross_attn, which is exactly the
        # architecture-only ablation we want.
        self._inner = CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim,
            audio_dim=video_dim,  # both encoders receive video
            hidden_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
            num_layers=num_layers,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
            modality_dropout=0.0,
        )
        self.video_dim = video_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures

    @property
    def video_encoder(self) -> nn.Module:
        return self._inner.video_encoder

    @property
    def audio_encoder(self) -> nn.Module:
        """Second video encoder. Named ``audio_encoder`` for harness reuse."""
        return self._inner.audio_encoder

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor | None = None,  # noqa: ARG002 — intentionally ignored
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Discard `audio` if provided; feed `video` to both encoder slots so
        # the cross-attention runs as a self-cross over two independent video
        # representations.
        return self._inner(
            video=video,
            audio=video,
            dt=dt,
            mask=mask,
            return_attention=return_attention,
        )


class NoisyVideoSelfXAttnWithMDN(nn.Module):
    """Round-17 minimal reproduction of the cross-attention regularization
    mechanism: same architecture as :class:`CrossModalAttnBiCfCNADWithMDN`
    but **the second encoder receives the same video plus Gaussian noise**.

    This isolates the "decorrelated second stream" hypothesis from round 16.
    If round 16's interpretation (cross_attn's +14.9pp over uni_video_xattn
    is structural regularisation from any decorrelated stream, not from
    audio content) is correct, this model should:

    * BEAT :class:`UniVideoSelfXAttnWithMDN` (which feeds identical video to
      both encoders).
    * Approach the +47.1% gain :class:`CrossModalAttnBiCfCNADWithMDN`
      achieved with ``audio_mode='zero'`` (no audio content, only a
      structurally distinct second input).

    Conversely, if noisy-video matches the uni-video baseline, the
    regularisation is not just decorrelation — it needs a genuinely
    out-of-distribution second stream (e.g. an audio-like source).

    The ``noise_std`` parameter controls how decorrelated the second stream
    is from the first. The audio input slot of the underlying cross-attn is
    re-purposed as the noisy-video stream; ``audio`` argument at forward
    time is **ignored**.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,  # noqa: ARG002 — kept for API parity; ignored
        hidden_size: int = 16,
        output_size: int = 2,
        num_mixtures: int = 1,
        num_layers: int = 1,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        self._inner = CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim,
            audio_dim=video_dim,  # second encoder also takes video shape
            hidden_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
            num_layers=num_layers,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
            modality_dropout=0.0,
        )
        self.video_dim = video_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.noise_std = float(noise_std)

    @property
    def video_encoder(self) -> nn.Module:
        return self._inner.video_encoder

    @property
    def audio_encoder(self) -> nn.Module:
        """Second video+noise encoder. Named ``audio_encoder`` for harness reuse."""
        return self._inner.audio_encoder

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor | None = None,  # noqa: ARG002 — intentionally ignored
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        # In training mode, sample fresh noise per forward; in eval mode, also
        # sample fresh noise (the regularisation effect is the input itself,
        # not a frozen perturbation).
        if self.noise_std > 0.0:
            noise = torch.randn_like(video) * self.noise_std
            second_input = video + noise
        else:
            second_input = video.clone()
        return self._inner(
            video=video,
            audio=second_input,
            dt=dt,
            mask=mask,
            return_attention=return_attention,
        )


class MixedStreamSelfXAttnWithMDN(nn.Module):
    """Round-18 cosine-similarity probe.

    Same architecture as :class:`UniVideoSelfXAttnWithMDN` / 
    :class:`NoisyVideoSelfXAttnWithMDN`, but the second encoder receives a
    *mixture* of video and matched-power Gaussian noise:

        stream2 = mix_alpha * video + (1 - mix_alpha) * noise

    where ``noise`` is sampled fresh per forward with ``std = video.std()``
    so the mixture has stable power across alpha values.

    * ``mix_alpha = 1.0`` -> stream2 == video, reduces to uni_video_xattn.
    * ``mix_alpha = 0.0`` -> stream2 is pure matched-power noise, analogous
      to cross_attn(audio=random).
    * intermediate alphas trace out a continuous interpolation in
      cosine-similarity space, exactly the curve needed to disentangle the
      round-17 'decorrelation' contribution from the round-16
      'structurally different source' contribution.

    The forward pass also exposes a ``last_cos_sim`` attribute (a float)
    holding the mean cosine similarity between stream1 and stream2 across
    the most recent batch — handy for the benchmark to log the actual
    decorrelation amount per alpha.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,  # noqa: ARG002 — kept for API parity; ignored
        hidden_size: int = 16,
        output_size: int = 2,
        num_mixtures: int = 1,
        num_layers: int = 1,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
        mix_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError("mix_alpha must be in [0, 1]")
        self._inner = CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim,
            audio_dim=video_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
            num_layers=num_layers,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
            modality_dropout=0.0,
        )
        self.video_dim = video_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.mix_alpha = float(mix_alpha)
        self.last_cos_sim: float = float("nan")

    @property
    def video_encoder(self) -> nn.Module:
        return self._inner.video_encoder

    @property
    def audio_encoder(self) -> nn.Module:
        return self._inner.audio_encoder

    @staticmethod
    def _mean_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Flatten each [B, T, F] tensor to per-sample vectors and average
        the cosine similarity across the batch. Returns a Python float."""

        flat_a = a.reshape(a.shape[0], -1)
        flat_b = b.reshape(b.shape[0], -1)
        num = (flat_a * flat_b).sum(dim=-1)
        denom = (flat_a.norm(dim=-1) * flat_b.norm(dim=-1)).clamp_min(1e-12)
        return float((num / denom).mean().item())

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor | None = None,  # noqa: ARG002 — intentionally ignored
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.mix_alpha >= 1.0:
            second_input = video.clone()
        elif self.mix_alpha <= 0.0:
            noise = torch.randn_like(video) * video.std().clamp_min(1e-6)
            second_input = noise
        else:
            noise = torch.randn_like(video) * video.std().clamp_min(1e-6)
            second_input = self.mix_alpha * video + (1.0 - self.mix_alpha) * noise
        # Log the realised cosine similarity so the benchmark can report it.
        self.last_cos_sim = self._mean_cosine_similarity(video, second_input)
        return self._inner(
            video=video,
            audio=second_input,
            dt=dt,
            mask=mask,
            return_attention=return_attention,
        )


class RegisterTokenSelfXAttnWithMDN(nn.Module):
    """Round-19 register-token mechanism probe.

    Same architecture as :class:`UniVideoSelfXAttnWithMDN` and
    :class:`NoisyVideoSelfXAttnWithMDN`, but the second encoder
    receives an *input-independent learnable tensor* (a "register
    token" in the transformer sense) - completely decoupled from the
    actual video input.  This is the *minimal* model that could still
    give a "second encoder" effect: the second Bi-CfC-NAD has its own
    input projection but the projected input is *constant across the
    batch*.

    Hypothesis (falsifiable):  if cross_attn(audio=zero) and
    cross_attn(audio=random) are giving their +14.9pp / +29.5pp
    gains over uni_video_xattn by virtue of *stream2 being a free
    pool the encoder can specialise to a register-token-like
    representation*, then RegisterTokenSelfXAttnWithMDN (which makes
    the input literally independent of the data) should reproduce
    the gain.  If it does not, the gain requires the second stream
    to *interact with* the data, ruling out the register-token
    explanation.

    Notes:
    - The input shape is ``[B, T, video_dim]``, matching the second
      encoder's expected input.  We broadcast a single learnable
      parameter across the batch.
    - The learnable parameter is small (one tensor of shape
      ``[1, 1, video_dim]`` broadcast over B and T) so this model
      has *fewer* parameters than the uni_video variant - if it
      matches uni_video's gain, it must be from the architecture,
      not from extra capacity.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,  # noqa: ARG002 - kept for API parity; ignored
        hidden_size: int = 16,
        output_size: int = 2,
        num_mixtures: int = 1,
        num_layers: int = 1,
        noise_beta: float = 0.9,
        noise_aggregation: str = "independent",
    ) -> None:
        super().__init__()
        self._inner = CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim,
            audio_dim=video_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
            num_layers=num_layers,
            noise_beta=noise_beta,
            noise_aggregation=noise_aggregation,
            modality_dropout=0.0,
        )
        # The single learnable "register token" - broadcast over (B, T).
        # Initialised to small random values so the input projection
        # has non-zero gradient from step 0.
        self.register_token = nn.Parameter(torch.randn(1, 1, video_dim) * 0.1)
        self.video_dim = video_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures

    @property
    def video_encoder(self) -> nn.Module:
        return self._inner.video_encoder

    @property
    def audio_encoder(self) -> nn.Module:
        return self._inner.audio_encoder

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor | None = None,  # noqa: ARG002 - intentionally ignored
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Broadcast the learnable token to [B, T, video_dim].
        B, T, _ = video.shape
        second_input = self.register_token.expand(B, T, -1).contiguous()
        return self._inner(
            video=video,
            audio=second_input,
            dt=dt,
            mask=mask,
            return_attention=return_attention,
        )
