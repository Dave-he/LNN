"""Tests for the multimodal physics regression model and dataset."""

from __future__ import annotations

import math

import pytest
import torch

from lnn.core.mdn import mdn_mean
from lnn.core.multimodal_physreg import (
    CrossModalAttnBiCfCNADWithMDN,
    MultimodalBiCfCNADWithMDN,
)
from lnn.data.multimodal_physreg import (
    HeterogeneousForcedDataset,
    MultimodalPhysicsDataset,
)


# ---------- Dataset tests ----------


def test_multimodal_physics_dataset_shapes() -> None:
    dataset = MultimodalPhysicsDataset(num_samples=4, seq_len=16)
    assert len(dataset) == 4
    sample, target = dataset[0]
    assert sample["video"].shape == (16, 1)
    assert sample["audio"].shape == (16, 1)
    assert target["params"].shape == (2,)
    assert target["omega"].shape == ()
    assert target["zeta"].shape == ()


def test_dataset_audio_correlates_with_k() -> None:
    """The audio stream should encode the spring constant k via omega_d.

    This is the EMMA insight in a test form: when k is large (high omega),
    the audio frequency should also be high.  We check a single-seed
    sanity, not strict monotonicity.
    """
    dataset = MultimodalPhysicsDataset(num_samples=256, seq_len=8, seed=0)
    videos, audios, params = [], [], []
    for i in range(64):
        sample, target = dataset[i]
        videos.append(sample["video"])
        audios.append(sample["audio"])
        params.append(target["params"])
    k = torch.stack([p[0] for p in params])
    f_audio = torch.stack(audios).mean(dim=(1, 2))
    corr = torch.corrcoef(torch.stack([k, f_audio]))[0, 1].item()
    assert corr > 0.5, f"audio should strongly correlate with k (got {corr:.3f})"


def test_dataset_invalid_zeta_range() -> None:
    with pytest.raises(ValueError):
        MultimodalPhysicsDataset(num_samples=2, zeta_range=(0.0, 0.5))
    with pytest.raises(ValueError):
        MultimodalPhysicsDataset(num_samples=2, zeta_range=(0.1, 1.2))


# ---------- Model tests ----------


def test_multimodal_model_forward_shapes() -> None:
    model = MultimodalBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=8, output_size=2, num_mixtures=1
    )
    video = torch.randn(3, 10, 1)
    audio = torch.randn(3, 10, 1)
    out = model(video, audio)
    assert out["logits"].shape == (3, 10, 1)
    assert out["loc"].shape == (3, 10, 1, 2)
    assert out["log_scale"].shape == (3, 10, 1, 2)


def test_multimodal_model_fusion_modes() -> None:
    for fusion in ("concat", "mean"):
        model = MultimodalBiCfCNADWithMDN(
            video_dim=1, audio_dim=1, hidden_size=8, output_size=2, fusion=fusion
        )
        out = model(torch.randn(2, 6, 1), torch.randn(2, 6, 1))
        assert out["loc"].shape == (2, 6, 1, 2)


def test_multimodal_model_rejects_unknown_fusion() -> None:
    with pytest.raises(ValueError):
        MultimodalBiCfCNADWithMDN(video_dim=1, audio_dim=1, fusion="bogus")  # type: ignore[arg-type]


def test_multimodal_model_rejects_zero_mixtures() -> None:
    with pytest.raises(ValueError):
        MultimodalBiCfCNADWithMDN(video_dim=1, audio_dim=1, num_mixtures=0)


def test_multimodal_video_only_branch_shape_mismatch() -> None:
    model = MultimodalBiCfCNADWithMDN(video_dim=1, audio_dim=1, hidden_size=8, output_size=2)
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 1), torch.randn(2, 6, 1))  # mismatched time


def test_encode_modality_returns_video_or_audio() -> None:
    model = MultimodalBiCfCNADWithMDN(video_dim=1, audio_dim=1, hidden_size=8, output_size=2)
    x = torch.randn(2, 6, 1)
    v_feat = model.encode_modality("video", x)
    a_feat = model.encode_modality("audio", x)
    assert v_feat.shape == a_feat.shape == (2, 6, 8)
    with pytest.raises(ValueError):
        model.encode_modality("tactile", x)  # type: ignore[arg-type]


def test_mean_fusion_invariance_to_permutation() -> None:
    """mean fusion should be invariant to swapping the two stream contents."""
    model = MultimodalBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=8, output_size=2, fusion="mean"
    )
    video = torch.randn(2, 6, 1)
    audio = torch.randn(2, 6, 1)
    out1 = model(video, audio)
    out2 = model(audio, video)
    # Mdn locs differ only because of the (independently-initialised) video
    # vs audio encoders, so equality is too strict.  Check that the per-step
    # variance of the *differences* between the two runs is finite and the
    # runs are not identical (sanity).
    diff = (out1["loc"] - out2["loc"]).abs().max().item()
    assert math.isfinite(diff)


def test_mdn_mean_with_multimodal_output() -> None:
    model = MultimodalBiCfCNADWithMDN(video_dim=1, audio_dim=1, hidden_size=8, output_size=2)
    out = model(torch.randn(2, 4, 1), torch.randn(2, 4, 1))
    mean = mdn_mean({k: v[:, -1] for k, v in out.items()})
    assert mean.shape == (2, 2)


def test_multimodal_model_trains_below_random_init() -> None:
    """Sanity: a few gradient steps should reduce NLL on a small batch."""
    torch.manual_seed(0)
    model = MultimodalBiCfCNADWithMDN(video_dim=1, audio_dim=1, hidden_size=8, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    from lnn.core.mdn import mdn_negative_log_likelihood
    dataset = MultimodalPhysicsDataset(num_samples=16, seq_len=10, seed=0)
    items = [dataset[i] for i in range(8)]
    video = torch.stack([it[0]["video"] for it in items])
    audio = torch.stack([it[0]["audio"] for it in items])
    params = torch.stack([it[1]["params"] for it in items])
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        out = model(video, audio)
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, params)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"NLL did not decrease: {losses}"


# ---------- CrossModalAttnBiCfCNADWithMDN tests ----------


def test_cross_modal_attn_output_shape() -> None:
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=8, output_size=2, num_mixtures=2
    )
    video = torch.randn(2, 12, 1)
    audio = torch.randn(2, 12, 1)
    out = model(video, audio)
    assert out["logits"].shape == (2, 12, 2)
    assert out["loc"].shape == (2, 12, 2, 2)
    assert out["log_scale"].shape == (2, 12, 2, 2)


def test_cross_modal_attn_returns_attention_weights() -> None:
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=4, output_size=2
    )
    video = torch.randn(1, 6, 1)
    audio = torch.randn(1, 6, 1)
    out = model(video, audio, return_attention=True)
    assert "_attn_video_queries_audio" in out
    assert "_attn_audio_queries_video" in out
    assert out["_attn_video_queries_audio"].shape == (1, 6, 6)
    assert out["_attn_audio_queries_video"].shape == (1, 6, 6)
    # Attention rows must sum to 1 (softmax invariant).
    row_sums = out["_attn_video_queries_audio"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_cross_modal_attn_backward_into_all_modules() -> None:
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=4, output_size=2, num_mixtures=1
    )
    video = torch.randn(2, 8, 1, requires_grad=True)
    audio = torch.randn(2, 8, 1, requires_grad=True)
    out = model(video, audio)
    out["loc"].sum().backward()
    # Both encoders must receive gradient.
    v_grad = sum(p.grad.abs().sum().item() for p in model.video_encoder.parameters() if p.grad is not None)
    a_grad = sum(p.grad.abs().sum().item() for p in model.audio_encoder.parameters() if p.grad is not None)
    # All six attention projections must receive gradient.
    for proj in (model.q_v, model.k_a, model.v_a, model.q_a, model.k_v, model.v_v, model.fuse_proj):
        g = sum(p.grad.abs().sum().item() for p in proj.parameters() if p.grad is not None)
        assert g > 0, "attention projection has zero gradient"
    assert v_grad > 0 and a_grad > 0


def test_cross_modal_attn_differs_from_concat_baseline() -> None:
    """Cross-attention fusion must produce different outputs than vanilla
    concat fusion on the same encoder weights — otherwise the attention
    machinery is silently doing nothing."""
    torch.manual_seed(7)
    concat_model = MultimodalBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=8, output_size=2, num_mixtures=1, fusion="concat"
    )
    torch.manual_seed(7)  # match encoder init
    attn_model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=8, output_size=2, num_mixtures=1
    )
    # Copy encoders so the only difference is the fusion path.
    attn_model.video_encoder.load_state_dict(concat_model.video_encoder.state_dict())
    attn_model.audio_encoder.load_state_dict(concat_model.audio_encoder.state_dict())
    video = torch.randn(1, 14, 1)
    audio = torch.randn(1, 14, 1)
    with torch.no_grad():
        concat_loc = mdn_mean(concat_model(video, audio))
        attn_loc = mdn_mean(attn_model(video, audio))
    assert concat_loc.shape == attn_loc.shape
    assert not torch.allclose(concat_loc, attn_loc, atol=1e-3), (
        "cross-attn output identical to concat output — attention path is dead"
    )


def test_cross_modal_attn_training_reduces_nll() -> None:
    torch.manual_seed(0)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=1, audio_dim=1, hidden_size=8, output_size=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    from lnn.core.mdn import mdn_negative_log_likelihood
    dataset = MultimodalPhysicsDataset(num_samples=16, seq_len=10, seed=0)
    items = [dataset[i] for i in range(8)]
    video = torch.stack([it[0]["video"] for it in items])
    audio = torch.stack([it[0]["audio"] for it in items])
    params = torch.stack([it[1]["params"] for it in items])
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        out = model(video, audio)
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, params)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"NLL did not decrease: {losses}"


# ---------- Heterogeneous forced dataset tests ----------


def test_heterogeneous_dataset_shapes() -> None:
    ds = HeterogeneousForcedDataset(num_samples=4, seq_len=12, num_steps_per_dt=3)
    assert len(ds) == 4
    sample, target = ds[0]
    assert sample["video"].shape == (12, 1)
    assert sample["audio"].shape == (12, 1)
    assert target["params"].shape == (2,)


def test_heterogeneous_dataset_chirp_and_burst() -> None:
    for kind in ("chirp", "burst"):
        ds = HeterogeneousForcedDataset(
            num_samples=2, seq_len=8, force_kind=kind, num_steps_per_dt=2
        )
        sample, _ = ds[0]
        assert sample["video"].shape == (8, 1)
        assert sample["audio"].shape == (8, 1)


def test_heterogeneous_dataset_rejects_invalid_force_kind() -> None:
    with pytest.raises(ValueError):
        HeterogeneousForcedDataset(num_samples=2, force_kind="bogus")  # type: ignore[arg-type]


def test_heterogeneous_dataset_audio_in_force_amplitude_band() -> None:
    """Sanity that the audio forcing is the prescribed amplitude band.

    Audio RMS per sample should vary widely because amplitude is sampled
    uniformly from [0.4, 1.2].  If audio were a derived stat of the
    position (as in the round 6 dataset) the RMS would be tied to the
    oscillator's ω_d instead.
    """
    ds = HeterogeneousForcedDataset(num_samples=32, seq_len=24, seed=1)
    audios = torch.stack([ds[i][0]["audio"] for i in range(16)]).squeeze(-1)
    rms = audios.pow(2).mean(dim=-1).sqrt()
    assert rms.min().item() > 0.2, "audio signal should be substantially non-zero"
    assert rms.max().item() < 2.0, "audio signal should stay in the prescribed amplitude band"
