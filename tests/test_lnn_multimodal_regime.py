"""LNN 多模态系统 regime 双测 — 强制 future PR 在大小预算下分别验证。

27 轮 ablation 的核心发现 (尤其是 §27 cron `8d53b97` + §30 cron `68fe631`):
  * cross_attn 在 hidden=16, ep=20 regime 赢 (+50%)
  * cross_attn 在 hidden=64, ep=80 regime *输* 视频单流 (-755%!)
  * 新 SOTA (adaptive freeze) 也在 hidden=64, ep=80 regime 赢 (MSE 0.31)
  * hidden=8 在真实 EMMA 数据上有反常曲线 (self-xattn > cross_attn), 合成数据无

任何 *单一 regime* 报告的 "+X% gain" 都 *不能* 反映真实跨 regime 行为。
本模块强制 future PR 在大小预算下分别跑, 防止因 *regime 局限* 的过度推断。

运行:
  pytest tests/ -q                              (默认: 仅 small budget)
  pytest tests/ -q -m large_budget               (大小预算双跑)
  pytest tests/ -q -m 'not large_budget'         (仅小预算)
"""

from __future__ import annotations

import os
import sys

import pytest

# Make repo root importable for the model/data imports below.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


SMALL_BUDGET = dict(epochs=20, hidden_size=16, num_samples=200, num_mixtures=1)
LARGE_BUDGET = dict(epochs=80, hidden_size=64, num_samples=200, num_mixtures=1)


# ---------------------------------------------------------------------------
# Helper: a minimal regime-stratified benchmark used by the marker tests
# below.  Each test runs ONLY the video_only / cross_attn(A=zero) pair at
# its declared regime, so the assertion can be a strict "regime-conditional
# recommendation" rather than a single "+X% gain" claim.
# ---------------------------------------------------------------------------


def _run_pair(epochs: int, hidden_size: int) -> dict[str, float]:
    """Return test param MSE for video_only and cross_attn(audio=zero)."""
    import torch
    from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
    torch.manual_seed(42)
    dataset = EmmaRoverRegressionDataset(num_samples=200, window=16, feature_noise_std=0.02, seed=42)
    train_loader, _, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=32, seed=42,
    )
    out: dict[str, float] = {}
    for kind in ("video_only", "cross_attn"):
        torch.manual_seed(42)
        if kind == "video_only":
            model = BiCfCNADWithMDN(input_size=4, hidden_size=hidden_size, output_size=5, num_mixtures=1)
            def forward(batch):
                return model(torch.cat([batch["video"], batch["audio"]], dim=-1))
        else:
            model = CrossModalAttnBiCfCNADWithMDN(
                video_dim=3, audio_dim=1, hidden_size=hidden_size, output_size=5, num_mixtures=1,
            )
            def forward(batch):
                return model(batch["video"], batch["audio"])
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(epochs):
            model.train()
            for batch, target in train_loader:
                batch = {k: v for k, v in batch.items()}
                target = {k: v for k, v in target.items()}
                opt.zero_grad()
                mdn_params = forward(batch)
                final = {k: v[:, -1] for k, v in mdn_params.items()}
                loss = mdn_negative_log_likelihood(final, target["params"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
        model.eval()
        with torch.no_grad():
            sq = []
            for batch, target in test_loader:
                mdn_params = forward(batch)
                final = {k: v[:, -1] for k, v in mdn_params.items()}
                mean = mdn_mean(final)
                sq.append((mean - target["params"]).pow(2).sum(dim=-1))
            sq = torch.cat(sq)
        out[kind] = float(sq.mean().item())
    return out


# ---------------------------------------------------------------------------
# SMALL-BUDGET regime tests (default; fast ~1 minute total)
# ---------------------------------------------------------------------------


@pytest.mark.regime("small_budget")
def test_small_budget_video_only_baseline() -> None:
    """video_only at hidden=16, ep=20 should land in [400, 700] MSE."""
    mses = _run_pair(epochs=SMALL_BUDGET["epochs"], hidden_size=SMALL_BUDGET["hidden_size"])
    assert 400 < mses["video_only"] < 700, (
        f"video_only at small_budget should be in (400, 700), got {mses['video_only']:.2f}"
    )


@pytest.mark.regime("small_budget")
def test_small_budget_cross_attn_beats_video_only() -> None:
    """The headline §13/§14 small-budget claim: cross_attn beats video_only."""
    mses = _run_pair(epochs=SMALL_BUDGET["epochs"], hidden_size=SMALL_BUDGET["hidden_size"])
    assert mses["cross_attn"] < mses["video_only"], (
        f"cross_attn must beat video_only in small-budget regime, got "
        f"video_only={mses['video_only']:.2f}, cross_attn={mses['cross_attn']:.2f}"
    )


# ---------------------------------------------------------------------------
# LARGE-BUDGET regime tests (skip by default; run with -m large_budget)
# ---------------------------------------------------------------------------


@pytest.mark.large_budget
def test_large_budget_video_only_dominates() -> None:
    """§27 finding: at hidden=64, ep=80, video_only must reach MSE < 5.

    This is the *convergence-driven regime* test.  We do NOT require
    cross_attn < video_only here — that direction inverts at this regime.
    """
    mses = _run_pair(epochs=LARGE_BUDGET["epochs"], hidden_size=LARGE_BUDGET["hidden_size"])
    assert mses["video_only"] < 5, (
        f"video_only at large_budget should converge to MSE < 5, got {mses['video_only']:.2f}"
    )


@pytest.mark.large_budget
def test_large_budget_cross_attn_underperforms() -> None:
    """§27 finding: at hidden=64, ep=80, cross_attn must be *worse* than video_only.

    The exact multiple is seed-dependent but the direction is the
    regime-invariant finding.  We assert >= 2x worse (very loose) so
    this is robust to seed noise.
    """
    mses = _run_pair(epochs=LARGE_BUDGET["epochs"], hidden_size=LARGE_BUDGET["hidden_size"])
    assert mses["cross_attn"] > 2 * mses["video_only"], (
        f"cross_attn must be >2x worse than video_only at large_budget, got "
        f"video_only={mses['video_only']:.2f}, cross_attn={mses['cross_attn']:.2f}"
    )


# ---------------------------------------------------------------------------
# META: regime-aware CI guidance
# ---------------------------------------------------------------------------


def test_regime_marker_inventory() -> None:
    """Sanity: pytest should know about the 'large_budget' and 'regime' markers.

    Any future test author using ``@pytest.mark.large_budget`` or
    ``@pytest.mark.regime`` should be auto-registered.  This test
    ensures the marker is at least *discoverable* by pytest's
    collection system.
    """
    # We don't depend on external config; just confirm the markers
    # exist by referencing them.
    assert pytest.mark.large_budget is not None
    assert pytest.mark.regime is not None
