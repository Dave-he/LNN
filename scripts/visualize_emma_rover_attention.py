#!/usr/bin/env python3
"""Cross-attention visualization on the EMMA rover real data.

Trains a ``CrossModalAttnBiCfCNADWithMDN`` briefly on the EMMA rover
sliding-window dataset, then visualises the per-step attention weights
on a single held-out window.  The goal is to *qualitatively* confirm
the EMMA-style hypothesis:

    "The audio modality carries hidden actuation information
    (motor RPM) that the video modality cannot see — so at every
    query step the cross-attention should put non-trivial weight on
    the audio side, especially at physical 'motor-on / wheel-startup'
    transitions."

Outputs
-------
* ASCII heatmaps of the two attention matrices
  (video-queries-audio, audio-queries-video)
* Per-step argmax of each attention matrix (the single most-attended
  partner time step)
* Entropy of each row (low entropy = concentrated attention;
  high entropy = spread across the partner modality)

The text heatmap is a pragmatic choice for a no-matplotlib CI
environment; the per-step numbers carry the actual finding.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.mdn import mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EMMA_ROVER_GROUND_TRUTH,
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _text_heatmap(matrix: torch.Tensor, name: str) -> str:
    """Render a ``[T_q, T_k]`` attention matrix as an ASCII heatmap.

    The heatmap character set goes from ' ' (zero weight) through
    '.:+*#@' to '#' (top weight) so even at 60-step widths the
    structure is visible.
    """
    chars = " .:+*#@"
    T_q, T_k = matrix.shape
    weights = matrix.cpu().numpy()
    # Normalise each row so the heatmap colours show within-row mass.
    row_max = weights.max(axis=-1, keepdims=True)
    row_max = row_max.clip(min=1e-9)
    normed = weights / row_max
    lines = [f"--- {name} (shape {list(matrix.shape)}) ---"]
    header = "  q\\k  " + " ".join(f"{k:>3d}" for k in range(T_k))
    lines.append(header)
    for q in range(T_q):
        cells = []
        for k in range(T_k):
            v = float(normed[q, k])
            idx = min(len(chars) - 1, int(v * (len(chars) - 1)))
            cells.append(chars[idx])
        lines.append(f"  q={q:>3d}  " + " ".join(cells))
    return "\n".join(lines)


def _per_step_stats(matrix: torch.Tensor) -> dict[str, list[float]]:
    """Argmax and entropy per query row."""
    import numpy as np
    weights = matrix.cpu().numpy()
    argmax = weights.argmax(axis=-1).tolist()
    eps = 1e-9
    # Normalised entropy: 0 = one-hot (degenerate), 1 = uniform.
    T_k = weights.shape[-1]
    log_T_k = float(np.log(T_k))
    entropies = (-(weights * np.log(weights + eps)).sum(axis=-1) / log_T_k).tolist()
    return {"argmax": argmax, "entropy": entropies}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-attention visualization on EMMA rover")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--window", type=int, default=60, help="use full 60-frame trajectory for viz")
    parser.add_argument("--feature-noise-std", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="analysis/emma_rover/attention_viz.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples,
        window=min(args.window, 60),
        feature_noise_std=args.feature_noise_std,
        seed=args.seed,
    )
    train_loader, _, _ = create_emma_rover_dataloaders(dataset, batch_size=32, seed=args.seed)

    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1,
        hidden_size=args.hidden_size, output_size=5, num_mixtures=args.num_mixtures,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for _ in range(1, args.epochs + 1):
        model.train()
        for batch, target in train_loader:
            optimizer.zero_grad()
            out = model(batch["video"], batch["audio"])
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Now do one forward on a single held-out window (the full trajectory)
    # to extract attention matrices.
    model.eval()
    with torch.no_grad():
        sample, _ = dataset[0]
        video = sample["video"].unsqueeze(0)
        audio = sample["audio"].unsqueeze(0)
        out = model(video, audio, return_attention=True)
    attn_va = out["_attn_video_queries_audio"][0]  # [T, T]
    attn_av = out["_attn_audio_queries_video"][0]  # [T, T]

    print(_text_heatmap(attn_va, "video-queries-audio (each row: video step q's attention over audio steps)"))
    print()
    print(_text_heatmap(attn_av, "audio-queries-video (each row: audio step q's attention over video steps)"))

    stats_va = _per_step_stats(attn_va)
    stats_av = _per_step_stats(attn_av)
    print()
    print("--- Per-step statistics ---")
    print("video-queries-audio: argmax =", stats_va["argmax"])
    print("                       entropy =", [f"{e:.2f}" for e in stats_va["entropy"]])
    print("audio-queries-video: argmax =", stats_av["argmax"])
    print("                       entropy =", [f"{e:.2f}" for e in stats_av["entropy"]])

    # Interpretation: if the model is using cross-modal info, argmax should
    # be spread (not always the same step).  High entropy = "paying a bit
    # of attention to everywhere" — the cross-modal signal is being used.
    # Low entropy + same argmax repeatedly = degenerate / collapsed attention.
    mean_entropy_va = sum(stats_va["entropy"]) / len(stats_va["entropy"])
    mean_entropy_av = sum(stats_av["entropy"]) / len(stats_av["entropy"])
    max_entropy = float(torch.log(torch.tensor(attn_va.shape[-1], dtype=torch.float32)).item())
    print()
    print(
        f"video-queries-audio: mean row entropy {mean_entropy_va:.2f} / max {max_entropy:.2f} "
        f"= {mean_entropy_va / max_entropy * 100:.1f}% of uniform"
    )
    print(
        f"audio-queries-video: mean row entropy {mean_entropy_av:.2f} / max {max_entropy:.2f} "
        f"= {mean_entropy_av / max_entropy * 100:.1f}% of uniform"
    )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "video_queries_audio": {
                    "argmax": stats_va["argmax"],
                    "entropy": stats_va["entropy"],
                    "mean_entropy": mean_entropy_va,
                    "max_entropy": max_entropy,
                },
                "audio_queries_video": {
                    "argmax": stats_av["argmax"],
                    "entropy": stats_av["entropy"],
                    "mean_entropy": mean_entropy_av,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
