#!/usr/bin/env python3
"""Three-way ablation runner for LiquidTAD-style detection heads.

Pits these heads on the same SyntheticLongSequenceDataset, same seed and
training budget:

* ``data_dependent``        : LiquidTADHead (per-step retain gate predicted
                              from the hidden state).
* ``hierarchical_decay``    : HierarchicalDecayLiquidTADHead with one shared
                              retain coefficient per channel per block; init
                              decays grow geometrically with depth.
* ``hierarchical_shared``   : same as hierarchical_decay but the retain
                              parameter is tied across all blocks.

Designed to be a quick, repeatable check that the structural decay sharing
prior from LiquidTAD (arXiv:2604.18274) trades parameters for accuracy in a
predictable way.  Output: a JSON record per run + a single Markdown summary
table you can drop into ``analysis/long_sequence/``.

Usage::

    python scripts/ablation_liquid_tad_heads.py --samples 192 --seq-len 96 \
        --hidden-size 32 --num-blocks 3 --epochs 8

PRD §8 task #2 stage C-lite (the full Stage C is THUMOS-14 subset; this is the
synthetic-data warm-up to set comparison plumbing in place).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.long_sequence import (
    HierarchicalDecayLiquidTADHead,
    LiquidTADHead,
)
from lnn.data.long_sequence import (
    SyntheticLongSequenceDataset,
    create_long_sequence_dataloaders,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _build_head(name: str, args: argparse.Namespace) -> torch.nn.Module:
    if name == "data_dependent":
        return LiquidTADHead(
            input_size=args.feature_size,
            num_classes=args.num_classes + 1,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
        )
    if name == "hierarchical_decay":
        return HierarchicalDecayLiquidTADHead(
            input_size=args.feature_size,
            num_classes=args.num_classes + 1,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            init_decay=args.init_decay,
            decay_growth=args.decay_growth,
            share_decay=False,
        )
    if name == "hierarchical_shared":
        return HierarchicalDecayLiquidTADHead(
            input_size=args.feature_size,
            num_classes=args.num_classes + 1,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            init_decay=args.init_decay,
            decay_growth=args.decay_growth,
            share_decay=True,
        )
    raise ValueError(f"unknown head: {name}")


def _move_targets(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in target.items()}


def _train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    batches = 0
    for features, target in loader:
        features = features.to(device)
        target = _move_targets(target, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(features, mask=target["mask"])
        class_loss = F.cross_entropy(
            output["frame_logits"].reshape(-1, output["frame_logits"].shape[-1]),
            target["frame_labels"].reshape(-1),
        )
        boundary_loss = F.binary_cross_entropy(output["boundaries"], target["boundaries"])
        loss = class_loss + 0.5 * boundary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def _evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    frame_correct = 0.0
    frame_count = 0.0
    boundary_mae = 0.0
    boundary_count = 0.0
    batches = 0
    for features, target in loader:
        features = features.to(device)
        target = _move_targets(target, device)
        output = model(features, mask=target["mask"])
        class_loss = F.cross_entropy(
            output["frame_logits"].reshape(-1, output["frame_logits"].shape[-1]),
            target["frame_labels"].reshape(-1),
        )
        boundary_loss = F.binary_cross_entropy(output["boundaries"], target["boundaries"])
        loss_sum += (class_loss + 0.5 * boundary_loss).item()
        preds = output["frame_logits"].argmax(dim=-1)
        frame_correct += float((preds == target["frame_labels"]).sum().item())
        frame_count += float(target["frame_labels"].numel())
        boundary_mae += float((output["boundaries"] - target["boundaries"]).abs().mean().item())
        boundary_count += 1.0
        batches += 1
    return {
        "loss": loss_sum / max(batches, 1),
        "frame_accuracy": frame_correct / max(frame_count, 1.0),
        "boundary_mae": boundary_mae / max(boundary_count, 1.0),
    }


def _run_one(name: str, args: argparse.Namespace, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    dataset = SyntheticLongSequenceDataset(
        num_samples=args.samples,
        seq_len=args.seq_len,
        feature_size=args.feature_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_long_sequence_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = _build_head(name, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parameters = sum(p.numel() for p in model.parameters())

    start = time.perf_counter()
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val = _evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val["loss"])
    elapsed = time.perf_counter() - start
    test = _evaluate(model, test_loader, device)
    return {
        "name": name,
        "parameters": parameters,
        "train_loss_final": train_losses[-1],
        "val_loss_final": val_losses[-1],
        "test_loss": test["loss"],
        "test_frame_accuracy": test["frame_accuracy"],
        "test_boundary_mae": test["boundary_mae"],
        "elapsed_seconds": elapsed,
    }


def _format_markdown(payload: dict) -> str:
    cfg = payload["config"]
    env = payload["environment"]
    lines = [
        f"# LiquidTAD 3-way Head Ablation — {payload['run_id']}",
        "",
        "## 环境",
        f"- device: {env['device']}",
        f"- torch: {env['torch_version']} (cuda available={env['cuda_available']})",
        f"- python: {env['python']}",
        "",
        "## 配置",
        f"- samples / seq_len / feature_size: {cfg['samples']} / {cfg['seq_len']} / {cfg['feature_size']}",
        f"- num_classes (foreground) / num_blocks / hidden_size: {cfg['num_classes']} / {cfg['num_blocks']} / {cfg['hidden_size']}",
        f"- epochs / batch_size / lr / seed: {cfg['epochs']} / {cfg['batch_size']} / {cfg['lr']} / {cfg['seed']}",
        f"- decay init / growth (for hierarchical heads): {cfg['init_decay']} / {cfg['decay_growth']}",
        "",
        "## 结果",
        "| Head | 参数量 | Test loss | Test frame acc | Test boundary MAE | 训练秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in payload["results"]:
        lines.append(
            f"| `{r['name']}` | {r['parameters']:,} | {r['test_loss']:.4f} | "
            f"{r['test_frame_accuracy']*100:.2f}% | {r['test_boundary_mae']:.4f} | {r['elapsed_seconds']:.2f} |"
        )

    baseline = next(r for r in payload["results"] if r["name"] == "data_dependent")
    lines.extend([
        "",
        "## 相对 baseline (`data_dependent`) 的变化",
        "| Head | Δparams | Δtest_loss | Δframe_acc (pp) |",
        "|---|---:|---:|---:|",
    ])
    for r in payload["results"]:
        if r["name"] == "data_dependent":
            continue
        d_params = (r["parameters"] - baseline["parameters"]) / baseline["parameters"] * 100.0
        d_loss = (r["test_loss"] - baseline["test_loss"]) / baseline["test_loss"] * 100.0
        d_acc = (r["test_frame_accuracy"] - baseline["test_frame_accuracy"]) * 100.0
        lines.append(
            f"| `{r['name']}` | {d_params:+.2f}% | {d_loss:+.2f}% | {d_acc:+.2f}pp |"
        )

    lines.extend([
        "",
        "## 解读模板",
        "- params 减少而 acc 不显著掉 → hierarchical prior 在该规模下生效;",
        "- params 减少且 acc 明显掉 → 容量上限,推大 hidden_size/epochs 再看;",
        "- params 减少且 acc 反而涨 → 论文 sharing prior 直接验证,记录为强证据。",
        "",
        f"产出 JSON: `{payload['json_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--feature-size", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-decay", type=float, default=0.80)
    parser.add_argument("--decay-growth", type=float, default=1.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/long_sequence")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    print("=== LiquidTAD 3-way Head Ablation ===")
    print(f"Device: {device} | Samples: {args.samples} | SeqLen: {args.seq_len} | Hidden: {args.hidden_size}")
    print(f"Epochs: {args.epochs} | Blocks: {args.num_blocks} | Seed: {args.seed}")

    results: list[dict] = []
    for head_name in ("data_dependent", "hierarchical_decay", "hierarchical_shared"):
        print(f"\n--- Training head: {head_name}")
        record = _run_one(head_name, args, device)
        print(
            f"    params={record['parameters']:,}  test_loss={record['test_loss']:.4f}"
            f"  acc={record['test_frame_accuracy']*100:.2f}%  elapsed={record['elapsed_seconds']:.2f}s"
        )
        results.append(record)

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_liquid_tad_head_ablation.json"
    md_path = output_dir / f"{run_id}_liquid_tad_head_ablation.md"

    rel_json = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path

    payload = {
        "run_id": run_id,
        "experiment": "liquid_tad_head_ablation",
        "generated_at": now.isoformat(),
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "python": sys.version.split()[0],
        },
        "config": vars(args),
        "results": results,
        "json_path": str(rel_json),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(payload), encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
