#!/usr/bin/env python3
"""Run a synthetic long-sequence Liquid-S4/LiquidTAD-style experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.long_sequence import (
    HierarchicalDecayLiquidTADHead,
    LiquidTADHead,
    LongSequenceLiquidClassifier,
)
from lnn.data.long_sequence import SyntheticLongSequenceDataset, create_long_sequence_dataloaders


ROOT = pathlib.Path(__file__).resolve().parents[1]


def move_targets(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in target.items()}


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for features, target in dataloader:
        features = features.to(device)
        target = move_targets(target, device)
        optimizer.zero_grad(set_to_none=True)
        if mode == "classification":
            logits = model(features, mask=target["mask"])
            loss = F.cross_entropy(logits, target["label"])
        else:
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
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mode: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0
    batches = 0
    for features, target in dataloader:
        features = features.to(device)
        target = move_targets(target, device)
        if mode == "classification":
            logits = model(features, mask=target["mask"])
            loss = F.cross_entropy(logits, target["label"])
            total_correct += (logits.argmax(dim=-1) == target["label"]).float().sum().item()
            total_count += target["label"].numel()
        else:
            output = model(features, mask=target["mask"])
            loss = F.cross_entropy(
                output["frame_logits"].reshape(-1, output["frame_logits"].shape[-1]),
                target["frame_labels"].reshape(-1),
            )
            total_correct += (
                output["frame_logits"].argmax(dim=-1) == target["frame_labels"]
            ).float().sum().item()
            total_count += target["frame_labels"].numel()
        total_loss += loss.item()
        batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": total_correct / max(total_count, 1.0),
    }


def write_report(payload: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}_long_sequence.json"
    md_path = output_dir / f"{run_id}_long_sequence.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config = payload["config"]
    test = payload["test"]
    lines = [
        "---",
        f"title: Liquid-S4/LiquidTAD 长序列运行记录 - {run_id}",
        f"date: {payload['run_date']}",
        "tags: [LNN, Liquid-S4, LiquidTAD, long-sequence]",
        "---",
        "",
        f"# Liquid-S4/LiquidTAD 长序列运行记录 - {run_id}",
        "",
        "## 任务",
        "- 数据：合成长序列 action segment，支持序列分类和 frame-level TAD smoke test。",
        "- 模型：并行 liquid relaxation + depthwise temporal mixing。",
        f"- 模式：{payload['mode']}",
        f"- 设备：{payload['device']}",
        "",
        "## 配置",
        f"- 样本数 / SeqLen：{config['samples']} / {config['seq_len']}",
        f"- Feature dim / Classes：{config['feature_size']} / {config['num_classes']}",
        f"- Hidden / Blocks：{config['hidden_size']} / {config['num_blocks']}",
        f"- Epoch / Batch：{config['epochs']} / {config['batch_size']}",
        "",
        "## 结果",
        f"- Test loss：{test['loss']:.6f}",
        f"- Test accuracy：{test['accuracy']:.4f}",
        f"- 参数量：{payload['parameters']}",
        f"- 训练耗时：{payload['elapsed_seconds']:.2f}s",
        "",
        "## 产物",
        f"- JSON：`{json_path.relative_to(ROOT)}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["classification", "tad"], default="classification")
    parser.add_argument(
        "--tad-head",
        choices=["data_dependent", "hierarchical_decay"],
        default="data_dependent",
        help=(
            "data_dependent = LiquidTADHead (per-step retain gate, default); "
            "hierarchical_decay = HierarchicalDecayLiquidTADHead (LiquidTAD paper, "
            "layer-shared exponential decay schedule)."
        ),
    )
    parser.add_argument(
        "--tad-init-decay",
        type=float,
        default=0.80,
        help="Initial decay coefficient for the first hierarchical_decay block.",
    )
    parser.add_argument(
        "--tad-decay-growth",
        type=float,
        default=1.05,
        help="Geometric growth factor for per-block decay (deeper layers integrate longer).",
    )
    parser.add_argument(
        "--tad-share-decay",
        action="store_true",
        help="Tie the retain parameter across all hierarchical_decay blocks.",
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--feature-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/long_sequence")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
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
    if args.mode == "classification":
        model = LongSequenceLiquidClassifier(
            input_size=args.feature_size,
            num_classes=args.num_classes,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
        ).to(device)
    elif args.tad_head == "hierarchical_decay":
        model = HierarchicalDecayLiquidTADHead(
            input_size=args.feature_size,
            num_classes=args.num_classes + 1,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            init_decay=args.tad_init_decay,
            decay_growth=args.tad_decay_growth,
            share_decay=args.tad_share_decay,
        ).to(device)
    else:
        model = LiquidTADHead(
            input_size=args.feature_size,
            num_classes=args.num_classes + 1,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("=== Liquid Long Sequence Experiment ===")
    print(f"Mode: {args.mode} | Device: {device} | Samples: {args.samples} | SeqLen: {args.seq_len}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.mode)
        val = evaluate(model, val_loader, device, args.mode)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | Train {train_loss:.4f} | "
                f"Val loss {val['loss']:.4f} | Val acc {val['accuracy']:.3f}"
            )

    elapsed_seconds = time.perf_counter() - start
    test = evaluate(model, test_loader, device, args.mode)
    print(f"Test loss: {test['loss']:.6f} | Test accuracy: {test['accuracy']:.4f}")

    now = dt.datetime.now()
    payload = {
        "run_id": now.strftime("%Y-%m-%d_%H%M%S"),
        "run_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "mode": args.mode,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed_seconds,
        "config": vars(args),
        "history": history,
        "test": test,
    }
    json_path, md_path = write_report(payload, ROOT / args.output_dir)
    print(f"JSON written: {json_path.relative_to(ROOT)}")
    print(f"Report written: {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
