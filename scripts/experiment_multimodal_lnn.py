#!/usr/bin/env python3
"""
Run a local multimodal LNN classification experiment.

The experiment uses synthetic data with three modalities:
    - sensor sequence
    - small image pattern
    - short token sequence

Usage:
    python scripts/experiment_multimodal_lnn.py --model cfc --epochs 12
    python scripts/experiment_multimodal_lnn.py --model ltc --epochs 5 --samples 300
"""

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
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.multimodal import MultimodalFusionLNN
from lnn.data.multimodal import SyntheticMultimodalDataset, create_multimodal_dataloaders
from lnn.utils.visualization import plot_training_curve


ROOT = pathlib.Path(__file__).resolve().parents[1]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def confusion_matrix(targets: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> list[list[int]]:
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1), strict=False):
        matrix[int(target), int(prediction)] += 1
    return matrix.tolist()


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, labels in dataloader:
        batch = move_batch(batch, device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    all_targets = []
    all_predictions = []

    for batch, labels in dataloader:
        batch = move_batch(batch, device)
        labels = labels.to(device)
        logits = model(batch)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=-1)
        total_loss += loss.item()
        batches += 1
        all_targets.append(labels.cpu())
        all_predictions.append(predictions.cpu())

    targets = torch.cat(all_targets, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    accuracy = (targets == predictions).float().mean().item()
    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(targets, predictions, num_classes),
    }


def write_report(payload: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}_multimodal_lnn.json"
    md_path = output_dir / f"{run_id}_multimodal_lnn.md"
    curve_path = output_dir / f"{run_id}_training_curve.png"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_training_curve(
        payload["history"]["train_loss"],
        payload["history"]["val_loss"],
        title=f"{payload['model'].upper()} Multimodal LNN Training",
        save_path=str(curve_path),
    )

    config = payload["config"]
    test = payload["test"]
    lines = [
        "---",
        f"title: 本机多模态 LNN 运行记录 - {run_id}",
        f"date: {payload['run_date']}",
        "tags: [LNN, multimodal, local-run, experiment]",
        "---",
        "",
        f"# 本机多模态 LNN 运行记录 - {run_id}",
        "",
        "## 任务",
        "- 模态：传感器序列 + 图像模式 + 文本 token",
        "- 目标：三分类，多模态编码后输入 CfC/LTC 风格 LNN",
        f"- 模型：{payload['model']}",
        f"- 设备：{payload['device']}",
        "",
        "## 配置",
        f"- 样本数：{config['samples']}",
        f"- 序列长度 / 传感器维度：{config['seq_len']} / {config['sensor_dim']}",
        f"- 图像尺寸 / 文本长度：{config['image_size']} / {config['text_len']}",
        f"- 融合维度 / 隐藏维度：{config['fusion_size']} / {config['hidden_size']}",
        f"- Epoch / Batch：{config['epochs']} / {config['batch_size']}",
        "",
        "## 结果",
        f"- 测试 Loss：{test['loss']:.6f}",
        f"- 测试 Accuracy：{test['accuracy']:.4f}",
        f"- 训练耗时：{payload['elapsed_seconds']:.2f}s",
        f"- 参数量：{payload['parameters']}",
        "",
        "## 混淆矩阵",
        "| true \\ pred | class 0 | class 1 | class 2 |",
        "|---|---:|---:|---:|",
    ]
    for index, row in enumerate(test["confusion_matrix"]):
        lines.append(f"| class {index} | " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## 产物",
            f"- JSON：`{json_path.relative_to(ROOT)}`",
            f"- 训练曲线：`{curve_path.relative_to(ROOT)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, curve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multimodal LNN experiment")
    parser.add_argument("--model", choices=["cfc", "ltc"], default="cfc")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--sensor_dim", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=16)
    parser.add_argument("--text_len", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=48)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--fusion_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a torch device string")
    parser.add_argument("--output_dir", default="analysis/multimodal")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    dataset = SyntheticMultimodalDataset(
        num_samples=args.samples,
        seq_len=args.seq_len,
        sensor_dim=args.sensor_dim,
        image_size=args.image_size,
        text_len=args.text_len,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_multimodal_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = MultimodalFusionLNN(
        sensor_dim=args.sensor_dim,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        fusion_size=args.fusion_size,
        hidden_size=args.hidden_size,
        recurrent_type=args.model,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=== Local Multimodal LNN Experiment ===")
    print(f"Model: {args.model} | Device: {device} | Samples: {args.samples}")
    print(
        f"Modalities: sensor({args.seq_len}x{args.sensor_dim}) + "
        f"image({args.image_size}x{args.image_size}) + text({args.text_len})"
    )
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | Train loss {train_loss:.4f} | "
                f"Val loss {val_metrics['loss']:.4f} | Val acc {val_metrics['accuracy']:.3f}"
            )

    elapsed_seconds = time.perf_counter() - start
    test_metrics = evaluate(model, test_loader, criterion, device, args.num_classes)
    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Confusion matrix: {test_metrics['confusion_matrix']}")

    now = dt.datetime.now()
    payload = {
        "run_id": now.strftime("%Y-%m-%d_%H%M%S"),
        "run_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "model": args.model,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed_seconds,
        "config": vars(args),
        "history": history,
        "test": test_metrics,
    }
    json_path, md_path, curve_path = write_report(payload, ROOT / args.output_dir)
    print(f"\nResults saved:")
    print(f"  {json_path.relative_to(ROOT)}")
    print(f"  {md_path.relative_to(ROOT)}")
    print(f"  {curve_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
