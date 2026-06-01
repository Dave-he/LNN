#!/usr/bin/env python3
"""Run a synthetic dynamic-graph GNN + LNN prediction experiment."""

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

from lnn.core.graph import GraphLNNPredictor
from lnn.data.graph_timeseries import SyntheticGraphTimeSeriesDataset, create_graph_dataloaders


ROOT = pathlib.Path(__file__).resolve().parents[1]


def move_graph_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def train_epoch(
    model: GraphLNNPredictor,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = move_graph_batch(batch, device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: GraphLNNPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = move_graph_batch(batch, device)
        target = target.to(device)
        prediction = model(batch)
        total_mse += F.mse_loss(prediction, target).item()
        total_mae += F.l1_loss(prediction, target).item()
        batches += 1
    return {
        "mse": total_mse / max(batches, 1),
        "mae": total_mae / max(batches, 1),
    }


def write_report(payload: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}_graph_lnn.json"
    md_path = output_dir / f"{run_id}_graph_lnn.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config = payload["config"]
    test = payload["test"]
    lines = [
        "---",
        f"title: GNN+LNN 动态图运行记录 - {run_id}",
        f"date: {payload['run_date']}",
        "tags: [LNN, graph, spatio-temporal, experiment]",
        "---",
        "",
        f"# GNN+LNN 动态图运行记录 - {run_id}",
        "",
        "## 任务",
        "- 数据：合成动态图扩散序列，含动态边和 per-step dt。",
        "- 目标：用 GNN snapshot encoder + LNN temporal core 预测下一步图级负载。",
        f"- 模型：Graph encoder + {payload['recurrent_type']}",
        f"- 设备：{payload['device']}",
        "",
        "## 配置",
        f"- 样本数 / SeqLen：{config['samples']} / {config['seq_len']}",
        f"- 节点数 / 节点特征：{config['num_nodes']} / {config['node_feature_size']}",
        f"- Graph dim / Hidden：{config['graph_feature_size']} / {config['hidden_size']}",
        f"- Epoch / Batch：{config['epochs']} / {config['batch_size']}",
        "",
        "## 结果",
        f"- Test MSE：{test['mse']:.6f}",
        f"- Test MAE：{test['mae']:.6f}",
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
    parser.add_argument("--recurrent", choices=["cfc", "ltc", "gru"], default="cfc")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--node-feature-size", type=int, default=3)
    parser.add_argument("--graph-feature-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/graph")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dataset = SyntheticGraphTimeSeriesDataset(
        num_samples=args.samples,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        node_feature_size=args.node_feature_size,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_graph_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = GraphLNNPredictor(
        node_feature_size=args.node_feature_size,
        graph_feature_size=args.graph_feature_size,
        hidden_size=args.hidden_size,
        recurrent_type=args.recurrent,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("=== GNN + LNN Dynamic Graph Experiment ===")
    print(f"Model: graph encoder + {args.recurrent} | Device: {device} | Samples: {args.samples}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history = {"train_loss": [], "val_mse": [], "val_mae": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val["mse"])
        history["val_mae"].append(val["mae"])
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val MSE {val['mse']:.4f} | Val MAE {val['mae']:.4f}")

    elapsed_seconds = time.perf_counter() - start
    test = evaluate(model, test_loader, device)
    print(f"Test MSE: {test['mse']:.6f} | Test MAE: {test['mae']:.6f}")

    now = dt.datetime.now()
    payload = {
        "run_id": now.strftime("%Y-%m-%d_%H%M%S"),
        "run_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "recurrent_type": args.recurrent,
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
