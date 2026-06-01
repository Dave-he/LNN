#!/usr/bin/env python3
"""Run a synthetic LNN behavior cloning experiment with MSE or MDN actions."""

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

from lnn.core.control import LNNImitationPolicy
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.data.robotics import SyntheticImitationDataset, create_imitation_dataloaders


ROOT = pathlib.Path(__file__).resolve().parents[1]


def unpack_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if len(batch) == 2:
        states, actions = batch
        metadata: dict[str, torch.Tensor] = {}
    else:
        states, actions, metadata = batch
    metadata = {name: value.to(device) for name, value in metadata.items()}
    return states.to(device), actions.to(device), metadata


def loss_and_prediction(
    output: torch.Tensor | dict[str, torch.Tensor],
    actions: torch.Tensor,
    head_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_type == "mdn":
        loss = mdn_negative_log_likelihood(output, actions)
        prediction = mdn_mean(output)
    else:
        prediction = output
        loss = F.mse_loss(prediction, actions)
    return loss, prediction


def train_epoch(
    model: LNNImitationPolicy,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    head_type: str,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in dataloader:
        states, actions, metadata = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(states, dt=metadata.get("dt"), mask=metadata.get("mask"))
        loss, _ = loss_and_prediction(output, actions, head_type)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: LNNImitationPolicy,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    head_type: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    batches = 0
    for batch in dataloader:
        states, actions, metadata = unpack_batch(batch, device)
        output = model(states, dt=metadata.get("dt"), mask=metadata.get("mask"))
        loss, prediction = loss_and_prediction(output, actions, head_type)
        total_loss += loss.item()
        total_mse += F.mse_loss(prediction, actions).item()
        batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "mse": total_mse / max(batches, 1),
    }


def write_report(payload: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}_imitation_lnn.json"
    md_path = output_dir / f"{run_id}_imitation_lnn.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config = payload["config"]
    test = payload["test"]
    lines = [
        "---",
        f"title: LNN 控制模仿学习运行记录 - {run_id}",
        f"date: {payload['run_date']}",
        "tags: [LNN, imitation-learning, MDN, control]",
        "---",
        "",
        f"# LNN 控制模仿学习运行记录 - {run_id}",
        "",
        "## 任务",
        "- 数据：合成低维状态轨迹，专家动作含隐藏双模态选择。",
        "- 目标：验证 CfC/LTC/AutoNCP recurrent core 与 MSE/MDN action head 的离线行为克隆链路。",
        f"- 模型：{payload['recurrent_type']} + {payload['head_type']}",
        f"- 设备：{payload['device']}",
        "",
        "## 配置",
        f"- 样本数 / Context：{config['samples']} / {config['context_len']}",
        f"- State dim / Action dim：{config['state_dim']} / {config['action_dim']}",
        f"- Hidden / Mixtures：{config['hidden_size']} / {config['num_mixtures']}",
        f"- Epoch / Batch：{config['epochs']} / {config['batch_size']}",
        "",
        "## 结果",
        f"- Test loss：{test['loss']:.6f}",
        f"- Test MSE：{test['mse']:.6f}",
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
    parser.add_argument("--recurrent", choices=["cfc", "ltc", "autoncp"], default="cfc")
    parser.add_argument("--head", choices=["mse", "mdn"], default="mdn")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/control")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dataset = SyntheticImitationDataset(
        num_samples=args.samples,
        context_len=args.context_len,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seed=args.seed,
        return_metadata=True,
    )
    train_loader, val_loader, test_loader = create_imitation_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = LNNImitationPolicy(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        recurrent_type=args.recurrent,
        head_type=args.head,
        num_mixtures=args.num_mixtures,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("=== LNN Imitation Learning Experiment ===")
    print(f"Model: {args.recurrent} + {args.head} | Device: {device} | Samples: {args.samples}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history = {"train_loss": [], "val_loss": [], "val_mse": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.head)
        val = evaluate(model, val_loader, device, args.head)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mse"].append(val["mse"])
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val loss {val['loss']:.4f} | Val MSE {val['mse']:.4f}")

    elapsed_seconds = time.perf_counter() - start
    test = evaluate(model, test_loader, device, args.head)
    print(f"Test loss: {test['loss']:.6f} | Test MSE: {test['mse']:.6f}")

    now = dt.datetime.now()
    payload = {
        "run_id": now.strftime("%Y-%m-%d_%H%M%S"),
        "run_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "recurrent_type": args.recurrent,
        "head_type": args.head,
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
