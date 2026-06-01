#!/usr/bin/env python3
"""Run a physics-informed LNN experiment on damped oscillator sequences."""

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

from lnn.core.physics import PhysicsInformedLNN, physics_informed_loss
from lnn.data.physics import DampedOscillatorDataset, create_physics_dataloaders


ROOT = pathlib.Path(__file__).resolve().parents[1]


def move_target(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in target.items()}


def train_epoch(
    model: PhysicsInformedLNN,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for states, target in dataloader:
        states = states.to(device)
        target = move_target(target, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(states, dt=target["dt"], mask=target["mask"])
        loss, _ = physics_informed_loss(
            prediction,
            target,
            param_weight=args.param_weight,
            rollout_weight=args.rollout_weight,
            residual_weight=args.residual_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: PhysicsInformedLNN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_param_mae = 0.0
    total_rollout_mse = 0.0
    total_residual = 0.0
    batches = 0
    for states, target in dataloader:
        states = states.to(device)
        target = move_target(target, device)
        prediction = model(states, dt=target["dt"], mask=target["mask"])
        loss, metrics = physics_informed_loss(
            prediction,
            target,
            param_weight=args.param_weight,
            rollout_weight=args.rollout_weight,
            residual_weight=args.residual_weight,
        )
        total_loss += loss.item()
        total_param_mae += F.l1_loss(prediction["params"], target["params"]).item()
        total_rollout_mse += metrics["rollout_loss"]
        total_residual += metrics["residual_loss"]
        batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "param_mae": total_param_mae / max(batches, 1),
        "rollout_mse": total_rollout_mse / max(batches, 1),
        "residual": total_residual / max(batches, 1),
    }


def write_report(payload: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}_physics_lnn.json"
    md_path = output_dir / f"{run_id}_physics_lnn.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config = payload["config"]
    test = payload["test"]
    lines = [
        "---",
        f"title: Physics-informed LNN 运行记录 - {run_id}",
        f"date: {payload['run_date']}",
        "tags: [LNN, physics-informed, LTC, oscillator]",
        "---",
        "",
        f"# Physics-informed LNN 运行记录 - {run_id}",
        "",
        "## 任务",
        "- 数据：随机参数 damped oscillator，观测 position/velocity 序列。",
        "- 目标：恢复 omega/damping，并预测未来 rollout，同时加入 ODE residual。",
        f"- 模型：{payload['recurrent_type']}",
        f"- 设备：{payload['device']}",
        "",
        "## 配置",
        f"- 样本数 / SeqLen / Horizon：{config['samples']} / {config['seq_len']} / {config['horizon']}",
        f"- Hidden / Epoch / Batch：{config['hidden_size']} / {config['epochs']} / {config['batch_size']}",
        f"- Loss weights：param={config['param_weight']}, rollout={config['rollout_weight']}, "
        f"residual={config['residual_weight']}",
        "",
        "## 结果",
        f"- Test loss：{test['loss']:.6f}",
        f"- 参数 MAE：{test['param_mae']:.6f}",
        f"- Rollout MSE：{test['rollout_mse']:.6f}",
        f"- Physics residual：{test['residual']:.6f}",
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
    parser.add_argument("--recurrent", choices=["cfc", "ltc", "gru"], default="ltc")
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--param-weight", type=float, default=1.0)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/physics")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dataset = DampedOscillatorDataset(
        num_samples=args.samples,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_physics_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = PhysicsInformedLNN(
        hidden_size=args.hidden_size,
        horizon=args.horizon,
        recurrent_type=args.recurrent,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("=== Physics-informed LNN Experiment ===")
    print(f"Model: {args.recurrent} | Device: {device} | Samples: {args.samples}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history = {"train_loss": [], "val_loss": [], "val_param_mae": [], "val_rollout_mse": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        val = evaluate(model, val_loader, device, args)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_param_mae"].append(val["param_mae"])
        history["val_rollout_mse"].append(val["rollout_mse"])
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | Train {train_loss:.4f} | "
                f"Val param MAE {val['param_mae']:.4f} | Val rollout MSE {val['rollout_mse']:.4f}"
            )

    elapsed_seconds = time.perf_counter() - start
    test = evaluate(model, test_loader, device, args)
    print(
        f"Test loss: {test['loss']:.6f} | Param MAE: {test['param_mae']:.6f} | "
        f"Rollout MSE: {test['rollout_mse']:.6f} | Residual: {test['residual']:.6f}"
    )

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
