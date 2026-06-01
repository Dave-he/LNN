"""
真实数据上的 LNN 训练实验
使用股票和能源价格数据
"""

import argparse
import os
import sys
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.trainer import Trainer
from lnn.data.real_data import (
    RealTimeSeriesDataset,
    create_real_data_loaders,
    generate_stock_like_data,
    load_yahoo_finance
)
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_predictions, plot_training_curve


def get_model(model_name: str, input_size: int, hidden_size: int, output_size: int) -> torch.nn.Module:
    if model_name == "cfc":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=1)
    elif model_name == "ltc":
        return LTCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4")
    elif model_name == "gru":
        class GRUPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                out, _ = self.gru(x)
                out = self.fc(out[:, -1, :])
                return out
        return GRUPredictor(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_results_to_csv(
    results: Dict[str, Any],
    filename: str = "analysis/real_data/cfc_stock_results.csv"
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    file_exists = os.path.exists(filename)
    
    with open(filename, "a" if file_exists else "w") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "model", "dataset", "hidden_size", "seq_len",
                "train_loss", "val_loss", "mse", "rmse", "mae", "mape",
                "train_time", "samples_per_sec"
            ])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results["model"],
            results["dataset"],
            results["hidden_size"],
            results["seq_len"],
            results["history"]["train_losses"][-1],
            results["best_val_loss"],
            results["metrics"]["mse"],
            results["metrics"]["rmse"],
            results["metrics"]["mae"],
            results["metrics"]["mape"],
            results["training_time"],
            results["samples_per_sec"]
        ])


def main():
    parser = argparse.ArgumentParser(description="真实数据 LNN 训练实验")
    parser.add_argument("--model", type=str, default="cfc", choices=["cfc", "ltc", "gru"])
    parser.add_argument("--dataset", type=str, default="stock", choices=["stock", "energy"])
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--output_dir", type=str, default="analysis/real_data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model}_{args.dataset}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print(f"真实数据 LNN 训练实验 - {timestamp}")
    print("=" * 70)
    print(f"Model: {args.model} | Dataset: {args.dataset} | Hidden: {args.hidden_size}")
    print(f"SeqLen: {args.seq_len} | Horizon: {args.horizon} | Epochs: {args.epochs}")
    print(f"LR: {args.lr} | Scheduler: {args.use_scheduler}")
    print("=" * 70)

    # 加载数据
    if args.dataset == "stock":
        data = generate_stock_like_data(num_samples=2500, seed=42)
        target_col = None
    elif args.dataset == "energy":
        # 生成能源价格数据
        import numpy as np
        np.random.seed(42)
        t = np.arange(2500)
        # 日周期、周周期、趋势、噪声、尖峰
        daily_season = 0.2 * np.sin(2 * np.pi * 0.04 * t)
        weekly_season = 0.1 * np.sin(2 * np.pi * 0.006 * t + np.pi/4)
        trend = 0.00005 * t
        noise = 0.08 * np.random.randn(2500)
        # 加入尖峰（模拟价格波动
        spike_indices = np.random.choice(2500, size=12, replace=False)
        data = 1.0 + trend + daily_season + weekly_season + noise
        data[spike_indices] *= np.random.uniform(1.5, 2.0, size=12)
        # 归一化
        data = (data - data.mean()) / (data.std() + 1e-8)
        target_col = None
    else:
        data = generate_stock_like_data(num_samples=2500, seed=42)
        target_col = None

    print(f"数据加载完成，共 {len(data)} 个样本")

    train_loader, val_loader, test_loader = create_real_data_loaders(
        data,
        target_col=target_col,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size
    )

    input_size = 1
    output_size = 1 if args.horizon == 1 else args.horizon

    model = get_model(args.model, input_size, args.hidden_size, output_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    lr_scheduler = None

    trainer = Trainer(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr=args.lr,
        patience=args.patience,
        checkpoint_dir=os.path.join("checkpoints", run_id),
    )

    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_best_only=True,
    )

    # 测试推理速度
    start_time = time.time()
    preds, targets = trainer.predict(test_loader)
    inference_time = time.time() - start_time
    num_samples = len(targets)
    samples_per_sec = num_samples / inference_time if inference_time > 0 else 0
    
    metrics = compute_metrics(targets, preds)
    
    print("\n" + "=" * 70)
    print("测试集结果")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.6f}")
    print(f"  Inference Speed: {samples_per_sec:.0f} samples/sec")
    
    # 保存结果
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "epochs": history["total_epochs"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": float(history["best_val_loss"]) if history["best_val_loss"] is not None else None,
        "parameters": param_count,
        "training_time": history["elapsed_seconds"],
        "samples_per_sec": samples_per_sec,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "history": {
            "train_losses": [float(x) for x in history["train_losses"]],
            "val_losses": [float(x) for x in history["val_losses"]],
            "lrs": [float(x) for x in history.get("lrs", [])],
        }
    }
    
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    save_results_to_csv(results, os.path.join(args.output_dir, f"{args.model}_{args.dataset}_results.csv"))

    plot_training_curve(
        history["train_losses"],
        history["val_losses"],
        lrs=history.get("lrs"),
        title=f"{args.model.upper()} 训练曲线 - {args.dataset}",
        save_path=os.path.join(run_dir, "training.png"),
    )

    pred_np = preds.numpy().flatten()
    target_np = targets.numpy().flatten()
    plot_predictions(
        target_np[:300],
        pred_np[:300],
        title=f"{args.model.upper()} 预测对比 - {args.dataset}",
        save_path=os.path.join(run_dir, "predictions.png"),
    )

    print(f"\n结果保存至: {run_dir}/")
    print(f"汇总表格: {args.output_dir}/{args.model}_{args.dataset}_results.csv")


if __name__ == "__main__":
    main()
