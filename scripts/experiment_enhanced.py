"""
Enhanced time series prediction experiment using LNN models.

This script includes:
- Learning rate scheduling
- Checkpoint saving/loading
- Realistic dataset (energy price simulation)
- Comprehensive visualization
- Multiple model comparison

Usage:
    python scripts/experiment_enhanced.py --model cfc --data energy
    python scripts/experiment_enhanced.py --model ltc --data mackey_glass --use_scheduler
"""

import argparse
import os
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.trainer import Trainer
from lnn.data.timeseries import create_dataloader, generate_mackey_glass, generate_sine_data, generate_lorenz
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_predictions, plot_training_curve


def generate_energy_price_data(
    num_samples: int = 3000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate realistic energy price time series with:
    - Daily seasonality
    - Weekly seasonality
    - Trend
    - Noise
    - Occasional price spikes
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    t = np.arange(num_samples, dtype=np.float32)
    
    daily_freq = 1 / 24.0
    weekly_freq = 1 / (24 * 7)
    
    base_trend = 0.0001 * t
    daily_seasonal = 0.15 * np.sin(2 * np.pi * daily_freq * t)
    weekly_seasonal = 0.1 * np.sin(2 * np.pi * weekly_freq * t + np.pi / 4)
    noise = 0.05 * rng.standard_normal(num_samples)
    
    price = 1.0 + base_trend + daily_seasonal + weekly_seasonal + noise
    
    spike_indices = rng.choice(num_samples, size=num_samples // 100, replace=False)
    price[spike_indices] *= 1.5 + rng.uniform(0, 0.5, size=len(spike_indices))
    
    price = (price - price.mean()) / (price.std() + 1e-8)
    
    return price.astype(np.float32)


def get_model(model_name: str, input_size: int, hidden_size: int, output_size: int) -> torch.nn.Module:
    if model_name == "cfc":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=1)
    elif model_name == "ltc":
        return LTCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4")
    elif model_name == "cfc_deep":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=2)
    elif model_name == "lstm":
        return nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced LNN Time Series Experiment")
    parser.add_argument("--model", type=str, default="cfc", choices=["cfc", "ltc", "cfc_deep", "lstm"])
    parser.add_argument("--data", type=str, default="energy", choices=["sine", "mackey_glass", "lorenz", "energy"])
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="analysis/enhanced")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model}_{args.data}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print(f"Enhanced LNN Time Series Experiment - {timestamp}")
    print("=" * 60)
    print(f"Model: {args.model} | Data: {args.data} | Hidden: {args.hidden_size}")
    print(f"SeqLen: {args.seq_len} | Horizon: {args.horizon} | Epochs: {args.epochs}")
    print(f"LR: {args.lr} | Scheduler: {args.use_scheduler} | Patience: {args.patience}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    if args.data == "sine":
        raw_data = generate_sine_data(num_samples=3000, freq=0.05, noise_std=0.05)
    elif args.data == "mackey_glass":
        raw_data = generate_mackey_glass(num_samples=3000, tau=17)
    elif args.data == "lorenz":
        raw_data = generate_lorenz(num_samples=3000)
    else:
        raw_data = generate_energy_price_data(num_samples=3000)

    split_train = int(len(raw_data) * 0.7)
    split_val = int(len(raw_data) * 0.85)
    train_data = raw_data[:split_train]
    val_data = raw_data[split_train:split_val]
    test_data = raw_data[split_val:]

    train_loader = create_dataloader(
        train_data, seq_len=args.seq_len, horizon=args.horizon, batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_data, seq_len=args.seq_len, horizon=args.horizon, batch_size=args.batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        test_data, seq_len=args.seq_len, horizon=args.horizon, batch_size=args.batch_size, shuffle=False
    )

    input_size = 1
    output_size = 1 if args.horizon == 1 else args.horizon

    model = get_model(args.model, input_size, args.hidden_size, output_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    lr_scheduler = None
    if args.use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

    trainer = Trainer(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr=args.lr,
        patience=args.patience,
        checkpoint_dir=os.path.join(args.checkpoint_dir, run_id),
    )

    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_best_only=True,
    )

    preds, targets = trainer.predict(test_loader)
    metrics = compute_metrics(targets, preds)
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.6f}")
    
    summary = {
        "model": args.model,
        "data": args.data,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "epochs": history["total_epochs"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "parameters": param_count,
        "training_time": history["elapsed_seconds"],
        "test_metrics": metrics,
    }
    
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    plot_training_curve(
        history["train_losses"],
        history["val_losses"],
        lrs=history.get("lrs"),
        title=f"{args.model.upper()} Training Curve ({args.data})",
        save_path=os.path.join(run_dir, "training.png"),
    )

    pred_np = preds.numpy().flatten()
    target_np = targets.numpy().flatten()
    plot_predictions(
        target_np[:200],
        pred_np[:200],
        title=f"{args.model.upper()} Prediction ({args.data})",
        save_path=os.path.join(run_dir, "prediction.png"),
    )

    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
