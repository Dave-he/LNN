"""
Time series prediction experiment using LNN models.

This script demonstrates a complete end-to-end workflow:
1. Generate/load time series data
2. Create train/val/test splits
3. Train CfC and LTC models
4. Evaluate and compare results
5. Save visualizations

Usage:
    python scripts/experiment_timeseries.py --model cfc --data sine --epochs 100
    python scripts/experiment_timeseries.py --model ltc --data mackey_glass --epochs 50
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.trainer import Trainer
from lnn.data.timeseries import create_dataloader, generate_mackey_glass, generate_sine_data
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_predictions, plot_training_curve


def get_model(model_name: str, input_size: int, hidden_size: int, output_size: int) -> torch.nn.Module:
    if model_name == "cfc":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=1)
    elif model_name == "ltc":
        return LTCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4")
    elif model_name == "cfc_deep":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="LNN Time Series Prediction Experiment")
    parser.add_argument("--model", type=str, default="cfc", choices=["cfc", "ltc", "cfc_deep"])
    parser.add_argument("--data", type=str, default="sine", choices=["sine", "mackey_glass"])
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="analysis/experiments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== LNN Time Series Experiment ===")
    print(f"Model: {args.model} | Data: {args.data} | Hidden: {args.hidden_size}")
    print(f"SeqLen: {args.seq_len} | Horizon: {args.horizon} | Epochs: {args.epochs}")

    if args.data == "sine":
        raw_data = generate_sine_data(num_samples=2000, freq=0.05, noise_std=0.05)
    else:
        raw_data = generate_mackey_glass(num_samples=2000, tau=17)

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
    print(f"Parameters: {param_count:,}")

    trainer = Trainer(model, lr=args.lr, patience=args.patience)
    history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

    preds, targets = trainer.predict(test_loader)
    metrics = compute_metrics(targets, preds)
    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.6f}")

    prefix = f"{args.model}_{args.data}"
    plot_training_curve(
        history["train_losses"],
        history["val_losses"],
        title=f"{args.model.upper()} Training Curve ({args.data})",
        save_path=os.path.join(args.output_dir, f"{prefix}_training.png"),
    )

    pred_np = preds.numpy().flatten()
    target_np = targets.numpy().flatten()
    plot_predictions(
        target_np[:200],
        pred_np[:200],
        title=f"{args.model.upper()} Prediction ({args.data})",
        save_path=os.path.join(args.output_dir, f"{prefix}_prediction.png"),
    )

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
