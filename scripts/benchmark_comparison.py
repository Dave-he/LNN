"""
Model comparison benchmark: LNN (CfC, LTC) vs LSTM vs GRU.

Runs all models on the same dataset with the same hyperparameters
and produces a comparative analysis with visualizations.

Usage:
    python scripts/benchmark_comparison.py --data mackey_glass --epochs 100
    python scripts/benchmark_comparison.py --data sine --epochs 50
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.trainer import Trainer
from lnn.data.timeseries import create_dataloader, generate_mackey_glass, generate_sine_data
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_model_comparison, plot_predictions, plot_training_curve


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


MODELS = {
    "CfC": lambda i, h, o: CfCNetwork(i, h, o, num_layers=1, return_sequences=False),
    "LTC": lambda i, h, o: LTCNetwork(i, h, o, num_layers=1),
    "LSTM": lambda i, h, o: LSTMModel(i, h, o),
    "GRU": lambda i, h, o: GRUModel(i, h, o),
}


def main():
    parser = argparse.ArgumentParser(description="LNN vs LSTM vs GRU Benchmark")
    parser.add_argument("--data", type=str, default="mackey_glass", choices=["sine", "mackey_glass"])
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="analysis/benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LNN vs LSTM vs GRU Benchmark")
    print("=" * 60)

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
    output_size = 1

    all_results = {}
    all_metrics = {}

    for name, model_fn in MODELS.items():
        print(f"\n--- Training {name} ---")
        model = model_fn(input_size, args.hidden_size, output_size)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        trainer = Trainer(model, lr=args.lr, patience=args.patience)
        start = time.time()
        history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs, verbose=True)
        elapsed = time.time() - start

        preds, targets = trainer.predict(test_loader)
        metrics = compute_metrics(targets, preds)
        metrics["params"] = param_count
        metrics["train_time"] = elapsed
        metrics["epochs"] = history["total_epochs"]

        all_results[name] = history
        all_metrics[name] = metrics

        print(f"  RMSE: {metrics['rmse']:.6f} | MAE: {metrics['mae']:.6f} | Time: {elapsed:.1f}s")

        plot_training_curve(
            history["train_losses"],
            history["val_losses"],
            title=f"{name} Training Curve ({args.data})",
            save_path=os.path.join(args.output_dir, f"{name}_{args.data}_training.png"),
        )

        pred_np = preds.numpy().flatten()
        target_np = targets.numpy().flatten()
        plot_predictions(
            target_np[:200],
            pred_np[:200],
            title=f"{name} Prediction ({args.data})",
            save_path=os.path.join(args.output_dir, f"{name}_{args.data}_prediction.png"),
        )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Model':<10} {'RMSE':<12} {'MAE':<12} {'Params':<10} {'Time(s)':<10} {'Epochs':<8}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(
            f"{name:<10} {m['rmse']:<12.6f} {m['mae']:<12.6f} "
            f"{m['params']:<10,} {m['train_time']:<10.1f} {m['epochs']:<8}"
        )

    plot_model_comparison(
        all_metrics,
        metric="rmse",
        title=f"Model Comparison - RMSE ({args.data})",
        save_path=os.path.join(args.output_dir, f"comparison_rmse_{args.data}.png"),
    )

    plot_model_comparison(
        all_metrics,
        metric="mae",
        title=f"Model Comparison - MAE ({args.data})",
        save_path=os.path.join(args.output_dir, f"comparison_mae_{args.data}.png"),
    )

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
