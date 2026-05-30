"""
Experiment script to train LNN models on real-world or realistic datasets.
"""

import argparse
import os
import sys
import time

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.trainer import Trainer
from lnn.data.datasets import (
    generate_stock_like_data,
    prepare_univariate_data,
    create_real_dataloader,
)
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_predictions, plot_training_curve


def get_model(model_name: str, input_size: int, hidden_size: int, output_size: int) -> torch.nn.Module:
    if model_name == "cfc":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=1)
    elif model_name == "ltc":
        return LTCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4")
    elif model_name == "gru":
        return torch.nn.GRU(input_size, hidden_size, batch_first=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class GRUWrapper(torch.nn.Module):
    """Wrapper to make GRU have same interface as our LNN models."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


def main():
    parser = argparse.ArgumentParser(description="LNN Real Data Experiment")
    parser.add_argument("--model", type=str, default="cfc", choices=["cfc", "ltc", "gru"])
    parser.add_argument("--data", type=str, default="stock", choices=["stock", "electricity", "air_quality"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="analysis/real_data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("=== LNN Real-World Data Experiment ===")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Data: {args.data}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Horizon: {args.horizon}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print()

    # Load/generate data
    print(f"Loading {args.data} data...")
    if args.data == "stock":
        data = generate_stock_like_data(num_samples=5000, num_features=1)
        df = pd.DataFrame(data, columns=["value"])
        column = "value"
    elif args.data == "electricity":
        df = download_electricity_data()
        column = df.columns[0]  # Use first client's data
    elif args.data == "air_quality":
        df = download_air_quality_data()
        column = "PM2.5"  # Use PM2.5 as target
    else:
        raise ValueError(f"Unknown data type: {args.data}")

    print(f"Data shape: {df.shape}")
    print(f"Using column: {column}")

    # Prepare data
    train_data, val_data, test_data = prepare_univariate_data(
        df, column, seq_len=args.seq_len, horizon=args.horizon
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create dataloaders
    train_loader = create_real_dataloader(
        train_data, seq_len=args.seq_len, horizon=args.horizon, 
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_real_dataloader(
        val_data, seq_len=args.seq_len, horizon=args.horizon, 
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = create_real_dataloader(
        test_data, seq_len=args.seq_len, horizon=args.horizon, 
        batch_size=args.batch_size, shuffle=False
    )

    # Create model
    input_size = 1
    output_size = 1 if args.horizon == 1 else args.horizon

    if args.model == "gru":
        model = GRUWrapper(input_size, args.hidden_size, output_size)
    else:
        model = get_model(args.model, input_size, args.hidden_size, output_size)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {param_count:,} parameters")

    # Train
    print("\nStarting training...")
    trainer = Trainer(model, lr=args.lr, patience=args.patience)
    
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs)
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # Evaluate speed
    print("\nMeasuring inference speed...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))[0]
        start_time = time.time()
        for _ in range(100):
            _ = model(test_batch)
        inference_time = (time.time() - start_time) / 100
        samples_per_sec = test_batch.shape[0] / inference_time
        print(f"Inference time per batch: {inference_time*1000:.2f} ms")
        print(f"Samples per second: {samples_per_sec:.2f}")

    # Evaluate
    print("\nEvaluating on test set...")
    preds, targets = trainer.predict(test_loader)
    metrics = compute_metrics(targets, preds)

    print("\n" + "=" * 40)
    print("Test Results:")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.6f}")
    print(f"\nTraining Time: {train_time:.2f}s")
    print(f"Inference Speed: {samples_per_sec:.2f} samples/s")

    # Save results
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

    # Save metrics
    results = {
        "model": args.model,
        "data": args.data,
        "params": param_count,
        "train_time": train_time,
        "samples_per_sec": samples_per_sec,
        **metrics,
    }
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, f"{prefix}_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\nAll results saved to {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
