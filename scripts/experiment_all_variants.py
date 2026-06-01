#!/usr/bin/env python3
"""
Comprehensive experiment testing all LNN model variants.

This script demonstrates the complete set of LNN variants we implemented.
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.variants import (
    StrictCfCNetwork,
    HybridCfCNetwork,
    CTLTCNetwork,
    LiquidS4Network,
    LRCNetwork,
    CfCDTNetwork,
    EulerLTCDTNetwork,
)
from lnn.core.trainer import Trainer
from lnn.data.timeseries import create_dataloader, generate_mackey_glass, generate_sine_data
from lnn.utils.metrics import compute_metrics


def get_all_models(input_size: int, hidden_size: int, output_size: int):
    """Return all implemented model variants."""
    models = {
        "LTC": LTCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4"),
        "CfC": CfCNetwork(input_size, hidden_size, output_size, num_layers=1),
        "StrictCfC": StrictCfCNetwork(input_size, hidden_size, output_size, num_layers=1),
        "HybridCfC": HybridCfCNetwork(input_size, hidden_size, output_size, num_layers=1),
        "CTLTC": CTLTCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4"),
        "LiquidS4": LiquidS4Network(input_size, hidden_size, output_size, num_layers=1),
        "LRC": LRCNetwork(input_size, hidden_size, output_size, num_layers=1, ode_method="rk4"),
        "CfC-DT": CfCDTNetwork(input_size, hidden_size, output_size, num_layers=1),
        "Euler-LTC-DT": EulerLTCDTNetwork(input_size, hidden_size, output_size, num_layers=1),
    }
    return models


def main():
    # Config
    hidden_size = 32
    seq_len = 32
    horizon = 1
    epochs = 50
    batch_size = 32
    lr = 1e-3
    patience = 10
    output_dir = "analysis/all_variants"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE LNN VARIANTS EXPERIMENT")
    print("=" * 80)

    # Generate data
    print("\nGenerating Mackey-Glass time series data...")
    raw_data = generate_mackey_glass(num_samples=2000, tau=17)
    split_train = int(len(raw_data) * 0.7)
    split_val = int(len(raw_data) * 0.85)
    train_data = raw_data[:split_train]
    val_data = raw_data[split_train:split_val]
    test_data = raw_data[split_val:]

    train_loader = create_dataloader(
        train_data, seq_len=seq_len, horizon=horizon, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_data, seq_len=seq_len, horizon=horizon, batch_size=batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        test_data, seq_len=seq_len, horizon=horizon, batch_size=batch_size, shuffle=False
    )

    input_size = 1
    output_size = 1

    # Get all models
    models = get_all_models(input_size, hidden_size, output_size)

    results = []

    # Test each model
    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Testing model: {name}")
        print(f"{'=' * 80}")

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        # Train
        trainer = Trainer(model, lr=lr, patience=patience)
        start_time = time.time()
        history = trainer.fit(train_loader, val_loader, num_epochs=epochs)
        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        preds, targets = trainer.predict(test_loader)
        infer_time = time.time() - start_time
        samples_per_sec = len(test_data) / infer_time

        # Metrics
        metrics = compute_metrics(targets, preds)

        # Save results
        result = {
            "model": name,
            "params": param_count,
            "train_time_sec": train_time,
            "infer_samples_per_sec": samples_per_sec,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "best_val_loss": history["val_loss"][-1],
        }
        results.append(result)

        # Print
        print("\nTest Results:")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.6f}")
        print(f"\nTraining Time: {train_time:.2f}s")
        print(f"Inference Speed: {samples_per_sec:.0f} samples/sec")

    # Save and analyze results
    df = pd.DataFrame(results)
    df = df.sort_values("mse")
    df.to_csv(os.path.join(output_dir, "all_variants_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MSE
    ax = axes[0, 0]
    ax.barh(df["model"], df["mse"], color="steelblue")
    ax.set_xlabel("MSE (lower is better)")
    ax.set_title("Mean Squared Error")
    ax.invert_yaxis()
    
    # RMSE
    ax = axes[0, 1]
    ax.barh(df["model"], df["rmse"], color="coral")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Root Mean Squared Error")
    ax.invert_yaxis()
    
    # Training Time
    ax = axes[1, 0]
    ax.barh(df["model"], df["train_time_sec"], color="forestgreen")
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Speed")
    ax.invert_yaxis()
    
    # Inference Speed
    ax = axes[1, 1]
    ax.barh(df["model"], df["infer_samples_per_sec"], color="purple")
    ax.set_xlabel("Samples per Second (higher is better)")
    ax.set_title("Inference Speed")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_variants_comparison.png"), dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {os.path.join(output_dir, 'all_variants_comparison.png')}")
    print(f"Results saved to {os.path.join(output_dir, 'all_variants_results.csv')}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
