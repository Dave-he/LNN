#!/usr/bin/env python3
"""
Tutorial: How to use LNN models and understand their advantages.

This script demonstrates:
1. Basic usage of all LNN variants
2. Comparison with traditional RNN/GRU
3. Key advantages of LNN models
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from lnn.core.ltc import LTCNetwork
from lnn.core.cfc import CfCNetwork
from lnn.core.variants import (
    StrictCfCNetwork,
    HybridCfCNetwork,
    CTLTCNetwork,
    LiquidS4Network,
    LRCNetwork,
    CfCDTNetwork,
    EulerLTCDTNetwork,
)
from lnn.data.timeseries import create_dataloader, generate_sine_data, generate_mackey_glass
from lnn.core.trainer import Trainer
from lnn.utils.metrics import compute_metrics


class TraditionalGRU(nn.Module):
    """Traditional GRU for comparison."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output)


def demonstrate_basic_usage():
    """Demonstrate basic usage of LNN models."""
    print("="*80)
    print("PART 1: BASIC USAGE")
    print("="*80)
    
    # Hyperparameters
    input_size = 1
    hidden_size = 32
    output_size = 1
    batch_size = 4
    seq_len = 16
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test all models
    models = {
        "LTC": LTCNetwork(input_size, hidden_size, output_size),
        "CfC": CfCNetwork(input_size, hidden_size, output_size),
        "StrictCfC": StrictCfCNetwork(input_size, hidden_size, output_size),
        "HybridCfC": HybridCfCNetwork(input_size, hidden_size, output_size),
        "CTLTC": CTLTCNetwork(input_size, hidden_size, output_size),
        "LiquidS4": LiquidS4Network(input_size, hidden_size, output_size),
        "LRC": LRCNetwork(input_size, hidden_size, output_size),
        "CfC-DT": CfCDTNetwork(input_size, hidden_size, output_size),
        "Euler-LTC-DT": EulerLTCDTNetwork(input_size, hidden_size, output_size),
        "GRU (Baseline)": TraditionalGRU(input_size, hidden_size, output_size),
    }
    
    print("\nTesting forward passes...\n")
    
    results = []
    for name, model in models.items():
        # Forward pass
        start = time.time()
        output = model(x)
        forward_time = (time.time() - start) * 1000
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"{name:20s} | Params: {num_params:6,d} | Time: {forward_time:6.2f}ms | Output: {output.shape}")
        results.append({"name": name, "params": num_params, "time": forward_time})
    
    return results


def train_and_compare():
    """Train models and compare performance."""
    print("\n" + "="*80)
    print("PART 2: TRAINING & PERFORMANCE COMPARISON")
    print("="*80)
    
    # Generate data - Mackey-Glass is a challenging chaotic time series
    print("\nGenerating Mackey-Glass chaotic time series...")
    raw_data = generate_mackey_glass(num_samples=1500, tau=17)
    
    # Split data
    split_train = int(len(raw_data) * 0.7)
    split_val = int(len(raw_data) * 0.85)
    train_data = raw_data[:split_train]
    val_data = raw_data[split_train:split_val]
    test_data = raw_data[split_val:]
    
    # Create dataloaders
    seq_len = 32
    horizon = 1
    batch_size = 32
    
    train_loader = create_dataloader(train_data, seq_len, horizon, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, seq_len, horizon, batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, seq_len, horizon, batch_size, shuffle=False)
    
    # Models to compare
    input_size = 1
    hidden_size = 32
    output_size = 1
    
    models = {
        "CfC (Fast & Accurate)": CfCNetwork(input_size, hidden_size, output_size),
        "LTC (Most Accurate)": LTCNetwork(input_size, hidden_size, output_size),
        "HybridCfC (Balanced)": HybridCfCNetwork(input_size, hidden_size, output_size),
        "Euler-LTC-DT (Edge)": EulerLTCDTNetwork(input_size, hidden_size, output_size),
        "GRU (Traditional)": TraditionalGRU(input_size, hidden_size, output_size),
    }
    
    all_results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")
        
        # Train
        trainer = Trainer(model, lr=1e-3, patience=10)
        start_train = time.time()
        history = trainer.fit(train_loader, val_loader, num_epochs=30)
        train_time = time.time() - start_train
        
        # Evaluate
        start_infer = time.time()
        preds, targets = trainer.predict(test_loader)
        infer_time = time.time() - start_infer
        
        metrics = compute_metrics(targets, preds)
        samples_per_sec = len(test_data) / infer_time
        
        print(f"\nResults:")
        print(f"  MSE:  {metrics['mse']:.8f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  Train Time: {train_time:.2f}s")
        print(f"  Inference:  {samples_per_sec:.0f} samples/sec")
        
        all_results.append({
            "name": name,
            "params": num_params,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "train_time": train_time,
            "samples_per_sec": samples_per_sec,
        })
    
    return all_results


def demonstrate_irregular_time():
    """Demonstrate CfC-DT with irregular time steps."""
    print("\n" + "="*80)
    print("PART 3: IRREGULAR TIME SAMPLING (CfC-DT ADVANTAGE)")
    print("="*80)
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    
    # Create model that supports variable time steps
    model = CfCDTNetwork(input_size, hidden_size, output_size)
    
    print("\nCfC-DT supports explicit time steps!")
    print("This is crucial for real-world data with irregular sampling.")
    
    # Create data with irregular time steps
    batch_size = 2
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Irregular time steps (e.g., missing sensor data)
    dt = torch.tensor([
        [0.1, 0.5, 0.2, 2.0, 0.3, 0.1, 1.5, 0.4, 0.6, 0.2],  # Batch 1
        [0.3, 0.1, 1.0, 0.2, 0.5, 0.8, 0.1, 0.3, 2.5, 0.2],  # Batch 2
    ]).unsqueeze(-1)
    
    # Forward pass with time steps
    output = model(x, dt=dt)
    
    print(f"\n✓ Irregular time steps handled successfully!")
    print(f"  Input shape: {x.shape}")
    print(f"  dt shape:    {dt.shape}")
    print(f"  Output shape: {output.shape}")
    print("\nThis is a key advantage over traditional RNNs/GRUs!")


def plot_comparison(results):
    """Plot comparison results."""
    print("\n" + "="*80)
    print("PART 4: VISUAL COMPARISON")
    print("="*80)
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MSE comparison
    ax = axes[0, 0]
    colors = ["steelblue" if "CfC" in n or "LTC" in n else "gray" for n in df["name"]]
    bars = ax.barh(df["name"], df["mse"], color=colors)
    ax.set_xlabel("MSE (lower is better)")
    ax.set_title("Prediction Accuracy (MSE)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    # Inference speed
    ax = axes[0, 1]
    bars = ax.barh(df["name"], df["samples_per_sec"], color=colors)
    ax.set_xlabel("Samples/Second (higher is better)")
    ax.set_title("Inference Speed")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    # Parameters vs MSE
    ax = axes[1, 0]
    scatter = ax.scatter(df["params"], df["mse"], s=200, c=range(len(df)), cmap="viridis")
    for i, row in df.iterrows():
        ax.annotate(row["name"], (row["params"], row["mse"]), 
                   xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("MSE")
    ax.set_title("Efficiency: Parameters vs Accuracy")
    ax.grid(alpha=0.3)
    
    # R² score
    ax = axes[1, 1]
    bars = ax.barh(df["name"], df["r2"], color=colors)
    ax.set_xlabel("R² Score (higher is better)")
    ax.set_title("Goodness of Fit (R²)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    output_file = "analysis/lnn_comparison.png"
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")


def print_advantages():
    """Print key advantages of LNN over large models."""
    print("\n" + "="*80)
    print("KEY ADVANTAGES OF LNN OVER TRADITIONAL LARGE MODELS")
    print("="*80)
    
    advantages = [
        {
            "category": "1. 🏆 Parameter Efficiency",
            "points": [
                "10-100x fewer parameters than transformers",
                "32-128 hidden units often sufficient",
                "No self-attention overhead",
            ]
        },
        {
            "category": "2. ⚡ Computational Speed",
            "points": [
                "CfC: O(N) time, no matrix exponentials",
                "Faster training and inference",
                "Real-time capable on edge devices",
            ]
        },
        {
            "category": "3. 🧠 Continuous-Time Understanding",
            "points": [
                "Naturally models physical systems",
                "Handles irregularly sampled data",
                "Better extrapolation beyond training data",
            ]
        },
        {
            "category": "4. 🎯 OOD Generalization",
            "points": [
                "Robust to distribution shifts",
                "Better at out-of-domain prediction",
                "Less sensitive to small input perturbations",
            ]
        },
        {
            "category": "5. 🔌 Edge & IoT Deployment",
            "points": [
                "Runs on microcontrollers (Euler-LTC-DT)",
                "Low memory footprint",
                "Energy-efficient inference",
            ]
        },
        {
            "category": "6. 📊 Interpretability",
            "points": [
                "Continuous dynamics are more interpretable",
                "Time constants have physical meaning",
                "Easier to debug and understand",
            ]
        },
    ]
    
    for adv in advantages:
        print(f"\n{adv['category']}")
        for point in adv["points"]:
            print(f"   • {point}")
    
    print("\n" + "="*80)
    print("WHEN TO USE LNN vs LARGE LANGUAGE MODELS")
    print("="*80)
    
    print("\nUse LNN for:")
    print("  • Time series prediction & control")
    print("  • Robotics & autonomous systems")
    print("  • IoT sensor data processing")
    print("  • Real-time edge applications")
    print("  • Physical system modeling")
    
    print("\nUse Large Models for:")
    print("  • Natural language understanding")
    print("  • Complex pattern recognition")
    print("  • Knowledge-intensive tasks")
    print("  • Creative generation")


def main():
    print("\n" + "="*80)
    print("LIQUID NEURAL NETWORKS - TUTORIAL & COMPARISON")
    print("="*80)
    
    # Part 1: Basic usage
    basic_results = demonstrate_basic_usage()
    
    # Part 2: Train and compare
    train_results = train_and_compare()
    
    # Part 3: Irregular time
    demonstrate_irregular_time()
    
    # Part 4: Plot comparison
    plot_comparison(train_results)
    
    # Part 5: Print advantages
    print_advantages()
    
    print("\n" + "="*80)
    print("TUTORIAL COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Try experiment_all_variants.py for full comparison")
    print("  2. Read OPTIMIZATION_STRATEGIES.md")
    print("  3. Test on your own time series data")


if __name__ == "__main__":
    main()
