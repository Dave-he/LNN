#!/usr/bin/env python3
"""
Comprehensive LNN Experimentation: implements all model variants from the papers.

This script implements and compares:
- LTC (Liquid Time-Constant)
- CfC (Closed-form Continuous-time)
- Strict CfC
- Hybrid CfC
- CT-LTC
- Liquid-S4
- LRC (Liquid Resistive-Capacitive)
- CfC-DT
- Euler-LTC-DT
- GRU (baseline)
- LSTM (baseline)

On tasks:
1. Non-stationary synthetic time series
2. Natural gas price forecasting
3. Irregularly sampled data
4. OOD (Out-of-Distribution) generalization
"""

import argparse
import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lnn.core.ltc import LTCNetwork
from lnn.core.cfc import CfCNetwork
from lnn.core.variants import (
    StrictCfCNetwork, HybridCfCNetwork, CTLTCNetwork,
    LiquidS4Network, LRCNetwork, CfCDTNetwork, EulerLTCDTNetwork
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_non_stationary_sequence(
    num_samples: int = 1000,
    seq_len: int = 50,
    num_features: int = 1,
    regime_changes: bool = True,
    noise_level: float = 0.1,
    seed: int = 42
):
    """
    Generate a non-stationary time series with changing dynamics.
    """
    np.random.seed(seed)
    t = np.linspace(0, 20 * np.pi, num_samples)
    
    # Changing frequency and amplitude
    freq_base = 0.2 + 0.5 * np.sin(t * 0.1)
    amp_base = 1.0 + 0.3 * np.sin(t * 0.05)
    data = amp_base * np.sin(freq_base * t)
    
    # Add regime changes
    if regime_changes:
        for i in range(3):
            start = (i + 1) * num_samples // 4
            end = start + num_samples // 4
            data[start:end] += 0.5 * np.sin(3 * freq_base[start:end] * t[start:end])
    
    # Add trend
    trend = 0.01 * t
    data += trend
    
    # Add noise
    data += noise_level * np.random.randn(num_samples)
    
    return data.reshape(-1, num_features)


def generate_irregular_dt(
    num_samples: int,
    base_dt: float = 1.0,
    min_dt: float = 0.5,
    max_dt: float = 2.0,
    seed: int = 42
):
    """Generate irregular time steps."""
    np.random.seed(seed)
    dt = np.random.uniform(min_dt, max_dt, num_samples)
    return dt.reshape(-1, 1)


class TimeSeriesDataset(Dataset):
    """Time series dataset for sequence prediction."""
    
    def __init__(
        self, data, seq_len: int = 50, horizon: int = 1, dt=None, include_dt: bool = False):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.dt = dt
        self.include_dt = include_dt
        self.num_sequences = len(data) - seq_len - horizon + 1
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.horizon]
        
        if self.include_dt and self.dt is not None:
            dt_seq = self.dt[idx:idx + self.seq_len]
            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor(dt_seq, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32))
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class GRUBaseline(nn.Module):
    """GRU baseline model."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMBaseline(nn.Module):
    """LSTM baseline model."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def get_model(model_name: str, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
    """Get model by name."""
    if model_name == "ltc":
        return LTCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "cfc":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "strict_cfc":
        return StrictCfCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "hybrid_cfc":
        return HybridCfCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "ct_ltc":
        return CTLTCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "liquid_s4":
        return LiquidS4Network(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "lrc":
        return LRCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "cfc_dt":
        return CfCDTNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "euler_ltc_dt":
        return EulerLTCDTNetwork(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "gru":
        return GRUBaseline(input_size, hidden_size, output_size, num_layers=num_layers)
    elif model_name == "lstm":
        return LSTMBaseline(input_size, hidden_size, output_size, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(
    model, train_loader, val_loader, device, num_epochs: int = 100, lr: float = 0.001, patience: int = 15, use_dt: bool = False):
    """Train a model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_dt:
                x, dt_seq, y = batch
                x = x.to(device)
                dt_seq = dt_seq.to(device)
                y = y.to(device)
                pred = model(x, dt=dt_seq)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            
            # Ensure prediction matches target shape
            if pred.dim() > 2 and y.dim() == 3:
                y = y.squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_dt:
                    x, dt_seq, y = batch
                    x = x.to(device)
                    dt_seq = dt_seq.to(device)
                    y = y.to(device)
                    pred = model(x, dt=dt_seq)
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                
                if pred.dim() > 2 and y.dim() == 3:
                    y = y.squeeze(1)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        scheduler.step(val_loss)
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    train_time = time.time() - start_time
    
    return model, train_losses, val_losses, best_val_loss, train_time


def evaluate_model(model, test_loader, device, use_dt: bool = False):
    """Evaluate model on test set."""
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if use_dt:
                x, dt_seq, y = batch
                x = x.to(device)
                dt_seq = dt_seq.to(device)
                y = y.to(device)
                pred = model(x, dt=dt_seq)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            
            if pred.dim() > 2 and y.dim() == 3:
                y = y.squeeze(1)
            
            loss = criterion(pred, y)
            test_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return test_loss, mae, all_preds, all_targets


def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(args):
    """Run comprehensive experiment comparing all models."""
    set_seed(args.seed)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate or load data
    print("Generating datasets...")
    
    # Task 1: Non-stationary synthetic data
    data1 = generate_non_stationary_sequence(
        num_samples=2000, seq_len=args.seq_len, seed=args.seed
    )
    dt1 = generate_irregular_dt(len(data1), seed=args.seed)
    
    # Split data
    split1 = int(len(data1) * 7 // 10)
    split2 = int(len(data1) * 85 // 100)
    train_data1, val_data1, test_data1 = data1[:split1], data1[split1:split2], data1[split2:]
    train_dt1, val_dt1, test_dt1 = dt1[:split1], dt1[split1:split2], dt1[split2:]
    
    # Create datasets
    train_dataset1 = TimeSeriesDataset(train_data1, seq_len=args.seq_len, horizon=args.horizon)
    val_dataset1 = TimeSeriesDataset(val_data1, seq_len=args.seq_len, horizon=args.horizon)
    test_dataset1 = TimeSeriesDataset(test_data1, seq_len=args.seq_len, horizon=args.horizon)
    
    train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True)
    val_loader1 = DataLoader(val_dataset1, batch_size=args.batch_size, shuffle=False)
    test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False)
    
    # Task 2: Irregularly sampled data (with dt)
    train_dataset2 = TimeSeriesDataset(
        train_data1, seq_len=args.seq_len, horizon=args.horizon, dt=train_dt1, include_dt=True
    )
    val_dataset2 = TimeSeriesDataset(
        val_data1, seq_len=args.seq_len, horizon=args.horizon, dt=val_dt1, include_dt=True
    )
    test_dataset2 = TimeSeriesDataset(
        test_data1, seq_len=args.seq_len, horizon=args.horizon, dt=test_dt1, include_dt=True
    )
    train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True)
    val_loader2 = DataLoader(val_dataset2, batch_size=args.batch_size, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=args.batch_size, shuffle=False)
    
    # Task 3: OOD data (different seed for testing)
    data_ood = generate_non_stationary_sequence(
        num_samples=500, seq_len=args.seq_len, seed=args.seed + 100, noise_level=0.2
    )
    ood_dataset = TimeSeriesDataset(data_ood, seq_len=args.seq_len, horizon=args.horizon)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model names
    all_models = [
        "ltc", "cfc", "strict_cfc", "hybrid_cfc",
        "ct_ltc", "liquid_s4", "lrc",
        "gru", "lstm"
    ]
    dt_models = ["cfc_dt", "euler_ltc_dt"]
    
    results = []
    
    # Run experiments for all models
    print("\n" + "="*80)
    print("Running experiments on regular data (Task 1 & 3)")
    print("="*80)
    
    for model_name in all_models:
        print(f"\nTraining {model_name}...")
        model = get_model(model_name, 1, args.hidden_size, args.horizon)
        num_params = count_parameters(model)
        print(f"  Parameters: {num_params:,}")
        
        model = model.to(device)
        
        # Train
        model, train_losses, val_losses, best_val_loss, train_time = train_model(
            model, train_loader1, val_loader1, device,
            num_epochs=args.epochs, lr=args.lr, patience=args.patience
        )
        
        # Evaluate on ID test
        test_loss, test_mae, preds, targets = evaluate_model(model, test_loader1, device)
        
        # Evaluate on OOD
        ood_loss, ood_mae, _, _ = evaluate_model(model, ood_loader, device)
        
        # Inference speed
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                for batch in test_loader1:
                    x, y = batch
                    x = x.to(device)
                    _ = model(x)
        infer_time = (time.time() - start_time) / 10
        samples_per_sec = len(test_dataset1) / infer_time if infer_time > 0 else 0
        
        results.append({
            "model": model_name,
            "parameters": num_params,
            "train_time": train_time,
            "best_val_loss": best_val_loss,
            "test_mse": test_loss,
            "test_mae": test_mae,
            "ood_mse": ood_loss,
            "ood_mae": ood_mae,
            "ood_degradation": ((ood_loss / test_loss) - 1.0 if test_loss > 0 else 0),
            "infer_samples_per_sec": samples_per_sec
        })
        
        print(f"  Test MSE: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
        print(f"  OOD MSE: {ood_loss:.6f}, OOD MAE: {ood_mae:.6f}")
        print(f"  Inference: {samples_per_sec:.1f} samples/sec")
    
    # Run experiments for dt-aware models
    print("\n" + "="*80)
    print("Running experiments on irregularly sampled data (Task 2)")
    print("="*80)
    
    for model_name in dt_models:
        print(f"\nTraining {model_name}...")
        model = get_model(model_name, 1, args.hidden_size, args.horizon)
        num_params = count_parameters(model)
        print(f"  Parameters: {num_params:,}")
        
        model = model.to(device)
        
        # Train
        model, train_losses, val_losses, best_val_loss, train_time = train_model(
            model, train_loader2, val_loader2, device,
            num_epochs=args.epochs, lr=args.lr, patience=args.patience, use_dt=True
        )
        
        # Evaluate on test with dt
        test_loss, test_mae, preds, targets = evaluate_model(model, test_loader2, device, use_dt=True)
        
        # Inference speed
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                for batch in test_loader2:
                    x, dt_seq, y = batch
                    x = x.to(device)
                    dt_seq = dt_seq.to(device)
                    _ = model(x, dt=dt_seq)
        infer_time = (time.time() - start_time) / 10
        samples_per_sec = len(test_dataset2) / infer_time if infer_time > 0 else 0
        
        results.append({
            "model": model_name,
            "parameters": num_params,
            "train_time": train_time,
            "best_val_loss": best_val_loss,
            "test_mse": test_loss,
            "test_mae": test_mae,
            "ood_mse": float('nan'),
            "ood_mae": float('nan'),
            "ood_degradation": float('nan'),
            "infer_samples_per_sec": samples_per_sec,
            "dt_aware": True
        })
        
        print(f"  Test MSE: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
        print(f"  Inference: {samples_per_sec:.1f} samples/sec")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_mse")
    
    output_file = os.path.join(args.output_dir, f"comprehensive_results_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Test MSE comparison
    ax = axes[0, 0]
    model_names = [r["model"] for r in results]
    test_mses = [r["test_mse"] for r in results]
    ax.barh(model_names, test_mses, color='steelblue')
    ax.set_xlabel('Test MSE (lower is better)')
    ax.set_title('Test MSE by Model')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: OOD performance
    ax = axes[0, 1]
    ood_mses = [r["ood_mse"] if not np.isnan(r["ood_mse"]) else 0 for r in results]
    model_names_filtered = [r["model"] for r in results if not np.isnan(r["ood_mse"])]
    ood_mses_filtered = [r["ood_mse"] for r in results if not np.isnan(r["ood_mse"])]
    ax.barh(model_names_filtered, ood_mses_filtered, color='darkorange')
    ax.set_xlabel('OOD MSE (lower is better)')
    ax.set_title('OOD Generalization by Model')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 3: Inference speed
    ax = axes[1, 0]
    infer_speeds = [r["infer_samples_per_sec"] for r in results]
    ax.barh(model_names, infer_speeds, color='forestgreen')
    ax.set_xlabel('Samples per Second (higher is better)')
    ax.set_title('Inference Speed by Model')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Parameter efficiency (MSE / parameters
    ax = axes[1, 1]
    params = [r["parameters"] for r in results]
    mse_per_param = [r["test_mse"] / max(r["parameters"], 1) * 1000 for r in results]
    ax.barh(model_names, mse_per_param, color='purple')
    ax.set_xlabel('(MSE / 1000 params) (lower better)')
    ax.set_title('Parameter Efficiency by Model')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(args.output_dir, f"comprehensive_plots_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {plot_file}")
    
    # Save configuration
    config = vars(args)
    config_file = os.path.join(args.output_dir, f"config_{timestamp}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print conclusions
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    best_model = results_df.iloc[0]
    print(f"\nBest model by test MSE: {best_model['model']}")
    print(f"  Test MSE: {best_model['test_mse']:.6f}")
    
    best_speed = results_df.iloc[results_df['infer_samples_per_sec'].idxmax()]
    print(f"\nFastest model: {best_speed['model']}")
    print(f"  Speed: {best_speed['infer_samples_per_sec']:.1f} samples/sec")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LNN Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--output-dir", type=str, default="analysis/comprehensive", help="Output directory")
    
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == "__main__":
    main()
