"""
Concept Drift Adaptation Experiment.

Core thesis: LNN's dynamic time constants enable real-time adaptation
to regime changes without retraining, unlike fixed-parameter models.

Experiment design:
1. Generate data with concept drift (regime A → regime B)
2. Train all models on regime A only
3. Test on full data including regime B
4. Measure adaptation capability across the drift boundary

Usage:
    python scripts/experiment_concept_drift.py --epochs 80
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.trainer import Trainer
from lnn.data.timeseries import create_dataloader, generate_concept_drift
from lnn.utils.interpretability import (
    extract_cfc_dynamics,
    plot_concept_drift_adaptation,
    plot_dynamics,
)
from lnn.utils.metrics import compute_metrics


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


MODELS = {
    "CfC": lambda i, h, o: CfCNetwork(i, h, o, return_sequences=False),
    "LTC": lambda i, h, o: LTCNetwork(i, h, o),
    "LSTM": lambda i, h, o: LSTMModel(i, h, o),
    "GRU": lambda i, h, o: GRUModel(i, h, o),
}


def main():
    parser = argparse.ArgumentParser(description="Concept Drift Adaptation Experiment")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="analysis/concept_drift")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Concept Drift Adaptation Experiment")
    print("Train: Regime A → Test: Regime A + Regime B (drift)")
    print("=" * 60)

    drift_point = 1000
    full_data, drift_labels = generate_concept_drift(
        num_samples=2000,
        drift_point=drift_point,
        freq_before=0.05,
        freq_after=0.12,
        amp_before=1.0,
        amp_after=0.6,
        noise_std=0.05,
    )

    regime_a = full_data[:drift_point]
    train_data = regime_a[:700]
    val_data = regime_a[700:]

    train_loader = create_dataloader(
        train_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )
    full_test_loader = create_dataloader(
        full_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )
    regime_b_loader = create_dataloader(
        full_data[drift_point:], seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )

    input_size = 1
    output_size = 1

    all_preds = {}
    all_metrics = {}

    for name, model_fn in MODELS.items():
        print(f"\n--- Training {name} on Regime A ---")
        model = model_fn(input_size, args.hidden_size, output_size)
        trainer = Trainer(model, lr=args.lr, patience=15)
        trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

        full_preds, full_targets = trainer.predict(full_test_loader)
        b_preds, b_targets = trainer.predict(regime_b_loader)

        full_m = compute_metrics(full_targets, full_preds)
        b_m = compute_metrics(b_targets, b_preds)
        all_metrics[name] = {"full": full_m, "regime_b": b_m}

        all_preds[name] = full_preds.numpy().flatten()

        print(f"  Full RMSE: {full_m['rmse']:.6f} | Regime B RMSE: {b_m['rmse']:.6f}")

    print("\n" + "=" * 60)
    print("CONCEPT DRIFT RESULTS")
    print("=" * 60)
    print(f"{'Model':<10} {'Full RMSE':<12} {'Regime B RMSE':<14} {'Drift Penalty':<14}")
    print("-" * 50)
    for name in MODELS:
        full_rmse = all_metrics[name]["full"]["rmse"]
        b_rmse = all_metrics[name]["regime_b"]["rmse"]
        print(f"{name:<10} {full_rmse:<12.6f} {b_rmse:<14.6f}")

    plot_concept_drift_adaptation(
        all_preds,
        full_data[args.seq_len:][:len(all_preds["CfC"])],
        drift_point=drift_point - args.seq_len,
        title="Concept Drift: Model Adaptation Comparison",
        save_path=os.path.join(args.output_dir, "concept_drift_adaptation.png"),
    )

    print("\n--- Extracting CfC Dynamics Across Drift ---")
    cfc_model = MODELS["CfC"](input_size, args.hidden_size, output_size)
    cfc_trainer = Trainer(cfc_model, lr=args.lr, patience=15)
    cfc_trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

    sample_x = next(iter(full_test_loader))[0]
    cfc_dynamics = extract_cfc_dynamics(cfc_model, sample_x)
    plot_dynamics(
        cfc_dynamics,
        model_type="cfc",
        title="CfC Dynamics Across Concept Drift",
        save_path=os.path.join(args.output_dir, "cfc_drift_dynamics.png"),
    )

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
