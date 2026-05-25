"""
OOD (Out-of-Distribution) Generalization Experiment.

Core thesis: LNN's dynamic time constants provide inherent robustness
to distribution shifts, unlike fixed-parameter models (LSTM/GRU).

Experiment design:
1. Train all models on in-distribution data (low freq, low noise sine)
2. Test on OOD data with shifted frequency, amplitude, and noise
3. Measure performance degradation ratio

Usage:
    python scripts/experiment_ood.py --epochs 80
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
from lnn.data.timeseries import create_dataloader, generate_ood_sine
from lnn.utils.interpretability import extract_cfc_dynamics, extract_ltc_dynamics, plot_dynamics, plot_ood_robustness
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_predictions


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
    parser = argparse.ArgumentParser(description="OOD Generalization Experiment")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="analysis/ood")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("OOD Generalization Experiment")
    print("Train: low freq/amp/noise → Test: shifted freq/amp/noise")
    print("=" * 60)

    id_data, ood_data = generate_ood_sine(
        num_train=1500,
        num_ood=500,
        train_freq=0.05,
        train_amp=1.0,
        train_noise=0.05,
        ood_freq_shift=0.03,
        ood_amp_shift=0.5,
        ood_noise_shift=0.15,
    )

    id_train = id_data[:1000]
    id_val = id_data[1000:]

    id_train_loader = create_dataloader(
        id_train, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True
    )
    id_val_loader = create_dataloader(
        id_val, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )
    id_test_loader = create_dataloader(
        id_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )
    ood_test_loader = create_dataloader(
        ood_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )

    input_size = 1
    output_size = 1

    id_metrics = {}
    ood_metrics = {}
    trained_models = {}

    for name, model_fn in MODELS.items():
        print(f"\n--- Training {name} ---")
        model = model_fn(input_size, args.hidden_size, output_size)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        trainer = Trainer(model, lr=args.lr, patience=15)
        trainer.fit(id_train_loader, id_val_loader, num_epochs=args.epochs)

        id_preds, id_targets = trainer.predict(id_test_loader)
        id_m = compute_metrics(id_targets, id_preds)
        id_metrics[name] = id_m

        ood_preds, ood_targets = trainer.predict(ood_test_loader)
        ood_m = compute_metrics(ood_targets, ood_preds)
        ood_metrics[name] = ood_m

        degradation = (ood_m["rmse"] - id_m["rmse"]) / id_m["rmse"] * 100
        print(f"  ID RMSE: {id_m['rmse']:.6f} | OOD RMSE: {ood_m['rmse']:.6f} | Degradation: {degradation:.1f}%")

        trained_models[name] = model

        plot_predictions(
            ood_targets.numpy().flatten()[:200],
            ood_preds.numpy().flatten()[:200],
            title=f"{name} OOD Prediction",
            save_path=os.path.join(args.output_dir, f"{name}_ood_prediction.png"),
        )

    print("\n" + "=" * 60)
    print("OOD GENERALIZATION RESULTS")
    print("=" * 60)
    print(f"{'Model':<10} {'ID RMSE':<12} {'OOD RMSE':<12} {'Degradation':<14}")
    print("-" * 48)
    for name in MODELS:
        degradation = (ood_metrics[name]["rmse"] - id_metrics[name]["rmse"]) / id_metrics[name]["rmse"] * 100
        print(f"{name:<10} {id_metrics[name]['rmse']:<12.6f} {ood_metrics[name]['rmse']:<12.6f} {degradation:<14.1f}%")

    plot_ood_robustness(
        id_metrics,
        ood_metrics,
        metric="rmse",
        save_path=os.path.join(args.output_dir, "ood_robustness.png"),
    )

    print("\n--- Extracting LNN Internal Dynamics ---")
    sample_x = next(iter(ood_test_loader))[0]

    cfc_model = trained_models["CfC"]
    cfc_dynamics = extract_cfc_dynamics(cfc_model, sample_x)
    plot_dynamics(
        cfc_dynamics,
        model_type="cfc",
        title="CfC Dynamics on OOD Data",
        save_path=os.path.join(args.output_dir, "cfc_ood_dynamics.png"),
    )

    ltc_model = trained_models["LTC"]
    ltc_dynamics = extract_ltc_dynamics(ltc_model, sample_x)
    plot_dynamics(
        ltc_dynamics,
        model_type="ltc",
        title="LTC Dynamics on OOD Data",
        save_path=os.path.join(args.output_dir, "ltc_ood_dynamics.png"),
    )

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
