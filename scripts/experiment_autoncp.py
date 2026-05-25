"""
ncps AutoNCP Experiment: Sparse Neural Circuit Wiring.

Core thesis: AutoNCP discovers biologically-inspired sparse connectivity
patterns that match C. elegans neural circuit structure:
    Sensory → Inter → Command → Motor neurons

This experiment:
1. Uses ncps AutoNCP to automatically wire CfC/LTC networks
2. Compares sparse vs dense connectivity
3. Visualizes the discovered wiring diagram
4. Analyzes parameter efficiency vs performance

Usage:
    python scripts/experiment_autoncp.py --epochs 80
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.trainer import Trainer
from lnn.data.timeseries import create_dataloader, generate_mackey_glass
from lnn.ncps_integration.ncps_models import NCPSAutoNCP, NCPSCfC
from lnn.utils.metrics import compute_metrics
from lnn.utils.visualization import plot_predictions, plot_training_curve


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


def main():
    parser = argparse.ArgumentParser(description="AutoNCP Sparse Wiring Experiment")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="analysis/autoncp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("AutoNCP Sparse Neural Circuit Experiment")
    print("=" * 60)

    raw_data = generate_mackey_glass(num_samples=2000, tau=17)
    split_train = int(len(raw_data) * 0.7)
    split_val = int(len(raw_data) * 0.85)
    train_data = raw_data[:split_train]
    val_data = raw_data[split_train:split_val]
    test_data = raw_data[split_val:]

    train_loader = create_dataloader(
        train_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        test_data, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )

    input_size = 1
    output_size = 1

    models = {
        "AutoNCP-CfC": NCPSAutoNCP(input_size, args.hidden_size, output_size, model_type="cfc"),
        "Dense-CfC": NCPSCfC(input_size, args.hidden_size, output_size),
        "LSTM": LSTMModel(input_size, args.hidden_size, output_size),
    }

    all_metrics = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        trainer = Trainer(model, lr=args.lr, patience=15)
        history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

        preds, targets = trainer.predict(test_loader)
        metrics = compute_metrics(targets, preds)
        metrics["params"] = param_count
        all_metrics[name] = metrics

        print(f"  RMSE: {metrics['rmse']:.6f} | MAE: {metrics['mae']:.6f}")

        plot_training_curve(
            history["train_losses"],
            history["val_losses"],
            title=f"{name} Training Curve",
            save_path=os.path.join(args.output_dir, f"{name.replace('-', '_')}_training.png"),
        )

        plot_predictions(
            targets.numpy().flatten()[:200],
            preds.numpy().flatten()[:200],
            title=f"{name} Prediction",
            save_path=os.path.join(args.output_dir, f"{name.replace('-', '_')}_prediction.png"),
        )

    print("\n" + "=" * 60)
    print("AutoNCP RESULTS")
    print("=" * 60)
    print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'Params':<10} {'Efficiency':<12}")
    print("-" * 61)
    for name, m in all_metrics.items():
        efficiency = 1.0 / (m["rmse"] * m["params"]) * 1e4
        print(f"{name:<15} {m['rmse']:<12.6f} {m['mae']:<12.6f} {m['params']:<10,} {efficiency:<12.4f}")

    try:
        autoncp_model = models["AutoNCP-CfC"]
        wiring = autoncp_model.rnn.wiring
        print("\n--- AutoNCP Wiring Structure ---")
        print(f"  Sensory neurons: {wiring.sensory_neurons}")
        print(f"  Inter neurons:   {wiring.inter_neurons}")
        print(f"  Command neurons: {wiring.command_neurons}")
        print(f"  Motor neurons:   {wiring.motor_neurons}")
        print(f"  Total connections: {sum(wiring.get_connections()[0].shape[0] for _ in [1])}")

        try:
            import matplotlib.pyplot as plt

            adj_matrix = wiring.get_adjacency_matrix()
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(adj_matrix, cmap="Blues", aspect="auto")
            ax.set_title("AutoNCP Wiring: Adjacency Matrix")
            ax.set_xlabel("Target Neuron")
            ax.set_ylabel("Source Neuron")
            plt.colorbar(cax)
            plt.tight_layout()
            fig.savefig(os.path.join(args.output_dir, "autoncp_wiring.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Wiring diagram saved to {args.output_dir}/autoncp_wiring.png")
        except Exception as e:
            print(f"  Could not plot wiring: {e}")
    except Exception as e:
        print(f"  Could not extract wiring info: {e}")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
