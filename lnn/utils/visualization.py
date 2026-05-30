import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Ground Truth",
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Ground Truth", alpha=0.8)
    ax.plot(y_pred, label="Prediction", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    lrs: list[float] | None = None,
    title: str = "Training Curve",
    save_path: str | None = None,
):
    if lrs is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(train_losses, label="Train Loss", color="#2196F3")
    if val_losses is not None:
        ax1.plot(val_losses, label="Val Loss", color="#4CAF50")
    ax1.set_title(title)
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    if lrs is not None:
        ax2.plot(lrs, color="#FF9800", marker="o", markersize=3, linewidth=1)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "rmse",
    title: str | None = None,
    save_path: str | None = None,
):
    if title is None:
        title = f"Model Comparison ({metric.upper()})"

    models = list(results.keys())
    values = [results[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"][:len(models)])
    ax.set_title(title)
    ax.set_ylabel(metric.upper())

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
