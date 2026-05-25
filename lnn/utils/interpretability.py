import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork


@torch.no_grad()
def extract_cfc_dynamics(
    model: CfCNetwork, x: torch.Tensor
) -> dict[str, np.ndarray]:
    """
    Extract internal dynamics from a CfC network during forward pass.

    Returns gate values, decay rates, and hidden states for each time step,
    providing insight into how the network adapts its time constants.

    Returns:
        Dictionary with keys: 'gates', 'decays', 'hidden_states', 'outputs'
    """
    model.eval()
    batch_size, seq_len, _ = x.shape
    cell = model.cells[0]

    h = torch.zeros(batch_size, cell.hidden_size, device=x.device, dtype=x.dtype)
    gates = []
    decays = []
    hidden_states = [h.cpu().numpy()]
    outputs = []

    for t in range(seq_len):
        x_t = x[:, t, :]
        combined = torch.cat([x_t, h], dim=-1)

        f = cell.f_gate(combined)
        g = cell.g_branch(combined)
        h_out = cell.h_branch(combined)
        decay = torch.sigmoid(-f * cell.time_scale)

        h = decay * g + (1.0 - decay) * h_out

        gates.append(f.cpu().numpy())
        decays.append(decay.cpu().numpy())
        hidden_states.append(h.cpu().numpy())
        outputs.append(model.output_proj(h).cpu().numpy())

    return {
        "gates": np.stack(gates, axis=1),
        "decays": np.stack(decays, axis=1),
        "hidden_states": np.stack(hidden_states[:-1], axis=1),
        "outputs": np.stack(outputs, axis=1),
    }


@torch.no_grad()
def extract_ltc_dynamics(
    model: LTCNetwork, x: torch.Tensor
) -> dict[str, np.ndarray]:
    """
    Extract internal dynamics from an LTC network during forward pass.

    Returns effective time constants (tau_eff), bias contributions,
    and hidden states for each time step.

    Returns:
        Dictionary with keys: 'tau_eff', 'bias', 'hidden_states', 'outputs'
    """
    model.eval()
    batch_size, seq_len, _ = x.shape
    cell = model.cells[0]
    ode_func = cell.ode_func

    h = torch.zeros(batch_size, cell.hidden_size, device=x.device, dtype=x.dtype)
    tau_effs = []
    biases = []
    hidden_states = [h.cpu().numpy()]
    outputs = []

    for t in range(seq_len):
        x_t = x[:, t, :]
        combined = torch.cat([x_t, h], dim=-1)
        f_out = ode_func.f_tau(combined)
        bias = ode_func.f_bias(combined)
        tau_eff = torch.abs(ode_func.tau_base) + f_out + 0.01

        t_span = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
        from torchdiffeq import odeint

        h = odeint(lambda t, state: ode_func(t, state, x_t), h, t_span, method=cell.ode_method)[-1]

        tau_effs.append(tau_eff.cpu().numpy())
        biases.append(bias.cpu().numpy())
        hidden_states.append(h.cpu().numpy())
        outputs.append(model.output_proj(h).cpu().numpy())

    return {
        "tau_eff": np.stack(tau_effs, axis=1),
        "bias": np.stack(biases, axis=1),
        "hidden_states": np.stack(hidden_states[:-1], axis=1),
        "outputs": np.stack(outputs, axis=1),
    }


def plot_dynamics(
    dynamics: dict[str, np.ndarray],
    model_type: str = "cfc",
    title: str = "LNN Internal Dynamics",
    save_path: str | None = None,
    num_neurons: int = 5,
):
    """
    Visualize the internal dynamics of an LNN model.

    For CfC: plots gate values and decay rates over time
    For LTC: plots effective time constants and bias over time
    """
    if model_type == "cfc":
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        gates = dynamics["gates"][0, :, :num_neurons]
        decays = dynamics["decays"][0, :, :num_neurons]
        hidden = dynamics["hidden_states"][0, :, :num_neurons]

        for i in range(min(num_neurons, gates.shape[1])):
            axes[0].plot(gates[:, i], label=f"Neuron {i}", alpha=0.8)
            axes[1].plot(decays[:, i], label=f"Neuron {i}", alpha=0.8)
            axes[2].plot(hidden[:, i], label=f"Neuron {i}", alpha=0.8)

        axes[0].set_title(f"{title} - Gate Values (f)")
        axes[0].set_ylabel("Gate")
        axes[1].set_title("Decay Rates (σ(-f·τ))")
        axes[1].set_ylabel("Decay")
        axes[2].set_title("Hidden States")
        axes[2].set_ylabel("h(t)")
        axes[2].set_xlabel("Time Step")

    elif model_type == "ltc":
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        tau_eff = dynamics["tau_eff"][0, :, :num_neurons]
        bias = dynamics["bias"][0, :, :num_neurons]
        hidden = dynamics["hidden_states"][0, :, :num_neurons]

        for i in range(min(num_neurons, tau_eff.shape[1])):
            axes[0].plot(tau_eff[:, i], label=f"Neuron {i}", alpha=0.8)
            axes[1].plot(bias[:, i], label=f"Neuron {i}", alpha=0.8)
            axes[2].plot(hidden[:, i], label=f"Neuron {i}", alpha=0.8)

        axes[0].set_title(f"{title} - Effective Time Constants (τ_eff)")
        axes[0].set_ylabel("τ_eff")
        axes[1].set_title("Bias Contributions")
        axes[1].set_ylabel("Bias")
        axes[2].set_title("Hidden States")
        axes[2].set_ylabel("h(t)")
        axes[2].set_xlabel("Time Step")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ood_robustness(
    id_metrics: dict[str, dict[str, float]],
    ood_metrics: dict[str, dict[str, float]],
    metric: str = "rmse",
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Compare model performance on in-distribution vs out-of-distribution data.

    Shows how much each model's performance degrades under distribution shift.
    """
    if title is None:
        title = f"OOD Robustness: ID vs OOD ({metric.upper()})"

    models = list(id_metrics.keys())
    id_vals = [id_metrics[m][metric] for m in models]
    ood_vals = [ood_metrics[m][metric] for m in models]
    degradation = [(ood - id_) / id_ * 100 for id_, ood in zip(id_vals, ood_vals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width / 2, id_vals, width, label="In-Distribution", color="#4CAF50")
    ax1.bar(x + width / 2, ood_vals, width, label="Out-of-Distribution", color="#F44336")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f"{metric.upper()}: ID vs OOD")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    colors = ["#4CAF50" if d < 50 else "#FF9800" if d < 100 else "#F44336" for d in degradation]
    bars = ax2.bar(models, degradation, color=colors)
    ax2.set_ylabel("Performance Degradation (%)")
    ax2.set_title("OOD Degradation (lower = more robust)")
    for bar, val in zip(bars, degradation):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_concept_drift_adaptation(
    predictions: dict[str, np.ndarray],
    ground_truth: np.ndarray,
    drift_point: int,
    title: str = "Concept Drift Adaptation",
    save_path: str | None = None,
):
    """
    Visualize how different models adapt to concept drift.

    Shows predictions before and after the drift point, with a vertical
    line marking the regime change.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ground_truth, label="Ground Truth", alpha=0.5, color="black", linewidth=0.8)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for idx, (name, pred) in enumerate(predictions.items()):
        ax.plot(pred, label=name, alpha=0.8, color=colors[idx % len(colors)], linewidth=1.0)

    ax.axvline(x=drift_point, color="red", linestyle="--", linewidth=2, label="Concept Drift")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
