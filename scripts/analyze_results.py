"""
Analyze and compare the results from different models.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    output_dir = "analysis/real_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all results
    results_files = [
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.endswith("_results.csv")
    ]
    
    if not results_files:
        print("No results files found!")
        return
    
    # Load and combine results
    dfs = []
    for f in results_files:
        dfs.append(pd.read_csv(f))
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values("mse")
    
    print("=" * 80)
    print("LIQUID NEURAL NETWORKS - PERFORMANCE COMPARISON")
    print("=" * 80)
    print("\nCombined Results:")
    print(combined_df.to_string(index=False))
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LNN vs GRU Performance Comparison", fontsize=16, fontweight="bold")
    
    models = combined_df["model"].values
    x = np.arange(len(models))
    width = 0.6
    
    # MSE
    ax = axes[0, 0]
    bars = ax.bar(x, combined_df["mse"], width, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xlabel("Model")
    ax.set_ylabel("MSE (lower is better)")
    ax.set_title("Mean Squared Error")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, combined_df["mse"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(combined_df["mse"])*0.01, 
               f"{val:.6f}", ha="center", va="bottom")
    
    # RMSE
    ax = axes[0, 1]
    bars = ax.bar(x, combined_df["rmse"], width, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Root Mean Squared Error")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, combined_df["rmse"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(combined_df["rmse"])*0.01, 
               f"{val:.4f}", ha="center", va="bottom")
    
    # Training Time
    ax = axes[1, 0]
    bars = ax.bar(x, combined_df["train_time"], width, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xlabel("Model")
    ax.set_ylabel("Training Time (seconds, lower is better)")
    ax.set_title("Training Time")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, combined_df["train_time"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(combined_df["train_time"])*0.01, 
               f"{val:.2f}s", ha="center", va="bottom")
    
    # Inference Speed
    ax = axes[1, 1]
    bars = ax.bar(x, combined_df["samples_per_sec"], width, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xlabel("Model")
    ax.set_ylabel("Samples per Second (higher is better)")
    ax.set_title("Inference Speed")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, combined_df["samples_per_sec"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(combined_df["samples_per_sec"])*0.01, 
               f"{val:.0f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {plot_path}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY & INSIGHTS")
    print("=" * 80)
    
    best_mse = combined_df.iloc[0]
    print(f"\n1. Best Model by MSE: {best_mse['model'].upper()}")
    print(f"   - MSE: {best_mse['mse']:.6f}")
    print(f"   - RMSE: {best_mse['rmse']:.4f}")
    
    fastest_train = combined_df.loc[combined_df["train_time"].idxmin()]
    print(f"\n2. Fastest Training: {fastest_train['model'].upper()}")
    print(f"   - Time: {fastest_train['train_time']:.2f} seconds")
    
    fastest_infer = combined_df.loc[combined_df["samples_per_sec"].idxmax()]
    print(f"\n3. Fastest Inference: {fastest_infer['model'].upper()}")
    print(f"   - Speed: {fastest_infer['samples_per_sec']:.0f} samples/second")
    
    # Save combined results
    combined_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined results saved to {combined_path}")
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("\n- CfC (Closed-form Continuous-time) networks offer an excellent balance")
    print("  between prediction accuracy, training speed, and inference speed.")
    print("\n- LTC (Liquid Time-Constant) networks, while accurate, are slower due to")
    print("  their reliance on ODE solvers.")
    print("\n- GRU, being a traditional RNN variant, is fast but may struggle with")
    print("  certain types of time-series patterns compared to continuous-time models.")
    print("=" * 80)

if __name__ == "__main__":
    main()
