#!/usr/bin/env python3
"""
Simple Demo: LNN Models and Their Key Advantages.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.variants import (
    CfCDTNetwork,
    EulerLTCDTNetwork,
)


class TraditionalGRU(nn.Module):
    """Traditional GRU for comparison."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output)


def main():
    print("\n" + "="*80)
    print("LNN - KEY ADVANTAGES OVER LARGE MODELS")
    print("="*80)
    
    # Setup
    input_size = 1
    hidden_size = 32
    output_size = 1
    batch_size = 32
    seq_len = 100
    
    # Create models
    models = {
        "CfC": CfCNetwork(input_size, hidden_size, output_size),
        "LTC": LTCNetwork(input_size, hidden_size, output_size),
        "CfC-DT": CfCDTNetwork(input_size, hidden_size, output_size),
        "Euler-LTC-DT": EulerLTCDTNetwork(input_size, hidden_size, output_size),
        "GRU (Baseline)": TraditionalGRU(input_size, hidden_size, output_size),
    }
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("\n" + "="*80)
    print("1. 🏆 PARAMETER EFFICIENCY (vs Large Models)")
    print("="*80)
    
    print("\nModel                Parameters    Relative Size")
    print("-" * 60)
    
    gru_params = sum(p.numel() for p in models["GRU (Baseline)"].parameters())
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        relative = num_params / gru_params
        
        print(f"{name:20s} {num_params:10,d}      {relative:6.2f}x")
    
    print("\n💡 Compare to GPT-2: ~124M parameters (40,000x larger!)")
    print("💡 Compare to LLaMA-7B: ~7B parameters (2,000,000x larger!)")
    
    print("\n" + "="*80)
    print("2. ⚡ SPEED ADVANTAGE")
    print("="*80)
    
    print("\nModel                Forward Time (ms)    Relative Speed")
    print("-" * 70)
    
    # Warmup
    for _ in range(10):
        _ = models["CfC"](x)
    
    times = {}
    for name, model in models.items():
        # Measure time
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            _ = model(x)
        elapsed = (time.time() - start) / n_runs * 1000
        times[name] = elapsed
    
    gru_time = times["GRU (Baseline)"]
    
    for name in models.keys():
        rel_speed = gru_time / times[name]
        print(f"{name:20s} {times[name]:15.2f} ms            {rel_speed:5.2f}x")
    
    print("\n" + "="*80)
    print("3. 📊 HANDLING IRREGULAR TIME (Real-World Data Advantage)")
    print("="*80)
    
    print("\nTraditional RNN/GRU: Assumes uniform time steps")
    print("LNN (CfC-DT): Supports explicit, irregular time steps\n")
    
    # Demonstrate CfC-DT
    model_dt = CfCDTNetwork(input_size, hidden_size, output_size)
    
    # Create data with missing/irregular time steps
    x_irregular = torch.randn(2, 10, input_size)
    dt = torch.tensor([
        [0.1, 0.5, 0.2, 2.0, 0.3, 0.1, 1.5, 0.4, 0.6, 0.2],
        [0.3, 0.1, 1.0, 0.2, 0.5, 0.8, 0.1, 0.3, 2.5, 0.2],
    ]).unsqueeze(-1)
    
    output = model_dt(x_irregular, dt=dt)
    
    print(f"✓ CfC-DT processes irregular time steps successfully!")
    print(f"  Input shape: {x_irregular.shape}")
    print(f"  dt shape:    {dt.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("\n" + "="*80)
    print("4. 🎯 COMPARISON SUMMARY")
    print("="*80)
    
    comparison = [
        {
            "aspect": "Parameters",
            "LNN": "Few thousand (32-128 hidden units)",
            "Large Models": "Millions - Billions",
            "winner": "LNN"
        },
        {
            "aspect": "Inference Speed",
            "LNN": "Very fast (O(N) time)",
            "Large Models": "Slow (O(N²) for attention)",
            "winner": "LNN"
        },
        {
            "aspect": "Training Data",
            "LNN": "Small datasets sufficient",
            "Large Models": "Needs massive datasets",
            "winner": "LNN"
        },
        {
            "aspect": "Edge Deployment",
            "LNN": "Runs on microcontrollers",
            "Large Models": "Needs GPUs/TPUs",
            "winner": "LNN"
        },
        {
            "aspect": "Continuous Time",
            "LNN": "Natively supports it",
            "Large Models": "Discrete-only",
            "winner": "LNN"
        },
        {
            "aspect": "Language Understanding",
            "LNN": "Limited",
            "Large Models": "Excellent",
            "winner": "Large Models"
        },
    ]
    
    print("\n" + "-"*100)
    print(f"{'Aspect':<25} {'LNN':<35} {'Large Models':<30} {'Winner':<10}")
    print("-"*100)
    
    for comp in comparison:
        winner_emoji = "✅" if comp["winner"] == "LNN" else "🤖"
        print(f"{comp['aspect']:<25} {comp['LNN']:<35} {comp['Large Models']:<30} {winner_emoji} {comp['winner']}")
    
    print("\n" + "="*80)
    print("5. 🚀 USE CASE GUIDELINES")
    print("="*80)
    
    print("\n✅ USE LNN FOR:")
    print("  • Time series prediction & forecasting")
    print("  • Robotics control & autonomous systems")
    print("  • IoT sensor data processing")
    print("  • Real-time edge applications")
    print("  • Physical system modeling")
    print("  • Energy-efficient AI")
    
    print("\n✅ USE LARGE MODELS FOR:")
    print("  • Natural language understanding & generation")
    print("  • Complex pattern recognition (images, audio)")
    print("  • Knowledge-intensive tasks")
    print("  • Creative generation & content creation")
    print("  • General intelligence tasks")
    
    print("\n" + "="*80)
    print("6. 👨‍💻 CODE EXAMPLE - USING LNN")
    print("="*80)
    
    code_example = '''
# Simple usage of LNN models
import torch
from lnn.core.cfc import CfCNetwork

# Create model (just 3,329 parameters!)
model = CfCNetwork(
    input_size=1,      # 1 sensor input
    hidden_size=32,    # 32 liquid units
    output_size=1,     # Predict 1 value
    num_layers=1       # Single layer is often enough!
)

# Create data
x = torch.randn(32, 100, 1)  # batch, seq_len, features

# Forward pass
output = model(x)  # Shape: (32, 100, 1)

# Done! That's it!
'''
    print(code_example)
    
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. See verify_all_models.py - Test all implementations")
    print("  2. See experiment_all_variants.py - Train and compare all")
    print("  3. Read docs/OPTIMIZATION_STRATEGIES.md - Optimization guide")
    print("  4. Read docs/IMPLEMENTATION_SUMMARY.md - Implementation overview")


if __name__ == "__main__":
    main()
