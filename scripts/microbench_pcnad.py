"""Microbenchmark: parallel-EMA vs streaming-EMA inference latency for CfC-NAD.

Yesterday's 2026-06-01 benchmark recorded a ~25% CPU latency overhead vs vanilla
CfC. That overhead was dominated by per-step Python tensor ops in the streaming
noise EMA loop. This microbenchmark times forward-only inference for the two
paths on identical inputs and weights, and writes a JSON snapshot under
``analysis/cfc_nad/``.

Usage:
    python scripts/microbench_pcnad.py --batch 32 --seq 64 --hidden 16 --repeat 200
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import torch

from lnn.core.cfc import CfCNetwork
from lnn.core.noise_adaptive_cfc import NoiseAdaptiveCfCNetwork


@torch.inference_mode()
def time_forward(
    model: torch.nn.Module, x: torch.Tensor, repeat: int, mask: torch.Tensor | None = None
) -> tuple[float, float]:
    """Return (mean ms per forward, std ms per forward)."""

    # Warmup
    for _ in range(5):
        _ = model(x, mask=mask)
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = model(x, mask=mask)
        samples.append((time.perf_counter() - t0) * 1e3)
    tensor = torch.tensor(samples)
    return float(tensor.mean()), float(tensor.std())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--output", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.seq, args.input_size)
    ones_mask = torch.ones(args.batch, args.seq)

    cfc = CfCNetwork(
        input_size=args.input_size,
        hidden_size=args.hidden,
        output_size=args.output,
        num_layers=args.num_layers,
        return_sequences=False,
    )
    nad = NoiseAdaptiveCfCNetwork(
        input_size=args.input_size,
        hidden_size=args.hidden,
        output_size=args.output,
        num_layers=args.num_layers,
        return_sequences=False,
    )

    cfc.eval()
    nad.eval()

    print(
        f"# microbenchmark batch={args.batch} seq={args.seq} input={args.input_size} "
        f"hidden={args.hidden} layers={args.num_layers} repeat={args.repeat}"
    )

    cfc_mean, cfc_std = time_forward(cfc, x, args.repeat)
    nad_par_mean, nad_par_std = time_forward(nad, x, args.repeat, mask=None)
    nad_stream_mean, nad_stream_std = time_forward(nad, x, args.repeat, mask=ones_mask)

    print(f"  CfC               : {cfc_mean:7.3f} ms (±{cfc_std:.3f})")
    print(f"  CfC-NAD parallel  : {nad_par_mean:7.3f} ms (±{nad_par_std:.3f})  "
          f"vs CfC: {(nad_par_mean / cfc_mean - 1) * 100:+.1f}%")
    print(f"  CfC-NAD streaming : {nad_stream_mean:7.3f} ms (±{nad_stream_std:.3f})  "
          f"vs CfC: {(nad_stream_mean / cfc_mean - 1) * 100:+.1f}%")
    speedup = (nad_stream_mean - nad_par_mean) / nad_stream_mean * 100
    print(f"  parallel speedup vs streaming: {speedup:+.1f}%")

    payload = {
        "config": vars(args),
        "results_ms": {
            "cfc": {"mean": cfc_mean, "std": cfc_std},
            "cfc_nad_parallel": {"mean": nad_par_mean, "std": nad_par_std},
            "cfc_nad_streaming": {"mean": nad_stream_mean, "std": nad_stream_std},
        },
        "summary": {
            "parallel_overhead_vs_cfc_pct": (nad_par_mean / cfc_mean - 1) * 100,
            "streaming_overhead_vs_cfc_pct": (nad_stream_mean / cfc_mean - 1) * 100,
            "parallel_speedup_vs_streaming_pct": speedup,
        },
    }
    out_dir = Path("analysis/cfc_nad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_pcnad_microbench.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
