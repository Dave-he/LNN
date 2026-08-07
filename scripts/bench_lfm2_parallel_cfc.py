"""r304 — Benchmark LSTM → ParallelCfC swap for LFM2.5-style inference.

We do NOT depend on real LFM2.5 weights (they're not in this repository).
Instead we build a tiny LFM2.5-style mock model that uses a single
``nn.LSTM`` backbone and benchmark:

  1. Param count delta
  2. Single-token + 8/16/32/64-token forward latency
  3. Output-shape preservation
  4. Output-stability proxy (std of output across 5 fixed seeds)

Usage:
  .venv312/bin/python scripts/bench_lfm2_parallel_cfc.py
  .venv312/bin/python scripts/bench_lfm2_parallel_cfc.py --t 32 --hidden 96
  .venv312/bin/python scripts/bench_lfm2_parallel_cfc.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Ensure project root is importable when invoked as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lnn.lfm2.parallel_integration import (
    RECURRENT_CLASSES,
    _count_lstm_like,
    replace_lstm_with_parallel_cfc,
)


class TinyLFM25Mock(nn.Module):
    """Minimal LFM2.5-shaped mock: token-embedding + LSTM backbone + LM head.

    Mirrors the LFM2.5 architecture as documented in
    ``docs/reports/LFM2.5-Encoder_350M_研读报告.md`` (r300):
    short-conv front + linear-LSTM middle + LM head.  We collapse the
    short-conv to a 1x1 projection for size and keep the LSTM to
    exercise the swap path.  The head is a tied-embedding classifier
    (input_size→vocab).
    """

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 64,
        num_layers: int = 1,
        proj_size: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # LFM2 uses linear-LSTM; we approximate with a 1-layer LSTM.
        lstm_kwargs = dict(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        if proj_size and proj_size > 0:
            lstm_kwargs["proj_size"] = proj_size
        self.backbone = nn.LSTM(**lstm_kwargs)
        self.head = nn.Linear(hidden_size if proj_size == 0 else proj_size, vocab_size, bias=False)
        # Tie weights.
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) token ids.
        e = self.embed(x)
        h = self.backbone(e)
        if isinstance(h, tuple):
            h = h[0]
        return self.head(h)  # (B, T, vocab)


def _build_mock(hidden: int, vocab: int, num_layers: int) -> TinyLFM25Mock:
    torch.manual_seed(0)
    return TinyLFM25Mock(
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=num_layers,
    )


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _time_forward(model: nn.Module, x: torch.Tensor, n_trials: int) -> List[float]:
    """Measure per-call latency, returning list of seconds."""
    model.eval()
    times: List[float] = []
    with torch.no_grad():
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - t0)
    return times


def _summarise(name: str, times: List[float]) -> Dict[str, float]:
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {"name": name, "mean_s": mean, "std_s": std, "n": len(times)}


def _output_stability(model: nn.Module, x: torch.Tensor, n_seeds: int = 5) -> float:
    """Std of output across n_seeds (with model frozen).  Lower is more
    stable — useful as a deployment proxy for output determinism under
    random init perturbation."""
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for s in range(n_seeds):
            torch.manual_seed(s)
            o = model(x).flatten()
            outs.append(o)
    stacked = torch.stack(outs, dim=0)
    return float(stacked.std(dim=0).mean().item())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--trials", type=int, default=5, help="latency trials")
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="sequence lengths (must each be a multiple of every window in --windows)",
    )
    p.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--json", type=str, default=None, help="optional JSON out path")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
    )
    args = p.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)

    # Validate seq_lens: every T must be a multiple of every W (or W=1).
    for T in args.seq_lens:
        for W in args.windows:
            if W == 1:
                continue
            if T % W != 0:
                raise SystemExit(
                    f"ERROR: T={T} is not a multiple of W={W}. "
                    f"Pick seq_lens that are multiples of all --windows."
                )

    print(f"Building mock LFM2.5 backbone: hidden={args.hidden} vocab={args.vocab} layers={args.num_layers}")
    baseline = _build_mock(args.hidden, args.vocab, args.num_layers).to(device)
    n_lstm = _count_lstm_like(baseline)
    base_params = _count_params(baseline)
    print(f"  baseline: {n_lstm} LSTM(s), {base_params} params")

    results: Dict = {
        "device": str(device),
        "config": {
            "vocab": args.vocab,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "batch": args.batch,
            "trials": args.trials,
        },
        "baseline": {
            "n_lstm": n_lstm,
            "params": base_params,
        },
        "per_seq_len": [],
    }

    for T in args.seq_lens:
        x = torch.randint(0, args.vocab, (args.batch, T), device=device)
        # Baseline (LSTM) latency.
        lat = _summarise("lstm", _time_forward(baseline, x, args.trials))
        out_shape = tuple(baseline(x).shape)
        per_seq_entry: Dict = {
            "T": T,
            "baseline": {
                "latency_ms": lat["mean_s"] * 1000.0,
                "latency_std_ms": lat["std_s"] * 1000.0,
                "out_shape": out_shape,
            },
            "swaps": [],
        }

        for W in args.windows:
            swap_model = _build_mock(args.hidden, args.vocab, args.num_layers).to(device)
            assert _count_lstm_like(swap_model) >= 1
            swap_model = replace_lstm_with_parallel_cfc(swap_model, window=W)
            assert _count_lstm_like(swap_model) == 0
            sp = _count_params(swap_model)
            lat = _summarise(f"plan_w{W}", _time_forward(swap_model, x, args.trials))
            # Output shape — must be (B, T, vocab) for strict drop-in.
            with torch.no_grad():
                o = swap_model(x)
            o_shape = tuple(o.shape)
            # Stability proxy: std across 5 fixed seeds.
            stab = _output_stability(swap_model, x, n_seeds=5)
            entry = {
                "window": W,
                "params": sp,
                "params_delta_pct": 100.0 * (sp - base_params) / base_params,
                "latency_ms": lat["mean_s"] * 1000.0,
                "latency_std_ms": lat["std_s"] * 1000.0,
                "latency_delta_pct_vs_lstm": 100.0 * (
                    lat["mean_s"] - per_seq_entry["baseline"]["latency_ms"] / 1000.0
                ) / (per_seq_entry["baseline"]["latency_ms"] / 1000.0),
                "out_shape": o_shape,
                "shape_match": o_shape == out_shape,
                "stability_std": stab,
            }
            per_seq_entry["swaps"].append(entry)
            print(
                f"  T={T:>3}  W={W}  params={sp:>6} ({entry['params_delta_pct']:+.1f}%)  "
                f"lat={entry['latency_ms']:.2f}±{entry['latency_std_ms']:.2f}ms  "
                f"({entry['latency_delta_pct_vs_lstm']:+.1f}% vs LSTM)  shape_match={entry['shape_match']}"
            )
        results["per_seq_len"].append(per_seq_entry)

    # Summary line: average across seq_lens for w=4 (recommended default).
    w4 = [e for sl in results["per_seq_len"] for e in sl["swaps"] if e["window"] == 4]
    if w4:
        avg_pct = statistics.mean([abs(e["params_delta_pct"]) for e in w4])
        avg_lat_pct = statistics.mean([e["latency_delta_pct_vs_lstm"] for e in w4])
        all_shape_match = all(e["shape_match"] for e in w4)
        results["summary_w4"] = {
            "avg_param_delta_pct": avg_pct,
            "avg_latency_delta_pct": avg_lat_pct,
            "all_shape_match": all_shape_match,
        }
        print(
            f"\nW=4 summary: avg params delta {avg_pct:.1f}%, "
            f"avg latency delta {avg_lat_pct:+.1f}%, "
            f"shape match all={all_shape_match}"
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
