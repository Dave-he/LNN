#!/usr/bin/env python3
"""End-to-end decision-loop latency harness for LNN backbones.

This is the Phase A deliverable of PRD-MSD-LNN: it complements
``scripts/jetson_lnn_benchmark.py`` (which only emits aggregate throughput)
by emitting **per-step p50 / p99 / p999 latency** histograms.

Latency is measured across the four stages of a real decision step:

    obs_prep  ->  forward (state-carry)  ->  act_post

The script sweeps four backbones (CfCStyle, LTC, PDNAPulse, GRU) over
hidden sizes, sequence lengths, and three precisions (FP32, FP16, INT8),
on the available hardware (Jetson CPU default, AGX CUDA optional,
Apple Silicon MPS optional).

Typical usage (Jetson CPU):

    python scripts/bench_decision_e2e_latency.py --hardware jetson-cpu --pareto --seeds 3

Output:
    - analysis/decisions/<timestamp>_decision_e2e_latency.json
    - analysis/decisions/<timestamp>_decision_e2e_latency.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.cfc import CfCCell, PDNAPulseHead  # noqa: E402
from lnn.core.ltc import LTCNetwork  # noqa: E402

try:
    from lnn.core.quantization import quantize_model_inplace
except Exception:  # pragma: no cover - import-tolerant
    quantize_model_inplace = None  # type: ignore


# ---------------------------------------------------------------------------
# Backbones (deliberately reuse the in-house cell primitives — no external
# LNN). Each backbone exposes a ``step(x_t, h, dt) -> (out, h_new)`` API so
# we can time the same decision step uniformly.
# ---------------------------------------------------------------------------


class _BackboneBase(nn.Module):
    """Common interface: ``step(x_t, h, dt) -> (out, h_new)``."""

    def step(self, x_t: torch.Tensor, h: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError


class CfCStyleBackbone(_BackboneBase):
    """In-house CfC closed-form cell (matches CfCStyleModel in jetson bench)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = CfCCell(1, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def step(self, x_t, h, dt):
        h_new = self.cell(x_t, h, dt)
        return self.readout(h_new), h_new


class LTCBackbone(_BackboneBase):
    """ODE-solver-based LTC (rk4, same as jetson_benchmark)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = LTCNetwork(
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=1,
            ode_method="rk4",
            return_sequences=False,
        )
        self.hidden_size = hidden_size

    def step(self, x_t, h, dt):
        # The in-house LTCNetwork runs the full sequence at once; emulate a
        # single-step pass through its inner cell for fair per-step timing.
        # ``LTCNetwork.cells[0]`` mirrors CfCCell's interface.
        cell = self.network.cells[0]
        h_new = cell(x_t, h, dt)
        out = self.network.output_proj(h_new.unsqueeze(1)).squeeze(1)
        return out, h_new


class PDNAPulseBackbone(_BackboneBase):
    """CfC + PDNA pulse modulation (iter#19 augmentation head)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = CfCCell(1, hidden_size)
        self.pulse = PDNAPulseHead(hidden_size, use_self_attend=False)
        self.readout = nn.Linear(hidden_size, 1)

    def step(self, x_t, h, dt):
        h_new = self.cell(x_t, h, dt)
        # Pulse head expects [B, T, H] — pass single-step, then read.
        h_aug = self.pulse(h_new.unsqueeze(1)).squeeze(1)
        return self.readout(h_aug), h_new


class GRUBackbone(_BackboneBase):
    """Vanilla GRU baseline (already known fastest throughput)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(1, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def step(self, x_t, h, dt):
        h_new = self.gru_cell(x_t, h)
        return self.readout(h_new), h_new


BACKBONES: dict[str, type[_BackboneBase]] = {
    "cfcs": CfCStyleBackbone,
    "ltc": LTCBackbone,
    "pdn": PDNAPulseBackbone,
    "gru": GRUBackbone,
}


# ---------------------------------------------------------------------------
# Latency harness
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    backbone: str
    hidden_size: int
    seq_len: int
    precision: str
    seed: int
    n_steps: int
    p50_ms: float
    p99_ms: float
    p999_ms: float
    mean_ms: float
    std_ms: float
    steps_per_sec: float
    obs_prep_p50_ms: float
    forward_p50_ms: float
    state_carry_p50_ms: float
    act_post_p50_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _resolve_device(hardware: str) -> torch.device:
    if hardware == "jetson-cpu":
        return torch.device("cpu")
    if hardware == "agx-gpu":
        if not torch.cuda.is_available():
            print("[warn] CUDA unavailable, falling back to CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    if hardware == "ios-mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            print("[warn] MPS unavailable, falling back to CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("mps")
    raise ValueError(f"unknown hardware: {hardware}")


def _apply_precision(model: nn.Module, precision: str) -> nn.Module:
    if precision == "fp32":
        return model
    if precision == "fp16":
        return model.to(torch.float16)
    if precision == "int8":
        if quantize_model_inplace is None:
            print("[warn] quantization unavailable, skipping int8", file=sys.stderr)
            return model
        # weight-only INT8 simulation (already verified at ±0.0001 MSE delta in
        # docs/reports/Int8_Quantization_N20_DLNet_Stage3_2026-08-05.md)
        quantize_model_inplace(model, per_channel=True)
        return model
    raise ValueError(f"unknown precision: {precision}")


def _time_one_step(model: nn.Module, x_t: torch.Tensor, h0: torch.Tensor, dt: float) -> tuple[float, float, float, float, torch.Tensor]:
    """Time the four phases of one decision step."""
    # obs_prep (already prepared by caller; we keep the slot to mirror real env)
    t0 = time.perf_counter()
    obs_t = x_t
    t_obs = (time.perf_counter() - t0) * 1000.0

    # forward (single-step recurrent pass)
    t0 = time.perf_counter()
    with torch.no_grad():
        out, h_new = model.step(obs_t, h0, dt)
    t_fwd = (time.perf_counter() - t0) * 1000.0

    # state_carry (clone to mimic what we'd hand to the next env.step)
    t0 = time.perf_counter()
    h_new.detach().clone()  # intentional state-carry work
    t_state = (time.perf_counter() - t0) * 1000.0

    # act_post (cast + clamp; this is the action layer of a PPO policy)
    t0 = time.perf_counter()
    action = out.detach().clamp(-1.0, 1.0)
    t_act = (time.perf_counter() - t0) * 1000.0

    return t_obs, t_fwd, t_state, t_act, action


def run_one_config(
    backbone: str,
    hidden_size: int,
    seq_len: int,
    precision: str,
    seed: int,
    device: torch.device,
    warmup: int = 20,
    measure: int = 200,
) -> LatencyResult:
    """Time a single backbone config; returns per-step histogram summary."""
    torch.manual_seed(seed)
    cls = BACKBONES[backbone]
    model = cls(hidden_size).to(device)
    model = _apply_precision(model, precision)
    model.eval()

    batch = 1  # decision loop is single-env
    h0 = torch.zeros(batch, hidden_size, device=device)
    dt = 1.0 / max(seq_len, 1)

    # Warmup (excluded from timing)
    x_t = torch.randn(batch, 1, device=device)
    h = h0
    for _ in range(warmup):
        _, h = model.step(x_t, h, dt)
        h = h.detach()
    # Re-init hidden after warmup to match real PPO loop semantics
    h = h0.clone()

    # Measure
    obs_t_list: list[float] = []
    fwd_t_list: list[float] = []
    state_t_list: list[float] = []
    act_t_list: list[float] = []
    total_t_list: list[float] = []

    for step_idx in range(measure):
        x_t = torch.randn(batch, 1, device=device) * (1.0 + step_idx * 0.001)
        t_obs, t_fwd, t_state, t_act, _ = _time_one_step(model, x_t, h, dt)
        obs_t_list.append(t_obs)
        fwd_t_list.append(t_fwd)
        state_t_list.append(t_state)
        act_t_list.append(t_act)
        total_t_list.append(t_obs + t_fwd + t_state + t_act)
        # carry state forward (mirror real env loop)
        h = model.step(x_t, h, dt)[1].detach()

    def pct(name: str, vals: list[float], p: float) -> float:
        return _percentile(sorted(vals), p)

    mean_ms = statistics.fmean(total_t_list)
    std_ms = statistics.pstdev(total_t_list) if len(total_t_list) > 1 else 0.0
    p50 = pct("p50", total_t_list, 50.0)
    p99 = pct("p99", total_t_list, 99.0)
    p999 = pct("p999", total_t_list, 99.9)
    sps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    return LatencyResult(
        backbone=backbone,
        hidden_size=hidden_size,
        seq_len=seq_len,
        precision=precision,
        seed=seed,
        n_steps=measure,
        p50_ms=p50,
        p99_ms=p99,
        p999_ms=p999,
        mean_ms=mean_ms,
        std_ms=std_ms,
        steps_per_sec=sps,
        obs_prep_p50_ms=pct("obs", obs_t_list, 50.0),
        forward_p50_ms=pct("fwd", fwd_t_list, 50.0),
        state_carry_p50_ms=pct("state", state_t_list, 50.0),
        act_post_p50_ms=pct("act", act_t_list, 50.0),
    )


def pareto_sweep(
    backbones: list[str],
    hidden_sizes: list[int],
    seq_lens: list[int],
    precisions: list[str],
    seeds: list[int],
    hardware: str,
) -> list[LatencyResult]:
    device = _resolve_device(hardware)
    results: list[LatencyResult] = []
    for bb in backbones:
        for h in hidden_sizes:
            for t in seq_lens:
                for p in precisions:
                    for s in seeds:
                        try:
                            r = run_one_config(bb, h, t, p, s, device)
                            results.append(r)
                            print(
                                f"[{bb:5s} h={h:2d} T={t:2d} {p:4s} s={s}] "
                                f"p50={r.p50_ms:.3f}ms p99={r.p99_ms:.3f}ms "
                                f"sps={r.steps_per_sec:.0f}",
                                flush=True,
                            )
                        except Exception as e:  # pragma: no cover - tolerant
                            print(f"[skip] {bb} h={h} T={t} {p} s={s}: {e}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def write_outputs(results: list[LatencyResult], hardware: str) -> tuple[Path, Path]:
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    json_path = out_dir / f"{stamp}_decision_e2e_latency.json"
    md_path = out_dir / f"{stamp}_decision_e2e_latency.md"

    json_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))

    lines = [f"# Decision-loop end-to-end latency ({hardware}) — {stamp}", ""]
    lines.append(f"Runs: {len(results)} configs. (Phase A of PRD-MSD-LNN.)")
    lines.append("")
    lines.append("Per-step timing split into 4 phases: `obs_prep | forward | state_carry | act_post`.")
    lines.append("All numbers in milliseconds unless noted.")
    lines.append("")
    lines.append("| backbone | h | T | precision | seed | p50 | p99 | p999 | mean | std | sps | obs | fwd | state | act |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.backbone} | {r.hidden_size} | {r.seq_len} | {r.precision} | {r.seed} | "
            f"{r.p50_ms:.3f} | {r.p99_ms:.3f} | {r.p999_ms:.3f} | "
            f"{r.mean_ms:.3f} | {r.std_ms:.3f} | {r.steps_per_sec:.0f} | "
            f"{r.obs_prep_p50_ms:.3f} | {r.forward_p50_ms:.3f} | "
            f"{r.state_carry_p50_ms:.3f} | {r.act_post_p50_ms:.3f} |"
        )

    # Pareto: smallest p99 per backbone × precision
    lines.extend(["", "## Pareto summary (best p99 per backbone × precision)", ""])
    by_key: dict[tuple[str, str], list[LatencyResult]] = {}
    for r in results:
        by_key.setdefault((r.backbone, r.precision), []).append(r)
    lines.append("| backbone | precision | best_p99_ms | config (h, T, seed) |")
    lines.append("|---|---|---:|---|")
    for (bb, p), rs in sorted(by_key.items()):
        rs.sort(key=lambda x: x.p99_ms)
        best = rs[0]
        lines.append(
            f"| {bb} | {p} | {best.p99_ms:.3f} | "
            f"h={best.hidden_size} T={best.seq_len} seed={best.seed} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--hardware", default="jetson-cpu",
                        choices=["jetson-cpu", "agx-gpu", "ios-mps"])
    parser.add_argument("--backbones", nargs="+",
                        default=["cfcs", "ltc", "pdn", "gru"],
                        choices=list(BACKBONES.keys()))
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1, 8, 16])
    parser.add_argument("--precisions", nargs="+",
                        default=["fp32", "fp16", "int8"],
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--pareto", action="store_true",
                        help="alias for default sweep; kept for CLI symmetry with jetson_lnn_benchmark.py")
    parser.add_argument("--quick", action="store_true",
                        help="smaller sweep for CI (1 seed, hidden=8,16, T=1,8)")
    parser.add_argument("--measure", type=int, default=200,
                        help="number of steps to measure per config (default 200)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--write-floor-table", action="store_true",
                        help="also write the canonical analysis/decisions/latency_floor_table.md")
    args = parser.parse_args(argv)

    if args.quick:
        args.hidden_sizes = [8, 16]
        args.seq_lens = [1, 8]
        args.seeds = args.seeds[:1]
        args.precisions = ["fp32"]

    print(f"[bench] hardware={args.hardware} "
          f"backbones={args.backbones} "
          f"hidden={args.hidden_sizes} T={args.seq_lens} "
          f"precisions={args.precisions} seeds={args.seeds}", flush=True)

    results = pareto_sweep(
        backbones=args.backbones,
        hidden_sizes=args.hidden_sizes,
        seq_lens=args.seq_lens,
        precisions=args.precisions,
        seeds=args.seeds,
        hardware=args.hardware,
    )

    json_path, md_path = write_outputs(results, args.hardware)
    print(f"[bench] wrote {json_path}")
    print(f"[bench] wrote {md_path}")

    if args.write_floor_table:
        floor_path = REPO_ROOT / "analysis" / "decisions" / "latency_floor_table.md"
        floor_path.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# Latency floor table (canonical)\n\n"
            "Source-of-truth table for end-to-end decision-loop latency. "
            "Updated by `scripts/bench_decision_e2e_latency.py --write-floor-table`.\n\n"
            f"Latest run: `{md_path.name}` ({len(results)} configs, hardware={args.hardware})."
        )
        floor_path.write_text(body + "\n")
        print(f"[bench] updated {floor_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())