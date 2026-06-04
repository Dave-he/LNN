#!/usr/bin/env python3
"""LNN-vs-LSTM time-series ablation runner (PRD §8 #5 v2).

Re-runs the project's canonical "CfC vs LTC vs GRU vs LSTM on Mackey-Glass"
comparison but with the multi-seed ablation pattern established in
``scripts/ablation_liquid_tad_heads.py`` (iter#4) and
``scripts/experiment_graph_lnn_molecule.py`` (iter#6).

What this script reports (per backbone, per seed):

- parameter count
- test MSE
- test MAE
- training wall-clock seconds
- inference samples/sec (warmed up, mean of K repeats)

Final Markdown summary aggregates mean±std across seeds. This is the
infrastructure update PRD §8 #5 calls for: the existing single-seed
``benchmark_comparison.py`` can't tell you whether a 5%% MSE gap is
real signal or seed luck.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.data.timeseries import (
    create_dataloader,
    generate_concept_drift,
    generate_gradual_multi_regime,
    generate_mackey_glass,
    generate_sine_data,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.proj(output[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.proj(output[:, -1, :])


def _build_model(name: str, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
    if name == "cfc":
        return CfCNetwork(input_size, hidden_size, output_size, num_layers=1, return_sequences=False)
    if name == "ltc":
        return LTCNetwork(input_size, hidden_size, output_size, num_layers=1)
    if name == "gru":
        return GRUModel(input_size, hidden_size, output_size)
    if name == "lstm":
        return LSTMModel(input_size, hidden_size, output_size)
    if name == "fhn_dynpmnn":
        # DynPMNN — arXiv 2605.08176v1 (Felipe-Sosa 2026) FitzHugh-Nagumo ODE
        # backbone. 4 Euler steps is a smoke-scale choice (paper default ~5).
        from lnn.core.dynpmnn import DynPMNNNetwork
        return DynPMNNNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=1, n_euler_steps=4, return_sequences=False,
        )
    raise ValueError(f"unknown backbone {name}")


def _load_split(args: argparse.Namespace, seed: int) -> tuple:
    """Generate train/val/test tensors for the chosen dataset."""
    if args.dataset == "mackey_glass":
        data = generate_mackey_glass(num_samples=args.samples, seed=seed)
    elif args.dataset == "sine":
        data = generate_sine_data(num_samples=args.samples, freq=0.07, noise_std=0.05, seed=seed)
    elif args.dataset == "concept_drift":
        # Regime A (lo-freq / hi-amp) → Regime B (hi-freq / lo-amp).
        # drift_point at 50% so train spans both regimes; val/test sees the
        # post-drift regime more heavily — exactly the non-stationary
        # boundary condition the LiquidNN paper claims to handle well.
        data, _ = generate_concept_drift(
            num_samples=args.samples,
            drift_point=args.samples // 2,
            seed=seed,
        )
    elif args.dataset == "gradual_multi_regime":
        # PRD §9 #2 phase-B: multiple regimes that gradually blend into each
        # other (cosine ramp), rather than a single sharp jump.  This is the
        # closer analogue of the clinical-style non-stationarity the LiquidNN
        # paper says it handles well — iter#9's negative finding was on a
        # sharper protocol than the paper actually claims to address.
        data, _ = generate_gradual_multi_regime(
            num_samples=args.samples,
            num_regimes=args.num_regimes,
            transition_frac=args.transition_frac,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    train_end = int(len(data) * 0.7)
    val_end = int(len(data) * 0.85)
    train, val, test = data[:train_end], data[train_end:val_end], data[val_end:]

    train_loader = create_dataloader(
        train, seq_len=args.seq_len, horizon=1, batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val, seq_len=args.seq_len, horizon=1, batch_size=args.batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        test, seq_len=args.seq_len, horizon=1, batch_size=args.batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


def _train_one(model: nn.Module, train_loader, args, device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # PRD §9 #2 phase-B: optional cosine schedule with linear warmup for the
    # first warmup_frac fraction of total steps.  iter#9 showed LTC suffers
    # without per-backbone lr adaptation — this helper keeps the call sites
    # identical while letting the recipe move.
    total_steps = max(args.epochs * max(len(train_loader), 1), 1)
    warmup_steps = max(int(total_steps * max(args.warmup_frac, 0.0)), 0)
    use_schedule = args.warmup_frac > 0.0

    def _lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        # cosine decay over remaining steps.
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)
        if use_schedule else None
    )
    criterion = nn.MSELoss()
    model.train()
    start = time.perf_counter()
    global_step = 0
    for _ in range(args.epochs):
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            if pred.dim() == 3:  # CfC / LTC may return [B, T, F] in some configs
                pred = pred[:, -1, :]
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
    return time.perf_counter() - start


@torch.no_grad()
def _evaluate(model: nn.Module, loader, device) -> tuple[float, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    count = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        pred = model(x)
        if pred.dim() == 3:
            pred = pred[:, -1, :]
        se_sum += float(((pred - y) ** 2).sum().item())
        ae_sum += float((pred - y).abs().sum().item())
        count += int(y.numel())
    return se_sum / max(count, 1), ae_sum / max(count, 1)


@torch.no_grad()
def _inference_throughput(model: nn.Module, loader, args, device) -> float:
    model.eval()
    # warm-up
    for batch in loader:
        x = batch[0].to(device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _ = model(x)
        break

    total_samples = 0
    start = time.perf_counter()
    for _ in range(max(args.inference_repeats, 1)):
        for batch in loader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            _ = model(x)
            total_samples += int(x.shape[0])
    elapsed = time.perf_counter() - start
    return total_samples / max(elapsed, 1e-9)


def _run_one(name: str, args, seed: int, device) -> dict:
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = _load_split(args, seed)
    model = _build_model(name, input_size=1, hidden_size=args.hidden_size, output_size=1).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    train_seconds = _train_one(model, train_loader, args, device)
    test_mse, test_mae = _evaluate(model, test_loader, device)
    val_mse, _ = _evaluate(model, val_loader, device)
    inf_throughput = _inference_throughput(model, test_loader, args, device)
    return {
        "backbone": name,
        "seed": seed,
        "parameters": parameters,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "val_mse": val_mse,
        "train_seconds": train_seconds,
        "inference_samples_per_sec": inf_throughput,
    }


def _aggregate(per_run: list[dict]) -> dict:
    """mean / std / median / min / max grouped by backbone (iter#11 lesson)."""
    by_backbone: dict[str, list[dict]] = {}
    for record in per_run:
        by_backbone.setdefault(record["backbone"], []).append(record)
    summary = {}
    for name, records in by_backbone.items():
        mse = [r["test_mse"] for r in records]
        mae = [r["test_mae"] for r in records]
        train = [r["train_seconds"] for r in records]
        infs = [r["inference_samples_per_sec"] for r in records]
        params = records[0]["parameters"]  # constant across seeds for fixed config
        mse_mean = statistics.fmean(mse)
        mse_std = statistics.stdev(mse) if len(mse) > 1 else 0.0
        summary[name] = {
            "parameters": params,
            "test_mse_mean": mse_mean,
            "test_mse_std": mse_std,
            "test_mse_median": statistics.median(mse),
            "test_mse_min": min(mse),
            "test_mse_max": max(mse),
            "test_mse_std_over_mean": (mse_std / mse_mean) if mse_mean > 1e-12 else 0.0,
            "test_mae_mean": statistics.fmean(mae),
            "test_mae_std": statistics.stdev(mae) if len(mae) > 1 else 0.0,
            "test_mae_median": statistics.median(mae),
            "train_seconds_mean": statistics.fmean(train),
            "train_seconds_std": statistics.stdev(train) if len(train) > 1 else 0.0,
            "inference_samples_per_sec_mean": statistics.fmean(infs),
            "inference_samples_per_sec_std": statistics.stdev(infs) if len(infs) > 1 else 0.0,
            "seeds": [r["seed"] for r in records],
            "n_seeds": len(records),
        }
    return summary


def _variance_flag(std_over_mean: float) -> str:
    """iter#11 lesson: std/mean > 1 -> high variance warning."""
    if std_over_mean > 1.0:
        return " ⚠️"
    if std_over_mean > 0.5:
        return " ⚠"
    return ""


def _small_n_flag(n_seeds: int) -> str:
    """iter#11 lesson: anything below 5 seeds is statistically suspect."""
    if n_seeds < 3:
        return " ⚠️ N<3"
    if n_seeds < 5:
        return " ⚠ N<5"
    return ""


def _format_markdown(payload: dict) -> str:
    cfg = payload["config"]
    summary = payload["summary"]
    # Order: cfc, ltc, gru, lstm for readable comparison.
    order = [n for n in ("cfc", "ltc", "gru", "lstm") if n in summary]
    any_n = max((summary[n]["n_seeds"] for n in order), default=0)
    n_warn = _small_n_flag(any_n)
    lines = [
        f"# LNN vs LSTM Time-Series Ablation — {payload['run_id']}",
        "",
        "## 任务",
        f"- dataset: `{cfg['dataset']}`",
        f"- samples / seq_len: {cfg['samples']} / {cfg['seq_len']}",
        f"- hidden_size / epochs / batch / lr: {cfg['hidden_size']} / {cfg['epochs']} / {cfg['batch_size']} / {cfg['lr']}",
        f"- seeds ({any_n}{n_warn}): {cfg['seeds']}",
        f"- device: {payload['environment']['device']}",
    ]
    if n_warn:
        lines.append("- ⚠️ **iter#11 lesson**: N<5 seed 的"
                     "Δ vs baseline 不可信 — 见 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] v5。")
    lines.extend([
        "",
        "## 跨 seed 鲁棒统计",
        "| Backbone | params | mean MSE | **median MSE** | std/mean | min / max | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in order:
        s = summary[name]
        warn = _variance_flag(s["test_mse_std_over_mean"])
        lines.append(
            f"| `{name}` | {s['parameters']:,} | "
            f"{s['test_mse_mean']:.5f} ± {s['test_mse_std']:.5f} | "
            f"{s['test_mse_median']:.5f} | "
            f"{s['test_mse_std_over_mean']:.2f}{warn} | "
            f"{s['test_mse_min']:.5f} / {s['test_mse_max']:.5f} | "
            f"{s['n_seeds']} |"
        )
    lines.extend([
        "",
        "## 训练/推理开销 (mean ± std)",
        "| Backbone | Train s | Inf samples/s |",
        "|---|---:|---:|",
    ])
    for name in order:
        s = summary[name]
        lines.append(
            f"| `{name}` | {s['train_seconds_mean']:.2f} ± {s['train_seconds_std']:.2f} | "
            f"{s['inference_samples_per_sec_mean']:.0f} ± {s['inference_samples_per_sec_std']:.0f} |"
        )

    if "lstm" in summary:
        lstm = summary["lstm"]
        lines.extend([
            "",
            "## 相对 LSTM baseline (mean + median 双视图)",
            "| Backbone | Δparams | Δmean_mse | Δmedian_mse | Δtrain_s | Δinf_throughput |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name in order:
            if name == "lstm":
                continue
            s = summary[name]
            d_params = (s["parameters"] - lstm["parameters"]) / lstm["parameters"] * 100.0
            d_mse_mean = (s["test_mse_mean"] - lstm["test_mse_mean"]) / max(lstm["test_mse_mean"], 1e-9) * 100.0
            d_mse_median = (s["test_mse_median"] - lstm["test_mse_median"]) / max(lstm["test_mse_median"], 1e-9) * 100.0
            d_train = (s["train_seconds_mean"] - lstm["train_seconds_mean"]) / max(lstm["train_seconds_mean"], 1e-9) * 100.0
            d_inf = (s["inference_samples_per_sec_mean"] - lstm["inference_samples_per_sec_mean"]) / max(lstm["inference_samples_per_sec_mean"], 1e-9) * 100.0
            sign_mean = "✅" if d_mse_mean < 0 else "❌"
            sign_median = "✅" if d_mse_median < 0 else "❌"
            lines.append(
                f"| `{name}` | {d_params:+.2f}% | {d_mse_mean:+.2f}% {sign_mean} | "
                f"{d_mse_median:+.2f}% {sign_median} | {d_train:+.2f}% | {d_inf:+.2f}% |"
            )

    lines.extend([
        "",
        "## 解读模板",
        "- mean 和 median 一致下降 → 真信号;只 mean 下降 → outlier 可能在拉低 baseline,看 median;",
        "- std/mean > 1 ⚠️ → 该 backbone 在该任务上 seed 极不稳定;",
        "- N<5 ⚠ → 单轮 Δ vs baseline 不可作结论(iter#11 教训)。",
        "",
        f"JSON 原数据: `{payload['json_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mackey_glass", "sine", "concept_drift", "gradual_multi_regime"], default="mackey_glass")
    parser.add_argument("--num-regimes", type=int, default=4,
                        help="Only used by --dataset gradual_multi_regime.")
    parser.add_argument("--transition-frac", type=float, default=0.15,
                        help="Cosine-blend width as a fraction of segment length (gradual_multi_regime only).")
    parser.add_argument("--warmup-frac", type=float, default=0.0,
                        help=">0 enables linear warmup + cosine decay over the full schedule. "
                             "PRD §9 #2 phase-B uses 0.1 to fix LTC's iter#9 OOD failure.")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", default="42,7,123",
                        help="Comma-separated seeds.")
    parser.add_argument("--backbones", default="cfc,ltc,gru,lstm")
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="analysis/timeseries_ablation")
    args = parser.parse_args()

    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    device = torch.device(args.device)
    print("=== LNN vs LSTM Time-Series Ablation ===")
    print(f"Dataset: {args.dataset} | Samples: {args.samples} | seq_len: {args.seq_len}")
    print(f"Backbones: {backbones} | Seeds: {args.seeds} | Device: {device}")

    per_run: list[dict] = []
    for backbone in backbones:
        for seed in args.seeds:
            print(f"\n--- {backbone} | seed {seed}")
            record = _run_one(backbone, args, seed, device)
            print(
                f"    params={record['parameters']:,}  test_mse={record['test_mse']:.5f}"
                f"  test_mae={record['test_mae']:.5f}  train={record['train_seconds']:.2f}s"
                f"  inf={record['inference_samples_per_sec']:.0f} samples/s"
            )
            per_run.append(record)

    summary = _aggregate(per_run)

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_lnn_vs_lstm.json"
    md_path = output_dir / f"{run_id}_lnn_vs_lstm.md"
    rel_json = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path

    payload = {
        "run_id": run_id,
        "experiment": "lnn_vs_lstm_timeseries_ablation",
        "generated_at": now.isoformat(),
        "config": {**vars(args), "seeds": args.seeds, "backbones": backbones},
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
        },
        "per_run": per_run,
        "summary": summary,
        "json_path": str(rel_json),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(payload), encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
