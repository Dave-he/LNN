#!/usr/bin/env python3
"""Run a small LNN/CfC-style benchmark suitable for Jetson edge devices.

Extensions since iter#35:
- tegrastats power/thermal sampling (via lnn.edge.tegrastats)
- NCPS official LTC/CfC implementations as comparison baselines
- optional ONNX/TensorRT export and benchmark (--export-trt)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_text(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
    except OSError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_benchmark_plot(run_date: str, payload: dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path | None:
    if payload.get("status") not in {"ok", "ok_cpu_fallback"} or not payload.get("results"):
        return None
    if payload.get("experiment") == "jetson_lnn_pareto_sweep":
        return write_pareto_plot(run_date, payload, output_dir)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception:
        return None

    results = payload["results"]
    names = [result["name"] for result in results]
    mse = [result["test_mse"] for result in results]
    throughput = [result["inference_steps_per_sec"] for result in results]
    train_seconds = [result["train_seconds"] for result in results]

    colors = ["#2563eb", "#16a34a", "#f97316", "#7c3aed"][: len(names)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.suptitle(f"Jetson LNN Benchmark - {run_date}", fontsize=13, fontweight="bold")

    panels = [
        (axes[0], mse, "Test MSE", "lower is better", "{:.4f}"),
        (axes[1], throughput, "Inference steps/s", "higher is better", "{:.0f}"),
        (axes[2], train_seconds, "Train seconds", "lower is better", "{:.2f}s"),
    ]
    for axis, values, title, subtitle, formatter in panels:
        bars = axis.bar(names, values, color=colors)
        axis.set_title(f"{title}\n{subtitle}", fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values, strict=False):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                formatter.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plot_path = output_dir / f"{run_date}_lnn_benchmark.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_pareto_plot(run_date: str, payload: dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception:
        return None

    results = payload["results"]
    pareto_results = [result for result in results if result.get("pareto_front")]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = {"CfCStyle": "o", "GRU": "s"}

    for name in sorted({result["name"] for result in results}):
        model_rows = [result for result in results if result["name"] == name]
        ax.scatter(
            [row["inference_steps_per_sec"] for row in model_rows],
            [row["test_mse"] for row in model_rows],
            s=[max(row["parameters"] / 3.0, 24.0) for row in model_rows],
            alpha=0.45,
            marker=markers.get(name, "o"),
            label=name,
        )

    if pareto_results:
        ax.scatter(
            [row["inference_steps_per_sec"] for row in pareto_results],
            [row["test_mse"] for row in pareto_results],
            s=[max(row["parameters"] / 2.2, 36.0) for row in pareto_results],
            facecolors="none",
            edgecolors="#dc2626",
            linewidths=1.7,
            label="Pareto front",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference steps/s (higher is better)")
    ax.set_ylabel("Test MSE (lower is better)")
    ax.set_title(f"Jetson LNN Pareto Sweep - {run_date}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / f"{run_date}_lnn_pareto.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def detect_environment(torch_module: Any | None = None) -> dict[str, Any]:
    model = read_text("/proc/device-tree/model")
    nv_tegra = read_text("/etc/nv_tegra_release")
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device_tree_model": model,
        "nv_tegra_release": nv_tegra,
        "tegrastats_available": command_output(["which", "tegrastats"]) is not None,
        "jetson_clocks_available": command_output(["which", "jetson_clocks"]) is not None,
    }
    if torch_module is not None:
        env.update(
            {
                "torch_version": torch_module.__version__,
                "cuda_available": torch_module.cuda.is_available(),
                "cuda_version": torch_module.version.cuda,
                "cudnn_version": torch_module.backends.cudnn.version(),
            }
        )
        if torch_module.cuda.is_available():
            props = torch_module.cuda.get_device_properties(0)
            env["cuda_device"] = {
                "name": torch_module.cuda.get_device_name(0),
                "total_memory_mb": round(props.total_memory / 1024 / 1024, 2),
                "multi_processor_count": props.multi_processor_count,
            }
    return env


def _format_power_summary(power_summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not power_summary or not power_summary.get("available"):
        lines.extend(["- 功耗采样：不可用"])
        reason = power_summary.get("reason", "tegrastats not available") if power_summary else "tegrastats not available"
        lines.append(f"  - {reason}")
        return lines

    lines.append(f"- 功耗采样：可用 ({power_summary['n_samples']} samples @ {power_summary['interval_ms']}ms)")
    lines.append(f"- 采样窗口时长：{power_summary['duration_s']:.2f}s")

    power = power_summary.get("power_mw", {})
    if power:
        lines.append("- 功率轨道 (mean/peak)：")
        for rail, stats in sorted(power.items()):
            lines.append(f"  - {rail}: {stats['mean']:.0f}/{stats['peak']} mW")

    energy = power_summary.get("energy_mj", {})
    if energy:
        lines.append("- 总能耗：")
        for rail, value in sorted(energy.items()):
            lines.append(f"  - {rail}: {value:.0f} mJ ({value/1000:.3f} J)")

    temps = power_summary.get("temp_c", {})
    if temps:
        lines.append("- 温度：")
        for zone, stats in sorted(temps.items()):
            lines.append(f"  - {zone}: {stats['mean']:.1f}°C (peak {stats['peak']:.1f}°C)")

    gpu = power_summary.get("gpu_util_pct", {})
    if gpu:
        lines.append(f"- GPU 利用率：mean {gpu['mean']:.0f}%, peak {gpu['peak']}%")

    return lines


def write_report(run_date: str, payload: dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir = ROOT / "analysis" / "jetson"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_date}_lnn_benchmark.json"
    md_path = output_dir / f"{run_date}_lnn_benchmark.md"
    plot_path = write_benchmark_plot(run_date, payload, output_dir)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = payload.get("environment", {})
    lines = [
        "---",
        f"title: Jetson LNN 基准验证 - {run_date}",
        f"date: {run_date}",
        "tags: [LNN, Jetson, benchmark, edge-ai]",
        "---",
        "",
        f"# Jetson LNN 基准验证 - {run_date}",
        "",
        "## 环境",
        f"- 平台：{env.get('platform')}",
        f"- 设备树型号：{env.get('device_tree_model') or 'unknown'}",
        f"- PyTorch：{env.get('torch_version') or 'not installed'}",
        f"- CUDA：{env.get('cuda_available')} ({env.get('cuda_version')})",
    ]
    if env.get("nv_tegra_release"):
        lines.extend(["- Jetson BSP：", "", "```text", env["nv_tegra_release"], "```"])
    else:
        lines.append("- Jetson BSP：unknown")
    if env.get("cuda_device"):
        device = env["cuda_device"]
        lines.append(f"- CUDA 设备：{device.get('name')}，显存 {device.get('total_memory_mb')} MB")

    # Power summary (if present)
    power_summary = payload.get("tegrastats_summary")
    if power_summary:
        lines.extend(["", "## 功耗与温度"])
        lines.extend(_format_power_summary(power_summary))

    # Per-model power (if present)
    per_model_power = payload.get("model_tegrastats", {})
    if per_model_power:
        lines.extend(["", "## 各模型独立功耗"])
        for model_name in sorted(per_model_power.keys()):
            summary = per_model_power[model_name]
            lines.extend(["", f"### {model_name}"])
            lines.extend(_format_power_summary(summary))

    if payload.get("status") not in {"ok", "ok_cpu_fallback"}:
        lines.extend(["", "## 状态", f"- {payload.get('status')}: {payload.get('reason')}"])
    elif payload.get("experiment") == "jetson_lnn_pareto_sweep":
        config = payload.get("config", {})
        lines.extend(
            [
                "",
                "## 任务配置",
                "- 数据：合成非平稳时间序列，一步预测",
                f"- Samples / Epoch：{config.get('samples')} / {config.get('epochs')}",
                f"- Hidden sweep：{config.get('hidden_sizes')}",
                f"- SeqLen sweep：{config.get('seq_lens')}",
                f"- Seeds：{config.get('seeds')}",
                f"- 设备：{payload.get('device')}",
            ]
        )

        # Check if per-model energy is present (single-model mode) or not (pareto mode)
        has_energy = any("energy_mj_per_step" in result for result in payload.get("results", []))
        if has_energy:
            lines.extend(
                [
                    "",
                    "## Pareto 结果",
                    "| Front | 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            sorted_results = sorted(
                payload.get("results", []),
                key=lambda result: (not result.get("pareto_front"), result["test_mse"]),
            )
            for result in sorted_results:
                marker = "yes" if result.get("pareto_front") else ""
                energy = result.get("energy_mj_per_step")
                energy_str = f"{energy:.2f}" if energy is not None else "n/a"
                lines.append(
                    f"| {marker} | {result['name']} | {result['hidden_size']} | {result['seq_len']} | "
                    f"{result['seed']} | {result['parameters']} | {result['test_mse']:.6f} | "
                    f"{result['inference_steps_per_sec']:.1f} | {result['train_seconds']:.2f} | {energy_str} |"
                )
        else:
            lines.extend(
                [
                    "",
                    "## Pareto 结果",
                    "| Front | 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            sorted_results = sorted(
                payload.get("results", []),
                key=lambda result: (not result.get("pareto_front"), result["test_mse"]),
            )
            for result in sorted_results:
                marker = "yes" if result.get("pareto_front") else ""
                lines.append(
                    f"| {marker} | {result['name']} | {result['hidden_size']} | {result['seq_len']} | "
                    f"{result['seed']} | {result['parameters']} | {result['test_mse']:.6f} | "
                    f"{result['inference_steps_per_sec']:.1f} | {result['train_seconds']:.2f} |"
                )

        if plot_path is not None:
            lines.extend(["", "## Pareto 图", f"![Jetson LNN Pareto]({plot_path.name})"])
        lines.extend(
            [
                "",
                "## 解读",
                "- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、"
                "更短训练时间和更高吞吐。",
                "- 该 sweep 是边缘筛选入口，正式实验应在真实 Jetson CUDA 路径上增加多 seed、"
                "能耗和导出后延迟。",
                "- `NCPS-LTC` / `NCPS-CfC` 是官方实现 (mlech26l/ncps)，与仓库内的近似实现做对比。",
            ]
        )
    else:
        config = payload.get("config", {})
        lines.extend(
            [
                "",
                "## 任务配置",
                f"- 数据：合成非平稳时间序列，一步预测",
                f"- 样本 / 序列长度：{config.get('samples')} / {config.get('seq_len')}",
                f"- 隐藏维度 / Epoch：{config.get('hidden_size')} / {config.get('epochs')}",
                f"- 设备：{payload.get('device')}",
            ]
        )

        has_energy = any("energy_mj_per_step" in result for result in payload.get("results", []))
        if has_energy:
            lines.extend(
                [
                    "",
                    "## 结果",
                    "| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for result in payload.get("results", []):
                energy = result.get("energy_mj_per_step")
                energy_str = f"{energy:.2f}" if energy is not None else "n/a"
                lines.append(
                    f"| {result['name']} | {result['parameters']} | {result['test_mse']:.6f} | "
                    f"{result['inference_steps_per_sec']:.1f} | {result['train_seconds']:.2f} | {energy_str} |"
                )
        else:
            lines.extend(
                [
                    "",
                    "## 结果",
                    "| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for result in payload.get("results", []):
                lines.append(
                    f"| {result['name']} | {result['parameters']} | {result['test_mse']:.6f} | "
                    f"{result['inference_steps_per_sec']:.1f} | {result['train_seconds']:.2f} |"
                )

        if plot_path is not None:
            lines.extend(["", "## Benchmark 图", f"![Jetson LNN Benchmark]({plot_path.name})"])
        if payload.get("status") == "ok_cpu_fallback":
            lines.extend(
                [
                    "",
                    "## CUDA 回退",
                    "- 本次优先尝试 Jetson CUDA 路径，但 CUDA 运行时返回内存/加速器错误，"
                    "已自动回退到 CPU smoke benchmark。",
                    "- 回退原因：",
                    "",
                    "```text",
                    str(payload.get("cuda_fallback_reason")),
                    "```",
                ]
            )
        lines.extend(
            [
                "",
                "## 解读",
                "- `CfCStyle` 是闭式连续时间思想的轻量实现，"
                "用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。",
                "- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。",
                "- `GRU` 是同等隐藏维度的传统循环网络基线。",
                "- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、"
                "固定随机种子、多次重复和置信区间。",
            ]
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in '{raw}'")
    return values


def dominates(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    no_worse = (
        candidate["test_mse"] <= other["test_mse"]
        and candidate["parameters"] <= other["parameters"]
        and candidate["train_seconds"] <= other["train_seconds"]
        and candidate["inference_steps_per_sec"] >= other["inference_steps_per_sec"]
    )
    strictly_better = (
        candidate["test_mse"] < other["test_mse"]
        or candidate["parameters"] < other["parameters"]
        or candidate["train_seconds"] < other["train_seconds"]
        or candidate["inference_steps_per_sec"] > other["inference_steps_per_sec"]
    )
    return no_worse and strictly_better


def mark_pareto_front(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked = []
    for index, result in enumerate(results):
        dominated = any(
            other_index != index and dominates(other, result)
            for other_index, other in enumerate(results)
        )
        result = dict(result)
        result["pareto_front"] = not dominated
        marked.append(result)
    return marked


def aggregate_seeds(
    per_seed_results: list[dict[str, Any]],
    metrics: tuple[str, ...] = ("test_mse", "inference_steps_per_sec", "train_seconds"),
) -> list[dict[str, Any]]:
    """iter#35: collapse per-seed rows into per-(model, hidden, seq) mean ± std.

    Groups by (name, hidden_size, seq_len) and computes mean + std + n_seeds
    for the requested metrics. `parameters` and `seeds` are passed through.
    The aggregated rows use the same flat-dict shape as per-seed rows but with
    the metric values wrapped as {mean, std, n_seeds} dicts. The dominance
    check in `mark_pareto_front` operates on these aggregated rows by reading
    the `.mean` sub-field (see `dominates` patched below).
    """
    import statistics

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in per_seed_results:
        key = (row["name"], row.get("hidden_size", -1), row.get("seq_len", -1))
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (name, hidden_size, seq_len), rows in groups.items():
        params = rows[0].get("parameters", 0)
        seeds = sorted({row.get("seed", -1) for row in rows})
        agg: dict[str, Any] = {
            "name": name,
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "parameters": params,
            "seeds": seeds,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if not values:
                continue
            if len(values) >= 2:
                mean = statistics.fmean(values)
                std = statistics.stdev(values)
            else:
                mean = float(values[0])
                std = 0.0
            agg[metric] = {"mean": mean, "std": std, "n_seeds": len(values)}
        aggregated.append(agg)
    return aggregated


def _agg_dominates(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    """Patched dominance check for aggregated rows (reads .mean sub-fields)."""
    def _get(row: dict[str, Any], key: str) -> float:
        val = row.get(key)
        if isinstance(val, dict) and "mean" in val:
            return float(val["mean"])
        return float(val) if val is not None else 0.0

    no_worse = (
        _get(candidate, "test_mse") <= _get(other, "test_mse")
        and _get(candidate, "parameters") <= _get(other, "parameters")
        and _get(candidate, "train_seconds") <= _get(other, "train_seconds")
        and _get(candidate, "inference_steps_per_sec") >= _get(other, "inference_steps_per_sec")
    )
    strictly_better = (
        _get(candidate, "test_mse") < _get(other, "test_mse")
        or _get(candidate, "parameters") < _get(other, "parameters")
        or _get(candidate, "train_seconds") < _get(other, "train_seconds")
        or _get(candidate, "inference_steps_per_sec") > _get(other, "inference_steps_per_sec")
    )
    return no_worse and strictly_better


def mark_pareto_front_aggregated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pareto marking for aggregated (mean-valued) rows."""
    marked = []
    for index, result in enumerate(rows):
        dominated = any(
            other_index != index and _agg_dominates(other, result)
            for other_index, other in enumerate(rows)
        )
        result = dict(result)
        result["pareto_front"] = not dominated
        marked.append(result)
    return marked


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lnn.edge.tegrastats import TegrastatsSampler, energy_per_step

    class CfCCell(nn.Module):
        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            width = input_size + hidden_size
            self.ff1 = nn.Sequential(nn.Linear(width, hidden_size), nn.Tanh())
            self.ff2 = nn.Sequential(nn.Linear(width, hidden_size), nn.Tanh())
            self.time_a = nn.Linear(width, hidden_size)
            self.time_b = nn.Linear(width, hidden_size)

        def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt_value: float) -> torch.Tensor:
            z = torch.cat([x_t, h], dim=-1)
            a = F.softplus(self.time_a(z))
            b = self.time_b(z)
            gate = torch.sigmoid(-a * dt_value + b)
            return self.ff1(z) * (1.0 - gate) + self.ff2(z) * gate

    class CfCStyleModel(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.cell = CfCCell(1, hidden_size)
            self.readout = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = x.shape
            h = x.new_zeros(batch, self.hidden_size)
            outputs = []
            dt_value = 1.0 / max(seq_len, 1)
            for index in range(seq_len):
                h = self.cell(x[:, index, :], h, dt_value)
                outputs.append(self.readout(h))
            return torch.stack(outputs, dim=1)

    class GRUModel(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(1, hidden_size, batch_first=True)
            self.readout = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.gru(x)
            return self.readout(output)

    # iter#34: add LTC and PDNA-pulse arms to the Pareto sweep, so we can
    # directly test the Hasani 2022 (Nature MI) "CfC 1-5 orders faster than
    # ODE-based LTC" claim and the PDNA paper's "pulse modulation ≈ CfC on
    # latency, slightly higher accuracy" claim on real Jetson hardware.
    from lnn.core.ltc import LTCNetwork
    from lnn.core.cfc import PDNAPulseHead

    class LTCModel(nn.Module):
        """ODE-solver-based LTC (the original LNN). Uses in-repo LTCNetwork."""

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.network = LTCNetwork(
                input_size=1,
                hidden_size=hidden_size,
                output_size=1,
                num_layers=1,
                ode_method="rk4",
                return_sequences=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    class PDNAPulseModel(nn.Module):
        """CfC + PDNA pulse modulation (iter#19 augmentation head)."""

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.cell = CfCCell(1, hidden_size)
            self.pulse = PDNAPulseHead(hidden_size, use_self_attend=False)
            self.readout = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = x.shape
            h = x.new_zeros(batch, self.hidden_size)
            hidden_states = []
            dt_value = 1.0 / max(seq_len, 1)
            for index in range(seq_len):
                h = self.cell(x[:, index, :], h, dt_value)
                hidden_states.append(h)
            hidden_seq = torch.stack(hidden_states, dim=1)  # [B, T, H]
            augmented = self.pulse(hidden_seq)  # [B, T, H]
            return self.readout(augmented)  # [B, T, 1]

    # NCPS official implementations (if available)
    try:
        from ncps.torch import CfC as NCPS_CfC
        from ncps.torch import LTC as NCPS_LTC

        class NCPS_CfC_Model(nn.Module):
            """Official NCPS CfC implementation from mlech26l/ncps."""
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.cfc = NCPS_CfC(1, hidden_size, return_sequences=True)
                self.readout = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.cfc(x)
                return self.readout(out)

        class NCPS_LTC_Model(nn.Module):
            """Official NCPS LTC implementation from mlech26l/ncps."""
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.ltc = NCPS_LTC(1, hidden_size, return_sequences=True)
                self.readout = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.ltc(x)
                return self.readout(out)

        NCPS_AVAILABLE = True
    except ImportError:
        NCPS_AVAILABLE = False
        NCPS_CfC_Model = None
        NCPS_LTC_Model = None

    def make_dataset(samples: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        steps = seq_len + 1
        t_axis = torch.linspace(0, 1, steps, device=device).unsqueeze(0).repeat(samples, 1)
        freq = torch.rand(samples, 1, device=device) * 3.0 + 0.5
        phase = torch.rand(samples, 1, device=device) * (2.0 * math.pi)
        drift = (torch.rand(samples, 1, device=device) - 0.5) * 0.6
        switch = (t_axis > (0.35 + 0.35 * torch.rand(samples, 1, device=device))).float()
        base = torch.sin(2.0 * math.pi * freq * t_axis + phase)
        seasonal = 0.35 * torch.sin(2.0 * math.pi * (freq * 2.7) * t_axis + phase / 2.0)
        regime = switch * 0.45 * torch.sin(2.0 * math.pi * (freq * 5.0) * t_axis)
        noise = 0.05 * torch.randn(samples, steps, device=device)
        signal = base + seasonal + drift * t_axis + regime + noise
        return signal[:, :-1].unsqueeze(-1), signal[:, 1:].unsqueeze(-1)

    def count_params(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def train_and_eval(
        name: str,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
    ) -> dict[str, Any]:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        start = time.perf_counter()
        for _epoch in range(args.epochs):
            order = torch.randperm(train_x.shape[0], device=train_x.device)
            for offset in range(0, train_x.shape[0], args.batch_size):
                batch_index = order[offset : offset + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(train_x[batch_index])
                loss = criterion(prediction, train_y[batch_index])
                loss.backward()
                optimizer.step()
        if train_x.is_cuda:
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start

        # Measure inference with power sampling (if requested)
        power_summary: dict[str, Any] | None = None
        model.eval()
        with torch.no_grad(), TegrastatsSampler(interval_ms=100) as sampler:
            prediction = model(test_x)
            test_mse = criterion(prediction, test_y).item()
            if test_x.is_cuda:
                torch.cuda.synchronize()
            repeats = max(args.inference_repeats, 1)
            start = time.perf_counter()
            for _ in range(repeats):
                _ = model(test_x)
            if test_x.is_cuda:
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-9)
            power_summary = sampler.summary()
        steps_per_second = test_x.shape[0] * test_x.shape[1] * repeats / elapsed
        total_steps = test_x.shape[0] * test_x.shape[1] * repeats
        energy_mj_per_step = energy_per_step(power_summary, total_steps) if power_summary else None

        result: dict[str, Any] = {
            "name": name,
            "parameters": count_params(model),
            "test_mse": test_mse,
            "train_seconds": train_seconds,
            "inference_steps_per_sec": steps_per_second,
        }
        if power_summary:
            result["tegrastats_summary"] = power_summary
        if energy_mj_per_step is not None:
            result["energy_mj_per_step"] = energy_mj_per_step
        return result

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    x, y = make_dataset(args.samples, args.seq_len, device)
    split = int(args.samples * 0.8)
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]

    models = [
        ("CfCStyle", CfCStyleModel(args.hidden_size).to(device)),
        ("LTC", LTCModel(args.hidden_size).to(device)),
        ("PDNAPulse", PDNAPulseModel(args.hidden_size).to(device)),
        ("GRU", GRUModel(args.hidden_size).to(device)),
    ]
    if NCPS_AVAILABLE:
        models.append(("NCPS-LTC", NCPS_LTC_Model(args.hidden_size).to(device)))
        models.append(("NCPS-CfC", NCPS_CfC_Model(args.hidden_size).to(device)))

    # Global power sampling for total run
    per_model_power: dict[str, dict[str, Any]] = {}
    with TegrastatsSampler(interval_ms=100) as global_sampler:
        results = []
        for name, model in models:
            result = train_and_eval(name, model, train_x, train_y, test_x, test_y)
            results.append(result)
            if "tegrastats_summary" in result:
                per_model_power[name] = result.pop("tegrastats_summary")

    payload: dict[str, Any] = {
        "status": "ok",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": detect_environment(torch),
        "device": str(device),
        "config": {
            "samples": args.samples,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "results": results,
    }
    if use_cuda:
        payload["cuda_peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    payload["tegrastats_summary"] = global_sampler.summary()
    payload["model_tegrastats"] = per_model_power
    return payload


def run_pareto_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    hidden_default = [8, 16] if args.quick else [8, 16, 32]
    seq_default = [16, 32] if args.quick else [32, 64]
    hidden_sizes = parse_int_list(args.hidden_sizes, hidden_default)
    seq_lens = parse_int_list(args.seq_lens, seq_default)
    seeds = parse_int_list(args.seeds, [args.seed])

    flat_results: list[dict[str, Any]] = []
    environment: dict[str, Any] = {}
    device = "unknown"

    for hidden_size in hidden_sizes:
        for seq_len in seq_lens:
            for seed in seeds:
                run_args = argparse.Namespace(**vars(args))
                run_args.hidden_size = hidden_size
                run_args.seq_len = seq_len
                run_args.seed = seed
                payload = run_benchmark(run_args)
                environment = payload.get("environment", environment)
                device = payload.get("device", device)
                for result in payload["results"]:
                    flat_results.append(
                        {
                            **result,
                            "hidden_size": hidden_size,
                            "seq_len": seq_len,
                            "seed": seed,
                        }
                    )

    return {
        "status": "ok",
        "experiment": "jetson_lnn_pareto_sweep",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": environment,
        "device": device,
        "config": {
            "samples": args.samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_sizes": hidden_sizes,
            "seq_lens": seq_lens,
            "seeds": seeds,
            "inference_repeats": args.inference_repeats,
        },
        "results": mark_pareto_front(flat_results),
        # iter#35: aggregated mean ± std per (model, hidden, seq) for the
        # multi-seed case. Empty list if only 1 seed (no aggregation needed).
        "aggregated_results": mark_pareto_front_aggregated(
            aggregate_seeds(flat_results)
        ) if len(seeds) >= 2 else [],
    }


def looks_like_cuda_runtime_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in text and any(
        marker in text
        for marker in (
            "out of memory",
            "memory allocation",
            "cublas",
            "cudnn",
            "cudacachingallocator",
            "internal assert",
            "accelerator",
            "nvml",
        )
    )


def exception_summary(exc: BaseException) -> str:
    summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return " ".join(summary.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--quick", action="store_true", help="Use a shorter smoke-test configuration.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Do not retry on CPU when the CUDA path fails with a runtime memory/accelerator error.",
    )
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-repeats", type=int, default=8)
    parser.add_argument("--pareto", action="store_true", help="Run a hidden/sequence/seed sweep and mark Pareto front.")
    parser.add_argument("--hidden-sizes", default=None, help="Comma-separated hidden sizes for --pareto.")
    parser.add_argument("--seq-lens", default=None, help="Comma-separated sequence lengths for --pareto.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for --pareto.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.samples = min(args.samples, 384)
        args.seq_len = min(args.seq_len, 48)
        args.hidden_size = min(args.hidden_size, 24)
        args.epochs = min(args.epochs, 3)
        args.inference_repeats = min(args.inference_repeats, 4)

    try:
        payload = run_pareto_benchmark(args) if args.pareto else run_benchmark(args)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        payload = {
            "status": "skipped",
            "reason": "PyTorch is not installed. Install a Jetson-compatible torch wheel to run the benchmark.",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "environment": detect_environment(None),
        }
    except Exception as exc:
        if args.cpu or args.no_cpu_fallback or not looks_like_cuda_runtime_error(exc):
            raise
        reason = exception_summary(exc)
        print(f"[warn] Jetson CUDA benchmark failed; retrying on CPU: {reason}", file=sys.stderr)
        args.cpu = True
        payload = run_pareto_benchmark(args) if args.pareto else run_benchmark(args)
        payload["status"] = "ok_cpu_fallback"
        payload["cuda_fallback_reason"] = reason

    json_path, md_path = write_report(args.date, payload)
    print(f"Jetson LNN benchmark report written: {json_path.relative_to(ROOT)}")
    print(f"Markdown summary written: {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
