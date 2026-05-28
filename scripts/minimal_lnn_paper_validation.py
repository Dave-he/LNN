#!/usr/bin/env python3
"""Validate the minimal LNN paper ideas on a Jetson-class device.

The experiment compresses the core LTC/CfC training ideas into a small,
auditable simulation:

* irregular non-stationary sequence data with per-step dt values;
* CfC-DT: closed-form continuous-time update that consumes dt;
* Euler-LTC-DT: lightweight Liquid Time-Constant ODE-style update;
* GRU+dt: conventional recurrent baseline that receives the same dt feature.

Outputs are written to analysis/jetson by default as JSON, Markdown, and a PNG.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import platform
import subprocess
import time
import traceback
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def detect_environment(torch_module: Any | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device_tree_model": read_text("/proc/device-tree/model"),
        "nv_tegra_release": read_text("/etc/nv_tegra_release"),
        "tegrastats_available": command_output(["which", "tegrastats"]) is not None,
        "jetson_clocks_available": command_output(["which", "jetson_clocks"]) is not None,
    }
    if torch_module is None:
        return env

    cuda_available = False
    cuda_device_count = 0
    cuda_error: str | None = None
    try:
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_device_count = int(torch_module.cuda.device_count())
    except Exception as exc:  # pragma: no cover - depends on Jetson runtime
        cuda_error = exception_summary(exc)

    env.update(
        {
            "torch_version": torch_module.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_version": torch_module.version.cuda,
            "cudnn_version": torch_module.backends.cudnn.version(),
            "cuda_probe_error": cuda_error,
        }
    )
    if cuda_available:
        try:
            props = torch_module.cuda.get_device_properties(0)
            env["cuda_device"] = {
                "name": torch_module.cuda.get_device_name(0),
                "total_memory_mb": round(props.total_memory / 1024 / 1024, 2),
                "multi_processor_count": props.multi_processor_count,
            }
        except Exception as exc:  # pragma: no cover - depends on Jetson runtime
            env["cuda_device_error"] = exception_summary(exc)
    elif cuda_device_count > 0:
        env["cuda_note"] = "CUDA device is visible, but torch.cuda.is_available() is false."
    return env


def exception_summary(exc: BaseException) -> str:
    summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return " ".join(summary.split())


def count_params(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_datasets(args: argparse.Namespace, device: Any) -> dict[str, Any]:
    import torch

    def make_split(samples: int, profile: str) -> tuple[Any, Any, Any]:
        steps = args.seq_len + 1
        if profile == "id":
            freq = torch.rand(samples, 1, device=device) * 1.3 + 0.7
            amp = torch.rand(samples, 1, device=device) * 0.4 + 0.8
            drift = (torch.rand(samples, 1, device=device) - 0.5) * 0.3
            noise_scale = 0.035
            dt_values = torch.rand(samples, steps, device=device) * 0.05 + 0.025
            regime_strength = 0.35
        else:
            freq = torch.rand(samples, 1, device=device) * 1.8 + 1.7
            amp = torch.rand(samples, 1, device=device) * 0.8 + 1.0
            drift = (torch.rand(samples, 1, device=device) - 0.5) * 0.9
            noise_scale = 0.09
            dt_values = torch.rand(samples, steps, device=device) * 0.11 + 0.015
            regime_strength = 0.65

        phase = torch.rand(samples, 1, device=device) * (2.0 * math.pi)
        time_axis = torch.cumsum(dt_values, dim=1)
        time_axis = time_axis / time_axis[:, -1:].clamp_min(1e-6)
        switch_at = 0.35 + 0.35 * torch.rand(samples, 1, device=device)
        switch = (time_axis > switch_at).float()

        base = amp * torch.sin(2.0 * math.pi * freq * time_axis + phase)
        seasonal = 0.25 * torch.sin(2.0 * math.pi * (freq * 2.3) * time_axis + phase * 0.3)
        regime = regime_strength * switch * torch.sin(2.0 * math.pi * (freq * 4.6) * time_axis)
        trend = drift * time_axis
        noise = noise_scale * torch.randn(samples, steps, device=device)
        signal = base + seasonal + regime + trend + noise

        x = signal[:, :-1].unsqueeze(-1)
        y = signal[:, 1:].unsqueeze(-1)
        step_dt = dt_values[:, 1:].unsqueeze(-1)
        return x, step_dt, y

    train_x, train_dt, train_y = make_split(args.samples, "id")
    id_x, id_dt, id_y = make_split(max(args.samples // 4, 32), "id")
    ood_x, ood_dt, ood_y = make_split(max(args.samples // 4, 32), "ood")

    value_mean = train_x.mean()
    value_std = train_x.std().clamp_min(1e-6)
    dt_mean = train_dt.mean().clamp_min(1e-6)

    def normalize(x: Any, step_dt: Any, y: Any) -> tuple[Any, Any, Any]:
        return (x - value_mean) / value_std, step_dt / dt_mean, (y - value_mean) / value_std

    train_x, train_dt, train_y = normalize(train_x, train_dt, train_y)
    id_x, id_dt, id_y = normalize(id_x, id_dt, id_y)
    ood_x, ood_dt, ood_y = normalize(ood_x, ood_dt, ood_y)

    return {
        "train": (train_x, train_dt, train_y),
        "id_test": (id_x, id_dt, id_y),
        "ood_test": (ood_x, ood_dt, ood_y),
        "normalization": {
            "value_mean": float(value_mean.detach().cpu()),
            "value_std": float(value_std.detach().cpu()),
            "dt_mean": float(dt_mean.detach().cpu()),
        },
    }


def build_models(hidden_size: int) -> list[tuple[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CfCDTCell(nn.Module):
        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            width = input_size + hidden_size
            self.ff1 = nn.Sequential(nn.Linear(width, hidden_size), nn.Tanh())
            self.ff2 = nn.Sequential(nn.Linear(width, hidden_size), nn.Tanh())
            self.time_a = nn.Linear(width, hidden_size)
            self.time_b = nn.Linear(width, hidden_size)

        def forward(self, x_t: Any, h: Any, dt_t: Any) -> Any:
            z = torch.cat([x_t, h], dim=-1)
            decay_rate = F.softplus(self.time_a(z)) + 1e-4
            gate = torch.sigmoid(-decay_rate * dt_t + self.time_b(z))
            return self.ff1(z) * (1.0 - gate) + self.ff2(z) * gate

    class CfCDTModel(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.cell = CfCDTCell(1, hidden_size)
            self.readout = nn.Linear(hidden_size, 1)

        def forward(self, x: Any, step_dt: Any) -> Any:
            batch, seq_len, _ = x.shape
            h = x.new_zeros(batch, self.hidden_size)
            outputs = []
            for index in range(seq_len):
                h = self.cell(x[:, index, :], h, step_dt[:, index, :])
                outputs.append(self.readout(h))
            return torch.stack(outputs, dim=1)

    class EulerLTCDTCell(nn.Module):
        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            width = input_size + hidden_size
            self.f_tau = nn.Sequential(nn.Linear(width, hidden_size), nn.Sigmoid())
            self.f_drive = nn.Sequential(nn.Linear(width, hidden_size), nn.Tanh())
            self.tau_base = nn.Parameter(torch.ones(hidden_size))
            self.amplitude = nn.Parameter(torch.ones(hidden_size) * 0.5)

        def forward(self, x_t: Any, h: Any, dt_t: Any) -> Any:
            z = torch.cat([x_t, h], dim=-1)
            tau = F.softplus(self.tau_base) + self.f_tau(z) + 0.05
            drive = self.f_drive(z) * self.amplitude
            dh = -h / tau + drive
            return h + dt_t * dh

    class EulerLTCDTModel(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.cell = EulerLTCDTCell(1, hidden_size)
            self.readout = nn.Linear(hidden_size, 1)

        def forward(self, x: Any, step_dt: Any) -> Any:
            batch, seq_len, _ = x.shape
            h = x.new_zeros(batch, self.hidden_size)
            outputs = []
            for index in range(seq_len):
                h = self.cell(x[:, index, :], h, step_dt[:, index, :])
                outputs.append(self.readout(h))
            return torch.stack(outputs, dim=1)

    class GRUDTModel(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(2, hidden_size, batch_first=True)
            self.readout = nn.Linear(hidden_size, 1)

        def forward(self, x: Any, step_dt: Any) -> Any:
            features = torch.cat([x, step_dt], dim=-1)
            output, _ = self.gru(features)
            return self.readout(output)

    return [
        ("CfC-DT", CfCDTModel(hidden_size)),
        ("Euler-LTC-DT", EulerLTCDTModel(hidden_size)),
        ("GRU+dt", GRUDTModel(hidden_size)),
    ]


def evaluate(model: Any, x: Any, step_dt: Any, y: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        pred = model(x, step_dt)
        mse = F.mse_loss(pred, y).item()
        mae = F.l1_loss(pred, y).item()
    return {"mse": mse, "mae": mae}


def train_one_model(name: str, model: Any, data: dict[str, Any], args: argparse.Namespace, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    train_x, train_dt, train_y = data["train"]
    id_x, id_dt, id_y = data["id_test"]
    ood_x, ood_dt, ood_y = data["ood_test"]

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    start = time.perf_counter()
    last_loss = 0.0
    for _epoch in range(args.epochs):
        order = torch.randperm(train_x.shape[0], device=device)
        for offset in range(0, train_x.shape[0], args.batch_size):
            batch_index = order[offset : offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[batch_index], train_dt[batch_index])
            loss = criterion(prediction, train_y[batch_index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    if train_x.is_cuda:
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    id_metrics = evaluate(model, id_x, id_dt, id_y)
    ood_metrics = evaluate(model, ood_x, ood_dt, ood_y)

    repeats = max(args.inference_repeats, 1)
    model.eval()
    with torch.no_grad():
        if id_x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(id_x, id_dt)
        if id_x.is_cuda:
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-9)
    steps_per_second = id_x.shape[0] * id_x.shape[1] * repeats / elapsed

    degradation_pct = ((ood_metrics["mse"] / max(id_metrics["mse"], 1e-12)) - 1.0) * 100.0
    return {
        "name": name,
        "parameters": count_params(model),
        "last_train_loss": last_loss,
        "id_mse": id_metrics["mse"],
        "id_mae": id_metrics["mae"],
        "ood_mse": ood_metrics["mse"],
        "ood_mae": ood_metrics["mae"],
        "ood_degradation_pct": degradation_pct,
        "train_seconds": train_seconds,
        "inference_steps_per_sec": steps_per_second,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    data = make_datasets(args, device)
    models = build_models(args.hidden_size)
    results = [train_one_model(name, model, data, args, device) for name, model in models]

    payload: dict[str, Any] = {
        "status": "ok",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "minimal_lnn_paper_validation",
        "paper_ideas": [
            "LTC: input-dependent continuous-time decay/time constants for non-stationary dynamics.",
            "CfC: closed-form continuous-time recurrence avoids expensive numerical ODE solving.",
            "Edge validation: compare parameter count, training time, inference throughput, ID/OOD error.",
        ],
        "environment": detect_environment(torch),
        "device": str(device),
        "config": {
            "samples": args.samples,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "inference_repeats": args.inference_repeats,
        },
        "normalization": data["normalization"],
        "results": results,
    }
    if device.type == "cuda":
        payload["cuda_peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    return payload


def write_plot(run_date: str, payload: dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path | None:
    if payload.get("status") != "ok" or not payload.get("results"):
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    results = payload["results"]
    names = [result["name"] for result in results]
    id_mse = [result["id_mse"] for result in results]
    ood_mse = [result["ood_mse"] for result in results]
    throughput = [result["inference_steps_per_sec"] for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle(f"Minimal LNN Paper Validation - {run_date}", fontsize=13, fontweight="bold")

    x_positions = list(range(len(names)))
    width = 0.36
    axes[0].bar([x - width / 2 for x in x_positions], id_mse, width, label="ID")
    axes[0].bar([x + width / 2 for x in x_positions], ood_mse, width, label="OOD")
    axes[0].set_title("Prediction MSE")
    axes[0].set_xticks(x_positions, names, rotation=12)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(names, [result["ood_degradation_pct"] for result in results], color="#f97316")
    axes[1].set_title("OOD degradation %")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(names, throughput, color="#16a34a")
    axes[2].set_title("Inference steps/s")
    axes[2].tick_params(axis="x", rotation=12)
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plot_path = output_dir / f"{run_date}_minimal_lnn_paper_validation.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_outputs(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path | None]:
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.date}_minimal_lnn_paper_validation.json"
    md_path = output_dir / f"{args.date}_minimal_lnn_paper_validation.md"
    plot_path = write_plot(args.date, payload, output_dir)

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = payload.get("environment", {})
    reproduce_command = (
        "python scripts/minimal_lnn_paper_validation.py "
        f"{'--cpu ' if args.cpu else ''}"
        f"--samples {args.samples} --seq-len {args.seq_len} --hidden-size {args.hidden_size} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} "
        f"--weight-decay {args.weight_decay} --grad-clip {args.grad_clip} "
        f"--seed {args.seed} --inference-repeats {args.inference_repeats}"
    )
    lines = [
        "---",
        f"title: Jetson 最小 LNN 论文思路验证 - {args.date}",
        f"date: {args.date}",
        "tags: [LNN, Jetson, CfC, LTC, simulation]",
        "---",
        "",
        f"# Jetson 最小 LNN 论文思路验证 - {args.date}",
        "",
        "## 验证目标",
        "- 用最小非平稳不规则采样序列验证 LTC/CfC 论文里的连续时间动态建模思路。",
        "- 比较 `CfC-DT`、`Euler-LTC-DT` 与传统 `GRU+dt` 在 ID/OOD 误差、参数量、训练时间和推理吞吐上的差异。",
        "- 本实验是本机 Jetson smoke validation，不等同于正式论文复现。",
        "",
        "## 环境",
        f"- 平台：{env.get('platform')}",
        f"- 机器架构：{env.get('machine')}",
        f"- 设备树型号：{env.get('device_tree_model') or 'unknown'}",
        f"- PyTorch：{env.get('torch_version') or 'not installed'}",
        f"- CUDA：{env.get('cuda_available')}，device_count={env.get('cuda_device_count')}，torch CUDA={env.get('cuda_version')}",
    ]
    if env.get("cuda_note"):
        lines.append(f"- CUDA 说明：{env.get('cuda_note')}")
    if env.get("nv_tegra_release"):
        lines.extend(["- Jetson BSP：", "", "```text", env["nv_tegra_release"], "```"])
    config = payload.get("config", {})
    lines.extend(
        [
            "",
            "## 配置",
            f"- 设备：{payload.get('device')}",
            f"- 样本数 / 序列长度：{config.get('samples')} / {config.get('seq_len')}",
            f"- 隐藏维度 / Epoch：{config.get('hidden_size')} / {config.get('epochs')}",
            f"- Batch / LR：{config.get('batch_size')} / {config.get('lr')}",
            f"- Seed：{config.get('seed')}",
            "",
            "## 结果",
            "| 模型 | 参数量 | ID MSE | OOD MSE | OOD 退化 | ID MAE | OOD MAE | 推理步/秒 | 训练秒 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload.get("results", []):
        lines.append(
            f"| {result['name']} | {result['parameters']} | {result['id_mse']:.6f} | "
            f"{result['ood_mse']:.6f} | {result['ood_degradation_pct']:.1f}% | "
            f"{result['id_mae']:.6f} | {result['ood_mae']:.6f} | "
            f"{result['inference_steps_per_sec']:.1f} | {result['train_seconds']:.2f} |"
        )
    if plot_path is not None:
        lines.extend(["", "## 图", f"![Minimal LNN Paper Validation]({plot_path.name})"])
    lines.extend(
        [
            "",
            "## 结论",
            "- `CfC-DT` 对应闭式连续时间思想，重点观察是否以较小参数和较高吞吐跑通不规则 `dt` 输入。",
            "- `Euler-LTC-DT` 对应 LTC 的输入依赖时间常数，用固定步 Euler 做 Jetson 友好的最小模拟。",
            "- `GRU+dt` 是同等输入信息的传统循环基线；若 LNN 方案没有在误差、退化率或延迟上占优，需要继续调 `hidden_size`、`seq_len`、学习率和数据难度。",
            "",
            "## 复现命令",
            "",
            "```bash",
            reproduce_command.strip(),
            "```",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--output-dir", default="analysis/jetson")
    parser.add_argument("--quick", action="store_true", help="Use a shorter Jetson smoke configuration.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-repeats", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.samples = min(args.samples, 240)
        args.seq_len = min(args.seq_len, 32)
        args.hidden_size = min(args.hidden_size, 12)
        args.epochs = min(args.epochs, 3)
        args.inference_repeats = min(args.inference_repeats, 4)

    payload = run_validation(args)
    json_path, md_path, plot_path = write_outputs(args, payload)
    print(f"Minimal LNN paper validation written: {json_path.relative_to(ROOT)}")
    print(f"Markdown summary written: {md_path.relative_to(ROOT)}")
    if plot_path is not None:
        print(f"Plot written: {plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
