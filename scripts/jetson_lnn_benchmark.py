#!/usr/bin/env python3
"""Run a small LNN/CfC-style benchmark suitable for Jetson edge devices."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time
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


def write_report(run_date: str, payload: dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir = ROOT / "analysis" / "jetson"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_date}_lnn_benchmark.json"
    md_path = output_dir / f"{run_date}_lnn_benchmark.md"
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

    if payload.get("status") != "ok":
        lines.extend(["", "## 状态", f"- {payload.get('status')}: {payload.get('reason')}"])
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
        lines.extend(
            [
                "",
                "## 解读",
                "- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。",
                "- `GRU` 是同等隐藏维度的传统循环网络基线，便于比较参数量、误差和吞吐。",
                "- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。",
            ]
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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

    def train_and_eval(name: str, model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> dict[str, Any]:
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

        model.eval()
        with torch.no_grad():
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
        steps_per_second = test_x.shape[0] * test_x.shape[1] * repeats / elapsed
        return {
            "name": name,
            "parameters": count_params(model),
            "test_mse": test_mse,
            "train_seconds": train_seconds,
            "inference_steps_per_sec": steps_per_second,
        }

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
        ("GRU", GRUModel(args.hidden_size).to(device)),
    ]
    results = [train_and_eval(name, model, train_x, train_y, test_x, test_y) for name, model in models]

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
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--quick", action="store_true", help="Use a shorter smoke-test configuration.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-repeats", type=int, default=8)
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
        payload = run_benchmark(args)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        payload = {
            "status": "skipped",
            "reason": "PyTorch is not installed. Install a Jetson-compatible torch wheel to run the benchmark.",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "environment": detect_environment(None),
        }

    json_path, md_path = write_report(args.date, payload)
    print(f"Jetson LNN benchmark report written: {json_path.relative_to(ROOT)}")
    print(f"Markdown summary written: {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
