#!/usr/bin/env python3
"""Export LNN/CfC/LTC models to ONNX and TensorRT for Jetson deployment.

This script:
- Defines the CfCStyle/LTC/PDNAPulse models from jetson_lnn_benchmark.py
- Exports them to ONNX with fixed input shapes
- Optionally converts ONNX to TensorRT (FP16/INT8)
- Benchmarks PyTorch eager vs ONNX Runtime vs TensorRT
- Writes a report to analysis/jetson/YYYY-MM-DD_lnn_tensorrt.json/md

Usage:
    # Quick smoke test (CPU only)
    python scripts/export_lnn_tensorrt.py --quick --cpu

    # Full export + TensorRT (on Jetson)
    python scripts/export_lnn_tensorrt.py --tensorrt --precision fp16 --precision int8

    # Specify hidden size and seq len
    python scripts/export_lnn_tensorrt.py --hidden-size 16 --seq-len 32 --samples 256
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--quick", action="store_true", help="Quick smoke configuration: small sizes, few epochs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden size for the recurrent cell.")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for export and benchmark.")
    parser.add_argument("--inference-repeats", type=int, default=50, help="Number of inference runs for timing.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "analysis" / "jetson"), help="Output directory.")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export to ONNX (default True).")
    parser.add_argument("--tensorrt", action="store_true", help="Convert ONNX to TensorRT.")
    parser.add_argument("--precision", type=str, action="append", default=["fp16"], help="Precisions for TensorRT (fp16, int8).")
    return parser.parse_args()


def build_models(hidden_size: int, device: str) -> dict[str, tuple[object, list[object]]]:
    """Build all benchmark models and return them with their export example inputs."""
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

    # Try to import NCPS models if available
    ncps_available = False
    try:
        from ncps.torch import CfC as NCPS_CfC
        from ncps.torch import LTC as NCPS_LTC

        class NCPS_CfC_Model(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.cfc = NCPS_CfC(1, hidden_size, return_sequences=True)
                self.readout = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.cfc(x)
                return self.readout(out)

        class NCPS_LTC_Model(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.ltc = NCPS_LTC(1, hidden_size, return_sequences=True)
                self.readout = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.ltc(x)
                return self.readout(out)

        ncps_available = True
    except ImportError:
        pass

    # Try to import repo internal LTC
    repo_ltc_available = False
    try:
        from lnn.core.ltc import LTCNetwork

        class LTCModel(nn.Module):
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

        repo_ltc_available = True
    except ImportError:
        pass

    models: dict[str, tuple[nn.Module, list[torch.Tensor]]] = {}

    # Example input (dynamic shape for ONNX export)
    device_obj = torch.device(device)
    fixed_example = torch.randn(1, 32, 1, device=device_obj)

    models["CfCStyle"] = (CfCStyleModel(hidden_size).to(device_obj), [fixed_example])
    models["GRU"] = (GRUModel(hidden_size).to(device_obj), [fixed_example])

    if repo_ltc_available:
        try:
            models["LTC"] = (LTCModel(hidden_size).to(device_obj), [fixed_example])
        except Exception:
            pass

    if ncps_available:
        try:
            models["NCPS-CfC"] = (NCPS_CfC_Model(hidden_size).to(device_obj), [fixed_example])
        except Exception:
            pass

    return models


def make_dataset(samples: int, seq_len: int, device: str) -> tuple[object, object]:
    """Make synthetic non-stationary dataset for evaluation."""
    import torch
    device_obj = torch.device(device)
    steps = seq_len + 1
    t_axis = torch.linspace(0, 1, steps, device=device_obj).unsqueeze(0).repeat(samples, 1)
    freq = torch.rand(samples, 1, device=device_obj) * 3.0 + 0.5
    phase = torch.rand(samples, 1, device=device_obj) * (2.0 * math.pi)
    drift = (torch.rand(samples, 1, device=device_obj) - 0.5) * 0.6
    switch = (t_axis > (0.35 + 0.35 * torch.rand(samples, 1, device=device_obj))).float()
    base = torch.sin(2.0 * math.pi * freq * t_axis + phase)
    seasonal = 0.35 * torch.sin(2.0 * math.pi * (freq * 2.7) * t_axis + phase / 2.0)
    regime = switch * 0.45 * torch.sin(2.0 * math.pi * (freq * 5.0) * t_axis)
    noise = 0.05 * torch.randn(samples, steps, device=device_obj)
    signal = base + seasonal + drift * t_axis + regime + noise
    return signal[:, :-1].unsqueeze(-1), signal[:, 1:].unsqueeze(-1)


def count_params(model: object) -> int:
    """Count number of trainable parameters."""
    import torch.nn as nn
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_onnx(
    model: object,
    example_input: object,
    output_path: pathlib.Path,
    model_name: str,
) -> dict[str, float]:
    """Export PyTorch model to ONNX, returning timing stats."""
    try:
        import torch

        start = time.perf_counter()
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size", 1: "seq_len"}},
        )
        export_time = time.perf_counter() - start
        size_mb = output_path.stat().st_size / (1024 * 1024)

        return {
            "available": True,
            "export_time_s": export_time,
            "onnx_size_mb": size_mb,
            "onnx_path": str(output_path.relative_to(ROOT)),
        }
    except ImportError as e:
        return {"available": False, "reason": f"ONNX export failed: missing dependency {e}"}
    except Exception as e:
        return {"available": False, "reason": f"ONNX export failed: {e}"}


def benchmark_onnxruntime(
    onnx_path: pathlib.Path,
    test_x: object,
    test_y: object,
    repeats: int,
    device: str,
) -> dict[str, float]:
    """Benchmark ONNX Runtime inference."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        return {"available": False, "reason": "onnxruntime not installed"}

    test_x_np = test_x.cpu().numpy() if hasattr(test_x, "cpu") else np.array(test_x)
    test_y_np = test_y.cpu().numpy() if hasattr(test_y, "cpu") else np.array(test_y)

    providers = []
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        return {"available": False, "reason": f"ONNX Runtime session failed: {e}"}

    # Warmup
    _ = sess.run(None, {"input": test_x_np})

    start = time.perf_counter()
    for _ in range(repeats):
        outputs = sess.run(None, {"input": test_x_np})
    elapsed = max(time.perf_counter() - start, 1e-9)

    predictions = outputs[0]
    mse = float(((predictions - test_y_np) ** 2).mean())
    steps_per_second = test_x_np.shape[0] * test_x_np.shape[1] * repeats / elapsed

    return {
        "available": True,
        "mse": mse,
        "inference_steps_per_sec": steps_per_second,
        "total_time_s": elapsed,
    }


def build_tensorrt_engine(
    onnx_path: pathlib.Path,
    output_engine_path: pathlib.Path,
    precision: str,
    batch_size: int,
    seq_len: int,
    device: str,
) -> dict[str, float]:
    """Build TensorRT engine from ONNX."""
    if device != "cuda":
        return {"available": False, "reason": "TensorRT requires CUDA"}

    try:
        import tensorrt as trt
    except ImportError:
        return {"available": False, "reason": "tensorrt Python bindings not installed"}

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def build_engine() -> trt.ICudaEngine:
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: proper calibration needed for accurate INT8
            # We just enable it for this script

        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for error in range(parser.num_errors):
                    errors.append(str(parser.get_error(error)))
                raise RuntimeError("\n".join(errors))

        # Set input profile for dynamic shapes
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)

        # Min, opt, max shapes (batch, seq_len, features)
        min_shape = (1, 1, 1)
        opt_shape = (batch_size, seq_len, 1)
        max_shape = (batch_size * 4, seq_len * 2, 1)

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        return builder.build_serialized_network(network, config)

    start = time.perf_counter()
    try:
        engine_data = build_engine()
        if engine_data is None:
            return {"available": False, "reason": "TensorRT engine build returned None"}

        with open(str(output_engine_path), "wb") as f:
            f.write(engine_data)

        build_time = time.perf_counter() - start
        size_mb = output_engine_path.stat().st_size / (1024 * 1024)

        return {
            "available": True,
            "build_time_s": build_time,
            "engine_size_mb": size_mb,
            "engine_path": str(output_engine_path.relative_to(ROOT)),
        }
    except Exception as e:
        return {"available": False, "reason": f"TensorRT build failed: {e}"}


def benchmark_tensorrt(
    engine_path: pathlib.Path,
    test_x: object,
    test_y: object,
    repeats: int,
    device: str,
) -> dict[str, float]:
    """Benchmark TensorRT inference."""
    if device != "cuda":
        return {"available": False, "reason": "TensorRT requires CUDA"}

    try:
        import tensorrt as trt
    except ImportError:
        return {"available": False, "reason": "tensorrt Python bindings not installed"}

    import torch
    import numpy as np

    test_x_np = test_x.cpu().numpy() if hasattr(test_x, "cpu") else np.array(test_x)
    test_y_np = test_y.cpu().numpy() if hasattr(test_y, "cpu") else np.array(test_y)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def load_engine() -> trt.ICudaEngine:
        with open(str(engine_path), "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    try:
        engine = load_engine()
    except Exception as e:
        return {"available": False, "reason": f"Engine load failed: {e}"}

    try:
        context = engine.create_execution_context()
        context.set_input_shape(engine.get_tensor_name(0), test_x_np.shape)
    except Exception as e:
        return {"available": False, "reason": f"Context creation failed: {e}"}

    # Allocate buffers
    outputs = []
    bindings = []
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if i == 0:
            # Input
            binding = torch.from_numpy(test_x_np).cuda().contiguous().data_ptr()
            bindings.append(binding)
        else:
            # Output
            shape = context.get_tensor_shape(tensor_name)
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(tensor_name)))
            output = np.zeros(shape, dtype=dtype)
            outputs.append(output)
            binding = torch.from_numpy(output).cuda().contiguous().data_ptr()
            bindings.append(binding)

    stream = torch.cuda.Stream()

    # Warmup
    try:
        context.execute_async_v2(bindings, stream.cuda_stream)
        stream.synchronize()
    except Exception:
        pass

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeats):
        context.execute_async_v2(bindings, stream.cuda_stream)
        stream.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    # Get prediction for MSE
    predictions = outputs[0]
    mse = float(((predictions - test_y_np) ** 2).mean())
    steps_per_second = test_x_np.shape[0] * test_x_np.shape[1] * repeats / elapsed

    return {
        "available": True,
        "mse": mse,
        "inference_steps_per_sec": steps_per_second,
        "total_time_s": elapsed,
    }


def detect_environment() -> dict[str, any]:
    """Detect environment (platform, CUDA, Jetson info)."""
    import platform
    import pathlib
    env = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["cuda_device"] = {"name": torch.cuda.get_device_name(0)}
    except ImportError:
        pass

    # Check Jetson
    nv_tegra_release = pathlib.Path("/etc/nv_tegra_release")
    if nv_tegra_release.exists():
        env["nv_tegra_release"] = nv_tegra_release.read_text().strip()

    return env


def write_report(run_date: str, payload: dict[str, any], output_dir: pathlib.Path):
    """Write JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_date}_lnn_tensorrt.json"
    md_path = output_dir / f"{run_date}_lnn_tensorrt.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = payload.get("environment", {})
    config = payload.get("config", {})

    lines = [
        "---",
        f"title: Jetson LNN ONNX/TensorRT Report - {run_date}",
        f"date: {run_date}",
        "tags: [LNN, Jetson, ONNX, TensorRT, benchmark]",
        "---",
        "",
        f"# Jetson LNN ONNX/TensorRT Export and Benchmark - {run_date}",
        "",
        "## Environment",
        f"- Platform: {env.get('platform')}",
        f"- Python: {env.get('python')}",
        f"- PyTorch: {env.get('torch_version')}",
        f"- CUDA available: {env.get('cuda_available')}",
    ]

    if env.get("nv_tegra_release"):
        lines.extend(["- Jetson BSP:", "", "```text", env["nv_tegra_release"], "```"])

    lines.extend([
        "",
        "## Configuration",
        f"- Hidden size: {config.get('hidden_size')}",
        f"- Sequence length: {config.get('seq_len')}",
        f"- Batch size: {config.get('batch_size')}",
        f"- Inference repeats: {config.get('inference_repeats')}",
        f"- Device: {payload.get('device')}",
        "",
        "## Results",
    ])

    model_results = payload.get("model_results", {})
    if model_results:
        lines.extend([
            "| Model | Parameters | PyTorch MSE | PyTorch steps/s | ONNX MSE | ONNX steps/s |",
            "|---|---:|---:|---:|---:|---:|",
        ])

        for name, result in sorted(model_results.items()):
            params = result.get("parameters", 0)
            pytorch_mse = result.get("pytorch_mse", 0)
            pytorch_speed = result.get("pytorch_steps_per_sec", 0)
            onnx_mse = result.get("onnx_mse", "n/a")
            onnx_speed = result.get("onnx_steps_per_sec", "n/a")

            onnx_mse_str = f"{onnx_mse:.6f}" if onnx_mse != "n/a" else "n/a"
            onnx_speed_str = f"{onnx_speed:.1f}" if onnx_speed != "n/a" else "n/a"

            lines.append(
                f"| {name} | {params} | {pytorch_mse:.6f} | {pytorch_speed:.1f} | {onnx_mse_str} | {onnx_speed_str} |"
            )

        lines.extend(["", "## TensorRT (if built)"])
        lines.extend([
            "| Model | TRT Precision | TRT steps/s | Speedup over PyTorch |",
            "|---|---|---:|---:|",
        ])

        for name, result in sorted(model_results.items()):
            trt_results = result.get("tensorrt", {})
            for precision, trt_result in sorted(trt_results.items()):
                if trt_result.get("available"):
                    trt_speed = trt_result.get("inference_steps_per_sec", 0)
                    pytorch_speed = result.get("pytorch_steps_per_sec", 1)
                    speedup = trt_speed / pytorch_speed
                    lines.append(
                        f"| {name} | {precision} | {trt_speed:.1f} | {speedup:.2f}x |"
                    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()

    if args.quick:
        args.hidden_size = min(args.hidden_size, 16)
        args.seq_len = min(args.seq_len, 16)
        args.inference_repeats = min(args.inference_repeats, 10)

    output_dir = pathlib.Path(args.output_dir)
    onnx_dir = output_dir / "onnx"
    trt_dir = output_dir / "tensorrt"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    trt_dir.mkdir(parents=True, exist_ok=True)

    # Detect environment
    import torch
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    env = detect_environment()

    # Make test dataset
    test_x, test_y = make_dataset(args.batch_size, args.seq_len, device)

    # Build models
    models = build_models(args.hidden_size, device)
    model_results = {}

    # Process each model
    for name, (model, example_inputs) in models.items():
        model.eval()
        example_input = example_inputs[0]

        # PyTorch eager benchmark
        import torch.nn.functional as F
        with torch.no_grad():
            # Warmup
            _ = model(test_x)

            start = time.perf_counter()
            for _ in range(args.inference_repeats):
                predictions = model(test_x)
            elapsed = max(time.perf_counter() - start, 1e-9)
            pytorch_mse = F.mse_loss(predictions, test_y).item()
            pytorch_steps_per_sec = test_x.shape[0] * test_x.shape[1] * args.inference_repeats / elapsed

        model_result = {
            "parameters": count_params(model),
            "pytorch_mse": pytorch_mse,
            "pytorch_steps_per_sec": pytorch_steps_per_sec,
            "pytorch_total_time_s": elapsed,
        }

        # ONNX export
        onnx_path = onnx_dir / f"{name}.onnx"
        onnx_result = export_onnx(model, example_input, onnx_path, name)
        model_result["onnx_export"] = onnx_result

        # ONNX Runtime benchmark
        ort_result = benchmark_onnxruntime(onnx_path, test_x, test_y, args.inference_repeats, device)
        model_result["onnx"] = ort_result
        model_result["onnx_mse"] = ort_result.get("mse", "n/a")
        model_result["onnx_steps_per_sec"] = ort_result.get("inference_steps_per_sec", "n/a")

        # TensorRT build and benchmark
        model_result["tensorrt"] = {}
        if args.tensorrt:
            for precision in args.precision:
                engine_path = trt_dir / f"{name}_{precision}.engine"
                trt_build = build_tensorrt_engine(onnx_path, engine_path, precision, args.batch_size, args.seq_len, device)
                trt_bench = {}
                if trt_build.get("available"):
                    trt_bench = benchmark_tensorrt(engine_path, test_x, test_y, args.inference_repeats, device)
                    trt_bench["build"] = trt_build
                model_result["tensorrt"][precision] = {**trt_build, **trt_bench}

        model_results[name] = model_result

    # Write report
    payload = {
        "status": "ok",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": env,
        "device": device,
        "config": {
            "hidden_size": args.hidden_size,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "inference_repeats": args.inference_repeats,
        },
        "model_results": model_results,
    }

    json_path, md_path = write_report(args.date, payload, output_dir)
    print(f"ONNX/TensorRT report written: {json_path.relative_to(ROOT)}")
    print(f"Markdown summary written: {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
