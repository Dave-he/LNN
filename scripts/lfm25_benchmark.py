#!/usr/bin/env python3
"""LFM2.5 benchmark for Jetson Orin Nano (and similar edge devices).

This script benchmarks LFM2.5 models on Jetson hardware:
- Measures model load time
- Measures prefill (first token) latency
- Measures decode (token/s) throughput
- Tracks memory usage
- Optionally samples power via tegrastats

Usage:
    # In container on Jetson
    python scripts/lfm25_benchmark.py --model LiquidAI/LFM2.5-350M --quick

    # With power sampling
    python scripts/lfm25_benchmark.py --model LiquidAI/LFM2.5-350M --power

Output:
    analysis/jetson/YYYY-MM-DD_lfm25_benchmark.json
    analysis/jetson/YYYY-MM-DD_lfm25_benchmark.md
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import pathlib
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-350M", help="Model name to benchmark.")
    parser.add_argument("--model-short", type=str, default=None, help="Short model name for report (auto if None).")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test: fewer tokens/repeats.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--dtype", type=str, default="float16", help="dtype for inference (float32, float16, bfloat16).")
    parser.add_argument("--prompt", type=str, default="Write a short poem about artificial intelligence:", help="Prompt for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repeats.")
    parser.add_argument("--power", action="store_true", help="Sample power via tegrastats.")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache for decode benchmark.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "analysis" / "jetson"), help="Output directory.")
    return parser.parse_args()


def detect_environment() -> dict[str, Any]:
    """Detect environment (platform, CUDA, Jetson info)."""
    import platform
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["cuda_device"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2),
            }
    except ImportError:
        pass

    nv_tegra_release = pathlib.Path("/etc/nv_tegra_release")
    if nv_tegra_release.exists():
        env["nv_tegra_release"] = nv_tegra_release.read_text().strip()

    return env


def benchmark_model(args: argparse.Namespace) -> dict[str, Any]:
    """Run the model benchmark."""
    import torch
    import torch.nn.functional as F

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float16)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers not installed; pip install transformers accelerate sentencepiece")

    from lnn.edge.tegrastats import TegrastatsSampler, energy_per_step

    # Download and load model
    print(f"Loading model: {args.model}")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with TegrastatsSampler(interval_ms=100) if args.power else contextlib.nullcontext() as sampler:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        model.eval()
    load_time = time.perf_counter() - load_start

    load_power_summary = sampler.summary() if args.power else None

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = total_bytes / (1024 * 1024)
    model_size_gb = total_bytes / (1024 * 1024 * 1024)

    print(f"Model loaded: {total_params:,} params, {model_size_mb:.1f} MB")

    # Warmup
    warmup_prompt = "Hello, world!"
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=8)

    # Benchmark prefill and decode
    prompts = [
        args.prompt,
        "Explain liquid neural networks in one sentence:",
        "Write a 10-line computer program that prints primes:",
    ]
    if args.quick:
        prompts = prompts[:1]

    all_results = []

    for repeat_idx in range(args.repeats):
        for prompt_idx, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            prompt_length = input_ids.shape[1]

            # Prefill benchmark (first token)
            prefill_start = time.perf_counter()
            with torch.no_grad():
                prefill_out = model(input_ids, use_cache=not args.no_cache)
                next_token_logits = prefill_out.logits[:, -1:, :]
            prefill_time = time.perf_counter() - prefill_start

            # Decode benchmark (generating N tokens)
            generate_max = args.max_new_tokens if not args.quick else min(args.max_new_tokens, 32)
            generate_start = time.perf_counter()

            with torch.no_grad(), TegrastatsSampler(interval_ms=100) if args.power else contextlib.nullcontext() as decode_sampler:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generate_max,
                    use_cache=not args.no_cache,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            generate_time = time.perf_counter() - generate_start
            decode_power_summary = decode_sampler.summary() if args.power else None

            tokens_generated = outputs.shape[1] - input_ids.shape[1]
            decode_time = max(generate_time - prefill_time * (1 / generate_max) if tokens_generated > 0 else 0.0, 1e-9)

            tokens_per_second = tokens_generated / decode_time
            prefill_tokens_per_second = prompt_length / prefill_time

            # Get generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            result = {
                "repeat": repeat_idx,
                "prompt_index": prompt_idx,
                "prompt_length": prompt_length,
                "tokens_generated": tokens_generated,
                "max_new_tokens": generate_max,
                "load_time_s": load_time,
                "prefill_time_s": prefill_time,
                "prefill_tokens_per_second": prefill_tokens_per_second,
                "generate_time_s": generate_time,
                "decode_time_s": decode_time,
                "tokens_per_second": tokens_per_second,
                "generated_text": generated_text,
            }

            if decode_power_summary:
                result["power_summary"] = decode_power_summary
                energy_mj = decode_power_summary.get("energy_mj", {}).get("VDD_IN")
                if energy_mj is not None and tokens_generated > 0:
                    result["energy_mj_per_token"] = energy_mj / tokens_generated

            all_results.append(result)

    # Aggregate
    all_prefill_times = [r["prefill_time_s"] for r in all_results]
    all_decode_times = [r["decode_time_s"] for r in all_results]
    all_tokens_per_second = [r["tokens_per_second"] for r in all_results]

    aggregated = {
        "model_name": args.model_short or args.model.split("/")[-1] if "/" in args.model else args.model,
        "model_path": args.model,
        "total_params": total_params,
        "model_size_mb": model_size_mb,
        "model_size_gb": model_size_gb,
        "device": device,
        "dtype": args.dtype,
        "load_time_s": load_time,
        "repeats": args.repeats,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "use_cache": not args.no_cache,
        "prefill_time_s": {
            "mean": sum(all_prefill_times) / len(all_prefill_times),
            "min": min(all_prefill_times),
            "max": max(all_prefill_times),
        },
        "tokens_per_second": {
            "mean": sum(all_tokens_per_second) / len(all_tokens_per_second),
            "min": min(all_tokens_per_second),
            "max": max(all_tokens_per_second),
        },
        "all_results": all_results,
    }

    if load_power_summary:
        aggregated["load_power"] = load_power_summary

    # Peak memory
    if use_cuda:
        try:
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            aggregated["memory_mb"] = {
                "peak_allocated_mb": max_allocated / (1024 * 1024),
                "peak_reserved_mb": max_reserved / (1024 * 1024),
            }
        except Exception:
            pass

    return aggregated


def write_report(run_date: str, payload: dict[str, Any], output_dir: pathlib.Path):
    """Write JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_date}_lfm25_benchmark.json"
    md_path = output_dir / f"{run_date}_lfm25_benchmark.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = payload.get("environment", {})
    bench = payload.get("benchmark", {})
    model_name = bench.get("model_name", "unknown")

    lines = [
        "---",
        f"title: Jetson LFM2.5 Benchmark - {run_date} - {model_name}",
        f"date: {run_date}",
        "tags: [LNN, LFM2.5, Jetson, benchmark, edge-ai]",
        "---",
        "",
        f"# Jetson LFM2.5 Benchmark - {run_date}",
        "",
        "## Environment",
        f"- Platform: {env.get('platform')}",
        f"- Python: {env.get('python')}",
        f"- PyTorch: {env.get('torch_version')}",
        f"- CUDA available: {env.get('cuda_available')}",
    ]

    if env.get("nv_tegra_release"):
        lines.extend(["- Jetson BSP:", "", "```text", env["nv_tegra_release"], "```"])
    if env.get("cuda_device"):
        dev = env["cuda_device"]
        lines.extend([
            f"- CUDA device: {dev.get('name')}",
            f"- Total memory: {dev.get('total_memory_mb')} MB",
        ])

    lines.extend([
        "",
        "## Model",
        f"- Name: {bench.get('model_path')}",
        f"- Parameters: {bench.get('total_params', 0):,}",
        f"- Size: {bench.get('model_size_mb', 0):.1f} MB ({bench.get('model_size_gb', 0):.2f} GB)",
        f"- dtype: {bench.get('dtype')}",
        f"- Device: {bench.get('device')}",
        f"- Load time: {bench.get('load_time_s', 0):.2f} s",
    ])

    mem = bench.get("memory_mb")
    if mem:
        lines.extend([
            f"- Peak allocated: {mem.get('peak_allocated_mb', 0):.1f} MB",
            f"- Peak reserved: {mem.get('peak_reserved_mb', 0):.1f} MB",
        ])

    tps = bench.get("tokens_per_second", {})
    prefill = bench.get("prefill_time_s", {})

    lines.extend([
        "",
        "## Performance",
        f"- Repeats: {bench.get('repeats')}",
        f"- Prompts per repeat: {bench.get('num_prompts')}",
        f"- Tokens generated per run: {bench.get('max_new_tokens')}",
        f"- KV cache enabled: {bench.get('use_cache')}",
        "",
        "| Metric | Mean | Min | Max |",
        "|---|---:|---:|---:|",
        f"| Tokens/s | {tps.get('mean', 0):.2f} | {tps.get('min', 0):.2f} | {tps.get('max', 0):.2f} |",
        f"| Prefill time (s) | {prefill.get('mean', 0):.3f} | {prefill.get('min', 0):.3f} | {prefill.get('max', 0):.3f} |",
    ])

    # Check if energy data is available
    all_results = bench.get("all_results", [])
    if all_results and any("energy_mj_per_token" in r for r in all_results):
        energies = [r.get("energy_mj_per_token") for r in all_results if r.get("energy_mj_per_token") is not None]
        if energies:
            e_mean = sum(energies) / len(energies)
            lines.extend([
                "",
                "## Energy (Jetson only)",
                f"- Energy per token: {e_mean:.1f} mJ/token",
            ])

    # Example generation
    if all_results:
        last_result = all_results[-1]
        example_text = last_result.get("generated_text", "")
        lines.extend([
            "",
            "## Example Generation",
            "",
            "```text",
            example_text,
            "```",
        ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()

    output_dir = pathlib.Path(args.output_dir)
    env = detect_environment()

    import contextlib

    benchmark = benchmark_model(args)

    payload = {
        "status": "ok",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": env,
        "benchmark": benchmark,
    }

    json_path, md_path = write_report(args.date, payload, output_dir)
    print(f"LFM2.5 benchmark report written: {json_path.relative_to(ROOT)}")
    print(f"Markdown summary written: {md_path.relative_to(ROOT)}")

    # Also print summary to stdout
    tps = benchmark.get("tokens_per_second", {})
    print(f"\nSummary: {benchmark.get('model_name')}")
    print(f"  Tokens/s: {tps.get('mean', 0):.2f} (mean)")
    print(f"  Model size: {benchmark.get('model_size_mb', 0):.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
