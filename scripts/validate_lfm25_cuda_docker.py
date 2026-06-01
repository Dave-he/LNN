#!/usr/bin/env python3
"""Validate LFM2.5 GGUF inference through a CUDA-enabled Jetson Docker image."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import shlex
import subprocess
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin"


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_command_safe(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        return run_command(command, timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "seconds": timeout,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "command timed out",
        }


def parse_int_list(value: str) -> list[int]:
    parsed: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        parsed.append(int(part))
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_llama_output(output: str) -> dict[str, Any]:
    response = ""
    response_match = re.search(
        r"\n> [^\n]+\n\n(?P<response>.*?)(?:llama_memory_breakdown_print:|\n\[ Prompt:)",
        output,
        flags=re.DOTALL,
    )
    if response_match:
        response = response_match.group("response").strip()
    timing: dict[str, float] = {}
    timing_match = re.search(
        r"\[ Prompt: (?P<prompt>[0-9.]+) t/s \| Generation: (?P<generation>[0-9.]+) t/s \]",
        output,
    )
    if timing_match:
        timing = {
            "prompt_tps": float(timing_match.group("prompt")),
            "generation_tps": float(timing_match.group("generation")),
        }
    cuda_line = ""
    memory_line = ""
    for line in output.splitlines():
        if "ggml_cuda_init: found" in line:
            cuda_line = line.strip()
        if "CUDA0" in line and "|" in line and "memory breakdown" not in line:
            memory_line = line.strip()
    if not response:
        response_text = re.split(r"llama_memory_breakdown_print:|\n\[ Prompt:", output, maxsplit=1)[0]
        response_lines = []
        for line in response_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith((">", "ggml_", "llama_", "common_", "srv ")):
                continue
            if stripped in {"Loading model...", "Failed to load the model"}:
                continue
            response_lines.append(stripped)
        response = "\n".join(response_lines).strip()
    return {
        "cuda_detected": bool(cuda_line),
        "cuda_line": cuda_line,
        "cuda_memory_line": memory_line,
        "response": response,
        "timing": timing,
    }


def classify_attempt(run: dict[str, Any], parsed: dict[str, Any], gpu_layers: int) -> str:
    combined_output = f"{run.get('stdout', '')}\n{run.get('stderr', '')}".lower()
    if run["returncode"] == 0 and parsed["cuda_detected"] and gpu_layers > 0:
        return "ok"
    if run["returncode"] == 0 and gpu_layers == 0:
        return "cpu_fallback"
    if "out of memory" in combined_output or "nvmapmem" in combined_output:
        return "cuda_oom"
    if "segmentation fault" in combined_output or "dumped core" in combined_output:
        return "segfault"
    if run["returncode"] == 124:
        return "timeout"
    return "run_failed"


def docker_llama_command(args: argparse.Namespace, case: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    inner = [
        "timeout",
        str(args.inner_timeout),
        "llama-cli",
        "-m",
        case["model"],
        "-p",
        args.prompt,
        "-n",
        str(candidate["max_new_tokens"]),
        "-c",
        str(candidate["ctx_size"]),
        "-t",
        str(args.threads),
        "--temp",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--repeat-penalty",
        str(args.repeat_penalty),
        "--no-display-prompt",
        "--no-warmup",
        "--single-turn",
        "--simple-io",
        "--n-gpu-layers",
        str(candidate["gpu_layers"]),
    ]
    return [
        "docker",
        "run",
        "--rm",
        "--runtime",
        "nvidia",
        "--network",
        "host",
        "--ipc",
        "host",
        "-e",
        "CUDA_MODULE_LOADING=LAZY",
        "-v",
        f"{ROOT}:/workspace/LNN",
        "-w",
        "/workspace/LNN",
        args.image,
        "bash",
        "-lc",
        shlex.join(inner),
    ]


def docker_probe_command(args: argparse.Namespace) -> list[str]:
    inner = [
        "bash",
        "-lc",
        "nvidia-smi || true; echo '--- llama-cli ---'; llama-cli --version",
    ]
    return [
        "docker",
        "run",
        "--rm",
        "--runtime",
        "nvidia",
        "--network",
        "host",
        "--ipc",
        "host",
        "-e",
        "CUDA_MODULE_LOADING=LAZY",
        args.image,
        "bash",
        "-lc",
        shlex.join(inner),
    ]


def collect_host_info() -> dict[str, Any]:
    commands = {
        "jetson_model": ["bash", "-lc", "tr -d '\\0' < /proc/device-tree/model 2>/dev/null || true"],
        "l4t_release": ["bash", "-lc", "head -n 1 /etc/nv_tegra_release 2>/dev/null || true"],
        "cuda_version": ["bash", "-lc", "cat /usr/local/cuda/version.json 2>/dev/null || nvcc --version 2>/dev/null || true"],
        "nvidia_smi": ["bash", "-lc", "nvidia-smi 2>/dev/null || true"],
        "memory": ["free", "-h"],
    }
    return {name: run_command_safe(command, timeout=10) for name, command in commands.items()}


def build_candidates(gpu_layers: list[int], ctx_sizes: list[int], max_new_tokens: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for ctx_size in ctx_sizes:
        for gpu_layer_count in gpu_layers:
            key = (gpu_layer_count, ctx_size, max_new_tokens)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "gpu_layers": gpu_layer_count,
                    "ctx_size": ctx_size,
                    "max_new_tokens": max_new_tokens,
                }
            )
    return candidates


def validate_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    model_path = ROOT / case["model"]
    result: dict[str, Any] = {
        "name": case["name"],
        "model": case["model"],
        "model_exists": model_path.exists(),
        "candidates": case["candidates"],
    }
    if not model_path.exists():
        result["status"] = "missing_model"
        return result
    attempts = []
    for candidate in case["candidates"]:
        run = run_command_safe(docker_llama_command(args, case, candidate), timeout=args.timeout)
        parsed = parse_llama_output(run["stdout"] + "\n" + run["stderr"])
        status = classify_attempt(run, parsed, candidate["gpu_layers"])
        attempt = {
            "status": status,
            **candidate,
            "run": run,
            **parsed,
        }
        attempts.append(attempt)
        if status == "ok":
            result.update(
                {
                    "status": "ok",
                    "selected": candidate,
                    "attempts": attempts,
                    "run": run,
                    **parsed,
                }
            )
            return result
    result.update(
        {
            "status": attempts[-1]["status"] if attempts else "run_failed",
            "attempts": attempts,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", default=f"analysis/lfm25/{dt.date.today().isoformat()}_lfm25_cuda_docker_validation.json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--inner-timeout", type=int, default=90)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--prompt", default="Say CUDA works.")
    parser.add_argument("--official-gpu-layers", type=parse_int_list, default=parse_int_list("4,3,2,1"))
    parser.add_argument("--official-ctx-sizes", type=parse_int_list, default=parse_int_list("256,128"))
    parser.add_argument("--dpo-gpu-layers", type=parse_int_list, default=parse_int_list("1"))
    parser.add_argument("--dpo-ctx-sizes", type=parse_int_list, default=parse_int_list("128"))
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [
        {
            "name": "official_instruct_q4_0",
            "model": "models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf",
            "candidates": build_candidates(args.official_gpu_layers, args.official_ctx_sizes, args.max_new_tokens),
        }
    ]
    if not args.skip_dpo:
        cases.append(
            {
                "name": "dpo_q4_0",
                "model": "models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf",
                "candidates": build_candidates(args.dpo_gpu_layers, args.dpo_ctx_sizes, args.max_new_tokens),
            }
        )
    payload = {
        "run_date": dt.date.today().isoformat(),
        "image": args.image,
        "workspace": str(ROOT),
        "host": collect_host_info(),
        "container_probe": None if args.skip_probe else run_command_safe(docker_probe_command(args), timeout=args.timeout),
        "cases": [validate_case(args, case) for case in cases],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if any(case["status"] != "ok" for case in payload["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
