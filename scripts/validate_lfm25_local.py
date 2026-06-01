#!/usr/bin/env python3
"""Validate local LFM2.5 GGUF and 1.2B DPO inference paths."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GGUF = ROOT / "models" / "lfm25" / "LFM2.5-1.2B-Instruct-Q4_0.gguf"
DEFAULT_LLAMA_CLI = ROOT / "projects" / "llama.cpp" / "build" / "bin" / "llama-cli"
DEFAULT_DPO = ROOT / "models" / "lfm25-dpo-s1"


def sha256_file(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: pathlib.Path | str) -> str | None:
    try:
        completed = subprocess.run(
            [str(command), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return None
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return output or None


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


def parse_llama_cli_output(output: str) -> dict[str, Any]:
    response = ""
    timing: dict[str, float] = {}
    response_match = re.search(r"\n> [^\n]+\n\n(?P<response>.+?)\n\n\[ Prompt:", output, flags=re.DOTALL)
    if response_match:
        response = response_match.group("response").strip()
    timing_match = re.search(
        r"\[ Prompt: (?P<prompt>[0-9.]+) t/s \| Generation: (?P<generation>[0-9.]+) t/s \]",
        output,
    )
    if timing_match:
        timing = {
            "prompt_tps": float(timing_match.group("prompt")),
            "generation_tps": float(timing_match.group("generation")),
        }
    return {"response": response, "timing": timing}


def validate_gguf(args: argparse.Namespace) -> dict[str, Any]:
    model_path = pathlib.Path(args.gguf_model).expanduser().resolve()
    llama_cli = pathlib.Path(args.llama_cli).expanduser().resolve()
    result: dict[str, Any] = {
        "model_path": str(model_path),
        "llama_cli": str(llama_cli),
        "expected_size_bytes": args.gguf_expected_size,
    }
    if not model_path.exists():
        result.update({"status": "missing_model"})
        return result
    if not llama_cli.exists():
        result.update({"status": "missing_llama_cli"})
        return result

    size = model_path.stat().st_size
    result.update(
        {
            "status": "file_ok",
            "size_bytes": size,
            "sha256": sha256_file(model_path),
            "llama_cli_version": command_version(llama_cli),
            "size_matches_expected": size == args.gguf_expected_size if args.gguf_expected_size else None,
        }
    )
    if args.skip_gguf_run:
        return result

    command = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-p",
        args.gguf_prompt,
        "-n",
        str(args.max_new_tokens),
        "-c",
        str(args.ctx_size),
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
    ]
    run = run_command(command, timeout=args.timeout)
    parsed = parse_llama_cli_output(run["stdout"] + "\n" + run["stderr"])
    result.update(
        {
            "status": "ok" if run["returncode"] == 0 and parsed["response"] else "run_failed",
            "run": run,
            **parsed,
        }
    )
    return result


def validate_dpo(args: argparse.Namespace) -> dict[str, Any]:
    model_ref = args.dpo_model
    result: dict[str, Any] = {"model": model_ref}
    model_path = pathlib.Path(model_ref).expanduser()
    if model_path.exists():
        result["model_path"] = str(model_path.resolve())
        safetensors = model_path / "model.safetensors"
        if safetensors.exists():
            result["model_safetensors_size_bytes"] = safetensors.stat().st_size
            if args.hash_dpo:
                result["model_safetensors_sha256"] = sha256_file(safetensors)
        else:
            result["status"] = "missing_model_safetensors"
            return result

    if args.skip_dpo_run:
        result["status"] = "metadata_only"
        return result

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        result.update({"status": "missing_dependency", "error": str(exc)})
        return result

    try:
        torch.set_num_threads(args.threads)
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        messages = [{"role": "user", "content": args.dpo_prompt}]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = args.dpo_prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cpu",
        )
        model.eval()
        load_seconds = time.perf_counter() - load_started
        generate_started = time.perf_counter()
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "top_k": args.top_k,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        generation_seconds = time.perf_counter() - generate_started
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        result.update(
            {
                "status": "ok",
                "response": response,
                "load_seconds": round(load_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "total_seconds": round(time.perf_counter() - started, 3),
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "output_tokens": int(outputs.shape[-1] - inputs["input_ids"].shape[-1]),
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve validation evidence.
        result.update({"status": "run_failed", "error": f"{type(exc).__name__}: {exc}"})
    return result


def environment() -> dict[str, Any]:
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "llama_cli_on_path": shutil.which("llama-cli"),
    }
    try:
        import torch

        env.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-model", default=str(DEFAULT_GGUF))
    parser.add_argument("--gguf-expected-size", type=int, default=695751488)
    parser.add_argument("--llama-cli", default=str(DEFAULT_LLAMA_CLI))
    parser.add_argument("--dpo-model", default=str(DEFAULT_DPO))
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-gguf-run", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--skip-dpo-run", action="store_true")
    parser.add_argument("--hash-dpo", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--gguf-prompt", default="Write one short sentence about liquid neural networks.")
    parser.add_argument("--dpo-prompt", default="Write one short sentence about liquid neural networks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = {
        "run_date": dt.date.today().isoformat(),
        "environment": environment(),
        "gguf": None if args.skip_gguf else validate_gguf(args),
        "dpo": None if args.skip_dpo else validate_dpo(args),
    }
    if args.output:
        output = pathlib.Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failures = [
        section
        for section in (payload.get("gguf"), payload.get("dpo"))
        if section is not None and section.get("status") not in {"ok", "file_ok", "metadata_only"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
