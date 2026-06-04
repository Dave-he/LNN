#!/usr/bin/env python3
"""Run a tiny reproducible local LLM micro-eval.

This is the local execution companion to ``build_llm_battlecard.py``.  It is
not a public leaderboard substitute; it is a fast gate that answers a narrower
question:

    "Can the model file that is currently deployed on this machine follow
    simple deterministic instructions before we spend time on lm-eval,
    OpenCompass, or a larger 30B+ comparison?"

The default target is the LFM2.5 1.2B GGUF already validated in this repo via
llama.cpp.  The same runner can be pointed at an LFM2.5-8B-A1B GGUF, a 30B+
llama.cpp quantized model, or an OpenAI-compatible chat endpoint served by
llama-server, vLLM, SGLang, or a remote API.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_CLI = ROOT / "projects" / "llama.cpp" / "build" / "bin" / "llama-cli"
DEFAULT_MODEL = ROOT / "models" / "lfm25" / "LFM2.5-1.2B-Instruct-Q4_0.gguf"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "llm_micro_eval"


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    category: str
    prompt: str
    expected: str
    grader: str
    max_new_tokens: int = 16


TASKS: tuple[EvalTask, ...] = (
    EvalTask(
        task_id="arith_2_plus_3",
        category="arithmetic",
        prompt="Answer with exactly the number. What is 2+3?",
        expected="5",
        grader="exact_text",
        max_new_tokens=8,
    ),
    EvalTask(
        task_id="arith_17_minus_9",
        category="arithmetic",
        prompt="Answer with exactly the number. What is 17-9?",
        expected="8",
        grader="exact_text",
        max_new_tokens=8,
    ),
    EvalTask(
        task_id="arith_12_times_4",
        category="arithmetic",
        prompt="Answer with exactly the number. What is 12*4?",
        expected="48",
        grader="exact_text",
        max_new_tokens=8,
    ),
    EvalTask(
        task_id="instr_exact_word",
        category="instruction",
        prompt="Reply with exactly this word: BLUE",
        expected="BLUE",
        grader="exact_text",
        max_new_tokens=8,
    ),
    EvalTask(
        task_id="instr_exact_two_words",
        category="instruction",
        prompt="Reply with exactly these two words: liquid networks",
        expected="liquid networks",
        grader="exact_text",
        max_new_tokens=8,
    ),
    EvalTask(
        task_id="json_color_blue",
        category="structured_output",
        prompt='Return only this JSON object and no other text: {"color":"blue"}',
        expected='{"color":"blue"}',
        grader="json_equal",
        max_new_tokens=24,
    ),
    EvalTask(
        task_id="abstain_unknown",
        category="abstention",
        prompt=(
            "If the answer is not contained in this prompt, reply exactly UNKNOWN. "
            "What is the launch code for Project Zephyr-917?"
        ),
        expected="UNKNOWN",
        grader="exact_text",
        max_new_tokens=12,
    ),
)


def normalize_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^`+|`+$", "", text)
    text = text.strip().strip("\"'")
    text = re.sub(r"\s+", " ", text)
    # Common llama.cpp/chat-template suffixes should not decide exact-match
    # tasks if the answer itself is otherwise exact.
    text = text.replace("<|im_end|>", "").replace("</s>", "").strip()
    return text


def first_json_object(value: str) -> Any:
    match = re.search(r"\{.*\}", value, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def grade_response(task: EvalTask, response: str) -> dict[str, Any]:
    normalized = normalize_text(response)
    expected = normalize_text(task.expected)
    if task.grader == "exact_text":
        passed = normalized.casefold() == expected.casefold()
        return {"passed": passed, "normalized": normalized, "expected_normalized": expected}
    if task.grader == "json_equal":
        try:
            actual_json = first_json_object(response)
            expected_json = json.loads(task.expected)
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "passed": False,
                "normalized": normalized,
                "expected_normalized": expected,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "passed": actual_json == expected_json,
            "normalized": json.dumps(actual_json, sort_keys=True, separators=(",", ":")),
            "expected_normalized": json.dumps(expected_json, sort_keys=True, separators=(",", ":")),
        }
    raise ValueError(f"unknown grader: {task.grader}")


def parse_llama_output(output: str, prompt: str) -> dict[str, Any]:
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

    response = ""
    escaped_prompt = re.escape(prompt)
    response_match = re.search(
        rf"\n> {escaped_prompt}\n\n(?P<response>.*?)(?:\n\n\[ Prompt:|\n\[ Prompt:)",
        output,
        flags=re.DOTALL,
    )
    if response_match:
        response = response_match.group("response").strip()
    else:
        before_timing = re.split(r"\n\[ Prompt:", output, maxsplit=1)[0]
        candidate_lines: list[str] = []
        capture = False
        for line in before_timing.splitlines():
            stripped = line.strip()
            if stripped == f"> {prompt}":
                capture = True
                continue
            if not capture:
                continue
            if not stripped:
                continue
            if stripped.startswith(("Loading model", "build", "model", "modalities", "available commands", "/")):
                continue
            candidate_lines.append(stripped)
        response = "\n".join(candidate_lines).strip()
    return {"response": response, "timing": timing}


def run_llama_task(args: argparse.Namespace, task: EvalTask) -> dict[str, Any]:
    command = [
        str(pathlib.Path(args.llama_cli).expanduser()),
        "-m",
        str(pathlib.Path(args.model).expanduser()),
        "-p",
        task.prompt,
        "-n",
        str(task.max_new_tokens),
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
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        seconds = round(time.perf_counter() - started, 3)
        output = f"{completed.stdout}\n{completed.stderr}"
        parsed = parse_llama_output(output, task.prompt)
        grade = grade_response(task, parsed["response"])
        status = "ok" if completed.returncode == 0 else "run_failed"
        return {
            "task_id": task.task_id,
            "category": task.category,
            "prompt": task.prompt,
            "expected": task.expected,
            "grader": task.grader,
            "status": status,
            "returncode": completed.returncode,
            "seconds": seconds,
            "response": parsed["response"],
            "timing": parsed["timing"],
            "grade": grade,
            "command": command,
            "stderr_tail": completed.stderr[-1000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "task_id": task.task_id,
            "category": task.category,
            "prompt": task.prompt,
            "expected": task.expected,
            "grader": task.grader,
            "status": "timeout",
            "returncode": 124,
            "seconds": args.timeout,
            "response": "",
            "timing": {},
            "grade": {"passed": False, "normalized": "", "expected_normalized": normalize_text(task.expected)},
            "command": command,
            "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "command timed out",
        }


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _extract_openai_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message.get("content") or "").strip()
    if first.get("text") is not None:
        return str(first.get("text") or "").strip()
    return ""


def run_openai_chat_task(args: argparse.Namespace, task: EvalTask) -> dict[str, Any]:
    """Run one task against an OpenAI-compatible /v1/chat/completions endpoint."""
    url = _chat_completions_url(args.openai_base_url)
    model = args.openai_model or args.model_name
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": task.prompt}],
        "temperature": args.temperature,
        "max_tokens": task.max_new_tokens,
    }
    if args.seed is not None:
        body["seed"] = args.seed
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_text = response.read().decode("utf-8")
            seconds = round(time.perf_counter() - started, 3)
            payload = json.loads(response_text)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return _failed_endpoint_result(args, task, "http_error", exc.code, time.perf_counter() - started, error_body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return _failed_endpoint_result(args, task, "run_failed", 1, time.perf_counter() - started, f"{type(exc).__name__}: {exc}")

    response_text = _extract_openai_response(payload)
    grade = grade_response(task, response_text)
    usage = payload.get("usage") or {}
    timing: dict[str, float] = {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and seconds > 0:
        timing["generation_tps"] = round(float(completion_tokens) / seconds, 3)
    return {
        "task_id": task.task_id,
        "category": task.category,
        "prompt": task.prompt,
        "expected": task.expected,
        "grader": task.grader,
        "status": "ok",
        "returncode": 0,
        "seconds": seconds,
        "response": response_text,
        "timing": timing,
        "usage": usage,
        "grade": grade,
        "endpoint": url,
        "model": model,
    }


def _failed_endpoint_result(
    args: argparse.Namespace,
    task: EvalTask,
    status: str,
    returncode: int,
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "category": task.category,
        "prompt": task.prompt,
        "expected": task.expected,
        "grader": task.grader,
        "status": status,
        "returncode": returncode,
        "seconds": round(elapsed, 3),
        "response": "",
        "timing": {},
        "grade": {"passed": False, "normalized": "", "expected_normalized": normalize_text(task.expected)},
        "endpoint": _chat_completions_url(args.openai_base_url),
        "model": args.openai_model or args.model_name,
        "error": error[-1000:],
    }


def run_task(args: argparse.Namespace, task: EvalTask) -> dict[str, Any]:
    if args.backend == "llama-cli":
        return run_llama_task(args, task)
    if args.backend == "openai-chat":
        return run_openai_chat_task(args, task)
    raise ValueError(f"unknown backend: {args.backend}")


def selected_tasks(task_filter: str) -> list[EvalTask]:
    if not task_filter or task_filter == "all":
        return list(TASKS)
    wanted = {part.strip() for part in task_filter.split(",") if part.strip()}
    selected = [task for task in TASKS if task.task_id in wanted or task.category in wanted]
    if not selected:
        raise ValueError(f"no tasks matched --tasks={task_filter!r}")
    return selected


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for result in results if result.get("grade", {}).get("passed"))
    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        cat = result["category"]
        stats = by_category.setdefault(cat, {"n": 0, "passed": 0})
        stats["n"] += 1
        stats["passed"] += int(bool(result.get("grade", {}).get("passed")))
    for stats in by_category.values():
        stats["accuracy"] = round(stats["passed"] / stats["n"], 4) if stats["n"] else 0.0
    generation_speeds = [
        float(result["timing"]["generation_tps"])
        for result in results
        if result.get("timing", {}).get("generation_tps") is not None
    ]
    return {
        "n": n,
        "passed": passed,
        "failed": n - passed,
        "accuracy": round(passed / n, 4) if n else 0.0,
        "by_category": by_category,
        "generation_tps_mean": round(statistics.fmean(generation_speeds), 3) if generation_speeds else None,
        "generation_tps_median": round(statistics.median(generation_speeds), 3) if generation_speeds else None,
    }


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    model = pathlib.Path(args.model).expanduser()
    llama_cli = pathlib.Path(args.llama_cli).expanduser()
    tasks = selected_tasks(args.tasks)
    safe_model_name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.model_name).strip("_") or "model"
    if args.backend == "llama-cli" and not args.allow_missing and not model.exists():
        raise FileNotFoundError(f"model not found: {model}")
    if args.backend == "llama-cli" and not args.allow_missing and not llama_cli.exists():
        raise FileNotFoundError(f"llama-cli not found: {llama_cli}")

    results = [] if args.dry_run else [run_task(args, task) for task in tasks]
    return {
        "run_id": f"{args.date}_{safe_model_name}_micro_eval",
        "date": args.date,
        "backend": args.backend,
        "model_name": args.model_name,
        "model_path": _display_path(model),
        "model_exists": model.exists(),
        "llama_cli": _display_path(llama_cli),
        "llama_cli_exists": llama_cli.exists(),
        "openai_base_url": args.openai_base_url if args.backend == "openai-chat" else None,
        "openai_model": (args.openai_model or args.model_name) if args.backend == "openai-chat" else None,
        "config": {
            "ctx_size": args.ctx_size,
            "threads": args.threads,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repeat_penalty": args.repeat_penalty,
            "seed": args.seed,
            "timeout": args.timeout,
            "tasks": args.tasks,
            "min_accuracy": args.min_accuracy,
            "dry_run": args.dry_run,
        },
        "summary": summarize_results(results) if results else {"n": len(tasks), "passed": 0, "failed": 0, "accuracy": None},
        "results": results,
    }


def format_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "---",
        f"title: LLM micro-eval - {payload['model_name']}",
        f"date: {payload['date']}",
        "tags: [LFM2.5, LNN, LLM, local-eval, llama.cpp, micro-benchmark]",
        "parent: [[PRD_LNN_Edge_Research]]",
        "---",
        "",
        f"# LLM micro-eval - {payload['model_name']}",
        "",
        "## Summary",
        "",
        f"- Backend: `{payload.get('backend', 'llama-cli')}`",
        f"- Model: `{payload['model_path']}`",
        f"- llama-cli: `{payload['llama_cli']}`",
        f"- Accuracy: **{_format_accuracy(summary.get('accuracy'))}** ({summary.get('passed')}/{summary.get('n')})",
        f"- Mean generation speed: `{summary.get('generation_tps_mean')}` tok/s",
        "",
        "## Category Split",
        "",
        "| Category | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for category, stats in sorted((summary.get("by_category") or {}).items()):
        lines.append(f"| {category} | {stats['passed']} | {stats['n']} | {stats['accuracy']:.1%} |")
    lines.extend([
        "",
        "## Tasks",
        "",
        "| Task | Category | Pass | Expected | Response |",
        "|---|---|---:|---|---|",
    ])
    for result in payload.get("results", []):
        passed = "yes" if result.get("grade", {}).get("passed") else "no"
        response = _md_cell(result.get("response", ""))
        expected = _md_cell(result.get("expected", ""))
        lines.append(f"| {result['task_id']} | {result['category']} | {passed} | `{expected}` | `{response}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "- This is a deployment sanity check. Passing it does not prove public leaderboard strength; "
            "failing it blocks any serious 3B-vs-30B claim until prompt/template/runtime issues are fixed."
        ),
        (
            "- Next gate: run the same script on `LFM2.5-8B-A1B-GGUF`, then run a public harness subset "
            "such as lm-eval or OpenCompass."
        ),
    ])
    if payload.get("backend") == "openai-chat":
        lines[lines.index(f"- Model: `{payload['model_path']}`")] = f"- OpenAI model: `{payload.get('openai_model')}`"
        lines[lines.index(f"- llama-cli: `{payload['llama_cli']}`")] = (
            f"- Endpoint: `{payload.get('openai_base_url')}`"
        )
    return "\n".join(lines) + "\n"


def _format_accuracy(value: Any) -> str:
    if value is None:
        return "not run"
    return f"{float(value):.1%}"


def _md_cell(value: str) -> str:
    return normalize_text(value).replace("|", "\\|").replace("\n", " ")[:160]


def write_outputs(payload: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{payload['run_id']}.json"
    md_path = output_dir / f"{payload['run_id']}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(payload), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--backend", choices=["llama-cli", "openai-chat"], default="llama-cli")
    parser.add_argument("--model-name", default="lfm25_1.2b_instruct_q4")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--llama-cli", default=str(DEFAULT_LLAMA_CLI))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tasks", default="all", help="Comma-separated task ids or categories; default: all.")
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    parser.add_argument("--dry-run", action="store_true", help="Build task/config payload without running llama-cli.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail early if model or llama-cli is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    if not args.no_write:
        json_path, md_path = write_outputs(payload, pathlib.Path(args.output_dir).expanduser())
        payload["json_path"] = _display_path(json_path)
        payload["markdown_path"] = _display_path(md_path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_markdown(payload))
    failed_runs = [result for result in payload.get("results", []) if result.get("status") != "ok"]
    if failed_runs:
        return 1
    accuracy = payload.get("summary", {}).get("accuracy")
    if accuracy is not None and float(accuracy) < args.min_accuracy:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
