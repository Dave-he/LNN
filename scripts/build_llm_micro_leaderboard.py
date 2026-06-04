#!/usr/bin/env python3
"""Build a reproducible micro-eval leaderboard from local LLM eval JSONs.

This aggregates outputs from ``scripts/run_llm_micro_eval.py`` into one table
that can hold local LFM/LNN-family candidates and 30B+ OpenAI-compatible
endpoint baselines.  It is intentionally a smoke-level leaderboard: useful for
deployment sanity checks and quick regressions, not a replacement for lm-eval,
OpenCompass, or a public leaderboard submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT_GLOB = "analysis/llm_micro_eval/*_micro_eval.json"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "llm_micro_eval"


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _model_text(payload: dict[str, Any]) -> str:
    parts = [
        payload.get("model_name"),
        payload.get("model_path"),
        payload.get("openai_model"),
    ]
    return " ".join(str(part) for part in parts if part).casefold()


def infer_comparison_role(payload: dict[str, Any]) -> str:
    """Infer leaderboard role from model identifiers only.

    The labels are intentionally conservative.  They are grouping hints for the
    report and do not prove parameter counts unless a separate source does.
    """
    text = _model_text(payload)
    if re.search(r"(30b|32b|34b|70b|72b|100b|120b)", text):
        return "30b_plus_baseline"
    if "8b-a1b" in text or "8b_a1b" in text:
        return "active_under_3b_moe_candidate"
    if re.search(r"(1\.2b|1_2b|1-2b|1\.17b|2b|3b)", text):
        return "under_3b_candidate"
    return "unknown"


def _task_signature(payload: dict[str, Any]) -> list[str]:
    results = payload.get("results") or []
    task_ids = [str(result.get("task_id")) for result in results if result.get("task_id")]
    if task_ids:
        return sorted(task_ids)
    tasks = payload.get("config", {}).get("tasks")
    if isinstance(tasks, str) and tasks:
        return [tasks]
    return []


def load_micro_eval(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict):
        return None
    accuracy = _safe_float(summary.get("accuracy"))
    n_tasks = _safe_int(summary.get("n")) or 0
    entry = {
        "rank": None,
        "run_id": payload.get("run_id") or path.stem,
        "date": payload.get("date"),
        "model_name": payload.get("model_name") or payload.get("openai_model") or "unknown",
        "backend": payload.get("backend", "llama-cli"),
        "comparison_role": infer_comparison_role(payload),
        "source_path": _display_path(path),
        "model_path": payload.get("model_path"),
        "openai_model": payload.get("openai_model"),
        "openai_base_url": payload.get("openai_base_url"),
        "task_signature": _task_signature(payload),
        "summary": {
            "n": n_tasks,
            "passed": _safe_int(summary.get("passed")) or 0,
            "failed": _safe_int(summary.get("failed")) or 0,
            "accuracy": accuracy,
            "generation_tps_mean": _safe_float(summary.get("generation_tps_mean")),
            "generation_tps_median": _safe_float(summary.get("generation_tps_median")),
        },
        "by_category": summary.get("by_category") or {},
    }
    return entry


def discover_entries(input_glob: str) -> list[dict[str, Any]]:
    paths = sorted(ROOT.glob(input_glob))
    entries = []
    for path in paths:
        entry = load_micro_eval(path)
        if entry is not None:
            entries.append(entry)
    return entries


def _rank_key(entry: dict[str, Any]) -> tuple[float, int, float, str]:
    summary = entry["summary"]
    accuracy = summary.get("accuracy")
    speed = summary.get("generation_tps_mean")
    return (
        float(accuracy) if accuracy is not None else -1.0,
        int(summary.get("n") or 0),
        float(speed) if speed is not None else -1.0,
        str(entry.get("model_name") or ""),
    )


def assign_ranks(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rankable = [entry for entry in entries if entry["summary"].get("accuracy") is not None]
    unranked = [entry for entry in entries if entry["summary"].get("accuracy") is None]
    ordered = sorted(rankable, key=_rank_key, reverse=True)
    last_score: tuple[float, int, float] | None = None
    last_rank = 0
    for index, entry in enumerate(ordered, start=1):
        score = _rank_key(entry)[:3]
        if score != last_score:
            last_rank = index
            last_score = score
        entry["rank"] = last_rank
    for entry in unranked:
        entry["rank"] = None
    return ordered + unranked


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    entries = assign_ranks(discover_entries(args.input_glob))
    rankable = [entry for entry in entries if entry.get("rank") is not None]
    top = rankable[0] if rankable else None
    roles: dict[str, int] = {}
    for entry in entries:
        roles[entry["comparison_role"]] = roles.get(entry["comparison_role"], 0) + 1
    return {
        "run_id": f"{args.date}_llm_micro_leaderboard",
        "date": args.date,
        "input_glob": args.input_glob,
        "summary": {
            "n_entries": len(entries),
            "n_rankable": len(rankable),
            "roles": roles,
            "top_model": top.get("model_name") if top else None,
            "top_accuracy": top.get("summary", {}).get("accuracy") if top else None,
            "top_generation_tps_mean": top.get("summary", {}).get("generation_tps_mean") if top else None,
        },
        "entries": entries,
    }


def _format_accuracy(value: Any) -> str:
    if value is None:
        return "not run"
    return f"{float(value):.1%}"


def _format_speed(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _source_link(path: str) -> str:
    md_path = pathlib.PurePosixPath(re.sub(r"\.json$", ".md", path)).name
    return f"[md]({md_path})"


def format_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "---",
        "title: LLM micro-eval leaderboard",
        f"date: {payload['date']}",
        "tags: [LFM2.5, LNN, LLM, local-eval, micro-benchmark, leaderboard]",
        "parent: [[PRD_LNN_Edge_Research]]",
        "---",
        "",
        f"# LLM micro-eval leaderboard - {payload['date']}",
        "",
        "## Summary",
        "",
        f"- Scanned: `{payload['input_glob']}`",
        f"- Entries: **{summary['n_entries']}** total, **{summary['n_rankable']}** rankable",
        f"- Current leader: `{summary.get('top_model') or 'none'}` "
        f"({_format_accuracy(summary.get('top_accuracy'))}, {_format_speed(summary.get('top_generation_tps_mean'))} tok/s)",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Backend | Role | Accuracy | Tasks | Mean tok/s | Source |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for entry in payload["entries"]:
        rank = entry["rank"] if entry["rank"] is not None else "-"
        s = entry["summary"]
        lines.append(
            "| {rank} | `{model}` | `{backend}` | `{role}` | {accuracy} | {passed}/{n} | {speed} | {source} |".format(
                rank=rank,
                model=entry["model_name"],
                backend=entry["backend"],
                role=entry["comparison_role"],
                accuracy=_format_accuracy(s.get("accuracy")),
                passed=s.get("passed"),
                n=s.get("n"),
                speed=_format_speed(s.get("generation_tps_mean")),
                source=_source_link(entry["source_path"]),
            )
        )
    lines.extend([
        "",
        "## Category Split",
        "",
        "| Model | arithmetic | instruction | structured_output | abstention |",
        "|---|---:|---:|---:|---:|",
    ])
    categories = ["arithmetic", "instruction", "structured_output", "abstention"]
    for entry in payload["entries"]:
        cells = []
        for category in categories:
            stats = entry.get("by_category", {}).get(category)
            if not stats:
                cells.append("-")
                continue
            cells.append(f"{stats.get('passed', 0)}/{stats.get('n', 0)}")
        lines.append(f"| `{entry['model_name']}` | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Ranking order is accuracy, then task coverage, then mean generation speed.",
        "- This is a local deployment sanity leaderboard, not a public benchmark.",
        "- Rows with different task signatures are useful for smoke checks but should not be used for dominance claims.",
        "- A real 30B+ comparison requires at least one `30b_plus_baseline` row from a live endpoint or local 30B+ runtime.",
    ])
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON payload instead of Markdown summary.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
