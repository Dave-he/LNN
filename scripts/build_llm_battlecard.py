#!/usr/bin/env python3
"""Build an LFM/LNN-related model battlecard against 30B+ LLM baselines.

This is deliberately a *claim-audit* tool, not a benchmark runner.  It
combines:

* local inference evidence already produced by ``validate_lfm25_local.py``;
* public benchmark snapshots from model cards/blog posts;
* a deterministic pass/fail readout for the "active <=3B beats 30B+" thesis.

The default candidate is LiquidAI/LFM2.5-8B-A1B because it is the closest
current LNN/LFM-family model to the user's "3B" target: 8.3B total parameters
but only 1.5B active parameters per token.  That is not the same claim as an
exact 3B dense model, so the generated report keeps both scope flags explicit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "llm_battlecard"
DEFAULT_LOCAL_GLOB = "analysis/lfm25/*_lfm25_local_validation.json"
DEFAULT_MICRO_EVAL_GLOB = "analysis/llm_micro_eval/*_micro_eval.json"
DEFAULT_MICRO_LEADERBOARD_GLOB = "analysis/llm_micro_eval/*_llm_micro_leaderboard.json"

HIGHER_IS_BETTER = "higher_is_better"


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    display_name: str
    family: str
    total_params_b: float
    active_params_b: float | None
    architecture: str
    source_url: str
    source_date: str
    notes: str
    metrics: dict[str, float]

    @property
    def active_or_total_b(self) -> float:
        return self.active_params_b if self.active_params_b is not None else self.total_params_b


METRIC_GROUPS: dict[str, list[str]] = {
    "knowledge_instruction": [
        "aa_omniscience_index",
        "aa_omniscience_accuracy",
        "aa_omniscience_non_hallucination",
        "ifeval",
        "ifbench",
        "multi_if",
    ],
    "math_agentic": [
        "math500",
        "aime25",
        "aime26",
        "bfclv3",
        "bfclv4",
        "tau2_telecom",
        "tau2_retail",
    ],
}


METRIC_LABELS = {
    "aa_omniscience_index": "AA-Omniscience Index",
    "aa_omniscience_accuracy": "AA-Omniscience Accuracy",
    "aa_omniscience_non_hallucination": "AA-Omniscience Non-Hallucination",
    "ifeval": "IFEval",
    "ifbench": "IFBench",
    "multi_if": "Multi-IF",
    "math500": "MATH500",
    "aime25": "AIME25",
    "aime26": "AIME26",
    "bfclv3": "BFCLv3",
    "bfclv4": "BFCLv4",
    "tau2_telecom": "Tau2 Telecom",
    "tau2_retail": "Tau2 Retail",
}


MODEL_SNAPSHOTS: dict[str, ModelSnapshot] = {
    "lfm25-8b-a1b": ModelSnapshot(
        model_id="LiquidAI/LFM2.5-8B-A1B",
        display_name="LFM2.5-8B-A1B",
        family="Liquid Foundation Model / LNN-related hybrid",
        total_params_b=8.3,
        active_params_b=1.5,
        architecture="MoE + double-gated LIV convolution + GQA",
        source_url="https://huggingface.co/LiquidAI/LFM2.5-8B-A1B",
        source_date="2026-05-28",
        notes="Closest current LFM-family target to active<=3B; not an exact 3B dense model.",
        metrics={
            "aa_omniscience_index": -24.70,
            "aa_omniscience_accuracy": 8.67,
            "aa_omniscience_non_hallucination": 63.47,
            "ifeval": 91.84,
            "ifbench": 56.47,
            "multi_if": 79.93,
            "math500": 88.76,
            "aime25": 42.53,
            "aime26": 50.00,
            "bfclv3": 64.79,
            "bfclv4": 49.73,
            "tau2_telecom": 88.07,
            "tau2_retail": 39.82,
        },
    ),
    "lfm25-1.2b-thinking": ModelSnapshot(
        model_id="LiquidAI/LFM2.5-1.2B-Thinking",
        display_name="LFM2.5-1.2B-Thinking",
        family="Liquid Foundation Model / LNN-related hybrid",
        total_params_b=1.17,
        active_params_b=1.17,
        architecture="double-gated LIV convolution + GQA",
        source_url="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking",
        source_date="2026-01-05",
        notes="Local-friendly model already validated in this repo; public 30B-overlap metrics are incomplete.",
        metrics={},
    ),
    "qwen3-30b-a3b-thinking-2507": ModelSnapshot(
        model_id="Qwen/Qwen3-30B-A3B",
        display_name="Qwen3-30B-A3B-Thinking-2507",
        family="Qwen3 MoE",
        total_params_b=30.5,
        active_params_b=3.3,
        architecture="Transformer MoE",
        source_url="https://huggingface.co/Qwen/Qwen3-30B-A3B",
        source_date="2025-05-13",
        notes="30B+ comparison row from LiquidAI's LFM2.5-8B-A1B benchmark table.",
        metrics={
            "aa_omniscience_index": -51.31,
            "aa_omniscience_accuracy": 18.80,
            "aa_omniscience_non_hallucination": 13.87,
            "ifeval": 90.82,
            "ifbench": 51.11,
            "multi_if": 79.04,
            "math500": 86.48,
            "aime25": 71.67,
            "aime26": 66.67,
            "bfclv3": 73.39,
            "bfclv4": 50.53,
            "tau2_telecom": 21.93,
            "tau2_retail": 56.14,
        },
    ),
    "qwen3-32b": ModelSnapshot(
        model_id="Qwen/Qwen3-32B",
        display_name="Qwen3-32B",
        family="Qwen3 dense",
        total_params_b=32.8,
        active_params_b=32.8,
        architecture="Transformer dense",
        source_url="https://huggingface.co/Qwen/Qwen3-32B",
        source_date="2025-05-13",
        notes="Metadata-only 30B+ baseline; no same-source overlap metrics in this snapshot.",
        metrics={},
    ),
}


def _latest_local_validation() -> pathlib.Path | None:
    candidates = sorted(ROOT.glob(DEFAULT_LOCAL_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_micro_eval() -> pathlib.Path | None:
    candidates = sorted(ROOT.glob(DEFAULT_MICRO_EVAL_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_micro_leaderboard() -> pathlib.Path | None:
    candidates = sorted(ROOT.glob(DEFAULT_MICRO_LEADERBOARD_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_local_validation(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable", "path": str(path)}
    return {
        "path": _display_path(path),
        "run_date": payload.get("run_date"),
        "environment": payload.get("environment", {}),
        "gguf": _summarize_local_section(payload.get("gguf")),
        "dpo": _summarize_local_section(payload.get("dpo")),
    }


def _load_micro_eval(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable", "path": str(path)}
    return {
        "path": _display_path(path),
        "date": payload.get("date"),
        "model_name": payload.get("model_name"),
        "model_path": payload.get("model_path"),
        "summary": payload.get("summary", {}),
    }


def _load_micro_leaderboard(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable", "path": str(path)}
    entries = []
    for entry in payload.get("entries", [])[:8]:
        summary = entry.get("summary") or {}
        entries.append({
            "rank": entry.get("rank"),
            "model_name": entry.get("model_name"),
            "backend": entry.get("backend"),
            "comparison_role": entry.get("comparison_role"),
            "accuracy": summary.get("accuracy"),
            "passed": summary.get("passed"),
            "n": summary.get("n"),
            "generation_tps_mean": summary.get("generation_tps_mean"),
            "source_path": entry.get("source_path"),
        })
    return {
        "path": _display_path(path),
        "date": payload.get("date"),
        "summary": payload.get("summary", {}),
        "entries": entries,
    }


def _summarize_local_section(section: dict[str, Any] | None) -> dict[str, Any] | None:
    if section is None:
        return None
    timing = section.get("timing") or {}
    return {
        "status": section.get("status"),
        "model_path": section.get("model_path") or section.get("model"),
        "size_bytes": section.get("size_bytes") or section.get("model_safetensors_size_bytes"),
        "prompt_tps": timing.get("prompt_tps"),
        "generation_tps": timing.get("generation_tps"),
        "input_tokens": section.get("input_tokens"),
        "output_tokens": section.get("output_tokens"),
        "generation_seconds": section.get("generation_seconds"),
        "response_preview": (section.get("response") or "")[:160],
    }


def compare_models(candidate: ModelSnapshot, baseline: ModelSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = sorted(set(candidate.metrics) & set(baseline.metrics))
    for metric in metric_names:
        cand = candidate.metrics[metric]
        base = baseline.metrics[metric]
        delta = cand - base
        rows.append({
            "metric": metric,
            "label": METRIC_LABELS.get(metric, metric),
            "direction": HIGHER_IS_BETTER,
            "candidate": cand,
            "baseline": base,
            "delta": round(delta, 4),
            "winner": "candidate" if delta > 0 else "baseline" if delta < 0 else "tie",
        })
    return rows


def summarize_comparison(rows: list[dict[str, Any]], candidate: ModelSnapshot, baseline: ModelSnapshot) -> dict[str, Any]:
    wins = sum(1 for row in rows if row["winner"] == "candidate")
    losses = sum(1 for row in rows if row["winner"] == "baseline")
    ties = sum(1 for row in rows if row["winner"] == "tie")
    group_summaries = {}
    for group, metrics in METRIC_GROUPS.items():
        group_rows = [row for row in rows if row["metric"] in metrics]
        group_summaries[group] = {
            "wins": sum(1 for row in group_rows if row["winner"] == "candidate"),
            "losses": sum(1 for row in group_rows if row["winner"] == "baseline"),
            "ties": sum(1 for row in group_rows if row["winner"] == "tie"),
            "n_metrics": len(group_rows),
        }
    win_rate = wins / len(rows) if rows else 0.0
    scope = {
        "candidate_total_under_3b": candidate.total_params_b <= 3.0,
        "candidate_active_under_3b": candidate.active_or_total_b <= 3.0,
        "baseline_total_30b_plus": baseline.total_params_b >= 30.0,
        "baseline_active_30b_plus": baseline.active_or_total_b >= 30.0,
    }
    if not rows:
        verdict = "insufficient_overlap"
    elif not scope["candidate_total_under_3b"]:
        verdict = "active_under_3b_not_dense_3b"
    elif win_rate >= 0.75 and losses == 0:
        verdict = "decisive_candidate_win"
    elif wins > losses:
        verdict = "mixed_candidate_edge"
    else:
        verdict = "not_proven"
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_metrics": len(rows),
        "win_rate": round(win_rate, 4),
        "groups": group_summaries,
        "scope": scope,
        "verdict": verdict,
        "interpretation": _interpret_verdict(verdict, wins, losses, candidate, baseline),
    }


def _interpret_verdict(verdict: str, wins: int, losses: int, candidate: ModelSnapshot, baseline: ModelSnapshot) -> str:
    if verdict == "active_under_3b_not_dense_3b":
        return (
            f"{candidate.display_name} beats {baseline.display_name} on {wins} shared metrics and loses on {losses}, "
            "but it is 8.3B total / 1.5B active, so this supports only the active<=3B MoE thesis, not an exact 3B dense claim."
        )
    if verdict == "mixed_candidate_edge":
        return (
            f"{candidate.display_name} has a positive shared-metric edge over {baseline.display_name}, "
            "but losses remain, so a dominance claim is too strong."
        )
    if verdict == "decisive_candidate_win":
        return f"{candidate.display_name} clears the configured decisive-win gate."
    if verdict == "insufficient_overlap":
        return "No shared benchmark metrics are available in the current snapshot."
    return "The current evidence does not prove the smaller model beats the 30B+ baseline."


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    candidate = MODEL_SNAPSHOTS[args.candidate]
    baseline = MODEL_SNAPSHOTS[args.baseline]
    local_arg = getattr(args, "local_validation", "")
    micro_arg = getattr(args, "local_micro_eval", "")
    leaderboard_arg = getattr(args, "local_micro_leaderboard", "")
    local_path = pathlib.Path(local_arg).expanduser() if local_arg else _latest_local_validation()
    micro_path = pathlib.Path(micro_arg).expanduser() if micro_arg else _latest_micro_eval()
    leaderboard_path = pathlib.Path(leaderboard_arg).expanduser() if leaderboard_arg else _latest_micro_leaderboard()
    rows = compare_models(candidate, baseline)
    summary = summarize_comparison(rows, candidate, baseline)
    return {
        "run_id": f"{args.date}_llm_battlecard",
        "date": args.date,
        "candidate_key": args.candidate,
        "baseline_key": args.baseline,
        "candidate": _model_to_dict(candidate),
        "baseline": _model_to_dict(baseline),
        "comparison": rows,
        "summary": summary,
        "local_validation": _load_local_validation(local_path),
        "local_micro_eval": _load_micro_eval(micro_path),
        "local_micro_leaderboard": _load_micro_leaderboard(leaderboard_path),
        "sources": [
            {
                "name": "LiquidAI LFM2.5-8B-A1B model card",
                "url": "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B",
                "used_for": "LFM2.5-8B-A1B metrics and architecture snapshot",
            },
            {
                "name": "LiquidAI LFM2.5-8B-A1B blog",
                "url": "https://www.liquid.ai/blog/lfm2-5-8b-a1b",
                "used_for": "release date, benchmark interpretation, inference support",
            },
            {
                "name": "Qwen3-30B-A3B model card",
                "url": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
                "used_for": "30.5B total / 3.3B active parameter metadata",
            },
            {
                "name": "LFM2.5-1.2B local validation",
                "url": "analysis/lfm25/2026-06-01_lfm25_local_inference_quantization.md",
                "used_for": "local GGUF/DPO smoke inference evidence already in this repo",
            },
            {
                "name": "Local LLM micro-eval",
                "url": "analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md",
                "used_for": "deterministic local deployment sanity check",
            },
            {
                "name": "Local LLM micro-eval leaderboard",
                "url": "analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.md",
                "used_for": "local smoke leaderboard across CLI and OpenAI-compatible endpoint runs",
            },
        ],
    }


def _model_to_dict(model: ModelSnapshot) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "display_name": model.display_name,
        "family": model.family,
        "total_params_b": model.total_params_b,
        "active_params_b": model.active_params_b,
        "architecture": model.architecture,
        "source_url": model.source_url,
        "source_date": model.source_date,
        "notes": model.notes,
        "metrics": model.metrics,
    }


def format_markdown(payload: dict[str, Any]) -> str:
    candidate = payload["candidate"]
    baseline = payload["baseline"]
    summary = payload["summary"]
    local = payload.get("local_validation")
    micro_eval = payload.get("local_micro_eval")
    micro_leaderboard = payload.get("local_micro_leaderboard")
    lines = [
        "---",
        "title: LFM/LNN-related 3B-vs-30B battlecard",
        f"date: {payload['date']}",
        "tags: [LFM2.5, LNN, LLM, benchmark, battlecard, active-3B, 30B-plus]",
        "parent: [[PRD_LNN_Edge_Research]]",
        "---",
        "",
        f"# LFM/LNN-related battlecard - {payload['date']}",
        "",
        "## Verdict",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Readout: {summary['interpretation']}",
        (
            "- Scope: candidate total<=3B = "
            f"{summary['scope']['candidate_total_under_3b']}; active<=3B = "
            f"{summary['scope']['candidate_active_under_3b']}; baseline total>=30B = "
            f"{summary['scope']['baseline_total_30b_plus']}."
        ),
        "",
        "## Models",
        "",
        "| Role | Model | Total params | Active params | Architecture | Source |",
        "|---|---|---:|---:|---|---|",
        _model_row("Candidate", candidate),
        _model_row("30B+ baseline", baseline),
        "",
        "## Shared Public Benchmark Snapshot",
        "",
        "| Metric | Candidate | Baseline | Delta | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["comparison"]:
        lines.append(
            "| {label} | {candidate:.2f} | {baseline:.2f} | {delta:+.2f} | {winner} |".format(**row)
        )
    lines.extend([
        "",
        f"Shared metric tally: **{summary['wins']} win / {summary['losses']} loss / {summary['ties']} tie** "
        f"(win rate {summary['win_rate']:.1%}).",
        "",
        "## Domain Split",
        "",
        "| Group | Wins | Losses | Ties | Metrics |",
        "|---|---:|---:|---:|---:|",
    ])
    for group, stats in summary["groups"].items():
        lines.append(f"| {group} | {stats['wins']} | {stats['losses']} | {stats['ties']} | {stats['n_metrics']} |")
    lines.extend([
        "",
        "## Local Evidence",
        "",
    ])
    if local:
        lines.append(f"- Local validation file: `{local.get('path')}`")
        for label in ("gguf", "dpo"):
            section = local.get(label)
            if not section:
                continue
            speed = _format_speed(section)
            lines.append(f"- {label.upper()}: status `{section.get('status')}`, {speed}")
    else:
        lines.append("- No local LFM2.5 validation JSON found.")
    if micro_eval:
        micro_summary = micro_eval.get("summary", {})
        accuracy = micro_summary.get("accuracy")
        accuracy_text = "not run" if accuracy is None else f"{float(accuracy):.1%}"
        lines.append(f"- Local micro-eval file: `{micro_eval.get('path')}`")
        lines.append(
            "- Micro-eval: "
            f"{accuracy_text} ({micro_summary.get('passed')}/{micro_summary.get('n')}), "
            f"mean generation {micro_summary.get('generation_tps_mean')} tok/s"
        )
    else:
        lines.append("- No local LLM micro-eval JSON found.")
    if micro_leaderboard:
        leaderboard_summary = micro_leaderboard.get("summary", {})
        roles = leaderboard_summary.get("roles") or {}
        roles_text = ", ".join(f"{role}={count}" for role, count in sorted(roles.items())) or "none"
        lines.append(f"- Micro leaderboard file: `{micro_leaderboard.get('path')}`")
        lines.append(
            "- Micro leaderboard: "
            f"{leaderboard_summary.get('n_entries')} entries, roles: {roles_text}; "
            f"leader `{leaderboard_summary.get('top_model')}` "
            f"({_format_accuracy(leaderboard_summary.get('top_accuracy'))}, "
            f"{leaderboard_summary.get('top_generation_tps_mean')} tok/s)"
        )
    else:
        lines.append("- No local LLM micro leaderboard JSON found.")
    lines.extend([
        "",
        "## Prediction",
        "",
        (
            "- Near-term target should be **agentic/RAG/tool-use and instruction-following**, where the current "
            "LFM-family public data is strongest."
        ),
        (
            "- Do **not** claim a general 3B model can beat 30B+ models yet: the strongest evidence here is "
            "active<=3B MoE, with clear losses on AIME and parts of BFCL/Tau Retail."
        ),
        (
            "- Next evidence gate: run local LFM2.5-8B-A1B GGUF on Jetson/desktop, then add a reproducible "
            "lm-eval or OpenCompass subset before any public leaderboard claim."
        ),
        "",
        "## Sources",
        "",
    ])
    for source in payload["sources"]:
        lines.append(f"- [{source['name']}]({source['url']}) - {source['used_for']}")
    return "\n".join(lines) + "\n"


def _model_row(role: str, model: dict[str, Any]) -> str:
    active = model["active_params_b"] if model["active_params_b"] is not None else model["total_params_b"]
    return (
        f"| {role} | {model['display_name']} | {model['total_params_b']:.2f}B | "
        f"{active:.2f}B | {model['architecture']} | [link]({model['source_url']}) |"
    )


def _format_speed(section: dict[str, Any]) -> str:
    if section.get("generation_tps") is not None:
        return f"generation {section['generation_tps']:.2f} tok/s"
    if section.get("output_tokens") is not None and section.get("generation_seconds"):
        tps = section["output_tokens"] / section["generation_seconds"]
        return f"generation {tps:.2f} tok/s"
    return "no timing parsed"


def _format_accuracy(value: Any) -> str:
    if value is None:
        return "not run"
    return f"{float(value):.1%}"


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
    parser.add_argument("--candidate", choices=sorted(MODEL_SNAPSHOTS), default="lfm25-8b-a1b")
    parser.add_argument("--baseline", choices=sorted(MODEL_SNAPSHOTS), default="qwen3-30b-a3b-thinking-2507")
    parser.add_argument("--local-validation", default="")
    parser.add_argument("--local-micro-eval", default="")
    parser.add_argument("--local-micro-leaderboard", default="")
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
