"""Tests for scripts/build_llm_micro_leaderboard.py."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_llm_micro_leaderboard  # noqa: E402


def _write_micro_eval(
    path: Path,
    *,
    run_id: str,
    model_name: str,
    backend: str,
    accuracy: float | None,
    passed: int,
    n: int,
    speed: float | None,
    openai_model: str | None = None,
) -> None:
    results = [
        {
            "task_id": f"task_{idx}",
            "category": "arithmetic" if idx == 0 else "instruction",
            "grade": {"passed": idx < passed},
        }
        for idx in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "run_id": run_id,
            "date": "2026-06-04",
            "backend": backend,
            "model_name": model_name,
            "model_path": f"models/{model_name}.gguf",
            "openai_model": openai_model,
            "openai_base_url": "http://127.0.0.1:8000/v1" if backend == "openai-chat" else None,
            "summary": {
                "n": n,
                "passed": passed,
                "failed": n - passed,
                "accuracy": accuracy,
                "by_category": {
                    "arithmetic": {"n": 1, "passed": 1 if passed else 0, "accuracy": 1.0 if passed else 0.0},
                    "instruction": {
                        "n": max(n - 1, 0),
                        "passed": max(passed - 1, 0),
                        "accuracy": 1.0 if passed == n else 0.0,
                    },
                },
                "generation_tps_mean": speed,
                "generation_tps_median": speed,
            },
            "results": results,
        }),
        encoding="utf-8",
    )


def test_discover_entries_and_infer_roles(monkeypatch, tmp_path):
    monkeypatch.setattr(build_llm_micro_leaderboard, "ROOT", tmp_path)
    eval_dir = tmp_path / "analysis" / "llm_micro_eval"
    _write_micro_eval(
        eval_dir / "2026-06-04_lfm25_1_2b_micro_eval.json",
        run_id="2026-06-04_lfm25_1_2b_micro_eval",
        model_name="lfm25_1.2b_instruct_q4",
        backend="llama-cli",
        accuracy=1.0,
        passed=7,
        n=7,
        speed=16.8,
    )
    _write_micro_eval(
        eval_dir / "2026-06-04_qwen3_30b_micro_eval.json",
        run_id="2026-06-04_qwen3_30b_micro_eval",
        model_name="qwen3_30b_endpoint",
        backend="openai-chat",
        accuracy=0.8571,
        passed=6,
        n=7,
        speed=22.5,
        openai_model="Qwen/Qwen3-30B-A3B",
    )
    _write_micro_eval(
        eval_dir / "2026-06-04_lfm25_dpo_micro_eval.json",
        run_id="2026-06-04_lfm25_dpo_micro_eval",
        model_name="lfm25_dpo_s1_q4",
        backend="llama-cli",
        accuracy=0.5714,
        passed=4,
        n=7,
        speed=11.4,
    )

    entries = build_llm_micro_leaderboard.discover_entries("analysis/llm_micro_eval/*_micro_eval.json")

    assert len(entries) == 3
    roles = {entry["model_name"]: entry["comparison_role"] for entry in entries}
    assert roles["lfm25_1.2b_instruct_q4"] == "under_3b_candidate"
    assert roles["lfm25_dpo_s1_q4"] == "under_3b_candidate"
    assert roles["qwen3_30b_endpoint"] == "30b_plus_baseline"


def test_assign_ranks_uses_accuracy_then_coverage_then_speed():
    entries = [
        {
            "model_name": "fast_two_task",
            "summary": {"accuracy": 1.0, "n": 2, "generation_tps_mean": 100.0},
        },
        {
            "model_name": "full_suite",
            "summary": {"accuracy": 1.0, "n": 7, "generation_tps_mean": 10.0},
        },
        {
            "model_name": "lower_accuracy",
            "summary": {"accuracy": 0.9, "n": 7, "generation_tps_mean": 200.0},
        },
        {
            "model_name": "dry_run",
            "summary": {"accuracy": None, "n": 7, "generation_tps_mean": None},
        },
    ]

    ranked = build_llm_micro_leaderboard.assign_ranks(entries)

    assert [entry["model_name"] for entry in ranked] == [
        "full_suite",
        "fast_two_task",
        "lower_accuracy",
        "dry_run",
    ]
    assert [entry["rank"] for entry in ranked] == [1, 2, 3, None]


def test_build_payload_and_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(build_llm_micro_leaderboard, "ROOT", tmp_path)
    eval_dir = tmp_path / "analysis" / "llm_micro_eval"
    _write_micro_eval(
        eval_dir / "2026-06-04_lfm25_1_2b_micro_eval.json",
        run_id="2026-06-04_lfm25_1_2b_micro_eval",
        model_name="lfm25_1.2b_instruct_q4",
        backend="llama-cli",
        accuracy=1.0,
        passed=7,
        n=7,
        speed=16.843,
    )
    args = argparse.Namespace(
        date="2026-06-04",
        input_glob="analysis/llm_micro_eval/*_micro_eval.json",
    )

    payload = build_llm_micro_leaderboard.build_payload(args)
    markdown = build_llm_micro_leaderboard.format_markdown(payload)

    assert payload["summary"]["n_entries"] == 1
    assert payload["summary"]["top_model"] == "lfm25_1.2b_instruct_q4"
    assert "| 1 | `lfm25_1.2b_instruct_q4`" in markdown
    assert "not a public benchmark" in markdown
    assert "30b_plus_baseline" in markdown


def test_write_outputs(tmp_path):
    payload = {
        "run_id": "2026-06-04_llm_micro_leaderboard",
        "date": "2026-06-04",
        "input_glob": "analysis/llm_micro_eval/*_micro_eval.json",
        "summary": {
            "n_entries": 1,
            "n_rankable": 1,
            "roles": {"under_3b_candidate": 1},
            "top_model": "lfm25",
            "top_accuracy": 1.0,
            "top_generation_tps_mean": 16.8,
        },
        "entries": [
            {
                "rank": 1,
                "model_name": "lfm25",
                "backend": "llama-cli",
                "comparison_role": "under_3b_candidate",
                "source_path": "analysis/llm_micro_eval/fake_micro_eval.json",
                "summary": {"n": 7, "passed": 7, "accuracy": 1.0, "generation_tps_mean": 16.8},
                "by_category": {},
            }
        ],
    }

    json_path, md_path = build_llm_micro_leaderboard.write_outputs(payload, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["top_model"] == "lfm25"
    assert "LLM micro-eval leaderboard" in md_path.read_text(encoding="utf-8")
