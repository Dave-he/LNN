"""Tests for scripts/build_llm_battlecard.py.

The battlecard is a claim-audit layer for the active<=3B vs 30B+ LLM
thesis.  These tests lock the important semantics:

1. Shared benchmark wins/losses are mechanically counted.
2. LFM2.5-8B-A1B is active<=3B but not total<=3B.
3. The default verdict stays scoped to active-parameter MoE evidence.
4. Local validation JSON is summarized without needing a model run.
5. Reports can be written outside the repo.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_llm_battlecard  # noqa: E402


def test_default_comparison_counts_public_snapshot():
    candidate = build_llm_battlecard.MODEL_SNAPSHOTS["lfm25-8b-a1b"]
    baseline = build_llm_battlecard.MODEL_SNAPSHOTS["qwen3-30b-a3b-thinking-2507"]

    rows = build_llm_battlecard.compare_models(candidate, baseline)
    summary = build_llm_battlecard.summarize_comparison(rows, candidate, baseline)

    assert len(rows) == 13
    assert summary["wins"] == 7
    assert summary["losses"] == 6
    assert summary["ties"] == 0
    assert summary["groups"]["knowledge_instruction"]["wins"] == 5
    assert summary["groups"]["math_agentic"]["wins"] == 2


def test_default_scope_does_not_claim_dense_3b_completion():
    candidate = build_llm_battlecard.MODEL_SNAPSHOTS["lfm25-8b-a1b"]
    baseline = build_llm_battlecard.MODEL_SNAPSHOTS["qwen3-30b-a3b-thinking-2507"]

    rows = build_llm_battlecard.compare_models(candidate, baseline)
    summary = build_llm_battlecard.summarize_comparison(rows, candidate, baseline)

    assert summary["scope"]["candidate_active_under_3b"] is True
    assert summary["scope"]["candidate_total_under_3b"] is False
    assert summary["scope"]["baseline_total_30b_plus"] is True
    assert summary["verdict"] == "active_under_3b_not_dense_3b"
    assert "not an exact 3B dense claim" in summary["interpretation"]


def test_build_payload_summarizes_local_validation(tmp_path):
    local_json = tmp_path / "local_validation.json"
    local_json.write_text(
        json.dumps({
            "run_date": "2026-06-04",
            "environment": {"python": "3.11.0"},
            "gguf": {
                "status": "ok",
                "model_path": "models/lfm25/model.gguf",
                "size_bytes": 123,
                "timing": {"generation_tps": 12.5},
                "response": "ok",
            },
            "dpo": {
                "status": "ok",
                "model": "models/lfm25-dpo",
                "output_tokens": 24,
                "generation_seconds": 12.0,
                "response": "ok",
            },
        }),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        date="2026-06-04",
        candidate="lfm25-8b-a1b",
        baseline="qwen3-30b-a3b-thinking-2507",
        local_validation=str(local_json),
        local_micro_eval=str(tmp_path / "missing_micro_eval.json"),
    )

    payload = build_llm_battlecard.build_payload(args)

    assert payload["local_validation"]["gguf"]["generation_tps"] == 12.5
    assert payload["local_validation"]["dpo"]["output_tokens"] == 24
    assert payload["summary"]["wins"] == 7


def test_build_payload_summarizes_local_micro_eval(tmp_path):
    micro_json = tmp_path / "micro_eval.json"
    micro_json.write_text(
        json.dumps({
            "date": "2026-06-04",
            "model_name": "lfm25_1.2b_instruct_q4",
            "model_path": "models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf",
            "summary": {
                "n": 7,
                "passed": 7,
                "failed": 0,
                "accuracy": 1.0,
                "generation_tps_mean": 31.7,
            },
        }),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        date="2026-06-04",
        candidate="lfm25-8b-a1b",
        baseline="qwen3-30b-a3b-thinking-2507",
        local_validation=str(tmp_path / "missing_local_validation.json"),
        local_micro_eval=str(micro_json),
    )

    payload = build_llm_battlecard.build_payload(args)

    assert payload["local_micro_eval"]["summary"]["accuracy"] == 1.0
    assert payload["local_micro_eval"]["summary"]["generation_tps_mean"] == 31.7


def test_build_payload_summarizes_local_micro_leaderboard(tmp_path):
    leaderboard_json = tmp_path / "leaderboard.json"
    leaderboard_json.write_text(
        json.dumps({
            "date": "2026-06-04",
            "summary": {
                "n_entries": 2,
                "n_rankable": 2,
                "roles": {"under_3b_candidate": 2},
                "top_model": "lfm25_1.2b_instruct_q4",
                "top_accuracy": 1.0,
                "top_generation_tps_mean": 16.843,
            },
            "entries": [
                {
                    "rank": 1,
                    "model_name": "lfm25_1.2b_instruct_q4",
                    "backend": "llama-cli",
                    "comparison_role": "under_3b_candidate",
                    "source_path": "analysis/llm_micro_eval/a.json",
                    "summary": {
                        "n": 7,
                        "passed": 7,
                        "accuracy": 1.0,
                        "generation_tps_mean": 16.843,
                    },
                },
                {
                    "rank": 2,
                    "model_name": "lfm25_1.2b_instruct_q4_http",
                    "backend": "openai-chat",
                    "comparison_role": "under_3b_candidate",
                    "source_path": "analysis/llm_micro_eval/b.json",
                    "summary": {
                        "n": 7,
                        "passed": 7,
                        "accuracy": 1.0,
                        "generation_tps_mean": 5.707,
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        date="2026-06-04",
        candidate="lfm25-8b-a1b",
        baseline="qwen3-30b-a3b-thinking-2507",
        local_validation=str(tmp_path / "missing_local_validation.json"),
        local_micro_eval=str(tmp_path / "missing_micro_eval.json"),
        local_micro_leaderboard=str(leaderboard_json),
    )

    payload = build_llm_battlecard.build_payload(args)

    assert payload["local_micro_leaderboard"]["summary"]["n_entries"] == 2
    assert payload["local_micro_leaderboard"]["entries"][1]["backend"] == "openai-chat"


def test_markdown_contains_prediction_gate():
    candidate = build_llm_battlecard.MODEL_SNAPSHOTS["lfm25-8b-a1b"]
    baseline = build_llm_battlecard.MODEL_SNAPSHOTS["qwen3-30b-a3b-thinking-2507"]
    rows = build_llm_battlecard.compare_models(candidate, baseline)
    payload = {
        "date": "2026-06-04",
        "candidate": build_llm_battlecard._model_to_dict(candidate),
        "baseline": build_llm_battlecard._model_to_dict(baseline),
        "comparison": rows,
        "summary": build_llm_battlecard.summarize_comparison(rows, candidate, baseline),
        "local_validation": None,
        "local_micro_eval": {
            "path": "analysis/llm_micro_eval/example.json",
            "summary": {"n": 7, "passed": 7, "accuracy": 1.0, "generation_tps_mean": 31.7},
        },
        "local_micro_leaderboard": {
            "path": "analysis/llm_micro_eval/leaderboard.json",
            "summary": {
                "n_entries": 2,
                "roles": {"under_3b_candidate": 2},
                "top_model": "lfm25",
                "top_accuracy": 1.0,
                "top_generation_tps_mean": 31.7,
            },
            "entries": [],
        },
        "sources": [],
    }

    markdown = build_llm_battlecard.format_markdown(payload)

    assert "Do **not** claim a general 3B model can beat 30B+ models yet" in markdown
    assert "7 win / 6 loss / 0 tie" in markdown
    assert "Micro-eval: 100.0% (7/7)" in markdown
    assert "Micro leaderboard: 2 entries" in markdown


def test_write_outputs_accepts_external_directory(tmp_path):
    candidate = build_llm_battlecard.MODEL_SNAPSHOTS["lfm25-8b-a1b"]
    baseline = build_llm_battlecard.MODEL_SNAPSHOTS["qwen3-30b-a3b-thinking-2507"]
    rows = build_llm_battlecard.compare_models(candidate, baseline)
    payload = {
        "run_id": "test_llm_battlecard",
        "date": "2026-06-04",
        "candidate": build_llm_battlecard._model_to_dict(candidate),
        "baseline": build_llm_battlecard._model_to_dict(baseline),
        "comparison": rows,
        "summary": build_llm_battlecard.summarize_comparison(rows, candidate, baseline),
        "local_validation": None,
        "local_micro_eval": None,
        "sources": [],
    }

    json_path, md_path = build_llm_battlecard.write_outputs(payload, tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["wins"] == 7
