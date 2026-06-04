"""Tests for scripts/run_llm_micro_eval.py."""

import argparse
import http.server
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_llm_micro_eval  # noqa: E402


def test_parse_llama_output_extracts_response_and_timing():
    output = """
Loading model...

> Answer with exactly the number. What is 2+3?

5

[ Prompt: 59.6 t/s | Generation: 44.2 t/s ]

Exiting...
"""

    parsed = run_llm_micro_eval.parse_llama_output(output, "Answer with exactly the number. What is 2+3?")

    assert parsed["response"] == "5"
    assert parsed["timing"]["prompt_tps"] == 59.6
    assert parsed["timing"]["generation_tps"] == 44.2


def test_grade_response_exact_and_json():
    exact = run_llm_micro_eval.EvalTask("t1", "instruction", "p", "BLUE", "exact_text")
    json_task = run_llm_micro_eval.EvalTask("t2", "structured_output", "p", '{"color":"blue"}', "json_equal")

    assert run_llm_micro_eval.grade_response(exact, '"blue"')["passed"] is True
    assert run_llm_micro_eval.grade_response(json_task, 'Here: {"color": "blue"}')["passed"] is True
    assert run_llm_micro_eval.grade_response(json_task, '{"color":"red"}')["passed"] is False


def test_selected_tasks_by_category():
    selected = run_llm_micro_eval.selected_tasks("arithmetic")

    assert {task.task_id for task in selected} == {
        "arith_2_plus_3",
        "arith_17_minus_9",
        "arith_12_times_4",
    }


def test_build_payload_dry_run_allow_missing():
    args = argparse.Namespace(
        date="2026-06-04",
        backend="llama-cli",
        model_name="missing_model",
        model="/tmp/does-not-exist.gguf",
        llama_cli="/tmp/does-not-exist-llama-cli",
        openai_base_url="http://127.0.0.1:1/v1",
        openai_model="",
        openai_api_key="",
        output_dir="/tmp",
        tasks="instruction",
        ctx_size=256,
        threads=4,
        temperature=0.0,
        top_k=1,
        repeat_penalty=1.05,
        seed=1,
        timeout=5,
        min_accuracy=1.0,
        no_write=True,
        json=True,
        dry_run=True,
        allow_missing=True,
    )

    payload = run_llm_micro_eval.build_payload(args)

    assert payload["summary"]["n"] == 2
    assert payload["summary"]["accuracy"] is None
    assert payload["model_exists"] is False
    assert payload["llama_cli_exists"] is False


def test_build_payload_sanitizes_run_id():
    args = argparse.Namespace(
        date="2026-06-04",
        backend="llama-cli",
        model_name="lfm25_1.2b/instruct:q4",
        model="/tmp/does-not-exist.gguf",
        llama_cli="/tmp/does-not-exist-llama-cli",
        openai_base_url="http://127.0.0.1:1/v1",
        openai_model="",
        openai_api_key="",
        output_dir="/tmp",
        tasks="instruction",
        ctx_size=256,
        threads=4,
        temperature=0.0,
        top_k=1,
        repeat_penalty=1.05,
        seed=1,
        timeout=5,
        min_accuracy=1.0,
        no_write=True,
        json=True,
        dry_run=True,
        allow_missing=True,
    )

    payload = run_llm_micro_eval.build_payload(args)

    assert payload["run_id"] == "2026-06-04_lfm25_1_2b_instruct_q4_micro_eval"


def test_build_payload_openai_chat_dry_run_does_not_require_local_files():
    args = argparse.Namespace(
        date="2026-06-04",
        backend="openai-chat",
        model_name="qwen3_30b_endpoint",
        model="/tmp/does-not-exist.gguf",
        llama_cli="/tmp/does-not-exist-llama-cli",
        openai_base_url="http://127.0.0.1:9999/v1",
        openai_model="Qwen/Qwen3-30B-A3B",
        openai_api_key="",
        output_dir="/tmp",
        tasks="instruction",
        ctx_size=256,
        threads=4,
        temperature=0.0,
        top_k=1,
        repeat_penalty=1.05,
        seed=1,
        timeout=5,
        min_accuracy=1.0,
        no_write=True,
        json=True,
        dry_run=True,
        allow_missing=False,
    )

    payload = run_llm_micro_eval.build_payload(args)

    assert payload["backend"] == "openai-chat"
    assert payload["openai_model"] == "Qwen/Qwen3-30B-A3B"
    assert payload["summary"]["n"] == 2


def test_run_llama_task_with_fake_cli(tmp_path):
    fake_cli = tmp_path / "fake_llama_cli.py"
    fake_cli.write_text(
        """#!/usr/bin/env python3
import sys
prompt = sys.argv[sys.argv.index("-p") + 1]
print(f"\\nLoading model...\\n\\n> {prompt}\\n\\n5\\n\\n[ Prompt: 100.0 t/s | Generation: 50.0 t/s ]\\n")
""",
        encoding="utf-8",
    )
    fake_cli.chmod(fake_cli.stat().st_mode | 0o111)
    fake_model = tmp_path / "model.gguf"
    fake_model.write_text("not a real model", encoding="utf-8")
    task = next(task for task in run_llm_micro_eval.TASKS if task.task_id == "arith_2_plus_3")
    args = argparse.Namespace(
        llama_cli=str(fake_cli),
        model=str(fake_model),
        ctx_size=256,
        threads=1,
        temperature=0.0,
        top_k=1,
        repeat_penalty=1.05,
        seed=1,
        timeout=5,
    )

    result = run_llm_micro_eval.run_llama_task(args, task)

    assert result["status"] == "ok"
    assert result["response"] == "5"
    assert result["grade"]["passed"] is True
    assert result["timing"]["generation_tps"] == 50.0


def test_run_openai_chat_task_with_fake_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib hook name
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert self.path == "/v1/chat/completions"
            assert body["model"] == "fake-30b"
            response = {
                "choices": [{"message": {"content": "5"}}],
                "usage": {"completion_tokens": 1},
            }
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        task = next(task for task in run_llm_micro_eval.TASKS if task.task_id == "arith_2_plus_3")
        args = argparse.Namespace(
            openai_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            openai_model="fake-30b",
            openai_api_key="",
            model_name="fake_endpoint",
            temperature=0.0,
            seed=1,
            timeout=5,
        )

        result = run_llm_micro_eval.run_openai_chat_task(args, task)
    finally:
        server.shutdown()
        server.server_close()

    assert result["status"] == "ok"
    assert result["response"] == "5"
    assert result["grade"]["passed"] is True
    assert result["timing"]["generation_tps"] is not None


def test_write_outputs(tmp_path):
    payload = {
        "run_id": "test_micro_eval",
        "date": "2026-06-04",
        "backend": "llama-cli",
        "model_name": "fake",
        "model_path": "model.gguf",
        "llama_cli": "llama-cli",
        "summary": {
            "n": 1,
            "passed": 1,
            "failed": 0,
            "accuracy": 1.0,
            "by_category": {"arithmetic": {"n": 1, "passed": 1, "accuracy": 1.0}},
            "generation_tps_mean": 50.0,
        },
        "results": [
            {
                "task_id": "arith_2_plus_3",
                "category": "arithmetic",
                "expected": "5",
                "response": "5",
                "grade": {"passed": True},
            }
        ],
    }

    json_path, md_path = run_llm_micro_eval.write_outputs(payload, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["accuracy"] == 1.0
    assert "LLM micro-eval" in md_path.read_text(encoding="utf-8")
