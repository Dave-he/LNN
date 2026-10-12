"""Unit tests for the Jetson benchmark's failure handling (2026-10-12).

The 2026-10-09..10-12 runs left no artifact at all: `import torch` raised
`ImportError: libcudss.so.0: cannot open shared object file`, which fell through
to the generic handler, failed `looks_like_cuda_runtime_error` (that predicate
requires the literal substring "cuda", absent from "libcudss"), and re-raised —
so `write_report` never ran and the daily run silently produced no report.

These tests pin the classifier and the CUDA-runtime-error predicate so a silent
day cannot regress unnoticed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "jetson_lnn_benchmark.py"


def _load_module():
    """Load jetson_lnn_benchmark.py by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("_jetson_lnn_benchmark", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_module()


class TestTorchLoadFailureClassifier(unittest.TestCase):
    def test_shared_object_import_error_is_torch_load_failure(self):
        exc = ImportError("libcudss.so.0: cannot open shared object file: No such file or directory")
        self.assertTrue(_mod._is_torch_load_failure(exc))

    def test_undefined_symbol_import_error_is_torch_load_failure(self):
        exc = ImportError("undefined symbol: cudaGraphAddNode")
        self.assertTrue(_mod._is_torch_load_failure(exc))

    def test_missing_module_is_not_torch_load_failure(self):
        # ModuleNotFoundError is handled earlier in main(); a plain ImportError
        # naming a missing module must not be misread as a dlopen failure.
        self.assertFalse(_mod._is_torch_load_failure(ImportError("No module named 'toml'")))

    def test_non_import_error_is_not_torch_load_failure(self):
        self.assertFalse(_mod._is_torch_load_failure(ValueError("bad config")))


class TestCudaRuntimeErrorPredicate(unittest.TestCase):
    def test_libcudss_message_does_not_match(self):
        # Documents *why* the separate ImportError handler is required: the
        # existing predicate looks for "cuda", which "libcudss" does not contain.
        exc = ImportError("libcudss.so.0: cannot open shared object file")
        self.assertFalse(_mod.looks_like_cuda_runtime_error(exc))

    def test_cuda_oom_still_matches(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertTrue(_mod.looks_like_cuda_runtime_error(exc))


class TestExceptionSummary(unittest.TestCase):
    def test_summary_is_single_line(self):
        summary = _mod.exception_summary(ImportError("line one\nline two"))
        self.assertNotIn("\n", summary)
        self.assertIn("line one", summary)


class TestStatusIsHonestAboutCpuRuns(unittest.TestCase):
    """A CPU run on a Jetson host must not report a bare "ok".

    2026-10-12 produced `status: "ok"` while CUDA was unavailable
    (`cuda_available: False`, driver 12060 vs torch cu130), which reads as GPU
    evidence. See [[analysis/pipeline_notes/2026-10-12_automation_audit]].
    """

    def _normalize(self, payload):
        return _mod.mark_cpu_only_run(payload)

    def test_cpu_run_is_renamed_ok_cpu(self):
        payload = self._normalize({"status": "ok", "device": "cpu"})
        self.assertEqual(payload["status"], "ok_cpu")
        self.assertIn("not GPU/edge performance evidence", payload["status_note"])

    def test_cuda_run_keeps_ok(self):
        payload = self._normalize({"status": "ok", "device": "cuda"})
        self.assertEqual(payload["status"], "ok")
        self.assertNotIn("status_note", payload)

    def test_cpu_fallback_status_is_preserved(self):
        # "ok_cpu_fallback" already means CPU and carries cuda_fallback_reason;
        # it must not be rewritten to a different name.
        payload = self._normalize({"status": "ok_cpu_fallback", "device": "cpu", "cuda_fallback_reason": "boom"})
        self.assertEqual(payload["status"], "ok_cpu_fallback")
        self.assertEqual(payload["cuda_fallback_reason"], "boom")


class TestSuccessStatusesCoverEveryProducibleStatus(unittest.TestCase):
    """Every status the run path can emit must still render the result tables.

    2026-10-12: `mark_cpu_only_run` started emitting `ok_cpu`, but `write_report`
    gated the results section on `{"ok", "ok_cpu_fallback"}`. The report then
    rendered only "## 状态 / ok_cpu: None" while the sibling JSON held all six
    model results and a Pareto PNG was on disk. A new status must not be added
    to the emitters without adding it here.
    """

    def test_ok_cpu_is_a_success_status(self):
        self.assertIn("ok_cpu", _mod.SUCCESS_STATUSES)

    def test_fallback_status_is_preserved(self):
        self.assertIn("ok_cpu_fallback", _mod.SUCCESS_STATUSES)

    def test_plain_ok_is_a_success_status(self):
        self.assertIn("ok", _mod.SUCCESS_STATUSES)


class TestOkCpuReportRendersResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_cleanup)

    def tmp_cleanup(self):
        self.tmp.cleanup()

    def test_ok_cpu_report_contains_result_rows_and_cpu_caveat(self):
        payload = _mod.mark_cpu_only_run(
            {
                "status": "ok",
                "device": "cpu",
                "experiment": "jetson_lnn_benchmark",
                "environment": {},
                "config": {},
                "results": [
                    {
                        "name": "LTC",
                        "parameters": 185,
                        "test_mse": 0.5,
                        "inference_steps_per_sec": 1234.5,
                        "train_seconds": 3.0,
                    }
                ],
            }
        )
        with mock.patch.object(_mod, "ROOT", pathlib.Path(self.tmp.name)):
            with mock.patch.object(_mod, "write_benchmark_plot", return_value=None):
                _, md_path = _mod.write_report("2099-01-01", payload)
        text = md_path.read_text(encoding="utf-8")

        self.assertIn("| LTC | 185 | 0.500000 | 1234.5 | 3.00 |", text)
        self.assertIn("CPU-only", text)
        self.assertNotIn("## 状态", text)

    def test_failed_status_still_reports_the_reason(self):
        with mock.patch.object(_mod, "ROOT", pathlib.Path(self.tmp.name)):
            with mock.patch.object(_mod, "write_benchmark_plot", return_value=None):
                _, md_path = _mod.write_report(
                    "2099-01-02",
                    {"status": "skipped", "reason": "torch import failed", "environment": {}},
                )
        text = md_path.read_text(encoding="utf-8")

        self.assertIn("## 状态", text)
        self.assertIn("torch import failed", text)


if __name__ == "__main__":
    unittest.main()
