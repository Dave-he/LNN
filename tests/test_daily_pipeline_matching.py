"""每日研究流水线的离线回归测试 (arXiv 重试 + 复现 dispatcher 匹配).

两个被修掉的真实缺陷:

1. ``daily_lnn_research.fetch_arxiv`` 旧实现只尝试一次, 一次 Varnish 抖动就让整轮
   退回前一日候选池 (09-22 / 09-23 / 10-12 / 10-13 / 10-14 连续命中)。
2. ``replicate_paper_dispatch`` 旧实现对 ``title + summary`` 做裸子串匹配, 于是摘要里
   "**limit**ation" 的子串 "imitation" 把 Liquid Fusion SOD (2606.26849, 与 imitation
   learning 无关) 误判成 imitation 实验。

用 unittest 而非 pytest: 本机 ``python3 -m pytest`` 会因为全局 deepeval 插件经
eventlet / pyOpenSSL 崩溃而**静默收集 0 条并返回退出码 0**, 断言形同虚设。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import daily_lnn_research as daily  # noqa: E402
import replicate_paper_dispatch as dispatch  # noqa: E402


class FakeRequestText:
    """替换 daily.request_text, 记录调用次数并按脚本抛异常."""

    def __init__(self, failures: int, result: str = "<feed/>") -> None:
        self.failures = failures
        self.result = result
        self.calls = 0

    def __call__(self, url: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("The read operation timed out")
        return self.result


class ArxivRetryTests(unittest.TestCase):
    def _patch(self, fake: FakeRequestText):
        original = daily.request_text
        daily.request_text = fake  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(daily, "request_text", original))

    def test_first_attempt_success_does_not_retry(self) -> None:
        fake = FakeRequestText(failures=0)
        self._patch(fake)
        self.assertEqual(daily.request_text_retry("http://x", backoff=()), fake.result)
        self.assertEqual(fake.calls, 1)

    def test_retries_transient_timeout_then_succeeds(self) -> None:
        fake = FakeRequestText(failures=2)
        self._patch(fake)
        # backoff 里用 0 占位: 实现只在 delay 非 0 时 sleep, 所以测试无需真等.
        self.assertEqual(daily.request_text_retry("http://x", backoff=(0, 0)), fake.result)
        self.assertEqual(fake.calls, 3)

    def test_empty_backoff_means_single_attempt(self) -> None:
        fake = FakeRequestText(failures=1)
        self._patch(fake)
        with self.assertRaises(TimeoutError):
            daily.request_text_retry("http://x", backoff=())
        self.assertEqual(fake.calls, 1)

    def test_raises_after_backoff_exhausted(self) -> None:
        fake = FakeRequestText(failures=99)
        self._patch(fake)
        # backoff=(1, 2) -> 立即 1 次 + 退避后 2 次 = 3 次尝试后抛原异常.
        with self.assertRaises(TimeoutError):
            daily.request_text_retry("http://x", backoff=(1, 2))
        self.assertEqual(fake.calls, 3)

    def test_backoff_schedule_is_60s_capped(self) -> None:
        # Varnish 清理窗口实测 30~60s, 退避上限不得超过 60s.
        self.assertEqual(daily.ARXIV_RETRY_BACKOFF, (8, 30, 60))
        self.assertLessEqual(max(daily.ARXIV_RETRY_BACKOFF), 60)


class DispatcherMatchTests(unittest.TestCase):
    def _match(self, title: str, summary: str = "") -> tuple[str | None, str | None]:
        for keywords, anchors, script, _extra in dispatch.REPRO_RULES:
            hit = next(
                (k for k in keywords if dispatch._word_pattern(k).search(title)),
                None,
            ) or next(
                (k for k in anchors if dispatch._word_pattern(k).search(summary)),
                None,
            )
            if hit is not None:
                return hit, script
        return None, None

    def test_limitation_substring_does_not_trigger_imitation(self) -> None:
        """回归: 2026-10-14 真实误判 —— "limitation" 含子串 "imitation"."""
        hit, script = self._match(
            "Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection",
            "a critical limitation of current approaches is their neglect of the spectral bias",
        )
        self.assertIsNone(hit)
        self.assertIsNone(script)

    def test_generic_abstract_words_never_trigger(self) -> None:
        for summary in ("limited generality", "robustness of baselines", "imitations"):
            self.assertIsNone(self._match("Sequential Modeling of Streams", summary)[0], summary)

    def test_imitation_learning_title_still_matches(self) -> None:
        hit, script = self._match("MDN-based Liquid Imitation Learning for Push-T")
        self.assertEqual(hit, "imitation")
        self.assertEqual(script, "scripts/experiment_imitation_lnn.py")

    def test_natural_gas_anchor_matches_in_abstract(self) -> None:
        hit, script = self._match(
            "Forecasting with Continuous-Time Networks", "we use daily Henry Hub prices"
        )
        self.assertEqual(hit, "henry hub")
        self.assertEqual(script, "scripts/replicate_paper_experiment.py")

    def test_temporal_dropout_title_matches(self) -> None:
        _, script = self._match("Temporal Dropout for Liquid Time-Constant Networks")
        self.assertEqual(script, "scripts/replicate_temporal_dropout.py")

    def test_strip_markdown_link_removes_url(self) -> None:
        raw = "[Liquid Fusion of Heterogeneous Representations](https://arxiv.org/abs/2606.26849v1)"
        self.assertEqual(
            dispatch.strip_markdown_link(raw), "Liquid Fusion of Heterogeneous Representations"
        )


class IndexEntryTests(unittest.TestCase):
    """每日索引的条目切分与容量截断 (2026-10-14 实测的数据丢失 bug)."""

    BLOCK = "\n".join(
        [
            "- **2026-10-14**：今日",
            "  - 复盘明细一",
            "  - 复盘明细二",
            "- **2026-10-06**：老条目",
            "- **workaround**：续行, 不是日期条目",
            "- **SSH push 第 3 次成功**：又一条续行",
            "- **2026-10-05**：再老一天",
        ]
    )

    def test_split_keeps_nested_lines_with_their_day(self) -> None:
        entries = daily.split_index_entries(self.BLOCK)
        self.assertEqual([c[0].split("**")[1] for c in entries], ["2026-10-14", "2026-10-06", "2026-10-05"])
        self.assertEqual(len(entries[0]), 3, "缩进复盘明细必须跟随其日期条目")

    def test_continuation_lines_do_not_start_new_entries(self) -> None:
        rows = [row for chunk in daily.split_index_entries(self.BLOCK) for row in chunk]
        self.assertIn("- **workaround**：续行, 不是日期条目", rows)

    def test_cap_counts_days_not_lines(self) -> None:
        rows = [row for chunk in daily.split_index_entries(self.BLOCK) for row in chunk]
        # 旧实现按行截断 29 行, 续行会吃光容量并挤掉真实日期条目.
        capped = daily.cap_index_entries(rows, max_days=2)
        days = [r for r in capped if daily.DAY_ENTRY_RE.match(r)]
        # 保留最近两天 (10-14 / 10-06), 而不是"前 29 行里恰好剩下的两行"
        self.assertEqual(days, ["- **2026-10-14**：今日", "- **2026-10-06**：老条目"])
        self.assertNotIn("- **2026-10-05**：再老一天", capped)
        # 被保留日期的子条目也必须一起留下
        self.assertIn("  - 复盘明细二", capped)

    def test_cap_keeps_sub_lines_of_dropped_days_out(self) -> None:
        rows = ["- **2026-10-14**：a", "  - 子", "- **2026-10-06**：b", "  - 子2"]
        capped = daily.cap_index_entries(rows, max_days=1)
        self.assertEqual(capped, ["- **2026-10-14**：a", "  - 子"])


class DispatcherInterpreterTests(unittest.TestCase):
    def _env(self, value: str | None) -> None:
        original = os.environ.get("PYTHON_BIN")
        if value is None:
            os.environ.pop("PYTHON_BIN", None)
        else:
            os.environ["PYTHON_BIN"] = value
        self.addCleanup(
            lambda: os.environ.__setitem__("PYTHON_BIN", original)
            if original is not None
            else os.environ.pop("PYTHON_BIN", None)
        )

    def test_prefers_explicit_interpreter_with_torch(self) -> None:
        self._env("good-python")
        original = dispatch._has_torch
        seen: list[str] = []

        def fake_has_torch(interpreter: str) -> bool:
            seen.append(interpreter)
            return interpreter == "good-python"

        dispatch._has_torch = fake_has_torch  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(dispatch, "_has_torch", original))
        self.assertEqual(dispatch.resolve_python(), "good-python")
        # PYTHON_BIN 排在候选首位, 命中后不应再探测后面的解释器.
        self.assertEqual(seen, ["good-python"])

    def test_skips_interpreter_without_torch(self) -> None:
        self._env(None)
        original = dispatch._has_torch
        dispatch._has_torch = lambda interpreter: interpreter == "python3"  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(dispatch, "_has_torch", original))
        self.assertEqual(dispatch.resolve_python(), "python3")

    def test_falls_back_to_explicit_python_bin_without_torch(self) -> None:
        """本机 (Jetson R36) 没有任何能 import torch 的解释器, 此时退回 PYTHON_BIN
        以便复现脚本至少留下可读的失败日志, 而不是 dispatcher 自己静默死掉."""
        self._env("torchless-python")
        original = dispatch._has_torch
        dispatch._has_torch = lambda interpreter: False  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(dispatch, "_has_torch", original))
        self.assertEqual(dispatch.resolve_python(), "torchless-python")


if __name__ == "__main__":
    unittest.main()
