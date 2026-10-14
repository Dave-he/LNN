#!/usr/bin/env python3
"""依据 digest 挑选可复现论文, 调用现有 reproduce 脚本执行.

支持的复现目标 (按 arxiv id 命中或关键词命中):
  - replicate_paper_experiment.py  : 天然气价格预测 (LNN vs LSTM, Henry Hub 数据)
  - replicate_temporal_dropout.py  : LNN vs LSTM 鲁棒性 + temporal dropout
  - experiment_imitation_lnn.py    : Liquid + MDN imitation learning (Push-T 等)

输出日志到 logs/pipeline/{date}_reproduce.log, 实验结果在 analysis/ 目录.

匹配规则: 关键词一律按 **词边界** 在**标题**里匹配; 只有高特异度的多词锚点
(如 "henry hub" / "behavior cloning") 才允许在摘要里匹配。旧实现对
``title + summary`` 做裸子串匹配, 2026-10-14 因此把 Liquid Fusion SOD
(2606.26849, 与 imitation learning 无关) 误判成 imitation 实验 —— 命中的是摘要里
"**limit**ation" 中的子串 "imitation"。裸子串匹配短词是不可靠的。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pipeline"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 规则: (标题关键词, 摘要锚点(高特异度多词), 复现脚本, 额外参数)
REPRO_RULES: list[tuple[list[str], list[str], str, list[str]]] = [
    (
        ["natural gas", "henry hub", "gas price", "gas spot price", "energy forecasting"],
        ["henry hub", "natural gas", "gas spot price"],
        "scripts/replicate_paper_experiment.py",
        ["--output_dir", "analysis/replication/natural_gas"],
    ),
    (
        ["temporal dropout", "robust", "robustness", "noise injection", "clinical utility"],
        ["temporal dropout", "noise injection"],
        "scripts/replicate_temporal_dropout.py",
        ["--output_dir", "analysis/replication/temporal_dropout"],
    ),
    (
        ["imitation", "mixture density", "mdn", "push-t", "robomimic", "behavior cloning"],
        ["mixture density", "push-t", "robomimic", "behavior cloning", "imitation learning"],
        "scripts/experiment_imitation_lnn.py",
        [],
    ),
]

# 与 run_daily_lnn_task.sh 的 resolve_bench_python 同一套候选: 复现脚本需要能
# import torch 的解释器, 裸 python3 在本机没有 torch, 会连续 exit 1。
PYTHON_CANDIDATES = [
    "python3",
    "/usr/bin/python3.10",
    "/home/hyx/.pyenv/versions/3.14.4/bin/python3",
    "python3.12",
    "python3.11",
]


def _has_torch(interpreter: str) -> bool:
    try:
        proc = subprocess.run(
            [interpreter, "-c", "import torch"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def resolve_python() -> str:
    """返回第一个能 import torch 的解释器; 都不可用时退回 sys.executable 并告警."""
    candidates = [sys.executable]
    explicit = os.environ.get("PYTHON_BIN", "").strip()
    if explicit:
        candidates.insert(0, explicit)
    candidates.extend(PYTHON_CANDIDATES)
    for candidate in candidates:
        if candidate and _has_torch(candidate):
            return candidate
    return explicit or sys.executable


def _word_pattern(keyword: str) -> re.Pattern[str]:
    # push-t / robomimic 带连字符, 用 \b 会把 "push-t" 的结尾 '-' 当边界, 可用;
    # 但要转义正则元字符以免 keyword 里的 . + 等被当通配.
    return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)


def strip_markdown_link(title: str) -> str:
    """digest 里 title 形如 ``[Title](https://arxiv.org/abs/2606.26849v1)``, 只留可读标题."""
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title or "")


def load_digest_papers(date: str) -> list[dict[str, str]]:
    """复用 select_papers_for_report 的解析逻辑, 简化返回."""
    from select_papers_for_report import parse_digest  # type: ignore

    return parse_digest(date).get("arxiv", [])


def pick_targets(date: str) -> list[dict[str, Any]]:
    papers = load_digest_papers(date)
    matched: list[dict[str, Any]] = []
    seen_scripts: set[str] = set()
    for paper in papers:
        title = strip_markdown_link(paper["title"])
        summary = paper.get("summary") or ""
        for keywords, summary_anchors, script, extra in REPRO_RULES:
            hit = next((k for k in keywords if _word_pattern(k).search(title)), None)
            if hit is None:
                hit = next((k for k in summary_anchors if _word_pattern(k).search(summary)), None)
            if hit is None:
                continue
            if script in seen_scripts:
                break
            seen_scripts.add(script)
            matched.append(
                {
                    "arxiv_id": paper["id"],
                    "title": paper["title"],
                    "matched_keyword": hit,
                    "script": script,
                    "extra_args": extra,
                }
            )
            break
    return matched


def run_one(target: dict[str, Any], date: str, log_path: pathlib.Path, python_bin: str) -> int:
    cmd = [python_bin, target["script"], *target["extra_args"]]
    header = (
        f"\n=== {dt.datetime.now():%F %T} "
        f"reproducing {target['arxiv_id']} via {target['script']} ===\n"
        f"title : {target['title']}\n"
        f"hit   : {target.get('matched_keyword', '?')}\n"
        f"cmd   : {' '.join(cmd)}\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(header)
    print(header, flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        msg = f"[timeout] {' '.join(cmd)} 跳过本次复现"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        print(msg)
        return 124
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[exit] {proc.returncode}\n\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="digest 日期")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = pick_targets(args.date)
    log_path = LOG_DIR / f"{args.date}_reproduce.log"
    if args.dry_run:
        print(json.dumps({"date": args.date, "targets": targets}, ensure_ascii=False, indent=2))
        return 0

    if not targets:
        msg = f"[{dt.datetime.now():%F %T}] digest {args.date} 无命中复现规则的论文, 跳过"
        log_path.write_text(msg + "\n", encoding="utf-8")
        print(msg)
        return 0

    print(f"匹配 {len(targets)} 篇可复现论文, 日志: {log_path}")
    python_bin = resolve_python()
    fail = 0
    for target in targets:
        code = run_one(target, args.date, log_path, python_bin)
        if code != 0:
            fail += 1
    print(f"复现完成: {len(targets) - fail}/{len(targets)} 成功, 失败 {fail}, 日志 {log_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
