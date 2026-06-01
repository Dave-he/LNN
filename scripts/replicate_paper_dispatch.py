#!/usr/bin/env python3
"""依据 digest 挑选可复现论文, 调用现有 reproduce 脚本执行.

支持的复现目标 (按 arxiv id 命中或关键词命中):
  - replicate_paper_experiment.py  : 天然气价格预测 (LNN vs LSTM, Henry Hub 数据)
  - replicate_temporal_dropout.py  : LNN vs LSTM 鲁棒性 + temporal dropout
  - experiment_imitation_lnn.py    : Liquid + MDN imitation learning (Push-T 等)

输出日志到 logs/pipeline/{date}_reproduce.log, 实验结果在 analysis/ 目录.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pipeline"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 规则: (触发关键词列表, 复现脚本, 额外参数)
REPRO_RULES: list[tuple[list[str], str, list[str]]] = [
    (
        ["natural gas", "henry hub", "gas price", "gas spot price", "energy forecasting"],
        "scripts/replicate_paper_experiment.py",
        ["--output-dir", "analysis/replication/natural_gas"],
    ),
    (
        ["temporal dropout", "robust", "robustness", "noise injection", "clinical utility"],
        "scripts/replicate_temporal_dropout.py",
        ["--output-dir", "analysis/replication/temporal_dropout"],
    ),
    (
        ["imitation", "mixture density", "mdn", "push-t", "robomimic", "behavior cloning"],
        "scripts/experiment_imitation_lnn.py",
        [],
    ),
]


def load_digest_papers(date: str) -> list[dict[str, str]]:
    """复用 select_papers_for_report 的解析逻辑, 简化返回."""
    from select_papers_for_report import parse_digest  # type: ignore

    return parse_digest(date).get("arxiv", [])


def pick_targets(date: str) -> list[dict[str, Any]]:
    papers = load_digest_papers(date)
    matched: list[dict[str, Any]] = []
    seen_scripts: set[str] = set()
    for paper in papers:
        blob = (paper["title"] + " " + paper["summary"]).lower()
        for keywords, script, extra in REPRO_RULES:
            if any(k in blob for k in keywords):
                if script in seen_scripts:
                    break
                seen_scripts.add(script)
                matched.append(
                    {
                        "arxiv_id": paper["id"],
                        "title": paper["title"],
                        "script": script,
                        "extra_args": extra,
                    }
                )
                break
    return matched


def run_one(target: dict[str, Any], date: str, log_path: pathlib.Path) -> int:
    cmd = ["python3", target["script"], *target["extra_args"]]
    header = (
        f"\n=== {dt.datetime.now():%F %T} "
        f"reproducing {target['arxiv_id']} via {target['script']} ===\n"
        f"title : {target['title']}\n"
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
    fail = 0
    for target in targets:
        code = run_one(target, args.date, log_path)
        if code != 0:
            fail += 1
    print(f"复现完成: {len(targets) - fail}/{len(targets)} 成功, 失败 {fail}, 日志 {log_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
