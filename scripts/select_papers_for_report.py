#!/usr/bin/env python3
"""从当日 digest 中挑选 1-3 篇高价值论文, 输出待研读清单.

该脚本被两层调用:
  1. LLM cron prompt 在生成研读报告前, 调用本脚本得到候选清单 + 关键摘要.
  2. 编排脚本 run_lnn_research_pipeline.sh 在 --skip-report=0 时本地烟测.

挑选规则 (按优先级):
  - 出现 'liquid' / 'CfC' / 'LTC' / 'NCP' / 'closed-form continuous-time' 等强关键词
  - 距今 < 30 天
  - keyword_score 高者优先
  - 已被研读过 (在 docs/reports/ 出现文件名) 的跳过
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "docs" / "daily"
REPORTS_DIR = ROOT / "docs" / "reports"
ARXIV_LINK_RE = re.compile(r"arxiv\.org/abs/([0-9]+\.[0-9]+(v[0-9]+)?)", re.IGNORECASE)
STRONG_KEYWORDS = re.compile(
    r"liquid neural|closed[- ]form continuous[- ]time|liquid time[- ]constant|"
    r"neural circuit polic|liquid structural state[- ]space|"
    r"\bCfC\b|\bLTC\b|\bLFM2|neural[- ]ode|continuous[- ]depth",
    re.IGNORECASE,
)


def parse_digest(date: str) -> dict[str, Any]:
    path = DAILY_DIR / f"{date}_LNN_research_digest.md"
    if not path.exists():
        return {"path": str(path), "arxiv": [], "repos": [], "models": []}

    text = path.read_text(encoding="utf-8")
    arxiv: list[dict[str, str]] = []
    repos: list[dict[str, str]] = []

    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            continue
        if not line.startswith("|"):
            continue
        if section.startswith("arxiv 候选论文"):
            match = ARXIV_LINK_RE.search(line)
            if not match:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4 or "arxiv" in cells[0].lower():
                continue
            arxiv.append(
                {
                    "id": match.group(1),
                    "date": cells[0],
                    "title": cells[1],
                    "authors": cells[2],
                    "summary": cells[3],
                    "url": f"https://arxiv.org/abs/{match.group(1)}",
                }
            )
        elif section.startswith("github 候选仓库"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4 or "更新" in cells[0]:
                continue
            repos.append(
                {
                    "updated": cells[0],
                    "name": cells[1],
                    "stars": cells[2],
                    "description": cells[3],
                }
            )

    return {"path": str(path), "arxiv": arxiv, "repos": repos}


def already_reported(arxiv_id: str) -> bool:
    if not REPORTS_DIR.exists():
        return False
    for md in REPORTS_DIR.glob("*.md"):
        if arxiv_id in md.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def score_paper(paper: dict[str, str]) -> int:
    blob = f"{paper['title']} {paper['summary']}".lower()
    score = 0
    score += 3 * len(STRONG_KEYWORDS.findall(blob))
    if "liquid neural" in blob:
        score += 5
    if "closed-form continuous-time" in blob or "closed form continuous time" in blob:
        score += 4
    if "imitation" in blob or "robot" in blob or "control" in blob:
        score += 2
    if "forecasting" in blob or "time series" in blob or "time-series" in blob:
        score += 2
    if "edge" in blob or "embedded" in blob or "jetson" in blob or "deployment" in blob:
        score += 2
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="digest 日期 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=3, help="返回前 N 篇")
    args = parser.parse_args()

    digest = parse_digest(args.date)
    candidates: list[dict[str, Any]] = []
    for paper in digest["arxiv"]:
        if already_reported(paper["id"]):
            continue
        s = score_paper(paper)
        if s <= 0:
            continue
        candidates.append({**paper, "score": s})
    candidates.sort(key=lambda p: (p["score"], p["date"]), reverse=True)

    payload = {
        "date": args.date,
        "digest_path": digest["path"],
        "candidates": candidates[: args.top],
        "n_total_arxiv": len(digest["arxiv"]),
        "n_skipped_reported": sum(
            1 for p in digest["arxiv"] if already_reported(p["id"])
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
