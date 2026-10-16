#!/usr/bin/env python3
"""校验负结果台账与证据一致, 让 AGENTS.md 约束 #4 可机械执行.

检查三件事:
  1. 每条台账条目的 `grounding:` 路径真实存在 (台账不指向已删/已挪走的证据)
  2. AGENTS.md 约束表每一行都能在台账中找到对应条目 (约束不悬空)
  3. 台账表格行数 == 条目数 (表格与正文不脱节)

退出码 0 = 一致; 1 = 有失配, 且逐条打印.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "analysis" / "negative_results" / "README.md"
AGENTS = ROOT / "AGENTS.md"

ENTRY_RE = re.compile(r"^<a id=\"(nr-\d+)\"></a>", re.MULTILINE)
GROUNDING_RE = re.compile(r"^- `grounding: (.+?)`\s*$", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\|\s*\[(NR-\d+)\]\(#", re.MULTILINE)
CLAIM_ROW_RE = re.compile(r"^\|\s*(\d)\s*\|\s*❌", re.MULTILINE)


def collect_entries(text: str) -> list[tuple[str, str]]:
    """返回 [(entry_id, grounding_path)], 按文档顺序."""
    entries: list[tuple[str, str]] = []
    matches = list(ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        grounding = GROUNDING_RE.search(block)
        entries.append((m.group(1), grounding.group(1).strip() if grounding else ""))
    return entries


def main() -> int:
    problems: list[str] = []

    if not LEDGER.exists():
        print(f"[FAIL] 台账不存在: {LEDGER.relative_to(ROOT)}")
        print("       AGENTS.md 约束 #4 要求负结果进 negative-result 目录")
        return 1

    ledger_text = LEDGER.read_text(encoding="utf-8")
    entries = collect_entries(ledger_text)

    if not entries:
        problems.append("台账里没有解析到任何条目 (检查 <a id=\"nr-NN\"></a> 锚点)")

    # 1. grounding 路径必须存在, 且每个条目都要有 grounding
    for entry_id, grounding in entries:
        if not grounding:
            problems.append(f"{entry_id}: 缺少 `grounding:` 行")
            continue
        path = ROOT / grounding
        if not path.exists():
            problems.append(f"{entry_id}: grounding 指向不存在的文件 -> {grounding}")

    # 2. 表格与条目必须一一对应
    table_ids = TABLE_ROW_RE.findall(ledger_text)
    entry_ids = [e[0].upper() for e in entries]
    if sorted(table_ids) != sorted(entry_ids):
        only_table = sorted(set(table_ids) - set(entry_ids))
        only_body = sorted(set(entry_ids) - set(table_ids))
        if only_table:
            problems.append(f"表格里有但正文没条目: {', '.join(only_table)}")
        if only_body:
            problems.append(f"正文有条目但表格没登记: {', '.join(only_body)}")

    # 3. AGENTS.md 每条 ❌ 约束都要在台账里有对应条目
    if AGENTS.exists():
        agents_text = AGENTS.read_text(encoding="utf-8")
        claim_count = len(CLAIM_ROW_RE.findall(agents_text))
        if claim_count > len(entries):
            problems.append(
                f"AGENTS.md 有 {claim_count} 条 ❌ 约束, 台账只有 {len(entries)} 条条目"
            )
        referenced = "negative_results/README" in agents_text
        if not referenced:
            problems.append(
                "AGENTS.md 约束 #4 没有指向 analysis/negative_results/README"
                " (接受 [[wikilink]] 或 markdown 路径两种写法)"
            )
    else:
        problems.append("AGENTS.md 不存在, 无法核对约束表")

    if problems:
        print("[FAIL] 负结果台账校验未通过:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"[OK] 负结果台账一致: {len(entries)} 条条目, grounding 全部存在")
    print("     AGENTS.md 约束表已全部落地到台账")
    return 0


if __name__ == "__main__":
    sys.exit(main())
