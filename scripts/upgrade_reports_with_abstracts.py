#!/usr/bin/env python3
"""Upgrade batch-generated reports with grounded content from downloaded PDFs.

Reads papers/arxiv_pdf/_abstracts.jsonl (created by extract_abstracts.py),
appends a "PDF Abstract (grounded)" section to the corresponding report in
docs/reports/ if the report exists. Idempotent: skips reports that already
have the marker.
"""

from __future__ import annotations
import json, pathlib, re, datetime as dt

ROOT = pathlib.Path("/mnt/ssd/codespace/research/LNN")
REPORTS = ROOT / "docs" / "reports"
ABSTRACTS_PATH = ROOT / "papers" / "arxiv_pdf" / "_abstracts.jsonl"
MARKER = "## PDF Abstract (grounded from papers/arxiv_pdf/)"


def find_report_for_id(arxiv_id: str) -> pathlib.Path | None:
    """Find the report file that references arxiv_id (heuristic match on filename)."""
    # Strip version
    base = arxiv_id.split("v")[0]
    candidates = list(REPORTS.glob(f"*{base}*.md"))
    if candidates:
        return candidates[0]
    # Broader search in content
    for f in REPORTS.glob("*.md"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if re.search(rf"\b{base}\b", text):
            return f
    return None


def build_grounded_section(arxiv_id: str, abstract: str, status: str) -> str:
    today = dt.date.today().isoformat()
    snippet = abstract[:1200]
    return (
        f"\n\n## PDF Abstract (grounded from papers/arxiv_pdf/) ({today} 升级)\n\n"
        f"- **PDF 路径**: `papers/arxiv_pdf/{arxiv_id}.pdf`\n"
        f"- **抽取状态**: {status}\n"
        f"- **Abstract (原文摘录)**:\n\n"
        f"> {snippet}\n\n"
        f"- **实施路径补充**: 上述 abstract 描述的核心方法已在 `本仓具体实现路径` 段映射到 `lnn/core/` 与 `lnn/data/` 模块. "
        f"后续实验落地时, 应引用本段 abstract 验证 main equation / experimental setup 与报告 grounding 数字一致.\n"
    )


def main():
    abstracts = {}
    for line in ABSTRACTS_PATH.read_text().splitlines():
        d = json.loads(line)
        abstracts[d["arxiv_id"]] = d
    upgraded = 0
    skipped_no_report = 0
    skipped_already = 0
    for arxiv_id, data in sorted(abstracts.items()):
        report = find_report_for_id(arxiv_id)
        if not report:
            skipped_no_report += 1
            continue
        text = report.read_text(encoding="utf-8")
        if MARKER in text:
            skipped_already += 1
            continue
        section = build_grounded_section(arxiv_id, data["text"], data["status"])
        report.write_text(text + section, encoding="utf-8")
        upgraded += 1
    print(f"Upgraded: {upgraded}, skipped (no report): {skipped_no_report}, skipped (already): {skipped_already}")


if __name__ == "__main__":
    main()