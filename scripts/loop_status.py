#!/usr/bin/env python3
"""Loop coverage status — PRD §8 task #8.

Built from 7 iterations of /loop 1h hands-on experience (see
docs/PRD_LNN_Edge_Research.md, analysis/jetson/2026-06-0*_loop_iteration*.md).
Each /loop fire used to re-scan the whole project to decide what to do;
this script formalises that scan so the next loop opens with a one-screen
"already done today / next candidate task" view.

What it scans:

* docs/daily/{date}_LNN_research_digest.md
* papers/daily/{date}_lnn_research.json
* analysis/repo_watchlist/{date}_lnn_open_source_watchlist.md
* analysis/jetson/{date}_lnn_benchmark.{json,md}
* analysis/jetson/{date}_loop_iteration*_*.md
* analysis/molecular/{date}_*.{json,md}
* analysis/long_sequence/{date}_*.{json,md}
* analysis/timeseries_ablation/{date}_*.{json,md}
* docs/PRD_LNN_Edge_Research.md §8 task table

What it emits:

* JSON (machine-readable) and Markdown (human) under analysis/loop_status/
* On stdout: a 1-screen status that the next /loop iteration can read first.

Usage::

    python scripts/loop_status.py                  # today, write report
    python scripts/loop_status.py --date 2026-06-04
    python scripts/loop_status.py --no-write       # stdout only
    python scripts/loop_status.py --json           # machine-readable

The script is intentionally side-effect-free outside the
``analysis/loop_status/`` directory.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----- file probes ----------------------------------------------------------


def _exists(path: pathlib.Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _glob(rel: str, date: str) -> list[dict]:
    """Return one record per file under ``rel`` whose name starts with date."""
    root = ROOT / rel
    if not root.exists():
        return []
    records = []
    for p in sorted(root.glob(f"{date}*")):
        if p.is_file():
            records.append(_exists(p))
    return records


def _scan_iterations(date: str) -> list[dict]:
    """Find all loop_iteration*_*.md reports for the date.

    These can live in analysis/jetson/, analysis/molecular/,
    analysis/long_sequence/, analysis/timeseries_ablation/, etc.
    """
    out: list[dict] = []
    pattern = re.compile(rf"^{re.escape(date)}_loop_iteration(\d+)_.+\.md$")
    for sub in ("analysis/jetson", "analysis/molecular", "analysis/long_sequence",
                "analysis/timeseries_ablation", "analysis/loop_status"):
        sub_dir = ROOT / sub
        if not sub_dir.exists():
            continue
        for p in sorted(sub_dir.glob(f"{date}_loop_iteration*_*.md")):
            m = pattern.match(p.name)
            if m:
                out.append({
                    "iteration": int(m.group(1)),
                    "title": p.stem,
                    "path": str(p.relative_to(ROOT)),
                })
    out.sort(key=lambda x: (x["iteration"], x["title"]))
    return out


# ----- git ------------------------------------------------------------------


def _git_commits_for_date(date: str) -> list[dict]:
    """Show local commits whose author date == ``date``.

    Reads via ``git log --since/--until``; date is YYYY-MM-DD local.
    """
    try:
        result = subprocess.run(
            [
                "git", "log",
                "--since", f"{date} 00:00:00",
                "--until", f"{date} 23:59:59",
                "--pretty=format:%h|%an|%s",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        commits = []
        for line in result.stdout.splitlines():
            if "|" in line:
                sha, author, subject = line.split("|", 2)
                commits.append({"sha": sha, "author": author, "subject": subject})
        return commits
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


# ----- PRD parsing ----------------------------------------------------------


def _parse_prd_tasks() -> list[dict]:
    """Read PRD §8 task table and classify each row by completion marker."""
    prd_path = ROOT / "docs" / "PRD_LNN_Edge_Research.md"
    if not prd_path.exists():
        return []
    text = prd_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_section = False
    rows: list[dict] = []
    for line in text:
        if line.strip().startswith("## 8."):
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("## "):
                break
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 4 or cells[0] in ("#", ":---") or cells[0].startswith(":-"):
                continue
            if not cells[0].isdigit():
                continue
            task_text = cells[1]
            # Detection rules from observed PRD edit history.  We default to
            # "pending" unless there's an explicit completion marker; this
            # under-counts only when a row's edit dropped its ✅ — preferable
            # to overstating done work and missing real follow-ups.
            done_markers = ("✅", "[done]", "loop#", "完成 ✅", "DONE",
                            "已 ✅", " done)", " done.", " done ", "完成]")
            pending_markers = ("pending", "PENDING", "TODO", "待")
            is_done = any(m in task_text for m in done_markers)
            is_pending_explicit = any(m in task_text for m in pending_markers) and not is_done
            if is_done:
                status = "completed"
            else:
                status = "pending"  # default — no done marker
            # explicit pending tag wins over silent default (no-op but documents intent)
            if is_pending_explicit:
                status = "pending"
            rows.append({
                "id": int(cells[0]),
                "title": task_text[:140],
                "status": status,
            })
    return rows


# ----- next-task suggestion -------------------------------------------------


def _suggest_next(prd: list[dict], iterations: list[dict]) -> str:
    pending = [r for r in prd if r["status"] == "pending"]
    if pending:
        ids = ", ".join(f"#{r['id']}" for r in pending)
        first = pending[0]
        return (
            f"Pending PRD tasks: {ids}. Next-up suggestion: "
            f"#{first['id']} — {first['title']}"
        )
    if iterations:
        last = iterations[-1]
        return (
            f"All PRD §8 tasks completed (or marker-undetected). "
            f"Most recent iteration today: #{last['iteration']} ({last['title']}). "
            "Consider extending an existing iteration or scoping a new PRD §9."
        )
    return "No PRD tasks pending and no iteration record for today. Pick a fresh thread."


# ----- markdown formatter ---------------------------------------------------


def _format_markdown(payload: dict) -> str:
    date = payload["date"]
    lines = [
        f"# Loop coverage status — {date}",
        "",
        "Generated by `scripts/loop_status.py` (PRD §8 task #8).",
        "Read this BEFORE starting the next /loop iteration to avoid redundant work.",
        "",
        "## 1. 今日固定产物",
        "| Artefact | exists | size (B) |",
        "|---|:---:|---:|",
    ]
    fixed = payload["fixed_artifacts"]
    for label, rec in fixed.items():
        mark = "✅" if rec["exists"] else "—"
        lines.append(f"| `{rec['path']}` ({label}) | {mark} | {rec['size_bytes']:,} |")

    lines.extend([
        "",
        "## 2. 今日 loop iteration 报告",
    ])
    if payload["iterations"]:
        lines.append("| # | 报告 | 路径 |")
        lines.append("|---:|---|---|")
        for it in payload["iterations"]:
            lines.append(f"| {it['iteration']} | {it['title']} | `{it['path']}` |")
    else:
        lines.append("_(尚无 iteration 报告)_")

    lines.extend([
        "",
        "## 3. 今日其他 analysis 产物",
    ])
    if payload["analysis_artifacts"]:
        lines.append("| 路径 | size (B) |")
        lines.append("|---|---:|")
        for rec in payload["analysis_artifacts"]:
            lines.append(f"| `{rec['path']}` | {rec['size_bytes']:,} |")
    else:
        lines.append("_(none)_")

    lines.extend([
        "",
        "## 4. 今日 git commits (local time)",
    ])
    if payload["git_commits"]:
        lines.append("| sha | author | subject |")
        lines.append("|---|---|---|")
        for c in payload["git_commits"]:
            subj = c["subject"].replace("|", "\\|")
            lines.append(f"| `{c['sha']}` | {c['author']} | {subj} |")
    else:
        lines.append("_(none — note: `git log` is local; remote-only commits won't show)_")

    lines.extend([
        "",
        "## 5. PRD §8 task table 状态",
        "| # | 状态 | 任务摘要 |",
        "|---:|:---:|---|",
    ])
    for r in payload["prd_tasks"]:
        emoji = {"completed": "✅", "pending": "⏳", "unknown": "❓"}.get(r["status"], "❓")
        lines.append(f"| {r['id']} | {emoji} | {r['title']} |")

    lines.extend([
        "",
        "## 6. 建议下一步",
        f"> {payload['suggestion']}",
        "",
        "## 7. 注意",
        "- 这是机械扫描结果,不代表 paper-level 的完整覆盖,只用于避免重复劳动。",
        "- `git_commits` 仅看 local repo;远程 EMMA agent 的提交需要 `git fetch + git log`。",
        "- iteration 检测靠 `analysis/**/{date}_loop_iteration*_*.md` 文件名约定,改名会漏。",
        "",
        f"产物 JSON: `{payload['json_path']}`",
    ])
    return "\n".join(lines) + "\n"


# ----- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(),
                        help="Date to scan (YYYY-MM-DD), defaults to today (local).")
    parser.add_argument("--no-write", action="store_true",
                        help="Skip writing reports to analysis/loop_status/.")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON to stdout instead of Markdown summary.")
    parser.add_argument("--output-dir", default="analysis/loop_status")
    args = parser.parse_args()

    date = args.date

    fixed = {
        "daily_digest": _exists(ROOT / "docs" / "daily" / f"{date}_LNN_research_digest.md"),
        "papers_json": _exists(ROOT / "papers" / "daily" / f"{date}_lnn_research.json"),
        "repo_watchlist": _exists(ROOT / "analysis" / "repo_watchlist"
                                  / f"{date}_lnn_open_source_watchlist.md"),
        "jetson_md": _exists(ROOT / "analysis" / "jetson" / f"{date}_lnn_benchmark.md"),
        "jetson_json": _exists(ROOT / "analysis" / "jetson" / f"{date}_lnn_benchmark.json"),
    }

    iterations = _scan_iterations(date)

    # analysis_artifacts: anything under analysis/* prefixed with date that
    # is NOT a fixed artifact and NOT an iteration report (already enumerated).
    fixed_paths = {rec["path"] for rec in fixed.values()}
    iter_paths = {it["path"] for it in iterations}
    other: list[dict] = []
    for sub in ("analysis/jetson", "analysis/molecular", "analysis/long_sequence",
                "analysis/timeseries_ablation", "analysis/repo_watchlist",
                "analysis/lfm25", "analysis/multimodal", "analysis/paper_replication"):
        for rec in _glob(sub, date):
            if rec["path"] in fixed_paths or rec["path"] in iter_paths:
                continue
            other.append(rec)

    git_commits = _git_commits_for_date(date)
    prd_tasks = _parse_prd_tasks()
    suggestion = _suggest_next(prd_tasks, iterations)

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    json_path = output_dir / f"{run_id}_loop_status_{date}.json"
    md_path = output_dir / f"{run_id}_loop_status_{date}.md"
    rel_json = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path

    payload = {
        "run_id": run_id,
        "date": date,
        "generated_at": now.isoformat(),
        "fixed_artifacts": fixed,
        "iterations": iterations,
        "analysis_artifacts": other,
        "git_commits": git_commits,
        "prd_tasks": prd_tasks,
        "suggestion": suggestion,
        "json_path": str(rel_json),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_format_markdown(payload), encoding="utf-8")

    # Always also print a short stdout summary so /loop can `tail` it.
    iter_count = len(iterations)
    done_artifacts = sum(1 for r in fixed.values() if r["exists"])
    pending_tasks = sum(1 for r in prd_tasks if r["status"] == "pending")
    print(f"=== Loop status {date} ===")
    print(f"  fixed artefacts: {done_artifacts}/{len(fixed)} present")
    print(f"  iteration reports today: {iter_count}")
    print(f"  other analysis files today: {len(other)}")
    print(f"  local git commits today: {len(git_commits)}")
    print(f"  PRD §8 pending: {pending_tasks} / {len(prd_tasks)}")
    print(f"  suggestion: {suggestion}")
    if not args.no_write:
        print(f"  wrote JSON: {json_path}")
        print(f"  wrote MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
