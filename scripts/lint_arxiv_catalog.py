#!/usr/bin/env python3
"""Lint the LNN repo for arXiv-ID hygiene issues (round 72).

Scans the repo's docs/ and papers/ directories for arXiv IDs and checks
each one against the canonical catalog in ``lnn.core.arxiv_catalog``.

Three classes of issue are reported:

* MISMATCH  — the ID is in the catalog and is flagged as a known
  misattribution (e.g. 2003.06236 mis-cited as CfC).
* UNKNOWN   — the ID is not in the catalog at all. Not necessarily wrong,
  but worth a human look.
* OK        — the ID is in the catalog and is correctly attributed.

Exit code is 0 in dry-run mode and 1 if any MISMATCH is found (so the
script can gate a CI step).

Usage::

    python scripts/lint_arxiv_catalog.py                  # lint the whole repo
    python scripts/lint_arxiv_catalog.py --dry-run        # exit 0 even on mismatch
    python scripts/lint_arxiv_catalog.py docs/            # scan a single subtree
    python scripts/lint_arxiv_catalog.py --include-tests  # also scan tests/ and lnn/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make the lnn package importable when run from any cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.arxiv_catalog import build_default_catalog  # noqa: E402

DEFAULT_INCLUDE_DIRS = ["docs", "papers"]
EXTENSIONS = {".md", ".json", ".txt", ".rst", ".py"}


def _maybe_harvest(catalog, args) -> int:
    """If --harvest is set, harvest from daily JSONs first.

    Returns the number of newly added entries. ``args.harvest is None``
    means the flag was not passed; ``args.harvest == []`` means it was
    passed without arguments (harvest from all daily JSONs).
    """
    if args.harvest is None:
        return 0
    daily_dir = ROOT / "papers" / "daily"
    if not daily_dir.exists():
        return 0
    if args.harvest == []:
        paths = sorted(daily_dir.glob("*_lnn_research.json"))
    else:
        paths = [Path(p) for p in args.harvest]
    total = 0
    for p in paths:
        new = catalog.harvest_from_daily_json(p)
        total += len(new)
    return total


def iter_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                out.append(p)
    return out


def classify(catalog, arxiv_id: str) -> str:
    res = catalog.verify_id(arxiv_id, offline=True)
    if res.is_mismatch:
        return "MISMATCH"
    if res.expected_title is None:
        return "UNKNOWN"
    return "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Subtrees to scan (default: docs/ and papers/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Always exit 0; print the report only",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Also scan tests/ and lnn/ subtrees",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a markdown table",
    )
    parser.add_argument(
        "--harvest",
        nargs="*",
        metavar="DAILY_JSON",
        help=(
            "Harvest arXiv metadata from daily JSONs into the catalog "
            "before linting. Pass specific file paths, or no arguments "
            "to harvest from all papers/daily/*_lnn_research.json."
        ),
    )
    args = parser.parse_args()

    roots = args.paths or [ROOT / d for d in DEFAULT_INCLUDE_DIRS]
    if args.include_tests:
        roots.extend([ROOT / "tests", ROOT / "lnn", ROOT / "scripts"])

    files = iter_files(roots)
    catalog = build_default_catalog()
    n_harvested = _maybe_harvest(catalog, args)
    if n_harvested:
        print(
            f"[lint] harvested {n_harvested} entries from daily JSONs into the catalog"
        )

    # (file, arxiv_id, classification, expected_title)
    rows: list[tuple[Path, str, str, str | None]] = []
    by_status: dict[str, list[tuple[Path, str, str, str | None]]] = defaultdict(list)

    for fp in files:
        try:
            text = fp.read_text(errors="ignore")
        except OSError:
            continue
        for aid in catalog.scan_text(text):
            cls = classify(catalog, aid)
            entry = catalog.get(aid)
            expected = entry.expected_title if entry else None
            row = (fp, aid, cls, expected)
            rows.append(row)
            by_status[cls].append(row)

    if args.json:
        report = {
            "scanned_files": len(files),
            "total_id_occurrences": len(rows),
            "unique_ids": len({r[1] for r in rows}),
            "by_status": {k: len(v) for k, v in sorted(by_status.items())},
            "rows": [
                {
                    "file": str(p.relative_to(ROOT)),
                    "arxiv_id": aid,
                    "status": cls,
                    "expected_title": expected,
                }
                for (p, aid, cls, expected) in rows
            ],
        }
        json.dump(report, sys.stdout, indent=2)
        print()
        return 0 if args.dry_run or not by_status["MISMATCH"] else 1

    # Markdown table output
    print(f"# arXiv catalog lint — {len(files)} files scanned")
    print()
    print(f"Total id occurrences: {len(rows)}  ({len({r[1] for r in rows})} unique)")
    print()
    for status in ("MISMATCH", "UNKNOWN", "OK"):
        n = len(by_status[status])
        print(f"## {status}  ({n})")
        if n == 0:
            print()
            continue
        print()
        print("| File | arXiv ID | Expected title |")
        print("|---|---|---|")
        for fp, aid, _, expected in by_status[status][:50]:
            rel = fp.relative_to(ROOT)
            if status == "MISMATCH":
                # Highlight: print the FIRST expected title with [MISMATCH] stripped.
                title = (expected or "").split("[MISMATCH")[0].strip() + "  ⚠ KNOWN MISMATCH"
            else:
                title = expected or "(not in catalog)"
            # Truncate long titles for readability.
            if len(title) > 80:
                title = title[:77] + "..."
            print(f"| {rel} | {aid} | {title} |")
        if n > 50:
            print(f"| ... | ({n - 50} more) | |")
        print()

    n_mismatch = len(by_status["MISMATCH"])
    if args.dry_run:
        return 0
    return 1 if n_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
