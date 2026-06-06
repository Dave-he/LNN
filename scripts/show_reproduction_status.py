#!/usr/bin/env python3
"""Show the arXiv catalog's reproduction_status distribution (round 72).

Prints a markdown table grouped by status, plus per-paper notes. Use this
before kicking off any new reproduction work to see which papers have
been re-run, which are still vendor-claim-only, and which have known
negative results.

Usage::

    python scripts/show_reproduction_status.py
    python scripts/show_reproduction_status.py --json
    python scripts/show_reproduction_status.py --status failed_to_reproduce
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.arxiv_catalog import REPRO_STATUS_VALUES, build_default_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--status",
        choices=sorted(REPRO_STATUS_VALUES),
        help="Only show papers with this status",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown",
    )
    args = parser.parse_args()

    catalog = build_default_catalog()
    counts = catalog.status_counts()

    if args.json:
        out = {
            "counts": counts,
            "papers": [
                {
                    "arxiv_id": e.arxiv_id,
                    "title": e.expected_title,
                    "year": e.year,
                    "status": e.reproduction_status,
                    "notes": e.notes,
                }
                for e in sorted(catalog.entries.values(), key=lambda x: x.arxiv_id)
                if (not args.status) or (e.reproduction_status == args.status)
            ],
        }
        json.dump(out, sys.stdout, indent=2)
        print()
        return 0

    print("# arXiv catalog — reproduction status")
    print()
    print(f"Total entries: {len(catalog.entries)}")
    print()
    print("| Status | Count |")
    print("|---|---:|")
    for s in sorted(REPRO_STATUS_VALUES):
        print(f"| {s} | {counts.get(s, 0)} |")
    print()
    print("## Known mismatches (linter will flag these as MISMATCH)")
    print()
    for e in sorted(catalog.known_mismatches(), key=lambda x: x.arxiv_id):
        print(f"- **{e.arxiv_id}** — {e.expected_title[:80]}")
        print(f"  - notes: {e.notes}")
    print()
    print("## Papers")
    print()
    by_status: dict[str, list] = defaultdict(list)
    for e in catalog.entries.values():
        by_status[e.reproduction_status].append(e)
    for s in sorted(REPRO_STATUS_VALUES):
        if args.status and s != args.status:
            continue
        entries = sorted(by_status[s], key=lambda x: x.arxiv_id)
        if not entries:
            continue
        print(f"### {s}  ({len(entries)})")
        print()
        for e in entries:
            title = e.expected_title
            if len(title) > 90:
                title = title[:87] + "..."
            year = f" ({e.year})" if e.year else ""
            print(f"- **{e.arxiv_id}**{year} — {title}")
            if e.notes:
                print(f"  - {e.notes}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
