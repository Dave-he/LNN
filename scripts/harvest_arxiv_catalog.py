#!/usr/bin/env python3
"""Harvest arXiv metadata from the daily-research JSONs into the local
catalog (round 72 extension).

The lint script (``scripts/lint_arxiv_catalog.py``) flags every arXiv ID
not in the local seed catalog as ``UNKNOWN``. This script closes that gap
by reading the daily-research JSON files (``papers/daily/*.json``),
which already contain the title, authors, year, and categories for every
paper fetched that day, and adding them as ``CatalogEntry`` stubs.

Each harvested entry is marked ``vendor_claim`` — the daily JSON is
itself a vendor/aggregator source, not a third-party reproduction. The
catalog stays honest about provenance.

Usage::

    python scripts/harvest_arxiv_catalog.py                         # latest daily JSON
    python scripts/harvest_arxiv_catalog.py papers/daily/2026-06-06_lnn_research.json
    python scripts/harvest_arxiv_catalog.py papers/daily/*.json    # all daily JSONs
    python scripts/harvest_arxiv_catalog.py --write                # also dump the catalog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.arxiv_catalog import (  # noqa: E402
    REPRO_STATUS_VENDOR,
    build_default_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "json_paths",
        nargs="*",
        type=Path,
        help="Daily-research JSON files to harvest. Default: papers/daily/*.json (newest first)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write the harvested catalog to "
            "analysis/arxiv_catalog/harvested_<date>.json for future CI use"
        ),
    )
    parser.add_argument(
        "--status",
        default=REPRO_STATUS_VENDOR,
        help="reproduction_status to assign to harvested entries (default: vendor_claim)",
    )
    args = parser.parse_args()

    paths = args.json_paths
    if not paths:
        # Default: all daily JSONs, newest first.
        daily_dir = ROOT / "papers" / "daily"
        paths = sorted(daily_dir.glob("*_lnn_research.json"), reverse=True)
    if not paths:
        print("No daily JSONs found.", file=sys.stderr)
        return 1

    cat = build_default_catalog()
    total_new = 0
    for p in paths:
        new = cat.harvest_from_daily_json(p, default_status=args.status)
        if new:
            print(f"[harvest] {p.name}: +{len(new)} new entries")
            total_new += len(new)
        else:
            print(f"[harvest] {p.name}: 0 new (all already in catalog)")

    counts = cat.status_counts()
    print()
    print(f"[harvest] done: +{total_new} new entries")
    print(f"[harvest] catalog now has {len(cat.entries)} entries total")
    print(
        f"[harvest] status: vendor_claim={counts[REPRO_STATUS_VENDOR]} "
        f"reproduced={counts['third_party_reproduced']} "
        f"failed={counts['failed_to_reproduce']} "
        f"untested={counts['untested']}"
    )

    if args.write:
        out_dir = ROOT / "analysis" / "arxiv_catalog"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Filename: harvested_YYYY-MM-DD_HHMMSS.json
        from datetime import datetime

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = out_dir / f"harvested_{stamp}.json"
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": [p.name for p in paths],
                    "counts": counts,
                    "entries": {aid: e.to_dict() for aid, e in cat.entries.items()},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"[harvest] wrote {out_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
