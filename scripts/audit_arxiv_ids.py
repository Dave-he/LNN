#!/usr/bin/env python3
"""CI-friendly wrapper around lint_arxiv_catalog.

Designed for pre-commit hooks and GitHub Actions: exit code 0 if the
repo is clean, 1 otherwise. Prints a short summary suitable for CI logs.

Usage::

    python scripts/audit_arxiv_ids.py                  # scan default roots
    python scripts/audit_arxiv_ids.py --quiet          # exit-code only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lint_arxiv_catalog import classify, iter_files  # noqa: E402
from lnn.core.arxiv_catalog import build_default_catalog  # noqa: E402

DEFAULT_ROOTS = ["docs", "papers"]


def _is_documented_mismatch(file_text: str, arxiv_id: str) -> bool:
    """Return True if the line mentioning ``arxiv_id`` also contains a
    keyword that signals the author is *explaining* the mismatch, not
    accidentally citing the wrong paper. Skips: MISMATCH, 错配, 误配, 误指,
    无关, NOT, mis-cited, misattributed, misattribution.
    """
    keywords = (
        "MISMATCH", "错配", "误配", "误指", "无关",
        "NOT ", "mis-cited", "misattributed", "misattribution",
    )
    for line in file_text.splitlines():
        if arxiv_id in line and any(k in line for k in keywords):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--quiet", action="store_true", help="Only print on issues"
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Also scan tests/ and lnn/",
    )
    parser.add_argument(
        "--allow-documented-mismatches",
        action="store_true",
        help=(
            "If the line mentioning a known mismatch also contains "
            "'MISMATCH' / '错配' / '误配' / '误指', treat it as an "
            "intentional documentation reference and do not flag it."
        ),
    )
    args = parser.parse_args()

    roots = [ROOT / d for d in DEFAULT_ROOTS]
    if args.include_tests:
        roots.extend([ROOT / "tests", ROOT / "lnn", ROOT / "scripts"])

    files = iter_files(roots)
    catalog = build_default_catalog()

    mismatches: list[tuple[Path, str]] = []
    documented: list[tuple[Path, str]] = []
    unknowns: list[tuple[Path, str]] = []
    ok_count = 0
    seen_ids: set[str] = set()

    for fp in files:
        try:
            text = fp.read_text(errors="ignore")
        except OSError:
            continue
        for aid in catalog.scan_text(text):
            seen_ids.add(aid)
            cls = classify(catalog, aid)
            if cls == "MISMATCH":
                if args.allow_documented_mismatches and _is_documented_mismatch(text, aid):
                    documented.append((fp, aid))
                else:
                    mismatches.append((fp, aid))
            elif cls == "UNKNOWN":
                unknowns.append((fp, aid))
            else:
                ok_count += 1

    if args.quiet and not mismatches:
        return 0

    print(f"[audit-arxiv] scanned {len(files)} files, {len(seen_ids)} unique IDs")
    print(
        f"[audit-arxiv] OK={ok_count}  UNKNOWN={len(unknowns)}  "
        f"MISMATCH={len(mismatches)}  documented_mismatch={len(documented)}"
    )

    if documented:
        print()
        print(
            f"ℹ️  {len(documented)} MISMATCH reference(s) are intentional "
            f"documentation (e.g. the deep-research report explaining the audit). "
            f"Skipped per --allow-documented-mismatches."
        )

    if mismatches:
        print()
        print("❌ Known arXiv ID mismatches found:")
        for fp, aid in mismatches:
            print(f"  - {aid} in {fp.relative_to(ROOT)}")
        print()
        print("Fix these references — the IDs are flagged in lnn/core/arxiv_catalog.py")
        print("as known misattributions. See docs/research/2026-06-06_LNN_2025-2026_deep_research_report.md")
        print("for the deep-research audit trail.")
        return 1

    if not args.quiet:
        print()
        print("✅ No actionable arXiv ID mismatches. (Unknown IDs are not blockers —")
        print("   they will be added to the catalog as they get verified.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
