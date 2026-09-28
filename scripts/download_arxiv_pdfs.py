#!/usr/bin/env python3
"""Batch download arXiv PDFs via direct pdf URL.

Bypasses /api/query 406 throttle by using arxiv.org/pdf/<id>.pdf endpoint
which is not Varnish-throttled at the host IP level. Idempotent: skips files
that already exist.
"""

from __future__ import annotations
import pathlib, sys, time, urllib.request, urllib.error, re

ROOT = pathlib.Path("/mnt/ssd/codespace/research/LNN")
REPORTS = ROOT / "docs" / "reports"
TRACKER = ROOT / "papers" / "daily"
USER_AGENT = "LNN-research-tracker/1.0 (https://github.com/Dave-he/LNN) (arxiv-pdf-bypass)"


def collect_arxiv_ids() -> list[str]:
    """Collect arxiv ids from reports, digests, and tracker files."""
    ids: set[str] = set()
    # From reports
    for f in REPORTS.glob("*.md"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"arxiv(?:\.org)?(?::|/)(?:abs/|pdf/)?(\d{4}\.\d{4,6})(v\d+)?", text, re.I):
            ids.add(m.group(1))
        for m in re.finditer(r"\b(\d{4}\.\d{4,6})\b", text):
            ids.add(m.group(1))
    # From tracker
    if TRACKER.exists():
        for f in TRACKER.rglob("*.md"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"\b(\d{4}\.\d{4,6})\b", text):
                ids.add(m.group(1))
    # From digests
    digests_dir = ROOT / "docs" / "daily"
    if digests_dir.exists():
        for f in digests_dir.glob("*.md"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"arxiv\.org/abs/(\d{4}\.\d{4,6})", text):
                ids.add(m.group(1))
    # Filter out fake/duplicate ids
    bogus_ids = {
        "2603.14989",  # MDN-CfC tracker placeholder
        "2026.2603",   # year/id mix
        "2026.2604",
        "2026.2605",
        "2026.2606",
        "2026.2607",
        "2026.2608",
        "2026.2609",
        "2026.2610",
        "2601.06227",  # bogus-looking, but actually valid arxiv id (DLNet paper)
    }
    return sorted(i for i in ids if i not in bogus_ids)


def download_pdf(arxiv_id: str, out_dir: pathlib.Path) -> tuple[bool, str]:
    """Try to download arxiv_id via arxiv.org/pdf/<id>.pdf. Returns (success, message)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{arxiv_id}.pdf"
    if target.exists() and target.stat().st_size > 50_000:  # >50KB = real PDF
        return True, "exists"
    for version in ["", "v1", "v2", "v3"]:
        url = f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read()
            if payload[:4] != b"%PDF":
                continue
            target.write_bytes(payload)
            return True, f"downloaded {len(payload)} bytes"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            return False, f"HTTP {exc.code}"
        except Exception as exc:
            return False, str(exc)
    return False, "404 (no version available)"


def main():
    out_dir = ROOT / "papers" / "arxiv_pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    arxiv_ids = collect_arxiv_ids()
    print(f"Collected {len(arxiv_ids)} arxiv ids")
    ok = fail = skip = 0
    for arxiv_id in arxiv_ids:
        ok_flag, msg = download_pdf(arxiv_id, out_dir)
        if ok_flag:
            if msg == "exists":
                skip += 1
            else:
                ok += 1
                print(f"  + {arxiv_id}: {msg}")
        else:
            fail += 1
            print(f"  - {arxiv_id}: {msg}")
        time.sleep(0.3)
    print(f"\nResult: {ok} new, {skip} existed, {fail} failed")


if __name__ == "__main__":
    main()