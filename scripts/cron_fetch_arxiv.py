#!/usr/bin/env python3
"""Fetch recent LNN-related papers from arXiv and trigger PDF download + git commit.

Cron job entry point. Executed daily by the LNN project's automated pipeline.
"""
from __future__ import annotations

import os
import re
import sys
import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path("/Users/hyx/workspace/LNN")
PAPERS_DIR = REPO_ROOT / "papers" / "daily"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_API = "https://export.arxiv.org/api/query"  # use HTTPS directly to avoid 301→rate-limit cycle
KEYWORDS = [
    "Liquid Neural Networks",
    "CfC",
    "LTC",
    "Neural ODEs",
]


def build_query() -> str:
    # arXiv API requires URL-encoded quotes (%22) inside the search_query value.
    # Using raw " yields HTTP 400 from export.arxiv.org.
    parts = [f'all:%22{urllib.parse.quote(kw)}%22' for kw in KEYWORDS]
    search_query = "+OR+".join(parts)
    return (
        f"{ARXIV_API}?search_query={search_query}"
        f"&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending"
    )


def fetch_arxiv(url: str) -> bytes:
    """Fetch arXiv API XML with retry/backoff for transient 429/5xx/network errors."""
    import time as _time
    last_exc: Exception | None = None
    delays = [10, 30, 90]  # total wait budget ≈ 130s; cron timeout = 300s
    for attempt, delay in enumerate([0] + delays):
        if delay:
            print(f"  ... waiting {delay}s before retry {attempt}", flush=True)
            _time.sleep(delay)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "LNN-Daily-Tracker/1.0 (mailto:tracker@example.com)",
                    "Accept": "application/atom+xml",
                },
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # 429 (rate limit) and 5xx are retriable; 4xx (except 429) are not
            if exc.code in (429, 500, 502, 503, 504) and attempt < len(delays):
                print(f"  ! arXiv HTTP {exc.code}, will retry", flush=True)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            print(f"  ! arXiv network error: {exc}", flush=True)
            if attempt < len(delays):
                continue
            raise
    raise last_exc  # type: ignore[misc]


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def parse_entries(xml_bytes: bytes, now: datetime) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    cutoff = now - timedelta(hours=24)
    entries: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id_raw = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        m = re.search(r"abs/([^v]+)(v\d+)?$", arxiv_id_raw)
        arxiv_id = m.group(1) if m else arxiv_id_raw
        title = " ".join(entry.find("atom:title", ATOM_NS).text.split()) if entry.find("atom:title", ATOM_NS) is not None else ""
        published_raw = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        if not published_raw:
            continue
        try:
            published = datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if published >= cutoff:
            entries.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "published": published.isoformat(),
            })
    return entries[:5]


def download_pdf(arxiv_id: str) -> Path | None:
    target = PAPERS_DIR / f"{arxiv_id}.pdf"
    if target.exists() and target.stat().st_size > 1024:
        return None  # already downloaded
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 LNN-Tracker"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            print(f"  ! {arxiv_id}: response is not a PDF (first 4 bytes: {data[:4]!r})", flush=True)
            return None
        target.write_bytes(data)
        print(f"  + {arxiv_id}.pdf ({len(data)} bytes)", flush=True)
        return target
    except Exception as exc:
        print(f"  ! {arxiv_id}: download failed: {exc}", flush=True)
        return None


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! cmd failed ({proc.returncode}): {' '.join(cmd)}\n    stderr: {proc.stderr.strip()[:500]}", flush=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> int:
    now = datetime.now(timezone.utc)
    print(f"[{now.isoformat()}] LNN daily tracker start", flush=True)

    try:
        xml_bytes = fetch_arxiv(build_query())
    except Exception as exc:
        print(f"arXiv query failed: {exc}", flush=True)
        return 1

    try:
        entries = parse_entries(xml_bytes, now)
    except Exception as exc:
        print(f"arXiv XML parse failed: {exc}", flush=True)
        return 2

    print(f"Found {len(entries)} entries within last 24h", flush=True)
    for e in entries:
        print(f"  - {e['arxiv_id']} ({e['published']}) {e['title']}", flush=True)

    downloaded: list[str] = []
    titles: list[str] = []
    for e in entries:
        path = download_pdf(e["arxiv_id"])
        if path is not None:
            downloaded.append(str(path))
            titles.append(e["title"])

    if not downloaded:
        print("No new PDFs to commit. Done.", flush=True)
        return 0

    # Add only the new PDFs
    for p in downloaded:
        rel = os.path.relpath(p, REPO_ROOT)
        rc, _ = run(["git", "add", rel])
        if rc != 0:
            print(f"  ! git add failed for {rel}", flush=True)

    title_blob = "; ".join(titles)
    commit_msg = f"Daily LNN papers: {now.strftime('%Y-%m-%d')} - {title_blob}"
    rc, out = run(["git", "commit", "-m", commit_msg])
    if rc != 0:
        print(f"  ! git commit failed: {out[:500]}", flush=True)
        return 3

    # Detect current branch
    rc, branch_out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    branch = branch_out.strip() if rc == 0 else "master"
    print(f"Pushing to origin/{branch}", flush=True)
    rc, out = run(["git", "push", "origin", branch])
    if rc != 0:
        print(f"  ! git push failed: {out[:500]}", flush=True)
        return 4

    print("Daily LNN tracker finished OK.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
