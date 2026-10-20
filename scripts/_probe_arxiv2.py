#!/usr/bin/env python3
"""Probe arXiv API directly with the script's USER_AGENT.

Uses the shared `build_arxiv_query` so the probe exercises the SAME query
syntax the daily pipeline uses (including boolean fragments). The old version
re-wrapped every term as `all:"..."`, which silently produced broken queries
once boolean fragments were introduced (2026-10-20).
"""
import pathlib
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import daily_lnn_research as m

q = m.build_arxiv_query([f for _g, _ts in m.ARXIV_TERM_GROUPS for f in _ts])
params = urllib.parse.urlencode({
    "search_query": q, "start": 0, "max_results": 25,
    "sortBy": "submittedDate", "sortOrder": "descending",
})
url = f"https://export.arxiv.org/api/query?{params}"
req = urllib.request.Request(url, headers={
    "User-Agent": m.USER_AGENT,
    "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.5",
})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")
        print(f"HTTP OK, size={len(body)}")
        print("entries:", body.count("<entry>"))
        for m2 in re.finditer(r"<id>(http[^<]+)</id>\s*<updated>([^<]+)</updated>\s*<title>([^<]+)</title>", body):
            print("  ", m2.group(1), m2.group(2), "::", m2.group(3)[:80])
            break
except Exception as e:
    print("FAIL:", e)
