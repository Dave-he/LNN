#!/usr/bin/env python3
"""Probe arXiv API directly with the script's USER_AGENT."""
import sys, urllib.request, urllib.parse
sys.path.insert(0, "/Users/hyx/workspace/LNN/scripts")
import daily_lnn_research as m

q = " OR ".join(f'all:"{t}"' for t in m.ARXIV_TERMS)
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
        # print first 2 entry ids+title
        import re
        for m2 in re.finditer(r"<id>(http[^<]+)</id>\s*<updated>([^<]+)</updated>\s*<title>([^<]+)</title>", body):
            print("  ", m2.group(1), m2.group(2), "::", m2.group(3)[:80])
            break
except Exception as e:
    print("FAIL:", e)
