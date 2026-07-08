#!/usr/bin/env python3
"""Debug helper: score + reported flag for every paper in today's digest."""
import sys
sys.path.insert(0, '/Users/hyx/workspace/LNN/scripts')
from select_papers_for_report import parse_digest, score_paper, already_reported

d = parse_digest('2026-07-09')
print(f'# arxiv entries parsed: {len(d["arxiv"])}')
for p in d['arxiv']:
    s = score_paper(p)
    rep = already_reported(p['id'])
    skip = 'skip' if (rep or s <= 0) else 'CANDIDATE'
    print(f'  [{skip:8s}] score={s:2d} reported={rep}  id={p["id"]}  date={p["date"]}  title={p["title"][:80]!r}')
