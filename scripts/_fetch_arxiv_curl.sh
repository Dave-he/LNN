#!/usr/bin/env bash
# Direct arXiv query using curl (which gets 200 even when urllib gets 406).
set -u
UA='LNN-research-tracker/1.0 mailto:hyx@nous.local (https://github.com/Dave-he/LNN)'
ACC='application/atom+xml,application/xml;q=0.9,*/*;q=0.5'
Q='all%3A%22liquid%20neural%22%20OR%20all%3A%22liquid%20time-constant%22%20OR%20all%3A%22closed-form%20continuous-time%22%20OR%20all%3A%22neural%20circuit%20policy%22%20OR%20all%3A%22liquid%20structural%20state-space%22%20OR%20all%3A%22CfC%22%20OR%20all%3A%22LTC%22'
curl -sS -o /tmp/arxiv_2026-09-27.xml -w 'HTTP %{http_code} size=%{size_download}\n' \
  -H "User-Agent: $UA" -H "Accept: $ACC" \
  "https://export.arxiv.org/api/query?search_query=$Q&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"
echo 'entries:' "$(grep -c '<entry>' /tmp/arxiv_2026-09-27.xml || echo 0)"
