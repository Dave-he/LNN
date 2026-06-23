#!/usr/bin/env bash
# Daily arXiv LNN tracker — fetches recent papers and pushes PDFs to git.
# Designed to be invoked from cron (no user interaction).
#
# Pipeline:
#   1. Query arXiv API for LNN / CfC / LTC / Neural-ODEs
#   2. Filter to entries published in the last 24 h (UTC)
#   3. Download up to 5 PDFs into ./papers/daily/
#   4. git add/commit/push the new PDFs
#
# Failure policy: every step is guarded so a network blip or a broken PDF
# download does NOT abort the rest of the run. The script logs everything to
# stdout/stderr — cron captures that.

set -u  # do NOT use -e; we want to continue past non-fatal errors
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DAILY_DIR="papers/daily"
ARXIV_API="http://export.arxiv.org/api/query"
QUERY_TERMS='all:"Liquid Neural Networks" OR all:"CfC" OR all:"LTC" OR all:"Neural ODEs"'
MAX_RESULTS="${MAX_RESULTS:-100}"
TOP_N="${TOP_N:-5}"
SKIP_HOURS="${SKIP_HOURS:-24}"
LOG_PREFIX="[lnn-cron]"

mkdir -p "$DAILY_DIR"

echo "$LOG_PREFIX $(date -u +%FT%TZ)  starting arXiv daily tracker"
echo "$LOG_PREFIX  cwd          = $ROOT_DIR"
echo "$LOG_PREFIX  branch       = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<no-git>')"
echo "$LOG_PREFIX  top_n        = $TOP_N  (window=${SKIP_HOURS}h)"

# -------------------------------------------------------------------------- #
# 1. Fetch the arXiv Atom feed.
# -------------------------------------------------------------------------- #
encoded_query=$(python3 -c "import sys, urllib.parse; sys.stdout.write(urllib.parse.quote(sys.argv[1], safe=':+()\" '))" "$QUERY_TERMS" 2>/dev/null || echo "")
if [[ -z "$encoded_query" ]]; then
  echo "$LOG_PREFIX ERROR: failed to URL-encode query terms" >&2
  exit 1
fi

url="${ARXIV_API}?search_query=${encoded_query}&start=0&max_results=${MAX_RESULTS}&sortBy=submittedDate&sortOrder=descending"
echo "$LOG_PREFIX  fetch        = ${url:0:120}..."

feed_xml="$(mktemp -t lnn_arxiv_XXXXXX.xml)"
trap 'rm -f "$feed_xml"' EXIT

http_code=""
for attempt in 1 2 3; do
  http_code=$(curl -sSL \
    --max-time 60 \
    -A "lnn-cron/1.0 (${USER:-cron})" \
    -o "$feed_xml" \
    -w "%{http_code}" \
    --retry 1 --retry-delay 5 \
    "$url" 2>/dev/null || echo "000")
  if [[ "$http_code" =~ ^2[0-9]{2}$ ]] && [[ -s "$feed_xml" ]] \
     && head -c 200 "$feed_xml" | grep -qi 'feed'; then
    break
  fi
  echo "$LOG_PREFIX  fetch attempt ${attempt} -> HTTP=${http_code} (retrying)" >&2
  sleep $((attempt * 5))
  http_code=""
done

if [[ -z "$http_code" ]] || ! [[ "$http_code" =~ ^2[0-9]{2}$ ]]; then
  echo "$LOG_PREFIX ERROR: arXiv API unreachable (HTTP=${http_code:-none}); skipping today." >&2
  exit 0
fi

# -------------------------------------------------------------------------- #
# 2. Parse + filter to last 24h, take top 5.
# -------------------------------------------------------------------------- #
parse_out="$(python3 - "$feed_xml" "$SKIP_HOURS" "$TOP_N" <<'PY' 2>&1
import sys, os, json, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

feed_path, skip_hours, top_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ns = {"a": "http://www.w3.org/2005/Atom"}
tree = ET.parse(feed_path)
root = tree.getroot()

now = datetime.now(timezone.utc)
cutoff = now - timedelta(hours=skip_hours)

entries = []
for e in root.findall("a:entry", ns):
    pub_s = e.findtext("a:published", default="", namespaces=ns)
    id_s  = e.findtext("a:id", default="", namespaces=ns)
    title_s = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
    title_s = " ".join(title_s.split())
    summary_s = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
    try:
        pub_dt = datetime.fromisoformat(pub_s.replace("Z", "+00:00"))
    except Exception:
        continue
    arxiv_id = id_s.rsplit("/", 1)[-1]
    if arxiv_id.endswith("v") or arxiv_id[-1].isdigit():
        # keep versioned id for filename
        pass
    entries.append({
        "id": arxiv_id,
        "title": title_s,
        "published": pub_dt.isoformat(),
        "summary": summary_s[:500],
    })

recent = [x for x in entries if datetime.fromisoformat(x["published"]) >= cutoff]
picked = recent[:top_n]

print(json.dumps({
    "now_utc": now.isoformat(),
    "cutoff_utc": cutoff.isoformat(),
    "total": len(entries),
    "recent_total": len(recent),
    "picked": picked,
}, ensure_ascii=False))
PY
)"

if [[ -z "$parse_out" ]]; then
  echo "$LOG_PREFIX ERROR: parser produced no output" >&2
  exit 0
fi
echo "$parse_out"

# Quick view via python (avoid jq dependency)
python3 - <<PY "$parse_out"
import json, sys
data = json.loads(sys.argv[1])
print(f"[lnn-cron]  parsed total={data['total']} recent(<24h)={data['recent_total']} picked={len(data['picked'])}")
PY

# Extract picked JSON list to a file the bash side can iterate over safely
picked_json="$(mktemp -t lnn_picked_XXXXXX.json)"
trap 'rm -f "$feed_xml" "$picked_json"' EXIT
python3 - <<PY "$parse_out" "$picked_json"
import json, sys
data = json.loads(sys.argv[1])
with open(sys.argv[2], "w") as f:
    json.dump(data["picked"], f, ensure_ascii=False)
PY

picked_count=$(python3 -c "import json,sys; print(len(json.load(open('$picked_json'))))")
if [[ "$picked_count" == "0" ]]; then
  echo "$LOG_PREFIX  no new papers in the last ${SKIP_HOURS}h — exiting without commit"
  exit 0
fi

# -------------------------------------------------------------------------- #
# 3. Download each PDF (skip if already present).
# -------------------------------------------------------------------------- #
declare -a new_files=()
declare -a new_titles=()

# Read the picked list with python, then iterate safely
python3 - "$picked_json" "$DAILY_DIR" <<'PY' >/tmp/lnn_dl_lines.txt
import json, sys, shlex, os
picked_path, daily_dir = sys.argv[1], sys.argv[2]
items = json.load(open(picked_path))
for it in items:
    aid = it["id"]
    title = it["title"]
    # Sanitize filename: keep arxiv id verbatim, no version-stripping
    pdf_name = f"{aid}.pdf"
    pdf_path = os.path.join(daily_dir, pdf_name)
    pdf_url = f"https://arxiv.org/pdf/{aid}"
    # Each line: <action>\t<local>\t<url>\t<title>
    # action: 'SKIP' or 'FETCH'
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1024:
        action = "SKIP"
    else:
        action = "FETCH"
    print("\t".join([action, pdf_path, pdf_url, title]))
PY

while IFS=$'\t' read -r action pdf_path pdf_url title; do
  [[ -z "$action" ]] && continue
  if [[ "$action" == "SKIP" ]]; then
    echo "$LOG_PREFIX  skip        = $pdf_path (already exists)"
    continue
  fi
  echo "$LOG_PREFIX  download    = $pdf_url"
  if curl -fsSL --max-time 90 -A "lnn-cron/1.0" -o "$pdf_path.tmp" "$pdf_url"; then
    if [[ -s "$pdf_path.tmp" ]] && head -c 4 "$pdf_path.tmp" | grep -q '%PDF'; then
      mv "$pdf_path.tmp" "$pdf_path"
      echo "$LOG_PREFIX  downloaded  = $pdf_path ($(wc -c < "$pdf_path") bytes)"
      new_files+=("$pdf_path")
      new_titles+=("$title")
    else
      echo "$LOG_PREFIX  ERROR       = $pdf_url did not return a PDF; removing partial file" >&2
      rm -f "$pdf_path.tmp"
    fi
  else
    echo "$LOG_PREFIX  ERROR       = curl failed for $pdf_url" >&2
    rm -f "$pdf_path.tmp"
  fi
done < /tmp/lnn_dl_lines.txt
rm -f /tmp/lnn_dl_lines.txt

if [[ ${#new_files[@]} -eq 0 ]]; then
  echo "$LOG_PREFIX  no new PDFs downloaded — exiting without commit"
  exit 0
fi

# -------------------------------------------------------------------------- #
# 4. git commit + push.
# -------------------------------------------------------------------------- #
# Build a commit message with the titles
msg_titles=""
for t in "${new_titles[@]}"; do
  # Strip anything that might break the commit message
  t="${t%%$'\n'*}"
  t="${t//;/,}"
  if [[ -z "$msg_titles" ]]; then
    msg_titles="$t"
  else
    msg_titles="${msg_titles}; ${t}"
  fi
done
today="$(date +%Y-%m-%d)"
commit_msg="Daily LNN papers: ${today} - ${msg_titles}"

# Only stage files we just downloaded
git add -- "${new_files[@]}"

if git diff --cached --quiet; then
  echo "$LOG_PREFIX  no staged changes — skipping commit"
  exit 0
fi

echo "$LOG_PREFIX  committing  = ${#new_files[@]} new PDF(s)"
git commit -m "$commit_msg"

branch="$(git rev-parse --abbrev-ref HEAD)"
echo "$LOG_PREFIX  pushing     = origin $branch"
if ! git push origin "$branch"; then
  echo "$LOG_PREFIX ERROR: git push failed; commit remains local" >&2
  exit 0
fi

echo "$LOG_PREFIX done. papers added: ${#new_files[@]}"