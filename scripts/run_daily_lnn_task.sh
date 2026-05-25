#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DATE="${RUN_DATE:-$(date +%F)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_RESULTS="${MAX_RESULTS:-25}"
PER_QUERY="${PER_QUERY:-8}"
COMMIT_AND_PUSH="${COMMIT_AND_PUSH:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-auto}"
DOWNLOAD_PDFS="${DOWNLOAD_PDFS:-0}"

research_args=(--date "$RUN_DATE" --max-results "$MAX_RESULTS" --per-query "$PER_QUERY")
if [[ "$DOWNLOAD_PDFS" == "1" ]]; then
  research_args+=(--download-pdfs)
fi

"$PYTHON_BIN" scripts/daily_lnn_research.py "${research_args[@]}"

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  "$PYTHON_BIN" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" --quick || true
elif [[ "$RUN_BENCHMARK" == "auto" ]]; then
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qiE 'jetson|nvidia'; then
    "$PYTHON_BIN" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" --quick || true
  fi
fi

if [[ "$COMMIT_AND_PUSH" == "1" ]]; then
  git add docs papers analysis
  if ! git diff --cached --quiet; then
    git commit -m "chore(daily): update LNN research digest ${RUN_DATE}"
    git push origin HEAD
  else
    echo "No daily LNN changes to commit."
  fi
fi
