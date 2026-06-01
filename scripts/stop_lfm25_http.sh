#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PID_FILE:-$ROOT/logs/lfm25_http.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file found: $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "LFM2.5 HTTP service is not running."
  rm -f "$PID_FILE"
  exit 0
fi

kill "$pid"
for _ in {1..20}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Stopped LFM2.5 HTTP service: pid=$pid"
    exit 0
  fi
  sleep 0.5
done

echo "Process did not exit after SIGTERM, sending SIGKILL: pid=$pid" >&2
kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
