#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_SERVER="${LLAMA_SERVER:-$ROOT/projects/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
CTX_SIZE="${CTX_SIZE:-2048}"
THREADS="${THREADS:-4}"
PARALLEL="${PARALLEL:-1}"
GPU_LAYERS="${GPU_LAYERS:-0}"
LOG_FILE="${LOG_FILE:-$ROOT/logs/lfm25_http.log}"
PID_FILE="${PID_FILE:-$ROOT/logs/lfm25_http.pid}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model file not found: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "LFM2.5 HTTP service is already running: pid=$existing_pid"
    echo "URL: http://$HOST:$PORT"
    exit 0
  fi
fi

cmd=(
  "$LLAMA_SERVER"
  -m "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  -c "$CTX_SIZE"
  -t "$THREADS"
  --parallel "$PARALLEL"
  --n-gpu-layers "$GPU_LAYERS"
  --fit off
  --no-webui
)

setsid "${cmd[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "Started LFM2.5 HTTP service: pid=$pid"
echo "URL: http://$HOST:$PORT"
echo "Model: $MODEL_PATH"
echo "Log: $LOG_FILE"
