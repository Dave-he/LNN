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
JETSON_USE_DOCKER="${JETSON_USE_DOCKER:-auto}"
JETSON_DOCKER_IMAGE="${JETSON_DOCKER_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
JETSON_BENCHMARK_ARGS="${JETSON_BENCHMARK_ARGS:---samples 64 --seq-len 16 --hidden-size 8 --epochs 1 --batch-size 8 --inference-repeats 2}"
JETSON_CUDA_ATTEMPTS="${JETSON_CUDA_ATTEMPTS:-2}"

research_args=(--date "$RUN_DATE" --max-results "$MAX_RESULTS" --per-query "$PER_QUERY")
if [[ "$DOWNLOAD_PDFS" == "1" ]]; then
  research_args+=(--download-pdfs)
fi

run_jetson_benchmark() {
  if [[ "$JETSON_USE_DOCKER" != "0" ]] && command -v docker >/dev/null 2>&1 && docker image inspect "$JETSON_DOCKER_IMAGE" >/dev/null 2>&1; then
    run_docker_benchmark() {
      local extra_args="${1:-}"
      docker run --rm --runtime nvidia --gpus all \
        -v "$ROOT_DIR":/workspace/LNN \
        -w /workspace/LNN \
        "$JETSON_DOCKER_IMAGE" \
        bash -lc "python3 scripts/jetson_lnn_benchmark.py --date '$RUN_DATE' $JETSON_BENCHMARK_ARGS $extra_args"
    }

    for ((attempt = 1; attempt <= JETSON_CUDA_ATTEMPTS; attempt++)); do
      if run_docker_benchmark "--no-cpu-fallback"; then
        chown "$(id -u):$(id -g)" analysis/jetson/"${RUN_DATE}"_lnn_benchmark.* 2>/dev/null || true
        return 0
      fi
      echo "[warn] Jetson CUDA benchmark attempt ${attempt}/${JETSON_CUDA_ATTEMPTS} failed." >&2
      sleep 2
    done

    if run_docker_benchmark ""; then
      chown "$(id -u):$(id -g)" analysis/jetson/"${RUN_DATE}"_lnn_benchmark.* 2>/dev/null || true
      return 0
    fi
    echo "[warn] Jetson Docker benchmark failed; retrying with host Python CPU smoke benchmark." >&2
    "$PYTHON_BIN" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" --quick --cpu
  else
    "$PYTHON_BIN" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" --quick
  fi
}

"$PYTHON_BIN" scripts/daily_lnn_research.py "${research_args[@]}"

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  run_jetson_benchmark || true
elif [[ "$RUN_BENCHMARK" == "auto" ]]; then
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qiE 'jetson|nvidia'; then
    run_jetson_benchmark || true
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
