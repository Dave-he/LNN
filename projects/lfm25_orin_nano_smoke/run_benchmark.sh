#!/bin/bash
# Run LFM2.5 benchmark in lnn-jetson-orin container.
# Usage:
#   cd projects/lfm25_orin_nano_smoke
#   ./run_benchmark.sh --quick
#   ./run_benchmark.sh --model LiquidAI/LFM2.5-350M --power

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATE_ARG=""
if [ -n "$1" ]; then
    DATE_ARG="--date $(date -I)"
fi

# Run in container with necessary mounts
docker run --rm --runtime nvidia --gpus all --network host \
    --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/LNN" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -w "/workspace/LNN" \
    lnn-jetson-orin \
    bash -c "pip install transformers accelerate sentencepiece 2>/dev/null || true; cd /workspace/LNN && python3 scripts/lfm25_benchmark.py ${DATE_ARG} $*"
