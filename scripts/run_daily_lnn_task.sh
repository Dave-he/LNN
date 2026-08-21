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
LOG_DIR="$ROOT_DIR/logs/pipeline"
LOG_FILE="$LOG_DIR/${RUN_DATE}_daily_task.log"
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# SSH key 推送: 显式指定私钥 + IdentitiesOnly, 避免 cron / 非交互 shell 找不到 key
# 优先用 id_github_dave-he (本仓库专用 key), 回退到 id_ed25519, 再不行尝试其他常见 key
# -----------------------------------------------------------------------------
SSH_KEY_OVERRIDE="${SSH_KEY_OVERRIDE:-}"
SSH_KEY=""
for _candidate in "$SSH_KEY_OVERRIDE" "$HOME/.ssh/id_github_dave-he" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/github_rsa" "$HOME/.ssh/id_rsa"; do
  if [[ -n "$_candidate" && -r "$_candidate" ]]; then
    SSH_KEY="$_candidate"
    break
  fi
done
if [[ -n "$SSH_KEY" ]]; then
  export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ProxyCommand=none"
  echo "[$(date '+%F %T')] ssh key: $SSH_KEY" >> "$LOG_FILE"
fi

# -----------------------------------------------------------------------------
# Git sync: 拉取 origin 并在分叉时重置到 origin/master,
# 避免 systemd 04:30 推送与 GH Action 06:30 (前日 22:30 UTC) 补推产生 "Updates were rejected".
# 根因: 历史上本地从未 pull, 累积落后后下一次 commit 即被 reject.
# -----------------------------------------------------------------------------
git_retry() {
  local max_attempts=5
  local attempt=1
  local sleep_s=4
  while (( attempt <= max_attempts )); do
    if "$@"; then
      return 0
    fi
    echo "[$(date '+%F %T')] [warn] $1 失败, 重试 $attempt/$max_attempts (sleep ${sleep_s}s)" >> "$LOG_FILE"
    sleep "$sleep_s"
    attempt=$((attempt+1))
    sleep_s=$((sleep_s+3))
  done
  return 1
}

sync_with_origin() {
  echo "[$(date '+%F %T')] [sync] 拉取 origin/master" >> "$LOG_FILE"
  if ! git_retry git fetch --no-tags origin master; then
    echo "[$(date '+%F %T')] [warn] git fetch 5 次重试均失败, 跳过 sync (push 可能被拒)" >> "$LOG_FILE"
    return 0
  fi
  if git merge-base --is-ancestor HEAD origin/master 2>/dev/null; then
    echo "[$(date '+%F %T')] [sync] 本地已是 origin/master 后裔, 无需 reset" >> "$LOG_FILE"
    return 0
  fi
  if git merge-base --is-ancestor origin/master HEAD 2>/dev/null; then
    echo "[$(date '+%F %T')] [sync] 本地领先 origin (fast-forward), 保留本地 commit" >> "$LOG_FILE"
    return 0
  fi
  echo "[$(date '+%F %T')] [sync] 本地与 origin 分叉, 重置到 origin/master (丢弃本地落后 commit, 当日 digest 会重新生成)" >> "$LOG_FILE"
  git reset --hard origin/master
}

if [[ "$COMMIT_AND_PUSH" == "1" ]]; then
  sync_with_origin
fi

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
    if ! git_retry git push origin HEAD; then
      echo "[$(date '+%F %T')] [error] git push 5 次重试均失败, 留待下次 cron 修复" >> "$LOG_FILE"
      exit 1
    fi
  else
    echo "No daily LNN changes to commit."
  fi
fi
# stale content
