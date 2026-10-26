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
# Jetson CUDA benchmark 的**候选镜像列表** (按顺序取第一个本地存在的)。
#
# 2026-10-26 修复 (三层):
#
# ① 原来这里写死单个 ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin, 而本机
#    **从来没 pull 过**它 —— `docker image inspect` 失败, 于是每天静默走
#    host Python 分支。**降级是静默的。**
#
# ② lnn-cuda126:local 装上了但**跑不了任何 kernel**: 它的 torch 是 cu126 索引
#    的 dev 夜轮版 2.13.0.dev20260610+cu126, 只编 sm_80/sm_90 且**显式排除
#    8.7**; Orin Nano 是 sm_87 → `no kernel image is available for execution
#    on the device` (而 `torch.cuda.is_available()` 仍然是 True, 所以只看它会被
#    骗)。所以**不能**把它当首选。
#
# ③ (2026-10-26 晚) **镜像根本不是必需的**: 宿主 `/usr/bin/python3.10` 的
#    torch 2.10.0 自带 sm_87 SASS, 实测能跑 kernel (此前一直没被选中,
#    因为解释器选择只测 `import torch`, 排序又把它排在 cu130 的 python3 之后)。
#    走宿主的产物是**真实 GPU 数据**, 不是降级。镜像只是"隔离环境"选项。
#
# ④ (2026-10-26 晚, 再更正) `lnn-sm87:local` **也是坏的, 而且坏得不一样**:
#    实测该镜像里的 torch 是 **2.7.0+cpu** (`Torch not compiled with CUDA
#    enabled`) —— 因为 **PyPI / TUNA 上的 aarch64 torch wheel 本身就是 CPU
#    构建** (文件名 `torch-2.7.0-cp312-cp312-manylinux_2_28_aarch64.whl` 里
#    没有 nvidia-* 依赖, 内嵌版本号带 `+cpu`)。
#    → Dockerfile.lnn-sm87 里"PyPI 官方 aarch64 wheel 默认带 sm_87 SASS,
#      2.7.0 → 默认 cu126"这句话是**错的**, 已在该文件里更正。
#    → 本机唯一被实测证明能跑 sm_87 kernel 的 torch, 是宿主 `~/.local` 下
#      那个 `linux_aarch64` 标签的 wheel (2.10.0, CUDA 12.6) —— 它**不是**
#      PyPI 的 manylinux wheel。
#    所以默认镜像列表里**不再列 lnn-sm87:local**; 想要隔离环境请自己按
#    Dockerfile.lnn-sm87 的说明重建并**用 kernel 预检验收**。
#
# 现场构建 (两步, 第一步已在本机完成):
#   docker build -f analysis/jetson/Dockerfile.lnn-cuda126 -t lnn-cuda126:local .
#   docker build -f analysis/jetson/Dockerfile.lnn-sm87    -t lnn-sm87:local    .
JETSON_DOCKER_IMAGES="${JETSON_DOCKER_IMAGES:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
# 兼容旧变量名 (单镜像)。
JETSON_DOCKER_IMAGE="${JETSON_DOCKER_IMAGE:-}"
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
# Git sync: 本地 systemd (04:30) 是 docs/ papers/ analysis/ 唯一的定时写入方。
# .github/workflows/daily-lnn-research.yml 的 schedule 已移除, 只保留手动触发,
# 因为两边提交相同路径 + 相同 message 会让本地与 origin 分叉, push 必然被拒
# (2026-09-03 lnn-daily-research.service 就是这样 exit 1 的)。
# 这里仍保留 sync/rebase, 是为了兼容手动 workflow_dispatch 或多机器写入。
# 注意: 分叉时会 git reset --hard, 工作区未提交的改动会一起丢失。
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
    echo "[$(date '+%F %T')] [error] git fetch 5 次重试均失败, 中止本次运行" >> "$LOG_FILE"
    echo "[error] git fetch 失败, 中止: 在落后的 HEAD 上提交只会让 push 必然被拒" >&2
    return 1
  fi
  # 判序必须先问"本地是否已包含 origin" —— 两边相同时两个 --is-ancestor 都成立,
  # 先问这一侧才不会把"相同"误判成"落后"。
  if git merge-base --is-ancestor origin/master HEAD 2>/dev/null; then
    echo "[$(date '+%F %T')] [sync] 本地已包含 origin/master (相同或领先), 保留本地 commit" >> "$LOG_FILE"
    return 0
  fi
  if git merge-base --is-ancestor HEAD origin/master 2>/dev/null; then
    echo "[$(date '+%F %T')] [sync] 本地落后 origin/master, 快进" >> "$LOG_FILE"
    git merge --ff-only origin/master
    return 0
  fi
  echo "[$(date '+%F %T')] [sync] 本地与 origin 分叉, 重置到 origin/master (丢弃本地落后 commit, 当日 digest 会重新生成)" >> "$LOG_FILE"
  echo "[warn] 即将 git reset --hard: 本仓库工作区里未提交的改动会一并丢失" >&2
  git reset --hard origin/master
}

# push 被拒的唯一安全解法是先 rebase 到远端再推, 而不是盲目重试同一个 ref。
push_with_rebase() {
  local max_attempts=5
  local attempt=1
  while (( attempt <= max_attempts )); do
    if git push origin HEAD; then
      return 0
    fi
    echo "[$(date '+%F %T')] [warn] push 被拒 ($attempt/$max_attempts), 先 rebase 到 origin/master 再重试" >> "$LOG_FILE"
    if ! git fetch --no-tags origin master; then
      sleep 4
      attempt=$((attempt+1))
      continue
    fi
    if ! git pull --rebase --autostash origin master; then
      echo "[$(date '+%F %T')] [error] rebase 冲突, 需要人工介入" >> "$LOG_FILE"
      git rebase --abort 2>/dev/null || true
      return 1
    fi
    attempt=$((attempt+1))
    sleep 3
  done
  return 1
}

if [[ "$COMMIT_AND_PUSH" == "1" ]]; then
  sync_with_origin || exit 1
fi

research_args=(--date "$RUN_DATE" --max-results "$MAX_RESULTS" --per-query "$PER_QUERY")
if [[ "$DOWNLOAD_PDFS" == "1" ]]; then
  research_args+=(--download-pdfs)
fi

# 候选解释器 (按优先级)。注意: **不能只看 `import torch` 成功**。
bench_python_candidates() {
  printf '%s\n' "$PYTHON_BIN" python3 /usr/bin/python3.10 \
                  /home/hyx/.pyenv/versions/3.14.4/bin/python3 python3.12 python3.11
}

# 判断某个解释器能否**真跑一个 CUDA kernel**。
#
# 2026-10-26 修正: 原来只测 `import torch`, 于是永远选中 python3 (torch
# 2.11.0+cu130)。而本机驱动 540.4.0 最高只支持 CUDA 12.6 → `cuda.is_available()`
# 为 False → 产物天天是 ok_cpu。**"CUDA 可见" 与 "CUDA 可用" 是两回事**:
# import 成功只说明 wheel 装上了, 不说明这个 wheel 编了本机 sm_87 的 SASS。
# 判据必须是真跑一个 kernel。
bench_cuda_kernel_ok() {
  local interp="$1"
  "$interp" -c "import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
x = torch.randn(64, 64, device='cuda')
torch.cuda.synchronize()
(x @ x).sum().item()" >/dev/null 2>&1
}

resolve_bench_python() {
  # The digest only needs stdlib, but the benchmark needs a torch that actually
  # imports. On this Jetson (R36) the /usr/bin/python3.10 cu130 wheel fails with
  # "libcudss.so.0: cannot open shared object file" every time, which is why
  # 2026-10-09..12 produced no benchmark artifact.
  #
  # 优先级: (1) 能跑 CUDA kernel 的解释器 → (2) 能 import torch 的 → (3) 兜底。
  # 本机 /usr/bin/python3.10 的 torch 2.10.0+cu126 带 sm_87 SASS, **能跑 kernel**,
  # 之前一直没被选中, 因为排序把它排在 python3 (cu130) 后面而判据又太松。
  if [[ -n "${PYTHON_BENCH_BIN:-}" ]] && bench_cuda_kernel_ok "$PYTHON_BENCH_BIN"; then
    echo "$PYTHON_BENCH_BIN"
    return 0
  fi
  local candidate
  while read -r candidate; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if bench_cuda_kernel_ok "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done < <(bench_python_candidates)
  while read -r candidate; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "import torch" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done < <(bench_python_candidates)
  echo "$PYTHON_BIN"
}

run_jetson_benchmark() {
  local bench_image=""
  if [[ "$JETSON_USE_DOCKER" != "0" ]] && command -v docker >/dev/null 2>&1; then
    # 单镜像兼容: 显式给了 JETSON_DOCKER_IMAGE 就只试它。
    local -a candidates=()
    if [[ -n "$JETSON_DOCKER_IMAGE" ]]; then
      candidates=("$JETSON_DOCKER_IMAGE")
    else
      # shellcheck disable=SC2206  # 故意按空白分词
      candidates=($JETSON_DOCKER_IMAGES)
    fi
    for candidate in "${candidates[@]}"; do
      if ! docker image inspect "$candidate" >/dev/null 2>&1; then
        echo "[info] Jetson CUDA 镜像本地不存在, 跳过: $candidate" >&2
        continue
      fi
      # **预检: 必须真跑一个 kernel。**
      # `torch.cuda.is_available() == True` 只说明驱动 + runtime 起来了, 不代表
      # 这个 torch build 编了你这张卡的 SASS —— 本机已有两个反例:
      #   - lnn-cuda126:local: is_available() 为 True、设备名 "Orin", 但任何
      #     kernel 都报 `no kernel image is available for execution on the device`
      #     (dev 夜轮 torch 只编 sm_80/sm_90 且显式排除 8.7, Orin 是 sm_87);
      #   - lnn-sm87:local: torch 是 **2.7.0+cpu**, 直接
      #     `AssertionError: Torch not compiled with CUDA enabled`
      #     (PyPI 的 aarch64 wheel 本身就是 CPU 构建)。
      # 预检失败 = 这个镜像**不可用**, 必须**跳过并继续试下一个**, 绝不能
      # 让它带着一份 CPU 产物混进"镜像路径"里 (那会把降级伪装成正常)。
      local preflight_log="$LOG_DIR/${RUN_DATE}_kernel_preflight_$(echo "$candidate" | tr -c 'A-Za-z0-9' '_').err"
      if docker run --rm --runtime nvidia --gpus all "$candidate" \
           bash -lc "python3 -c \"import torch; x=torch.randn(64,64,device='cuda'); torch.cuda.synchronize(); (x@x).sum().item(); print('KERNEL_OK')\"" \
           >/dev/null 2>"$preflight_log"; then
        bench_image="$candidate"
        break
      fi
      echo "[warn] 镜像 $candidate 的 CUDA kernel 预检失败 —— 跳过该镜像 (它的 torch 跑不了本机 sm_87 的 SASS kernel)。" >&2
      echo "[warn] 预检输出 (前 3 行):" >&2
      head -3 "$preflight_log" >&2 || true
    done
    if [[ -z "$bench_image" ]]; then
      # 降级必须**留痕**: 否则读者只会看到产物里的 ok_cpu, 不知道为什么。
      echo "[warn] 本机没有通过 kernel 预检的 Jetson CUDA 镜像 (已试: ${candidates[*]}), 回退到宿主 Python。" >&2
      echo "[info] 宿主上仍有能跑 sm_87 kernel 的解释器时, 产物仍然是真实 GPU 数据 —— 回退不等于降级。" >&2
      echo "[warn] 需要隔离环境的镜像 (可选, 两步):" >&2
      echo "[warn]   docker build -f analysis/jetson/Dockerfile.lnn-cuda126 -t lnn-cuda126:local ." >&2
      echo "[warn]   docker build -f analysis/jetson/Dockerfile.lnn-sm87    -t lnn-sm87:local    ." >&2
    fi
  fi

  if [[ -n "$bench_image" ]]; then
    run_docker_benchmark() {
      local extra_args="${1:-}"
      docker run --rm --runtime nvidia --gpus all \
        -v "$ROOT_DIR":/workspace/LNN \
        -w /workspace/LNN \
        "$bench_image" \
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
    "$(resolve_bench_python)" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" --quick --cpu
  else
    local bench_python
    bench_python="$(resolve_bench_python)"
    echo "[info] benchmark interpreter: $bench_python" >&2
    # 说清楚"为什么是 CPU": 探一下该解释器的 torch 到底能不能**跑 kernel**,
    # 否则日志里只有一句 "CUDA was unavailable", 读者无法区分"本机没 CUDA"
    # 与"选错了解释器 / 装了 cu130 wheel"。
    if bench_cuda_kernel_ok "$bench_python"; then
      echo "[info] 该解释器的 torch 能跑 CUDA kernel → 走 CUDA 路径" >&2
      # **不传 --no-cpu-fallback**: CUDA 在运行时才失败 (最常见是 NvMap 分配失败
      # —— 系统内存压力下 GPU 拿不到连续物理页) 时, 让脚本自己回退到 CPU 并把
      # **失败原因**写进产物; 强行 --no-cpu-fallback 只会让整条流水线崩掉, 当天
      # 连 CPU 数据都没有, 更糟。
      "$bench_python" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" $JETSON_BENCHMARK_ARGS
    else
      echo "[warn] 本机没有能跑 CUDA kernel 的解释器 → 本次产物是 CPU 数据, **不是边缘性能证据**" >&2
      echo "[warn] 宿主 torch 是 cu130 而驱动 540.4.0 最高只支持 CUDA 12.6" >&2
      "$bench_python" scripts/jetson_lnn_benchmark.py --date "$RUN_DATE" --quick
    fi
  fi
}

"$PYTHON_BIN" scripts/daily_lnn_research.py "${research_args[@]}"

# -----------------------------------------------------------------------------
# 负结果台账门禁 (AGENTS.md 约束 #4)
#
# 台账漂移 (grounding 指向已删文件 / 条目与 AGENTS.md 约束表脱节) 会让一条看起来
# 有据、实际已过期的负结果继续生效, 所以这里**大声报**, 不用 `|| true` 吞掉。
# 但它**不阻断**当天的研究产物产出 —— 数据采集不该因为文档索引漂移而停摆。
# -----------------------------------------------------------------------------
LEDGER_STATUS="ok"
if ! "$PYTHON_BIN" scripts/check_negative_results.py; then
  LEDGER_STATUS="FAIL"
  echo "[$(date '+%F %T')] [error] 负结果台账门禁未通过 (详见上方输出); 研究产物照常产出, 但 AGENTS.md 约束 #4 当前不可执行" >> "$LOG_FILE"
  echo "[error] 负结果台账门禁未通过, 请运行 python3 scripts/check_negative_results.py 排查" >&2
fi
echo "negative-results ledger: $LEDGER_STATUS" >> "$LOG_FILE"

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  run_jetson_benchmark || true
elif [[ "$RUN_BENCHMARK" == "auto" ]]; then
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qiE 'jetson|nvidia'; then
    run_jetson_benchmark || true
  fi
fi

if [[ "$COMMIT_AND_PUSH" == "1" ]]; then
  # -----------------------------------------------------------------------------
  # 仓库完整性门禁 (阻断式)
  #
  # 与台账门禁不同, 这里**必须阻断**: 2026-10-17 有 5 个被跟踪的文件带着 merge
  # 冲突标记进了 origin, 潜伏两天没人发现, 直到下游读 JSON 才炸。根因是一次带
  # 冲突的合并被提交 —— 产物一旦损坏, 流水线却仍"照常产出", 所以不能只报不拦。
  # -----------------------------------------------------------------------------
  if ! "$PYTHON_BIN" scripts/check_repo_integrity.py; then
    echo "[$(date '+%F %T')] [error] 仓库完整性门禁未通过 (冲突标记或坏 JSON), 拒绝提交" >> "$LOG_FILE"
    echo "[error] 仓库完整性门禁未通过, 拒绝提交; 详见 python3 scripts/check_repo_integrity.py" >&2
    exit 1
  fi

  git add docs papers analysis
  if ! git diff --cached --quiet; then
    git commit -m "chore(daily): update LNN research digest ${RUN_DATE}"
    if ! push_with_rebase; then
      echo "[$(date '+%F %T')] [error] push 重试 + rebase 均失败, 留待下次 cron 修复" >> "$LOG_FILE"
      exit 1
    fi
  else
    echo "No daily LNN changes to commit."
  fi
fi
# stale content
