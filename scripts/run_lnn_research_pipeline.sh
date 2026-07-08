#!/usr/bin/env bash
# =============================================================================
# LNN daily research pipeline orchestrator
# -----------------------------------------------------------------------------
#  1. digest  : 聚合 arXiv / GitHub / HF 的 LNN 相关更新 (scripts/daily_lnn_research.py)
#  2. report  : 解析 digest, 对高价值论文生成独立研读报告 (cron prompt 调用 paper-analyzer)
#  3. commit  : 提交 docs/ papers/ 并 git push
#  4. reproduce: 推送成功后, 依据 digest 挑选可复现论文, 跑对应 reproduce 脚本
#
# 该脚本可独立运行 (供本地烟测), 也可被 cron job 触发 (默认在 cron prompt 中
# 跳过 report 步骤, 改由 LLM 用 paper-analyzer 技能执行).
#
# 用法:
#   RUN_DATE=2026-06-01 SKIP_REPORT=0 SKIP_REPRO=0 \
#     bash scripts/run_lnn_research_pipeline.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DATE="${RUN_DATE:-$(date +%F)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_RESULTS="${MAX_RESULTS:-25}"
PER_QUERY="${PER_QUERY:-8}"
SKIP_REPORT="${SKIP_REPORT:-0}"        # 1 = 跳过研读报告阶段 (cron 由 LLM 接管)
SKIP_DIGEST="${SKIP_DIGEST:-0}"        # 1 = 跳过 digest 阶段
SKIP_COMMIT="${SKIP_COMMIT:-0}"        # 1 = 跳过 git commit + push
SKIP_REPRO="${SKIP_REPRO:-0}"          # 1 = 跳过论文复现阶段
DRY_RUN="${DRY_RUN:-0}"                # 1 = 只 print 不真跑
LOG_DIR="$ROOT_DIR/logs/pipeline"
LOG_FILE="$LOG_DIR/${RUN_DATE}_pipeline.log"

# Helper: 把脚本里 log/run 改名为 plog/prun, 避免与 macOS 系统命令 /usr/bin/log 冲突.
plog() {
  local ts; ts="$(date '+%F %T')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE"
}

prun() {
  if [[ "$DRY_RUN" == "1" ]]; then
    plog "[dry-run] $*"
  else
    plog "exec: $*"
    eval "$@"
  fi
}

# Retry 帮助: GitHub SSH 经常出现 "Connection closed by remote host / flush packet" 抖动,
# prun_retry 默认重试 5 次, 每次间隔递增, 用于 git fetch / git pull / git push 等.
prun_retry() {
  local max_attempts="${RETRY_ATTEMPTS:-5}"
  local attempt=1
  local sleep_s=4
  while (( attempt <= max_attempts )); do
    if prun "$@"; then
      return 0
    fi
    plog "  retry $attempt/$max_attempts after ${sleep_s}s ..."
    sleep "$sleep_s"
    attempt=$((attempt+1))
    sleep_s=$((sleep_s+3))
  done
  plog "  [error] command failed after $max_attempts attempts: $*"
  return 1
}

# SSH key 推送: 显式指定私钥 + IdentitiesOnly, 避免 cron / 非交互 shell 找不到 key
# 优先用 id_github_dave-he (本仓库专用 key), 回退到 id_ed25519, 再不行尝试其他常见 key
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
  plog "ssh key: $SSH_KEY"
else
  plog "[warn] 没找到任何可读 SSH key ($SSH_KEY_OVERRIDE / id_github_dave-he / id_ed25519 / github_rsa / id_rsa), git push 可能失败"
fi

mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# 0. 前置: 拉取最新代码, 避免落后 origin
# -----------------------------------------------------------------------------
plog "===== LNN daily pipeline for $RUN_DATE ====="
plog "repo: $ROOT_DIR"
plog "skip: digest=$SKIP_DIGEST report=$SKIP_REPORT commit=$SKIP_COMMIT repro=$SKIP_REPRO"

if [[ "$SKIP_COMMIT" != "1" ]]; then
  if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    prun_retry "git fetch --no-tags origin" || plog "[warn] git fetch 失败, 继续 (后续 pull 用 --rebase 兜底)"
    prun_retry "git pull --ff-only origin HEAD" || prun "git pull --rebase origin HEAD" || plog "[warn] git pull 也失败, 继续 (本地提交可能产生非快进)"
  fi
fi

# -----------------------------------------------------------------------------
# 1. digest: 抓取 LNN 相关更新 -> docs/daily/${DATE}_LNN_research_digest.md
# -----------------------------------------------------------------------------
if [[ "$SKIP_DIGEST" != "1" ]]; then
  plog "[1/4] digest: 抓取 arXiv / GitHub / Hugging Face 更新"
  prun "$PYTHON_BIN scripts/daily_lnn_research.py \
        --date '$RUN_DATE' \
        --max-results '$MAX_RESULTS' \
        --per-query '$PER_QUERY'"
else
  plog "[1/4] digest: SKIPPED"
fi

# -----------------------------------------------------------------------------
# 2. report: 解析 digest, 生成研读报告
#    - 默认 (cron 调用) 时此阶段由 LLM 在 cron prompt 中用 paper-analyzer 技能执行
#    - 本地烟测可用 --llm none 让脚本仅打印待研读清单
# -----------------------------------------------------------------------------
if [[ "$SKIP_REPORT" != "1" ]]; then
  plog "[2/4] report: 列出待研读候选 (cron 由 LLM 接管生成研读报告)"
  prun "$PYTHON_BIN scripts/select_papers_for_report.py \
        --date '$RUN_DATE' \
        --top 3"
else
  plog "[2/4] report: SKIPPED"
fi

# -----------------------------------------------------------------------------
# 3. commit + push: docs/ papers/ analysis/ 的当日变更
# -----------------------------------------------------------------------------
if [[ "$SKIP_COMMIT" != "1" ]]; then
  plog "[3/4] commit: 提交当日 docs/ papers/ 变更"
  prun "git add docs papers"
  if git diff --cached --quiet; then
    plog "  无 staged 变更, 跳过 commit"
  else
    prun "git commit -m 'chore(daily): LNN digest + 研读报告 ${RUN_DATE}'"
    prun_retry "git push origin HEAD" || plog "[warn] git push 失败, 留待后续重试"
  fi
else
  plog "[3/4] commit: SKIPPED"
fi

# -----------------------------------------------------------------------------
# 4. reproduce: 推送成功后, 依据 digest 挑选可复现论文, 跑对应 reproduce 脚本
# -----------------------------------------------------------------------------
if [[ "$SKIP_REPRO" != "1" ]]; then
  plog "[4/4] reproduce: 按 digest 挑选并跑论文复现"
  prun "$PYTHON_BIN scripts/replicate_paper_dispatch.py \
        --date '$RUN_DATE'"
else
  plog "[4/4] reproduce: SKIPPED"
fi

plog "===== pipeline done ====="
