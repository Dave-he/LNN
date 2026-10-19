#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="$SYSTEMD_USER_DIR/lnn-daily-research.service"
TIMER_PATH="$SYSTEMD_USER_DIR/lnn-daily-research.timer"
ENV_PATH="$SYSTEMD_USER_DIR/lnn-daily-research.env"
# 04:30 是现场调试出来的值 (2026-08-20 方案 A), 06:30 会和早上的人工操作撞车.
# 改这个默认值前请先确认没有别的调度已经占了这个点.
ON_CALENDAR="${ON_CALENDAR:-*-*-* 04:30:00}"

mkdir -p "$SYSTEMD_USER_DIR"

# GitHub search API 未认证只有 10 次/分且按出口 IP 计, 8 条查询挤在一分钟内很容易
# 被吃光预算 (2026-10-19 有 3/8 条查询这么静默失败). systemd service 不继承登录
# shell 的环境变量, 所以 token 必须显式落一个 600 的 EnvironmentFile —— 否则
# 无论 shell 里有没有 GITHUB_TOKEN, 定时任务都是匿名跑.
if [[ -f "$ENV_PATH" ]]; then
  echo "Reusing existing env file: $ENV_PATH"
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
  umask 077
  printf 'GITHUB_TOKEN=%s\n' "$GITHUB_TOKEN" >"$ENV_PATH"
  chmod 600 "$ENV_PATH"
  echo "Wrote GITHUB_TOKEN into $ENV_PATH (mode 600)"
else
  echo "No GITHUB_TOKEN in environment; writing env file without a token."
  echo "GitHub queries will run unauthenticated (10 req/min) until you add one:"
  echo "  printf 'GITHUB_TOKEN=ghp_xxx\\n' >> $ENV_PATH && chmod 600 $ENV_PATH"
fi

cat >"$SERVICE_PATH" <<SERVICE
[Unit]
Description=Daily LNN research tracking and Jetson smoke benchmark
Documentation=https://github.com/Dave-he/LNN

[Service]
Type=oneshot
WorkingDirectory=$ROOT_DIR
EnvironmentFile=-$ENV_PATH
Environment=PYTHON_BIN=/usr/bin/python3.10
# The benchmark needs a torch that imports; /usr/bin/python3.10's cu130 wheel
# fails with libcudss.so.0 on this Jetson, so pin a working interpreter here.
# run_daily_lnn_task.sh:resolve_bench_python auto-detects if this one is stale.
Environment=PYTHON_BENCH_BIN=/home/hyx/.pyenv/versions/3.14.4/bin/python3
Environment=RUN_BENCHMARK=auto
Environment=DOWNLOAD_PDFS=0
Environment=COMMIT_AND_PUSH=1
ExecStart=$ROOT_DIR/scripts/run_daily_lnn_task.sh
SERVICE

cat >"$TIMER_PATH" <<TIMER
[Unit]
Description=Run the LNN daily research task

[Timer]
OnCalendar=$ON_CALENDAR
Persistent=true
RandomizedDelaySec=10m

[Install]
WantedBy=timers.target
TIMER

systemctl --user daemon-reload
systemctl --user enable --now lnn-daily-research.timer

echo "Installed user timer: lnn-daily-research.timer"
systemctl --user list-timers lnn-daily-research.timer
