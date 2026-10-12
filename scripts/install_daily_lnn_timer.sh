#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="$SYSTEMD_USER_DIR/lnn-daily-research.service"
TIMER_PATH="$SYSTEMD_USER_DIR/lnn-daily-research.timer"
ON_CALENDAR="${ON_CALENDAR:-*-*-* 06:30:00}"

mkdir -p "$SYSTEMD_USER_DIR"

cat >"$SERVICE_PATH" <<SERVICE
[Unit]
Description=Daily LNN research tracking and Jetson smoke benchmark
Documentation=https://github.com/Dave-he/LNN

[Service]
Type=oneshot
WorkingDirectory=$ROOT_DIR
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
