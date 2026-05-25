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
