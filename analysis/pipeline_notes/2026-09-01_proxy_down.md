# LNN Pipeline Notes — 2026-09-01 (cron run)

## Critical: GitHub SSH proxy is DOWN

Symptom (every git operation that talks to `git@github.com`):
```
Ncat: Error reading proxy response Status-Line.
Connection closed by UNKNOWN port 65535
```

Root cause: `~/.ssh/config` line 34 routes `github.com` via:
```
ProxyCommand ncat --proxy 192.168.6.25:7890 %h %p
```
The proxy at `192.168.6.25:7890` accepts TCP (`nc -G 4 -zv 192.168.6.25 7890` → succeeds)
but does not speak any proxy protocol correctly (HTTP CONNECT, SOCKS4, SOCKS5 all fail).

Verified fallbacks all fail:
- `ncat --proxy 192.168.6.25:7890 %h %p` (HTTP CONNECT) → "Error reading proxy response Status-Line"
- `ncat --proxy-type socks5 --proxy 192.168.6.25:7890 %h %p` → "wrong SOCKS version in connect response"
- `ncat --proxy-type socks4 --proxy 192.168.6.25:7890 %h %p` → "Proxy connection failed."

Direct connectivity to github.com is fine:
- `ping github.com` → 0% loss
- `nc -zv github.com 22` → succeeded
- `curl -sI https://github.com` → 200 OK (out of band — host network egress works)

**Conclusion**: the proxy is the bottleneck, not github. Likely the user's local
proxy client (Clash / Surge / Shadowsocks / etc.) is down or restarted without
proxy entry for SSH. To restore git push: restart the proxy daemon or temporarily
edit `~/.ssh/config` to remove the ProxyCommand for github.com.

## Today's pipeline outcome

| step | status | detail |
|------|--------|--------|
| 1. digest | OK | `docs/daily/2026-09-01_LNN_research_digest.md` (25/40/16 candidates) |
| 2. select | OK (empty result) | All 12 arXiv papers already covered in `docs/reports/` → candidates=[] |
| 3. report | n/a | no candidates, no new reports needed |
| 4. commit | partial | `d7a9524 chore(daily): LNN digest + 研读报告 2026-09-01` committed locally; **push blocked by proxy** |
| 5. reproduce | failed | matched paper `2606.26849v1` → `experiment_imitation_lnn.py` → `ModuleNotFoundError: No module named 'torch'` because `replicate_paper_dispatch.py:77` hardcodes `python3` (3.11.15 without torch) instead of `.venv312/bin/python` (3.12 with torch 2.2.2) |
| 6. archive | n/a | no new `analysis/replication/` artifacts |

## Recommended fix (for the user, when proxy restored)

1. Edit `scripts/replicate_paper_dispatch.py` line 77 from
   ```python
   cmd = ["python3", target["script"], *target["extra_args"]]
   ```
   to
   ```python
   import shutil
   py = shutil.which("torch_3_12") or shutil.which("python3")
   cmd = [py, target["script"], *target["extra_args"]]
   ```
   Or simpler: add a symlink `~/.local/bin/torch_3_12 -> /Users/hyx/workspace/LNN/.venv312/bin/python`.

2. Once proxy is back: `cd /Users/hyx/workspace/LNN && git push origin HEAD`.

## Files touched today

- `docs/daily/2026-09-01_LNN_research_digest.md` (new, committed)
- `papers/daily/2026-09-01_lnn_research.json` (new, committed)
- `analysis/repo_watchlist/2026-09-01_lnn_open_source_watchlist.md` (new, untracked)
- `logs/pipeline/2026-09-01_pipeline.log` (new)
- `logs/pipeline/2026-09-01_reproduce.log` (new)
- `analysis/pipeline_notes/2026-09-01_proxy_down.md` (this file, new)
