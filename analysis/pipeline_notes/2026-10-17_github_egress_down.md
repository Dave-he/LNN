# LNN Pipeline 2026-10-17 — GitHub SSH Egress Hard-Down

## 现象
当日 cron 跑到 step 1 push 阶段时, GitHub SSH (22 和 443) 都返回 `Connection closed by 198.18.0.92/152 port 22/443`,
所有 `git fetch` / `git pull` / `git push` 都失败. 编排脚本 5 次重试 (~50s) 全部 `Connection closed by remote host`,
`Connection closed by UNKNOWN port 65535` 间歇出现. GitHub HTTPS 端点同样 `SSL_ERROR_SYSCALL`.
DNS 显示 `github.com -> 198.18.0.92` (TUN 假 IP, 198.18.0.0/15 fake-IP 段, 与 trap 2026-10-09 narrow-7step 一致).

## 与历史 trap 的关系
- **Trap #N+14 (2026-09-10/11/12/13)** — GitHub 侧 push 路径 IP-throttle (post-accept kill at git-receive-pack)
- **2026-10-09 narrow-7step** — TUN/VPN egress outage (SSH 22 + 443 双死)
- **2026-10-10 narrow-7step** — 推积 5 个 commit, 仍是 trap #N+14 复发
- **2026-10-17 (本次)** — 同 2026-10-09 形态 (TUN 假 IP), 但今天 fetch 也失败 (不只是 push), 范围更广

## 本次策略
1. 步骤 1 在本地完成 digest + commit `c27ee54` (digest: 24 papers from arXiv, GitHub/HF 因 SSL_ERROR_SYSCALL 失败但 arXiv 工作)
2. 步骤 2 selector `n_total_arxiv=24, n_skipped_reported=24` — 所有 24 篇分数 ≤ 0, 候选池空 (上游 OK + 全已研读 saturation state)
3. 步骤 3 报告生成跳过 (无候选)
4. 步骤 4 commit (无 staged 变化, 跳过)
5. 步骤 5 reproduce dispatcher auto-skip (digest 无命中规则的论文)
6. 步骤 6 archive 跳过 (无新复现产物)

## 决策
- 不切 HTTPS (cron 明令保持 SSH); 本地 commit 已落, 待 GitHub 推送路径恢复后下次 cron 自然追上
- 不重启任何服务; 不动 AGENTS.md / 已有研读报告 / 已有 reproduce 脚本
- 已记录此文件供下次会话判读

## 异常告警
- ⚠️ **GitHub SSH egress 已硬挂 ≥ 2 天** (2026-10-10 narrow-7step 也是同一周期). 若连续 3 天仍无推送, 提示用户考虑:
  (a) `gh auth login` 重置 token (2026-10-09 note 中 keyring token 已知无效)
  (b) 临时切回 HTTPS 直到 TUN egress 恢复
  (c) SSH over 443 + `ssh.github.com` (曾测试, 仍 post-accept kill)
- ⚠️ **GitHub API/HTTPS 也挂** (SSL_ERROR_SYSCALL) — 比 SSH 22 范围更广, 不只是 push throttling
- 🟢 arXiv 工作正常 (24 篇候选)
- 🟢 本地 commit 链: `c27ee54` → 已在前; 落 `c27ee54` 后 `master` 仍领先 `origin/master` 2 commits (`5b7d19a` 之前还有 4 个未推送)

## 文件
- 本地日志: `logs/pipeline/2026-10-17_reproduce.log` (生成)
- digest: `docs/daily/2026-10-17_LNN_research_digest.md` (已 commit)
- 候选池: `papers/daily/2026-10-17_lnn_research.json` (已 commit)
- watchlist: `analysis/repo_watchlist/2026-10-17_lnn_open_source_watchlist.md` (已 commit)