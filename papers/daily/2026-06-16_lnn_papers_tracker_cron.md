# LNN 每日论文追踪 Cron 报告 — 2026-06-16

**运行时间**: 2026-06-16T01:12:04Z (UTC)
**工作目录**: /Users/hyx/workspace/LNN
**Git 分支**: master（远端 `origin` = `git@github.com:Dave-he/LNN.git`，默认分支 `master`，**非** `main`）
**今日 (UTC)**: 2026-06-16 = **Tuesday**

## 步骤结果

| 步骤 | 结果 | 备注 |
|---|---|---|
| 1. 切目录 | ✅ | cwd = /Users/hyx/workspace/LNN |
| 2. arXiv API | ✅ | HTTPS, HTTP 200, **232,432 bytes** (100 entries). 首发 `urllib` 连撞 `HTTPError 503` / `RemoteDisconnected` × 5（边缘 bucket 限流，curl HTTP2 framing 也有同样指纹），改 UA + 6 次退避（20s/35s/50s/65s/80s/95s）后第二次 UA 在 attempt #1 即 200。换用 UA `LNN-research-tracker/1.0`（与 `scripts/daily_lnn_research.py` 同款）即可首发 200 — 与 `2026-06-13` 06-14 指纹一致。 |
| 3. 24h 过滤 | ⚠️ **0 篇** | 最近 1 条匹配论文 = `2606.15807v1` "Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liqu…"，`published = 2026-06-14T13:19:45Z`，距 now **35.9h**，**严格 24h 窗口外**。0 篇 = 合法结果（pitfall #4 / #17），未放宽到 48h/7d。 |
| 4. 下载 | ⏭️ | 跳过 — 无 24h 内新条目 |
| 5. commit | ⏭️ | 跳过 — 无新增 PDF（按 prompt "若无新论文则跳过提交"） |
| 6. push | ⏭️ | 跳过 — 无 commit |
| 7. 错误处理 | ✅ | 0 篇 ≠ 错误；首发 HTTP 503/connection drop 已按既定退避重试恢复，已按 `lnn-papers-tracker-cron` pitfall #4 / #10 / #17 处理 |

## 旁路信号

- 24h 内匹配关键词 `Liquid Neural Networks | CfC | LTC | Neural ODEs` 的 arXiv 论文 = **0**。
- 100 条最近匹配 feed 中最新 10 条的 `age`（与 `2026-06-14` 报告对比，feed 整体只滚动了 1 条新结果 `2606.15807v1`，arXiv 6 月 14–15 日 UTC 周末窗口索引延迟明显）：
  - `2606.15807v1` (35.9h) — Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liqu… ⭐ **LNN 直接命中**
  - `2606.15469v1` (52.3h) — Learning Context-Aware Neural ODE Dynamics for Adaptive Robotic Control
  - `2606.14313v1` (87.4h) — Nonlocal Bayesian Modeling of Continuous Spatio-Temporal Dynamics
  - `2606.14136v1` (91.4h) — Environment-Aware Stable Neural Koopman Dynamics Learning for Input-Driven Syste…
  - `2606.13571v1` (104.2h) — Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States…
  - `2606.12240v1` (129.3h) — Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training ⭐ **LNN 强相关**
  - `2606.11397v1` (149.6h) — Invariant Price of Anarchy and Multiplicative Smoothness
  - `2606.11162v1` (151.5h) — COGENT: Continuous Graph Emulators with Neural Ordinary Differential Equations…
  - `2606.10596v1` (160.2h) — Embedding Hybrid Systems into Continuous Latent Vector Fields
  - `2606.09588v1` (178.2h) — Probabilistically Checking Quantum Proofs, with Interaction
- **周末/节假日指纹（pitfall #17）**：本次跑点在 UTC Tuesday，按理不是周末；但 arXiv 6 月 14–15 日周末窗口新提交极慢，feed 头部条目整体卡在 35–60h 外，**0 篇不是 bug**，与 `2026-06-12` / `06-13` / `06-14` 三轮同款判定。
- 限流指纹（pitfall #10）：首发 `urllib` HTTPS 受 503 + `RemoteDisconnected` 双指纹影响，与 2026-06-12 跑点完全同款；切换 UA + 完整退避后 200。与 06-04 / 06-05 / 06-07 / 06-09 / 06-11 等连撞桶的轮次可互证"边缘 bucket 限流指纹"。

## 决策

- 严格遵守 spec "若无新论文则跳过提交" → 不创建空 commit、不 `git push`。
- 注意到 `2606.15807v1` (35.9h) 是新出现且直接命中 LNN（"Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liqu…"），**严格 24h 窗口外**，按 pitfall #4 仍不抓取，仅在报告里点名。
- `2606.12240v1` (Multi-Rate MoE for Accelerating LNN Training) 已连续 4 天卡在 24h 窗口外，**按 pitfall #4 仍不抓取**。
- 本报告 untracked 留盘（与 `2026-06-04` / `06-07` / `06-09` / `06-11` / `06-12` / `06-13` / `06-14` 同款约定）。

## 工作区状态（无关本次 commit，仅记录）

- Modified: `analysis/replication/temporal_dropout/temporal_dropout_report.md`, `analysis/replication/temporal_dropout/temporal_dropout_results.json`
- Untracked: `lnn/core/learned_beta_ps_ln_khlfft_stablessm_cfc.py`, `scripts/bench_learned_beta_ps_ln_khlfft_stablessm_cfc.py`, `tests/test_learned_beta_ps_ln_khlfft_stablessm_cfc.py`, `analysis/control/2026-06-15_*.json`, `analysis/control/2026-06-15_*.md`, 以及上游研究 pipeline 产物若干。
- HEAD = `master` @ `fd9adee round 208: stable diagonal SSM (sigmoid-A) on CfC (NEG, stability OK but regresses)`（master 较 2026-06-16 daily digest 提交 `a30db74` 又前进了 5 轮 CfC 实验）
- 这些都是上游 pipeline 产物，本次 cron 流程不动它们。
