# LNN 每日论文追踪 Cron 报告 — 2026-06-13

**运行时间**: 2026-06-13T01:01:17Z (UTC)
**工作目录**: /Users/hyx/workspace/LNN
**Git 分支**: master（远端 `origin` = `git@github.com:Dave-he/LNN.git`，默认分支 `master`，**非** `main`）
**今日 (UTC)**: 2026-06-13 = **Saturday** → arXiv 美国东岸周末/节假日窗口生效

## 步骤结果

| 步骤 | 结果 | 备注 |
|---|---|---|
| 1. 切目录 | ✅ | cwd = /Users/hyx/workspace/LNN |
| 2. arXiv API | ✅ | HTTPS, HTTP 200, **232,810 bytes** (100 entries)。**首次请求即 200**，未触发限流退避（与 2026-06-04 / 06-05 / 06-07 / 06-09 / 06-11 几次连撞桶不同，本桶已经冷下来）。 |
| 3. 24h 过滤 | ⚠️ **0 篇** | 最近 1 条匹配论文 = `2606.13571v1` "Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States…"，`published = 2026-06-11T16:59:42Z`，距 now **32.0h**，**严格 24h 窗口外**。0 篇 = 合法结果（pitfall #4 / #17），未放宽到 48h/7d。 |
| 4. 下载 | ⏭️ | 跳过 — 无 24h 内新条目 |
| 5. commit | ⏭️ | 跳过 — 无新增 PDF（按 prompt "若无新论文则跳过提交"） |
| 6. push | ⏭️ | 跳过 — 无 commit |
| 7. 错误处理 | ✅ | 0 篇 ≠ 错误，已按 `lnn-papers-tracker-cron` pitfall #4 / #17 处理 |

## 旁路信号

- 24h 内匹配关键词 `Liquid Neural Networks | CfC | LTC | Neural ODEs` 的 arXiv 论文 = **0**。
- 100 条最近匹配 feed 中最新 10 条的 `age`：
  - `2606.13571v1` (32.0h) — Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States…
  - `2606.12240v1` (57.3h) — Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training ⭐ **LNN 强相关**
  - `2606.11397v1` (77.4h) — Invariant Price of Anarchy and Multiplicative Smoothness
  - `2606.11162v1` (79.3h) — COGENT: Continuous Graph Emulators with Neural ODEs for Long-Term…
  - `2606.10596v1` (88.0h) — Embedding Hybrid Systems into Continuous Latent Vector Fields
  - `2606.09588v1` (106.0h) — Probabilistically Checking Quantum Proofs, with Interaction
  - `2606.08431v1` (141.8h) — Control-Theoretic View of Neural ODEs: Empirical Controllability and Observability
  - `2606.07798v1` (173.6h) — Reconstructing and forecasting disease trajectories of patients with Alzheimer's disease…
  - `2606.07247v2` (179.7h) — Theory of learning of high-dimensional controlled non-linear dynamical systems (I)
  - `2606.06351v1` (200.7h) — Function-Space Priors for Bayesian Neural ODEs with Application to Vessel Trajectory Prediction

- **周末/节假日指纹（pitfall #17）**：今天是 UTC Saturday，arXiv 美国东岸周末窗口内（周五 14:00 → 周一 14:00 暂停新提交），feed 最新一条只到 32h 前，符合预期。**0 篇不是 bug**，与 `2026-06-09` 同款判定。
- 限流指纹（pitfall #10）：本次首发即 200，未出现 HTTP 000 / 429 / `Error in the HTTP2 framing layer` 等指纹。`--http1.1` + `HTTPS` + `-L` + UA 全开，curl 一次成功。

## 决策

- 严格遵守 spec "若无新论文则跳过提交" → 不创建空 commit、不 `git push`。
- 注意到 `2606.12240v1` (57.3h) 仍是高度相关的 LNN 训练加速论文（"Multi-Rate MoE for Accelerating Liquid Neural Network Training"），与 `2026-06-12` 报告点名的同一条；它已连续 2 天卡在 24h 窗口外，**按 pitfall #4 仍不抓取**。
- 同样 `2606.11162v1` (COGENT: Neural ODEs for Long-Term…) 与 `2606.08431v1` (Control-Theoretic View of Neural ODEs) 都是 Neural ODEs 方向相关论文，但均超 24h，不抓。
- 本报告 untracked 留盘（与 `2026-06-04` / `2026-06-07` / `2026-06-09` / `2026-06-11` / `2026-06-12` 同款约定）。

## 工作区状态（无关本次 commit，仅记录）

- Modified: `analysis/replication/temporal_dropout/temporal_dropout_report.md`, `analysis/replication/temporal_dropout/temporal_dropout_results.json`
- Untracked: `analysis/control/2026-06-12_071358_imitation_lnn.{json,md}`, `analysis/control/2026-06-13_064104_imitation_lnn.{json,md}`, `analysis/repo_watchlist/2026-06-13_lnn_open_source_watchlist.md`
- HEAD = `master`
- 这些都是上游 pipeline 产物，本次 cron 流程不动它们。