# 每日 LNN 论文追踪 — 2026-06-20

## 运行结果

**今日 (UTC 2026-06-20) 候选论文已由 `scripts/daily_lnn_research.py` 抓取并落入 `docs/daily/2026-06-20_LNN_research_digest.md`, 共 25 篇 arXiv 候选 / 41 仓库 / 20 HF 模型。**

## 执行明细

1. **工作目录**：`/Users/hyx/workspace/LNN` (分支: `master`，远程: `origin` -> `git@github.com:Dave-he/LNN.git`).
2. **数据源**：`scripts/daily_lnn_research.py` 第一次跑时 arXiv 抓取超时（`The read operation timed out`），GitHub 部分 query 报 `SSL: UNEXPECTED_EOF_WHILE_READING`；第二次重跑 arXiv 恢复正常（25 篇），GitHub 41 仓库稳定。
3. **筛选规则**：强关键词 (`liquid neural` / `closed-form continuous-time` / `liquid time-constant` / `CfC` / `LTC` / `NCP`) + 距今 ≤ 30 天 + score > 0 + 已有研读报告的跳过。
4. **结果**：
   - `python3 scripts/select_papers_for_report.py --date 2026-06-20 --top 3` 默认打分下命中 1 篇 (GazeLNN, score=2)；手工把 FlowFake (2606.19579, 标题"Liquid Networks"被默认打分忽略) 补入候选池。
   - 真正新增候选 (无既有研读): **GazeLNN (2606.20491v1, 2026-06-18)** + **FlowFake (2606.19579v1, 2026-06-17)**；其余 10 篇 arXiv 候选均已被 round 130-134 既有报告覆盖 (Multi-Rate MoE / LiquidTAD / 3DGS Deformation / MA-GLTC / Liquid Random Feature / DynPMNN / EMMA / LNN vs LSTM / Natural Gas / Nonasymptotic BC).

## 新生成研读报告

- [[docs/reports/GazeLNN_2606.20491_研读报告.md|GazeLNN — 轻量级 CfC 驱动的注视扫描路径预测与主动感知机器人导航]]
- [[docs/reports/FlowFake_LTC_2606.19579_研读报告.md|FlowFake — 液态时间常数网络在跨数据集音频深度伪造检测中的应用]]

## PDF 落盘

- `papers/daily/GazeLNN_2606.20491.pdf` (4.6 MB, 8 pages)
- `papers/daily/FlowFake_LTC_2606.19579.pdf` (522 KB, 11 pages)

## 跳过提交说明

本次 cron prompt 已将 commit + push 交由后续步骤 (Step 4) 处理, 故此 tracker 不直接 commit.

## 最近 LNN 相关候选（供下个 cron 关注，已在外层 daily digest 中跟踪）

| arXiv ID | <published> (UTC) | 标题 | 距今 |
|---|---|---|---|
| 2606.20491v1 | 2026-06-18 | Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation (GazeLNN) | ~2 天 |
| 2606.19579v1 | 2026-06-17 | FlowFake: Liquid Networks for Audio Deepfake Detection | ~3 天 |
| 2606.15807v1 | 2026-06-14 | Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph LTC (MA-GLTC) | ~6 天 |
| 2606.15571v1 | 2026-06-14 | Liquid Random Feature Methods for Time-Dependent PDEs | ~6 天 |

> 注：本仓库默认分支为 `master`（非 `main`），远程仓库 `origin → git@github.com:Dave-he/LNN.git`。
