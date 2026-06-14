# LNN 每日论文追踪 — 2026-06-14

> Cron 任务：`lnn-papers-tracker-cron`
> 执行时间（UTC）：2026-06-14 01:02
> 检索窗口：最近 24 小时（2026-06-13 01:02 UTC ~ 2026-06-14 01:02 UTC）

## arXiv 检索摘要

- 检索 URL：
  `http://export.arxiv.org/api/query?search_query=all:%22Liquid%20Neural%20Networks%22+OR+all:%22CfC%22+OR+all:%22LTC%22+OR+all:%22Neural%20ODEs%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending`
- 关键词：`Liquid Neural Networks`, `CfC`, `LTC`, `Neural ODEs`
- arXiv 端总匹配数：`opensearch:totalResults = 966`
- 本次拉取条数：100（按 `submittedDate` 倒序）

## 24 小时窗口内结果

| 序号 | arXiv ID | 标题 | 发表时间（UTC） | 入库 |
| --- | --- | --- | --- | --- |
| — | — | — | — | — |

**结论**：本轮在最近 24 小时内未发现满足关键词条件的新论文。最接近窗口边界的一条是 `2606.13571v1`（发表于 2026-06-11 16:59 UTC，距窗口起点 56 小时，已在窗口外）。

## 近端候选（窗口外，但仍有参考价值）

下列条目虽然不在 24 小时窗口内，但为 6 月 LNN / Neural ODEs 主题最相关候选，可作为下一轮或人工跟进参考：

1. `2606.13571v1` — *Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States in Time Series Forecasting*（2026-06-11，Neural ODEs 相关）
2. `2606.12240v1` — *Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training*（2026-06-09，**直接命中 LNN**）
3. `2606.11162v1` — *COGENT: Continuous Graph Emulators with Neural Ordinary Differential Equations …*（2026-06-09）
4. `2606.10596v1` — *Embedding Hybrid Systems into Continuous Latent Vector Fields*（2026-06-08）
5. `2606.08431v1` — *Control-Theoretic View of Neural ODEs: Empirical Controllability and Observability*（2026-06-07）
6. `2606.07670v1` — *Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamical …*（2026-06-05，**直接命中 LNN**）

## 操作记录

- 工作目录：`/Users/hyx/workspace/LNN`
- 当前分支：`master`（非 `main`，push 目标对应调整）
- 下载目录：`papers/daily/`
- 本次新增 PDF：**0**
- Git 提交：已按 cron 规则跳过（`若无新论文则跳过提交`）
- Git 推送：未执行
- 错误：本次运行无错误

## 下次任务

- 当 arXiv 在 6 月 14 日 (UTC) 之后推送新的 LNN / Neural ODEs 条目时，将自动进入 24 小时窗口。
- 如需立即补充窗口外的高相关论文（如 `2606.12240v1`、`2606.07670v1`），可由人工触发 `scripts/daily_lnn_research.py` 或手动归档到 `papers/daily/`。
