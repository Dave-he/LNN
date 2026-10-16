---
title: LNN 研究笔记 2026-10-16 - 主题饱和 + 外部负结果 + 流水线盲区修复
date: 2026-10-16
tags: [LNN, negative-result, pipeline-fix, saturation]
---

# LNN 研究笔记 2026-10-16

## 0. 今日状态一句话

**arXiv 侧 0 篇新论文（真饱和），调度层健康，但发现并修复了一个让"新论文永远进不了研读队列"的流水线盲区。**

---

## 1. 调度层核验（健康）

| 项 | 值 |
|---|---|
| timer | `lnn-daily-research.timer` enabled / active (waiting) |
| 今日自动跑 | 2026-10-16 04:36:24 CST 启动，04:38:44 结束 |
| 产物 | digest 24/42/17，benchmark 落 `analysis/jetson/2026-10-16_*` |
| push | `1c522e5` 已上 origin（`9a0d74b..1c522e5`） |
| 下次触发 | 2026-10-17 04:32:22 CST |

`errors: []`，无数据源失败。**定时任务无需重启，已在按天执行。**

## 2. 今日 arXiv：0 篇新论文，饱和是领域事实

选择器输出 `pool_source=json total=24 skip=24 cand=0` —— 候选池 24 篇**全部**已在往日研读过（`docs/reports/` 共 120 篇）。

对照 2026-10-14 的判断：**10-08 之后确无新 LNN 论文入池，"0 new" 是真实领域事实，不是抓取失效**。佐证：GitHub / HuggingFace 侧同期仍有高频更新（LFM2.5-24B-A2B 权重 10-13 上架），说明数据源本身通畅。

## 3. 外部独立负结果（今日唯一实质研究产出）

### julschmidtJJL/lnn-vs-classical-forecasting（更新 2026-10-15，2★）

Leuphana University Lüneburg 本科毕业论文（B.Sc. Business Informatics 2026，评级 1.0），附 89 页论文 PDF。

**结论：不能。** 11 个液态变体 × 8 个 baseline × 3 个美股标的，**120 次校正比较中 0 次**显示液态模型显著优于传统 baseline。

**方法学（可复核，已逐条读 README 确认）：**

| 项 | 设置 |
|---|---|
| 标的 / 目标 | AAPL / JPM / SPY，十年日频；one-step-ahead log return |
| 验证 | expanding-window walk-forward，750 日初始窗，21 日 fold |
| 种子 | 5 个（seed 42–46），指标报 mean ± std |
| 显著性 | Diebold–Mariano（Harvey–Leybourne–Newbold 小样本校正）+ Holm 多重比较校正 |
| 液态模型 | LTC / CfC 各变体，横跨 gating 模式、wiring（全连接 vs 稀疏 NCP）、memory |
| baseline | LSTM、CT-RNN、Neural ODE、Naive、四个 ARMA 规格 |
| 输入 | 价格（yfinance）+ 宏观（FRED，滞后一日以规避发布延迟）+ 基本面（SEC EDGAR） |
| 可追溯性 | 每次运行记录 git commit / config / seed / 输入快照 hash |

**三条关键子结论：**

1. **无显著优势。** 任何液态变体在任何标的上都未显著超过传统 baseline（DM + Holm 校正后）。
2. **"最优模型"不是真赢。** 点指标上最强的纯 gating 稀疏 CfC 在三个标的上都排第一，但该领先**不显著**，且其方向性优势**完全由市场漂移解释**——其方向命中率与"永远预测上涨"这个基线持平。
3. **液态模型确实与其他神经模型不同。** 在神经模型内部，液态模型分离清晰且显著；实测表达度（轨迹长度）能对神经模型场排序，但**这种表达度换不来对传统基准的技能**。

**作者归因：** 限制来自数据而非模型 —— 日频收益率携带的信号太少，额外容量无法兑现。

### 与本项目结论的关系（important）

- ✅ **正向印证约束 #6**：LNN 在标准任务上未显著超过传统/transformer 路线。本条是**外部独立第三方**的同向证据，不是本项目自证。
- ✅ **印证"长尾工业任务才是 LNN 的稳定选择区"**：作者把失败归因于**日频收益率信号太稀疏**。这与本项目观察（LNN 仅在 D-3DGS / FJSP / EEG 等信号密集的长尾工业任务上稳定胜出）在机理上一致 —— **判据是信号密度，不是任务名称**。
- ⚠️ **不得反向升级**：这是一条负结果，**不能**被引用为"LNN 已解决/已证伪表达力"这类更强结论。作者本人也只声称"表达度排序不转化为技能"。

**作者主动披露的局限（引用时必须一并带上）：** ARMA baseline 拟合时未加截距（statsmodels SARIMAX 默认行为）；论文声明补上截距重拟合后零结果不变。本项目引用该仓库任何数字时必须同时披露此点。

**未验证项（诚实标注）：** 我方**尚未**复现该仓库（未 clone、未跑 grid）。本笔记仅核验了 README 的方法学陈述与结论自洽性，**不对其 120 次比较的数字本身背书**。若要纳入 r-series 负结果台账，需先本地复现。

## 4. 流水线修复：digest 表格截断导致的选择器盲区

### 症状

今日 digest 摘要写 `arXiv 候选论文：24 篇`，但 arXiv 表格**只有 12 行**；GitHub / HF 同理（42 → 12 行，17 → 12 行）。

### 根因

`scripts/daily_lnn_research.py` 有三处硬编码 `[:12]` 渲染表格，而 `scripts/select_papers_for_report.py` **只解析 markdown 表格**。

后果链条：
1. 摘要报全量 24，表格只给 12 → digest 读起来像"24 篇已覆盖"，实际只暴露 12 篇。
2. 选择器只见 12 篇 → **当日新论文若排在第 13 名之后，永远进不了研读队列，且无任何告警**。

这是"静默截断"类 bug，与 2026-10-14 的索引截断事故同类（那次是按行截断，这次是候选池截断）。今日未爆是因为 12 篇可见的恰好都已研读过；换一天排名变动就会漏。

### 修复

1. `daily_lnn_research.py`：新增 `DIGEST_TABLE_LIMIT = 12`，替换三处硬编码；摘要区**显式标注**"下表每类仅渲染前 12 条；全量候选池见 `papers/daily/<date>_lnn_research.json`，研读选择器以该 JSON 为准"。
2. `select_papers_for_report.py`：新增 `parse_json_pool()`，**以 JSON 全量池为准**，markdown 仅作回退；输出新增 `pool_source` 字段（`json` / `digest_markdown`）以便审计。

### 验证

- `pool_source=json total=24`（修复前 `total=12`），10-13 ~ 10-16 四天均确认读到全量 24。
- 合成渲染测试（20 条输入）：渲染行数 = 12，标注行存在，摘要总数仍为 20 —— 截断可见且不再伪装成覆盖。

## 5. 文档与现实漂移（**已解决**，同日闭环）

排查时发现：`AGENTS.md` 约束节声称"负结果已建 `negative-result` 目录（r104/r126/r128/r130-133/r306/r307）"，但：

- `analysis/negative_results/` **不存在**；
- 点名的 **r104 / r126 / r128 / r130–r133 在全库无任何实体** —— 全库实际只有 **r301–r307**
  （r301–r305 是 PLAN/Midpoint/STE/LFM2.5 并行 CfC 等**正向**实验，只有 r306/r307 是负结果）；
- 负结果实际**散落**在 `docs/reports/`、`analysis/research/`、`analysis/timeseries_ablation/`、
  `analysis/loop_status/`、`docs/research/` 等处，无统一台账。

即 AGENTS.md 约束 #4（"负结果必须进 negative-result 目录"）**无法机械执行**，且其证据清单本身是错的。

### 已做的修复

1. **建 `analysis/negative_results/README.md` 负结果台账**：9 条已判定负结果（NR-01~NR-09），
   每条含被否证主张 / 判定 / 证据 / grounding 路径 / 引用纪律 / 复现状态。
   采用**指针式**设计（只存判定 + 路径，不复制原文）—— 搬移会打断既有 `[[wikilink]]`，
   复制必然漂移，而**漂移后的台账比没有台账更危险**。
2. **建 `scripts/check_negative_results.py` 门禁**：校验 grounding 路径真实存在、
   表格与正文一一对应、`AGENTS.md` 约束表未悬空。**已做负向测试**（改坏路径 / 删 grounding /
   断开 AGENTS.md 引用，三种破坏均正确失败退出 1），确认是真门禁而非恒过桩。
3. **接入每日流水线** `run_daily_lnn_task.sh`：失败时**大声报 + 写日志**，但**不阻断**
   当天研究产出（数据采集不该因文档索引漂移而停摆）。刻意**不用** `|| true` 吞掉。
4. **修正 AGENTS.md**：约束表新增 NR-xx 列与第 9 条（NR-09 外部负结果），
   约束 #4 改为指向真实台账，删掉不存在的 r104/r126/r128/r130-133 引用，
   并把台账 + 门禁登记进"已落地自动化能力"。

### 教训

**"约束"与"落地实体"必须能互相校验，否则约束只存在于文字里。**
本项目此前已栽在同类的"文字声称已建、实际没建"上（本次是负结果目录）。
门禁脚本 + 负向测试是让约束具备牙齿的最低成本做法。

---

## 关联

- 调度说明：[[每日自动化任务与Jetson验证]]
- 全局索引：[[LNN_深度研读报告]]
- 外部来源：`https://github.com/julschmidtJJL/lnn-vs-classical-forecasting`（含 89 页论文 PDF release）
