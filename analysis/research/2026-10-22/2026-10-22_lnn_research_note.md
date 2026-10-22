---
title: 2026-10-22 LNN 每日研究笔记
date: 2026-10-22
tags: [LNN, Daily-Note, Coverage-Audit, Negative-Results, NR-05, NR-09, LFM2, LFM2.5-24B-A2B, Repo-Triage]
---

# 2026-10-22 LNN 研究笔记

**生成**：2026-10-22（digest 由 04:30 timer 自动产出，笔记由当日深度研读补全）
**Digest**：`docs/daily/2026-10-22_LNN_research_digest.md`
**原始池**：`papers/daily/2026-10-22_lnn_research.json`（92 篇 arXiv / 42 仓 / 20 模型 / errors=0）

---

## 0. 一句话结论

**今天没有新 arXiv 论文，但有两个比论文更硬的东西：LiquidAI 自己在 LFM2 架构搜索里
正面测过 CfC/SSM 并否决了它（NR-05 升级 + off-by-one 纠正），
以及 NR-09 的 120 次比较首次被我们从提交数据层面独立复核（0 显著，与来源方一致）。**

---

## 1. 覆盖度自检（AGENTS.md 约束 #7 / #8）

**必答问题**："arXiv 没有新论文"是**领域事实**，还是**我们的覆盖盲区**？

### 1.1 三条独立路径交叉核对

| 路径 | 结果 |
|---|---|
| ① 当日候选池 | 92 篇，与**历史全部每日池**逐 id 求差集 = **0 篇新**（`papers/daily/2026-*.json` 全量比对） |
| ② arXiv 搜索 UI | `https://arxiv.org/search/?searchtype=all&query="liquid neural network"` → **38 条**，最新 = ContiLNN (2026-10-08)，与协议 §3.1 记录的基线 38 完全吻合 |
| ③ arXiv 搜索 UI（LFM 系） | `query=LFM2` → 6 条；`query="Liquid Foundation Models"` → 3 条（均为已研读条目） |

**判定：主线（LNN/LTC/CfC/NCP）"无新论文" = 领域事实，不是覆盖盲区。**
与 2026-10-20 审计的结论一致：8 组查询词已补齐后，主线在 8/9/10 三个月全量扫描
（33,155 条）中只有 2 篇新投稿。**今日池 92 篇全为已知条目，说明扩词修复后
覆盖已稳定，不再有"看不见"的空间。**

> ⚠️ 但**不能据此宣称整个 LNN 领域饱和**：HF 侧今天有 **LiquidAI 官方新模型**
> （LFM2.5-24B-A2B，16,939 downloads / 543 likes / 2026-10-21 更新）。
> **arXiv 静默 ≠ 生态静默** —— 厂商的模型发布往往不发论文。

### 1.2 覆盖盲区残余风险（本轮不做，留待观察）

- `ARXIV_TERM_GROUPS` 未改，本轮**未触发约束 #8 的自检义务**。
- 协议 §3.1 基线 `liquid neural network = 38` 今日复核仍为 38，说明
  **该基线当前有效**。若未来某日 UI 计数与池内条目数背离，需重跑 §3.3。
- 尚未接入 Semantic Scholar（雪球法）—— 10-20 审计已挂账，本轮未动。

---

## 2. 本日最高价值发现：LiquidAI 自己否决了液态算子

### 2.1 触发路径

`selector --date 2026-10-22` 返回 **LFM2 Technical Report (2511.23404)，41 分，全场最高** ——
正是 10-20 覆盖度审计点名"下一步应优先研读、本仓至今无独立研读"的那一篇。
本轮完成独立研读 → [[docs/reports/LFM2_Technical_Report_2511.23404_研读报告]]。

### 2.2 核心发现（一手证据，出自原文）

LFM2 §2.1 的**硬件在环架构搜索**，搜索空间里**显式包含**液态/亚二次家族：

> "Liquid-Time Constant networks such as CfC (Hasani et al., 2022) … state-space variants
> such as S4, Liquid-S4, S5, RTF, Mamba, Mamba2 … linear attention"

而 §2.1 Outcomes 的结论是：

> "Under identical on-device performance budgets, augmenting these stacks (as in recent
> hybrid variants) with linear-attention, state-space, or additional convolution operators
> **does not improve aggregate quality** on the evaluation suite and **typically worsens
> device metrics**."

最终选中的 backbone = **纯 gated short conv + 少量 GQA**。

### 2.3 顺带纠正本仓一处 off-by-one（NR-05）

本仓原文写"LFM2 → LFM2.5 把 ODE + RK4/Euler solver 换成 double-gated conv + GQA"。
核对原文后**版本归属错误**：

- **LFM2 (2025-11-28) 的 abstract 本身就是** "gated short convolutions with a small
  number of grouped query attention blocks" —— **LFM2 里没有 ODE**，切换发生在
  **LFM1 → LFM2**，比本仓记录早一代。
- LFM2 = **gated** short conv（单门控，§3.3 公式）；"double-gated" 是 **LFM2.5** 的 conv 变体。
- 且不是"悄悄替换"，是**架构搜索里正面测过并否决**（输在**质量**轴，不只延迟轴）。

### 2.4 今日新证据：LFM2.5-24B-A2B 的 `config.json`

2026-10-21 官方发布，2026-10-22 直接拉取核对：

| 字段 | 值 |
|---|---|
| `architectures` | `Lfm2MoeForCausalLM` |
| `model_type` | `lfm2_moe` |
| `num_hidden_layers` | 40 |
| `layer_types` | **30 × `conv` + 10 × `full_attention`** |
| `num_experts` / `num_experts_per_tok` | 64 / 4 |
| `hidden_size` / `vocab_size` / `max_position_embeddings` | 2048 / 128000 / 131072 |

**40 层中零层是 ODE / LTC / CfC。** 模型卡自述 LFM2.5-24B-A2B 保持 LFM2-24B-A2B 架构，
只有上下文 32K→131K、词表 65K→128K 两处变化。

**对 NR-05 的意义**：从"厂商把液态降级为组件"升级为
"**液态算子被放进对照实验并输掉**"，且是**上游一手证据**。

---

## 3. NR-09 首次做到数据级独立复核

### 3.1 之前的状态（10-16 笔记）

> "❌ **我方未复现**（未 clone、未跑 grid）。本笔记仅核验了 README 的方法学陈述与
> 结论自洽性，**不对其 120 次比较的数字本身背书**。"

### 3.2 今日做了什么

来源仓库 `julschmidtJJL/lnn-vs-classical-forecasting` 现在**提交了结果数据表**，
因此无需跑 grid 即可做数据级复核。已从 `main` 分支 raw 拉取并逐行核对：

| 文件 | 内容 | 核对结果 |
|---|---|---|
| `src/docs/figures/data/dm_cfc_vs_sarimax_audit.csv` | 120 行 DM 校正比较 | ✅ **120 行，`significant` 全 0，`p_corrected` 全为 1.0** |
| `src/docs/figures/data/mae_run2a.csv` | 57 行 MAE 排名 | ✅ `cfc_pure_ncp` 在 AAPL/JPM/SPY 三个标的均排第 1，但**领先幅度极小**（AAPL 0.013039 vs `sarimax` 0.013095，差 0.4%） |
| `src/docs/figures/data/directional_skill.csv` | 方向命中率 vs 基准 | ✅ `edge_vs_always_long` **三个标的全为负**（AAPL −0.0012 / SPY −0.0018 / JPM −0.0003） |

**与来源方 README / thesis 的三处声明逐条吻合**：0/120 显著、`p_corrected` 最小值 = 1.0、
"最优模型"的方向命中率**低于"永远预测上涨"基准**。

**方法学实现也已核对**（`src/evaluation/evaluator.py`）：
- `dm_test_stat(e1, e2, h=1)`：平方误差损失差分 $d=e_1^2-e_2^2$，
  Newey-West HAC（$h=1$ 时退化为普通方差），HLN 小样本校正因子
  $\sqrt{(T+1-2h+h(h-1)/T)/T}$，$t(T-1)$ 分布求 p 值
- `apply_holm_bonferroni`：`statsmodels.multipletests(method="holm")`
- 显著性族为 **30 个 (arm, ticker, seed) 族 × 4 个 baseline = 120**

### 3.3 来源方自述的局限（引用必须一并带上）

来源仓库的 `src/docs/methods_dm_test_limitations.md` 与 `thesis_results.md` **主动披露**：

1. **DM 检验的 HAC 带宽问题**：不重叠 walk-forward fold 产生的损失差分存在
   自相关，朴素方差估计会**高估**显著性（即 p 偏小、偏乐观）
2. **"k of 5 seeds" 不是独立重复**：seed 间不构成独立重采样
3. **"未检测到差异" ≠ "等效"**：缺等效性检验
4. **ARMA baseline 拟合未加截距**（statsmodels SARIMAX 默认），来源方声明补截距后零结果不变

### 3.4 第二个独立来源交叉印证（今日新发现）

`niharmaddina/Liquidity`（2026-10-12，非本仓昨日跟踪条目）**独立**得出同向结论。
已拉取其提交的 `results/walk_forward.md` 核对（502 个交易日 × 4 标的 × 3 训练/测试切分）：

| 标的 | Persistence RMSE | LNN+ODE fixed | LNN+ODE refit | 方向命中率 vs 上涨日占比 |
|---|---:|---:|---:|---|
| AAPL | **2.54** | 7.03 | 3.95 | 48.8% vs 56.2% |
| TSLA | **8.69** | 10.50 | 11.18 | 49.6% vs 52.6% |
| MSFT | **4.94** | 15.96 | 6.89 | 49.4% vs 54.4% |
| SPY | **3.79** | 15.40 | 5.97 | 47.4% vs 57.4% |

LNN+ODE 在**全部 4 个标的**上都劣于 persistence（最好情况仍差 30%），且方向命中率
**一致低于"永远预测上涨"**。**两个独立团队、方法不同、标的不同的外部研究同向**。

### 3.5 判定更新（诚实边界）

| 项 | 状态 |
|---|---|
| 来源方"0/120 显著"数字 | ✅ **数据级独立复核通过**（从提交 CSV 逐行核对） |
| 来源方方法学陈述 | ✅ **已核对实现源码**（DM+HLN / Holm 族结构） |
| 我方本地复现其 grid | ❌ **仍未做**（未 clone 全仓、未重跑 11 变体 × 8 baseline × 3 标的 × 5 seed） |
| NR-09 判定 | 维持 **证伪（外部，我方未复现）**，但**证据等级从"仅 README 自洽"升到"数据级复核 + 第二独立来源"** |

> **不改判为"一手负结果"** —— 台账 §1 门槛第 2 条要求"本项目跑的 benchmark，
> **或**可复核的外部来源"。来源方数据可复核 ✅，但**我方复现**仍未发生。
> 措辞上把"仅核验 README 自洽性"更新为"已从提交数据表复核 120 行 + 源码核对方法学
> + 第二独立来源印证"，**同时保留"我方未复现"标注**。

---

## 4. 新候选分诊（今日新增，非已研读条目）

来源：2026-10-20/21 更新的 GitHub 仓库 + 2026-10-21 更新的 HF 模型。
**均为只读核查，未 clone 训练、未改动任何外部仓库。**

### 4.1 GitHub

| 仓库 | Star | 判定 | 依据 |
|---|---:|---|---|
| `RESMP-DEV/lfm25-embedding-trainer` | 7 | **worth_watching** | 真代码（非 README 空壳），含 training/evaluation scripts + requirements.txt；但基座是 `LFM2.5-Encoder-350M`，**未用 LTC/CfC 动力学**，"liquid" 只是品牌名 |
| `blacklike02/Liquid-Network-Model` | 4 | **high_value** | **真·LTC + CfC 实现**（`ltc.py` / `cfc.py` / `recurrent.py` / `triton_kernels.py`），4.6M 参数，硬件在环评估声称 140/140 测试通过，**体量适合 Jetson**。⚠️ 其 140/140 与 31.58%→75.20% recall 提升均为**作者自述，未复核** |
| `julschmidtJJL/lnn-vs-classical-forecasting` | 2 | 见 §3 | NR-09 来源 |
| `pinion05/decision-lfm2.5-1.2b` | 1 | **worth_watching** | 真代码：LM head 换成标量 decision head + LoRA，Open-Jev-v1.1，test brier 0.36。**与 LNN 动力学无关** |
| `niharmaddina/Liquidity` | 1 | 见 §3.4 | NR-09 第二来源 |
| `api-evangelist/liquid-ai` | 1 | **keyword_noise** | 第三方 API 说明书，无 LNN 代码 |
| `loom-ai-org/lfm2.5-audio-*` | — | **worth_watching** | LFM2.5 音频衍生（ASR/TTS），1.5B 档，未核实架构细节 |

### 4.2 Hugging Face

| 模型 | 下载 | 判定 | 依据 |
|---|---:|---|---|
| `LiquidAI/LFM2.5-24B-A2B` | 16,939 | **high_value** | 官方；`config.json` 已核对（见 §2.4）；24B **不适合 Orin Nano** |
| `LiquidAI/LFM2.5-24B-A2B-DSpark` | 220 | **worth_watching** | 投机解码 draft 模型；模型卡仍写 "Coming soon"（stale），但权重已发布（655MB） |
| `electroglyph/LFM2.5-24B-A2B-uncensored` | 88 | **duplicate_of_existing** | 第三方微调衍生 |
| `qualcomm/LFM2.5-1.2B-Instruct` | 0 | **keyword_noise** | **空占位仓库**（仅 `.gitattributes`，无权重无 config） |
| `loom-ai-org/lfm2.5-audio-1.5b-{asr,tts}-loom` | 138/75 | **worth_watching** | 音频衍生 |

### 4.3 分诊小结

今日新增候选**没有一个是"新的液态动力学"**：
生态热点集中在 **LFM2.5 的下游衍生**（量化 / 音频 / 决策头 / 检索微调），
这与 §2 的结论一致 —— **液态不再是 backbone 的竞争选项，而是被降级为"可挂载组件"**。
真正含 LTC/CfC 实现的只有 `blacklike02/Liquid-Network-Model`（4★）一个。

### 4.4 顺带修掉的一个流水线噪声：HF 空壳占位仓库

`qualcomm/LFM2.5-1.2B-Instruct` 进了当日 digest 的 HF 候选表并占掉一个可见行，
但它**是个空壳**：detail 端点实测 `usedStorage=0`、只有 1 个文件（`.gitattributes`）。

**已修**（`scripts/daily_lnn_research.py::fetch_huggingface_models`）：
新增 `_hf_model_is_empty()`，只对**零下载零点赞**的边缘项打 detail 端点，
按**实际存储/文件数**判空，查不动则**保留**（宁可留噪声也不误杀真模型）。

> ⚠️ **刻意不用 downloads/likes/tags 启发式**：实测
> `ryugyosoft/LFM2.5-24B-A2B-onwx-kit` 是 downloads=0 / likes=0 / tags 里没有任何
> safetensors|transformers|gguf 标记，但**有 5.4GB 权重、134 个文件** ——
> 启发式会把它误杀。唯一可靠判据是 detail 端点的实际存储。
>
> **实跑验证**（per_query=15）：空壳被跳过 ✅，`ryugyosoft`(5.4GB) /
> `electroglyph` / 官方 24B 全部保留 ✅，保留 33 个模型。

---

## 5. 诚实边界

- ❌ 未 clone 任何外部仓库跑训练；§3 的复核**仅限已提交的数据文件与源码**。
- ❌ 未复现 LFM2 §2.1 架构搜索（内部 50+ 评测套件未公开）。
- ❌ 未在 Jetson 上实测 LFM2.5-24B-A2B（24B 档不适合本仓 Orin Nano 目标）。
- ✅ 已按 AGENTS.md 约束 §7 区分"领域事实 vs 覆盖盲区"，未把 arXiv 静默宣称为全领域饱和。
- ✅ 未重复任何 NR-01~NR-09 的不可重复 claim。

---

## 6. 明日建议

1. **优先**：`blacklike02/Liquid-Network-Model` 结构化调研 —— 它是今日唯一
   含真 LTC/CfC 实现、且体量适合 Jetson 的新仓，直接对接本仓路线 A。
2. **次优先**：评估是否本地复现 NR-09 的 grid（`analysis/replication/` 已有模板）。
   复现成功可将 NR-09 从"外部"升级为"一手负结果"。
3. **低优先**：接入 Semantic Scholar（雪球法），补 §1.2 的残余盲区。

---

**关联**

- 全局研读索引：[[docs/LNN_深度研读报告]] §0
- 负结果台账：[[analysis/negative_results/README]]（NR-05 / NR-09 已更新）
- 本轮研读：[[docs/reports/LFM2_Technical_Report_2511.23404_研读报告]]
- 覆盖度审计：[[analysis/research/2026-10-20/2026-10-20_lnn_research_note]]
