---
title: LNN 研究笔记 2026-10-20 - 流水线覆盖审计: arXiv 论文侧对 LFM 主线失明 + GitHub fork 噪声挤掉高价值仓库
date: 2026-10-20
tags: [LNN, pipeline-fix, coverage-audit, arxiv, github, negative-result]
related_digest: "docs/daily/2026-10-20_LNN_research_digest.md"
---

# LNN 研究笔记 2026-10-20 — 流水线覆盖审计

## 0. 今日状态一句话

**调度层健康、台账门禁通过；原以为"零新论文 = 领域饱和"，审计后证明是**覆盖盲区**——
arXiv 查询只覆盖协议 §3 的 4/7 组，导致 LFM/LFM2 论文侧 100% 失明；
GitHub 候选表被同名 fork 刷屏，把 29★ 的头号仓库挤出可见区。
修复后同一天重跑：候选池 24 → 92 篇，selector 从 0 候选变成 3 个真候选。**"饱和"是假的。**

---

## 1. 调度层核验（健康）

| 项 | 值 |
|---|---|
| timer | `lnn-daily-research.timer` 04:30 触发，04:33 产出 |
| digest | `docs/daily/2026-10-20_LNN_research_digest.md`（修复前 24 篇 / 42 仓 / 14 模型 → **修复后 92 篇 / 42 仓 / 20 模型**） |
| watchlist | `analysis/repo_watchlist/2026-10-20_lnn_open_source_watchlist.md` |
| benchmark | `analysis/jetson/2026-10-20_lnn_benchmark.*`（`ok_cpu`，CUDA 不可用） |
| 台账门禁 | `python3 scripts/check_negative_results.py` → **OK**（9 条，grounding 全在） |
| 完整性门禁 | `python3 scripts/check_repo_integrity.py` → **OK**（3102 文件，无冲突标记） |

今日 06:36 补跑一次（`bfe8610`），HF 5 个查询全部 timeout，脚本按设计**保留上一轮成功数据**，
不伪装成"全成功" —— 这条降级行为在 `errors` 字段留痕，符合 10-17 建立的纪律。

## 2. "零新论文"的真相：是覆盖盲区，不是领域事实

选择器最初输出：`n_total_arxiv=24, n_skipped_reported=24, candidates=[]` —— 24 篇全部已研读过。
前几日的笔记把它记为"真饱和"。**这个判断是错的**，本次审计推翻了它。

**当时用的证据不足**：只说"GitHub/HF 同期仍在更新，所以抓取是通的"——
这只证明**网络**通，不证明**查询词**覆盖全。

**本次的独立验证（3 条互相独立的路径，全部绕开被限流的 `/api/query`）**：

| 验证方法 | 覆盖范围 | 结论 |
|---|---|---|
| arXiv 月度列表全量（cs.LG/AI/CL/CV/RO/SY/NE/DC，2026-08/09/10） | 33,155 条唯一条目 | 主线只有 2 篇（MTLiquid、ContiLNN）→ **主线确实饱和** |
| arXiv RSS 分类 feed（cs.LG/AI/NE/RO/SY） | 902 条（近两日，含完整摘要） | LNN/LFM 命中 0 |
| arXiv 搜索 UI 全库计数 | 逐词 | `LFM2`=6、`"liquid foundation model"`=3 —— **有真实论文存在** |

> 探针方法论备忘：arXiv `/api/query` 在本机当前处于 429 硬限流（多轮 4 次重试全 429），
> 但 **RSS feed、月度 HTML 列表、搜索 UI 三条路径均可用**。因此本次能给出结论而不是
> "抓不到所以不知道"。

**结论分层**：
- **主线（LNN/LTC/CfC/NCP）确实饱和** —— 3 个月全量扫描只有 2 篇，与候选池记录吻合。
- **但整体覆盖是残缺的** —— 协议 §3 明写的三组词从未进查询，
  LFM/LFM2 论文侧命中数为 **0/3356**。修复后同一个脚本立刻抓到 92 篇。



## 3. 缺陷一：arXiv 查询与协议 §3 不一致 —— LFM 主线 100% 失明（已修并实跑验证）

### 3.1 事实

`scripts/daily_lnn_research.py` 的 arXiv 查询只有 **9 个词组**（全在 LNN/LTC/CfC/NCP 主线）。
但 `docs/LNN_持续研究协议.md` §3 的关键词表定义了 **7 组**，其中 3 组**从未进过查询**：

| 协议 §3 分组 | 协议要求的词 | 是否在 arXiv 查询里 |
|---|---|---|
| 核心概念 | liquid neural network(s) | ✅ |
| LTC | liquid time-constant / liquid time constant | ✅ |
| CfC | closed-form continuous-time | ✅ |
| NCP | neural circuit policy(ies) | ✅ |
| 相关方法 | neural ODE / continuous-time RNN / state-space liquid | ❌ **缺** |
| 基础模型 | Liquid AI / LFM / LFM2 / liquid foundation model | ❌ **缺** |
| 应用场景 | LNN edge / time series / forecasting / medical / finance / robotics | ❌ **缺** |

### 3.2 后果（已量化）

- 候选池历史全量 **3356 条**记录里，含 `LFM2` / `liquid foundation model` / `Liquid AI` 的：**0 条**。
- 同一天 Hugging Face 侧 14 个模型里 **12 个是 LFM2.5**。**论文侧对生态主线完全失明。**
- 实际存在于 arXiv 的 LFM 论文因此从未进池：
  - `2511.23404` **LFM2 Technical Report** —— 全库唯一一篇"liquid foundation model"正文命中，
    只因 HF 模型卡 tags 里带 `arxiv:2511.23404` 才被 watchlist 间接看到，**从未作为论文进入研读队列**。
  - `2603.03517` **MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery** —— 仓库内零引用。
- 而 `KEYWORD_RE` 里**本来就含 `\bLFM2`**：说明**下游过滤器早就准备接收 LFM 论文，只是上游查询从来没送来过**。

### 3.3 修复

按协议 §3 的分组补齐查询词，并**拆成 8 个小查询组**（而非一条 21 词 OR）：
`/api/query` 按查询串长度限流，单条 21 词 OR（~810 字符）会稳定触发 406/429 ——
今日审计期间实测到 429 硬限流，拆分后单组失败不牵连其他组。
`KEYWORD_RE` 同步补上三组判定词，否则新查询抓回的论文会被 `keyword_score` 判 0 丢弃。

**新词逐个用 arXiv 搜索 UI 全库实测（2026-10-20），据此取舍**：

| 词 | 实测命中 | 处置 | 理由 |
|---|---:|---|---|
| `LFM2` | 6 | ✅ 采纳 | 含 `2511.23404` **LFM2 Technical Report**（从未进池） |
| `liquid foundation model` | 3 | ✅ 采纳 | 含 `2511.23404` + `2603.03517` MMAI Gym（从未进池） |
| `continuous-time RNN` | 6 | ✅ 采纳 | 含 `1811.00321`（LTC 作为 RNN 通用逼近） |
| `LNN time series` | 8 | ✅ 采纳 | 应用场景主通道 |
| `LNN robotics` | 4 | ✅ 采纳 | |
| `LNN forecasting` | 3 | ✅ 采纳 | |
| `LNN edge` | 2 | ✅ 采纳 | |
| `LNN medical` | 2 | ✅ 采纳 | |
| `Liquid AI` | 1 | ❌ 剔除 | 唯一命中是金融 "**Illiquid** Markets" 论文 —— 纯误伤 |
| `neural ODE` | 715 | ❌ 剔除 | 噪声淹没主线；且 selector 的 `STRONG_KEYWORDS` 本就含 `neural[- ]ode`，放开会把 715 篇灌进研读队列 |
| `state-space liquid` | 0 | ❌ 剔除 | 无命中 |
| `LNN finance` | 0 | ❌ 剔除 | 无命中 |
| `NCP liquid` | 0 | ❌ 剔除 | 无命中 |

**关键语法坑（本次踩到并修正）**：arXiv API 的 `all:"..."` 是**连续短语**匹配，
而搜索 UI 的 `"X" AND "Y"` 是**布尔**匹配（X、Y 各自出现即可，位置不限）。
**两者不等价**：实测 `all:"LNN edge"` → 0 篇，但 `all:LNN AND all:edge` → 2 篇。
所以应用场景组必须写成布尔式 `all:LNN AND all:edge`，不能写成 `all:"LNN edge"` ——
否则该组会**静默返回 0 篇**，看起来像"该场景没论文"，实际是查询写错了。
含 `AND` 的 fragment 一律加括号，不依赖 arXiv 的运算符优先级实现细节。

### 3.4 端到端验证（已通过）

限流窗口过后，用**真实 `/api/query`** 跑通两个新组，验证查询语法与信噪比：

**`应用场景` 组**（`all:LNN AND all:edge` 等 5 个布尔式，url_len=306）→ **18 篇命中**：

| 日期 | ID | score | 标题 | 是否已研读 |
|---|---|---:|---|---|
| 2026-08-13 | 2608.13215v1 | 1 | History-informed Lagrangian Neural Networks | — |
| 2026-06-27 | **2606.28827v1** | 3 | **LNN-Fly: Continuous-Time UAV Navigation for Robust Obstacle Avoidance** | **否（新发现）** |
| 2026-06-25 | 2606.26849v1 | 1 | Liquid Fusion of Heterogeneous Representations… | 已研读 (LFNet) |
| 2026-06-12 | 2606.14284v1 | 1 | Hierarchical ODE… | 已研读 |
| 2026-06-10 | 2606.12240v1 | 6 | Multi-Rate MoE for Accelerating LNN Training | 已研读 |
| 2026-05-26 | 2605.27467v1 | 7 | Comparative Analysis of LNN and LSTM | 已研读 |

**`基础模型` 组**（url_len=162）→ **8 篇命中**：

| 日期 | ID | score | 标题 | 是否已研读 |
|---|---|---:|---|---|
| 2026-07-16 | 2607.15232v1 | 2 | In-Place Tokenizer Expansion for Pre-trained LLMs | 否 |
| 2026-04-27 | 2604.24346v1 | 1 | SycoPhantasy… | 否 |
| 2026-04-06 | 2604.04579v2 | 1 | Firebolt-VL… | 否 |
| 2026-03-03 | **2603.03517v2** | 2 | **MMAI Gym for Science: Training LFM for Drug Discovery** | **否（新发现）** |
| 2026-02-25 | 2603.02250v1 | 1 | SGPA: Spectrogram-Guided Phonetic Alignment | 否 |
| 2025-11-28 | **2511.23404v1** | **14** | **LFM2 Technical Report** | **否（新发现，score 最高）** |

**关键结论**：
1. **布尔语法正确** —— `all:LNN AND all:edge` 真实返回 18 篇，证明"应用场景组写成短语会静默返回 0"的判断成立。
2. **信噪比可接受** —— 18 篇里 5 篇已研读、1 篇新发现（`LNN-Fly`），噪声可控。
   基础模型组 8 篇里 6 篇未研读，且 **`2511.23404` LFM2 Technical Report 的
   `keyword_score=14` 是全场最高** —— 印证了"论文侧对生态主线失明"这个诊断。
3. **两个新组带来 3+ 篇从未见过的论文**（`LNN-Fly` / `MMAI Gym` / `LFM2 Technical Report`），
   其中 LFM2 Technical Report 是生态主线论文，本仓至今没有独立研读报告。

### 3.5 诚实边界

- **已验证**：新分组在真实 `/api/query` 上返回正确结果；GitHub 修复实跑 42 仓 0 错误。
- **仍待观察**：新词组会让候选池**变大**（主线 24 篇 → 可能 40+ 篇）。
  selector 的 `score_paper` 会排序，但 `LNN` 缩写类论文的分数天然偏低
  （`LNN edge` +2 / `LNN time series` +2），可能排在已研读论文之后 ——
  **明早那次运行后必须复核一次队列实际长什么样**，不要默认"抓到 = 会去研读"。
- 该风险已按"分组隔离 + 失败留痕"设计兜底：任一组失败只丢该组，其余 7 组照常产出；
  失败**记进 `errors` 字段**（与 GitHub/HF 侧同一纪律，2026-10-19 `a556476` 立的规矩），
  绝不把"5/8 组成功"显示成"查询全成功"；全部组失败时才抛错，触发既有的
  RSS fallback + 前一日数据保留。



## 4. 缺陷二：GitHub fork 刷屏挤掉高价值仓库（已修）

### 4.1 事实

10-20 候选池 42 个仓库里，**8 个是同一个 `bebelm` 仓库的 fork**（不同 owner、同名、同描述、0★）。
digest 表只渲染前 12 行，结果：

| 现状（渲染前 12 行） | | |
|---|---|---|
| `bebelm` fork × 8 | 0★ | 占 2/3 表格 |
| `di-omics/liquid-policy-robustness` | 0★ | |
| `BuczynskiRafal/ltc-hydrology-flood-prediction` | 0★ | |
| **`Ipsedo/LiquidNetworks`** | **29★** | **被挤到第 13 行 —— 人类看不见** |
| **`siddharthk27/Neural-Circuit-Policies`** | **4★** | **被挤到第 38 行** |
| **`Noah-hjl/ContiLNN`** | **3★** | **被挤到第 21 行（官方 ContiLNN 实现！）** |

GitHub search API 的 `best match` 顺序把 fork 群排在前面，而表格又按这个顺序截断。
**结果是：star 最高的仓库、最相关的官方实现，全部落在 12 行窗口之外。**

该模式在 10-14 起持续放大（同名簇 2 → 10），属**趋势性劣化**而非偶发。

### 4.2 修复

1. 抓取时记录 GitHub 的 `fork` 标志（原先没取）。
2. `rank_repos()`：同名 repo 名（stem 相同）折叠为一簇，保留 star 最多者为主条目，
   记录 `fork_variants` 计数（**不丢信息**，JSON 里仍可查每个变体及其 `fork_of`）。
3. 排序改为**研究价值序**（star 降序 → 最近更新降序），fork 变体一律排到普通条目之后。

**修复后同一池子的前 12 行**：`Ipsedo/LiquidNetworks` (29★) → `blacklike02/Liquid-Network-Model` (4★)
→ `siddharthk27/Neural-Circuit-Policies` (4★) → `Noah-hjl/ContiLNN` (3★) → …
`bebelm` 折叠为 1 条主条目 + 7 个变体。

**实跑验证**（修复后重新抓取 GitHub，42 仓 / 0 错误）：

| 修复前位置 | 仓库 | Star | 修复后位置 |
|---:|---|---:|---:|
| 13 | `Ipsedo/LiquidNetworks` | 29 | **1** |
| — | `RESMP-DEV/lfm25-embedding-trainer`（本次新发现） | 7 | **2** |
| 38 | `siddharthk27/Neural-Circuit-Policies` | 4 | **4** |
| 21 | `Noah-hjl/ContiLNN` | 3 | **5** |
| 5-12 | `bebelm` × 8 (0★) | 0 | 折叠为 1 条 |

## 5. 研究方向与目标是否正确 —— 评估

**结论：研究方向正确，项目定位（§0 三角区）经得起检验；缺陷在"研究执行的覆盖面"，不在"方向"。**

### 5.1 定位判断仍然成立

- 9 条负结果台账（NR-01~NR-09）+ `AGENTS.md` 约束表双向绑定 + 可执行门禁 ——
  **这是本仓最强的方法论资产**，今日两道门禁均通过。
- 诚实负结果常态化（r306/r307 严格复现后撤回）—— ML 社区罕见，值得沉淀。
- 昨日 105K 档实测（`aa2836a`）进一步支持"轻量 ≠ 实时"：
  0.40 MB / 107K 参数，**batch=1 推理 248.9 ms（4 Hz）** —— 单样本达不到多数控制回路。
  这条与 §0 "端侧 + 小数据 + 时序" 三角区并不矛盾，但补上了一个此前被"0.40 MB"数字掩盖的边界。

### 5.2 真正的偏差：跟踪对象与生态重心错位

| 维度 | 项目预期 | 生态实际（今日实测） | 判定 |
|---|---|---|---|
| 论文侧重心 | LNN/LTC/CfC 主线 | 主线 3 个月仅 2 篇 | ✅ 真实饱和 |
| **LFM/LFM2** | 协议列为"基础模型"组 | **论文侧查询根本没覆盖**（0/3356）；HF 侧 12/14 是 LFM2.5 | ❌ **覆盖缺失（已修，实跑 +283%）** |
| GitHub 可见性 | 追踪高 Star 仓库 | 29★ 头号仓库被挤出窗口 | ❌ **可见性缺失（已修）** |
| 历史存量 | — | 修复后立刻捞出 2 篇 2023 年旧文 | ❌ **存量也漏（已修）** |
| 快照/恢复 / 持续学习 | §0 三角区之一 | MT-LNN 单人项目、已自撤回 4 条 | ⚠️ 只作思想来源 |
| 雪球法扩展 | 协议 §8 每周/每月 | Semantic Scholar / OpenAlex / OpenReview **从未实现**（`grep` 零命中） | ❌ **待扩展项长期挂账** |

### 5.3 下一步建议（落地导向）

| 目标 | 建议 | 入口 |
|---|---|---|
| 补齐 LFM 论文侧 | **已修并实跑验证**（池 24→92，selector 0→3 候选） | `scripts/daily_lnn_research.py` |
| 仓库可见性 | **已修并实跑验证** | 同上 |
| **真正的新证据** | **优先研读 `2511.23404` LFM2 Technical Report** —— selector 给它 41 分（全场最高），它是生态主线的技术报告，本仓至今没有独立研读报告 | `skills/paper-analyzer` |
| 降本 | 协议 §4 标注"待扩展"的 4 个数据源**从未实现**；与其继续扩 arXiv 词表，不如先接 **Semantic Scholar** 做引用雪球 —— 它补的是"关键词检索根本找不到的论文" | `scripts/` 新增 |
| 保持诚实 | 任何 LFM2.5 相关结论仍受 NR-05 约束：**"Liquid + X 混合"不是纯液态模型** | `analysis/negative_results/README.md` |

## 6. 今日产出

| 类型 | 路径 |
|---|---|
| 研读报告 | **0 篇**（修复前候选池 24/24 已研读 → `candidates=[]`；**修复后池子 92 篇，selector 给出 3 个真候选**，见 §6.1） |
| 研究笔记 | 本文 `analysis/research/2026-10-20/2026-10-20_lnn_research_note.md` |
| 流水线修复 | `scripts/daily_lnn_research.py`：arXiv 分组查询（8 组）+ 协议 §3 三组补齐 + GitHub fork 折叠/价值排序 |
| 协议对齐 | `docs/LNN_持续研究协议.md` §3.1（查询词↔实现对应表 + 实测命中数）、§3.2（arXiv 语法坑）、§3.3（覆盖度自检）、§4（待扩展数据源提醒） |
| Agent 约束 | `AGENTS.md` 新增「覆盖度自检约束」#7/#8/#9 |
| digest | `docs/daily/2026-10-20_LNN_research_digest.md`（**已用修复后的脚本重新生成**） |
| 负结果 | **未新增**（今日无新的可证伪主张 + 收敛判定；见台账 §1 门槛） |

### 6.1 修复后重跑的实际效果（同一天、同一脚本、修复前后对比）

| 指标 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---|
| arXiv 候选池 | 24 篇 | **92 篇** | **+283%** |
| 8 个查询组成功数 | 1（单条 OR） | **8/8** | 全组通 |
| selector 候选 | `[]` | **3 个真候选** | 从 0 到有 |
| GitHub 表前 12 行里的 0★ fork | 8 条 | **0 条** | 噪声清除 |
| GitHub 表首位 | `blacklike02/…` (4★) | **`Ipsedo/LiquidNetworks` (29★)** | 头号仓库可见 |
| HF 模型数 | 14 | 20 | — |

**selector 返回的 3 个新候选**（此前从未进入研读队列）：

| score | ID | 标题 | 备注 |
|---:|---|---|---|
| **41** | `2511.23404` | **LFM2 Technical Report** | 生态主线论文；`keyword_score` 全场最高 |
| 35 | `2304.08691` | LTC-SE: Expanding the Potential of Liquid Time-Constant Neural Networks… | 2023 年旧文，此前漏检 |
| 26 | `2307.04772` | Digital Twins for Patient Care via Knowledge Graphs and Closed-Form Continuous-Time LNNs | CfC × 医疗数字孪生 |

> **诚实边界**：这 3 篇是**候选**，不是**已研读**。今日未生成研读报告 ——
> 研读需要逐篇读原文，本轮时间预算用在了覆盖度审计与修复上。
> **明天的例行任务应当优先处理 `2511.23404` LFM2 Technical Report**（本仓至今没有它的独立研读报告，
> 而它正是生态主线的技术报告）。
> 注意其中 2 篇是 2023 年旧文 —— 这说明"候选池 24 篇"不只是"漏了新论文"，
> 也**漏了历史存量**，进一步说明原查询覆盖面偏窄。

> **未新增台账条目的理由**（遵守台账 §1 三门槛）：本次发现的是**流水线覆盖缺陷**，
> 不是"某个技术主张被实测否证"。台账收录的是"技术负结果"，不是"工程缺陷清单"。
> 流水线缺陷的正确去处是本笔记 + commit message + 修复本身。


## 7. 关联

- 每日 digest：[[docs/daily/2026-10-20_LNN_research_digest.md]]
- 项目定位：[[docs/LNN_深度研读报告]] §0
- 路线分叉：[[docs/research/2026-09-24_technical_route_landscape]] §四 / §五
- 负结果台账：[[analysis/negative_results/README]]
- 上一日笔记：[[analysis/research/2026-10-16/2026-10-16_lnn_research_note.md]]
