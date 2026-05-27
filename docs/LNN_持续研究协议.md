---
title: LNN 持续研究协议
date: 2026-05-27
tags: [LNN, research-protocol, living-review, automation, skill]
---

# LNN 持续研究协议

> 本协议配合 [[AGENTS]] 与 `skills/living-field-researcher` 使用，用于把 LNN 领域的每日搜索、筛选、研读、代码调研和实验队列沉淀为可持续迭代的 GitHub 知识库。

## 1. 范围与研究问题

领域范围：
- 液态神经网络（Liquid Neural Networks, LNN）。
- 液态时间常数网络（Liquid Time-Constant, LTC）。
- 闭式连续时间网络（Closed-form Continuous-time, CfC）。
- 神经电路策略（Neural Circuit Policy, NCP）。
- 与 LNN 相关的 Neural ODE、连续时间序列模型、边缘部署和 Liquid Foundation Models（LFM/LFM2）。

核心研究问题：
1. LNN / LTC / CfC 的核心架构和数学机制如何演进？
2. 哪些论文给出了可复现的性能、鲁棒性、参数效率或边缘部署证据？
3. 哪些开源仓库、模型或数据集值得进入 `projects/` 与 `analysis/` 的复现实验队列？
4. LNN 在非平稳时间序列、机器人控制、视觉导航、医疗金融、边缘 AI 和长序列建模中的真实优势与局限是什么？
5. LFM/LFM2 与传统 LNN/CfC/LTC 的关系、可部署性和实验验证路径是什么？

暂不纳入：
- 只讨论普通 RNN/Transformer/Neural ODE 且没有 LNN/LTC/CfC/NCP/LFM 关联的材料。
- 无代码、无数据、无明确实验设置，且与当前研究路线弱相关的低质量仓库。
- 纯营销内容，除非来自核心机构并包含可验证技术细节。

## 2. 方法框架

本项目采用轻量化组合框架：

- **Living Review**：每日或定期搜索，保持快速变化领域的证据更新。
- **PRISMA-lite**：记录数据源、查询式、日期、候选数量、筛选状态和排除原因，不冒充正式系统综述。
- **雪球法（Snowballing）**：从高价值种子论文做 backward/forward citation 扩展，补足关键词搜索漏检。
- **Progressive Summarization**：原始数据 -> 每日摘要 -> 单篇研读报告 -> 概念笔记/对比表 -> 全局综合报告。
- **Zettelkasten / MOC**：只把高复用概念沉淀为原子笔记或导航页，避免把每日流水账伪装成知识。

详细方法维护在 `skills/living-field-researcher/references/frameworks.md`。

## 3. 关键词与查询式

| 组别 | 关键词 / 查询式 | 用途 | 噪声风险 |
|---|---|---|---|
| 核心概念 | `Liquid Neural Networks`, `liquid neural network`, `liquid neural networks` | 捕获 LNN 主线论文与项目 | 低 |
| LTC | `Liquid Time-Constant`, `liquid time constant`, `LTC neural network` | 捕获 LTC 论文、教程、实现 | 中，LTC 有非神经网络歧义 |
| CfC | `Closed-form Continuous-time`, `closed form continuous time`, `CfC neural network` | 捕获 CfC 与连续时间闭式模型 | 中，CfC 缩写有歧义 |
| NCP | `Neural Circuit Policy`, `neural circuit policies`, `NCP liquid` | 捕获神经电路策略和机器人控制 | 中 |
| 相关方法 | `Neural ODE`, `continuous-time RNN`, `liquid structural state-space`, `state-space liquid` | 补充理论和相邻路线 | 高，需二次筛选 |
| 基础模型 | `Liquid AI`, `LFM`, `LFM2`, `liquid foundation model` | 捕获 Liquid AI / LFM 系列模型进展 | 中 |
| 应用场景 | `LNN time series`, `LNN robotics`, `LNN edge`, `LNN forecasting`, `LNN medical`, `LNN finance` | 捕获落地论文与工程项目 | 中高 |

## 4. 数据源与输出

| 数据源 | 当前方式 | 频率 | 输出位置 | 备注 |
|---|---|---|---|---|
| arXiv | `scripts/daily_lnn_research.py` | daily | `papers/daily/`, `docs/daily/` | 主论文候选来源 |
| GitHub | GitHub Search API | daily | `analysis/repo_watchlist/` | 仓库、教程、复现项目 |
| Hugging Face | Models API | daily | `analysis/repo_watchlist/` | LFM/LFM2 与相关模型 |
| Semantic Scholar / OpenAlex | 待扩展 | weekly | `papers/daily/` 或 `analysis/citations/` | 引用、forward snowballing |
| Papers with Code / OpenReview | 待扩展 | weekly/monthly | `docs/daily/` | benchmark 与会议论文 |
| Liquid AI / MIT CSAIL / 项目主页 | 手动或脚本扩展 | weekly | `docs/daily/` | 官方技术报告和发布说明 |

当前已落地入口：
- `scripts/daily_lnn_research.py`
- `scripts/run_daily_lnn_task.sh`
- `.github/workflows/daily-lnn-research.yml`

## 5. 纳入与排除规则

纳入：
- 标题、摘要、模型卡或 README 明确涉及 LNN/LTC/CfC/NCP/LFM。
- 对连续时间建模、非平稳序列、边缘部署、鲁棒控制、可解释动态系统有直接贡献。
- 提供代码、数据、复现实验、benchmark 或清晰数学公式。
- 来自核心作者、核心机构、高引用论文、活跃仓库或重要模型发布。

排除：
- 只有关键词堆砌但内容不相关。
- 仓库长期无人维护、无 README、无 license、无运行入口，且没有明显研究价值。
- 论文仅泛泛使用 RNN/ODE/Transformer，没有 LNN 相关机制或对比。
- 无法追溯来源或缺少基本元数据的内容。

## 6. 评分与优先级

| 信号 | 说明 | 分值 |
|---|---|---:|
| 主题相关性 | 是否命中 LNN/LTC/CfC/NCP/LFM 核心路线 | 0-3 |
| 新颖性 | 是否提出新模型、新公式、新任务、新数据或新结论 | 0-3 |
| 证据强度 | 是否有清晰实验、对比、消融、指标或理论证明 | 0-3 |
| 可复现性 | 是否有代码、数据、配置、环境说明 | 0-2 |
| 影响潜力 | 作者/机构、引用、Star、下载、社区关注 | 0-2 |
| 本仓库契合度 | 是否能转化为报告、代码、Jetson 实验或 benchmark | 0-3 |

优先级标签：
- `read_now`：总分 >= 10，且主题相关性 >= 2。
- `repo_analyze`：仓库或模型可复现性 >= 1，且本仓库契合度 >= 2。
- `experiment`：存在明确指标、数据或实现路径，可落到 `scripts/` 与 `analysis/`。
- `watch`：相关但证据不足，或需要等待代码/论文更新。
- `ignore`：明显不相关、重复、低质量或不可追溯。

## 7. 输出约定

| 产物 | 路径 | 维护规则 |
|---|---|---|
| 每日摘要 | `docs/daily/YYYY-MM-DD_LNN_research_digest.md` | 自动生成，供人工筛选 |
| 原始数据 | `papers/daily/YYYY-MM-DD_lnn_research.json` | 保留数据源、URL、摘要和候选列表 |
| PDF 归档 | `papers/daily/YYYY-MM-DD/` | 仅下载高相关或待精读论文 |
| 单篇研读报告 | `docs/reports/<论文文件名或slug>_研读报告.md` | 使用 `paper-analyzer` 结构 |
| 全局深度索引 | `docs/LNN_深度研读报告.md` | 追加精简版、链接和综合洞察 |
| 开源观察 | `analysis/repo_watchlist/` | 记录仓库、模型、复现成本 |
| 实验结果 | `analysis/` | 记录 benchmark、图表、JSON 和结论 |
| 代码入口 | `scripts/` | 只放可重复执行的抓取、分析、实验脚本 |

## 8. 迭代节奏

每日：
- 运行 `scripts/daily_lnn_research.py` 或 `scripts/run_daily_lnn_task.sh`。
- 检查当天高优先级候选，标记 `read_now` / `repo_analyze` / `experiment`。

每周：
- 从 `docs/daily/` 选择 1-3 个高价值候选生成研读报告或代码调研。
- 检查关键词噪声和漏检，必要时更新本协议与脚本查询式。
- 对新增仓库或模型补充复现成本、依赖栈、Jetson 可行性。

每月：
- 更新 [[LNN_深度研读报告]] 的路线图、方法对比和实验结论。
- 做一次雪球法扩展：从核心论文和近期高价值论文检查引用与被引。
- 汇总新的实验假设，明确下月 `scripts/` 与 `analysis/` 的优先级。

## 9. 当前种子集合

| 类型 | 名称 | 链接/位置 | 重要性 | 后续动作 |
|---|---|---|---|---|
| paper | Liquid Time-Constant Networks | https://arxiv.org/abs/2006.04439 | LTC 主线理论 | 雪球法扩展 |
| paper | Closed-form Continuous-time Neural Models | https://arxiv.org/abs/2106.13898 | CfC 工程效率路线 | 与 LTC/LSTM/GRU 对比 |
| paper | Neural Circuit Policies Enabling Auditable Autonomy | https://www.nature.com/articles/s42256-020-00237-3 | 控制与可解释自治 | 关联机器人控制实验 |
| repo | `raminmh/liquid_time_constant_networks` | https://github.com/raminmh/liquid_time_constant_networks | 经典 LTC 实现 | 代码调研 |
| repo | `mlech26l/ncps` | https://github.com/mlech26l/ncps | PyTorch/Keras LTC/CfC 实现 | 纳入实验依赖评估 |
| model | Liquid AI LFM/LFM2 | https://huggingface.co/docs/transformers/model_doc/lfm2 | 液态基础模型路线 | Jetson 推理与量化验证 |

## 10. Skill 维护规则

当持续研究流程出现可复用改进时：
- 如果是 LNN 专属关键词、数据源或筛选规则，更新本文档或 `scripts/daily_lnn_research.py`。
- 如果是任意领域都适用的工作流、模板或判断标准，更新 `skills/living-field-researcher/`。
- `SKILL.md` 只放高频执行规则；详细框架和模板放入 `references/`，避免技能上下文膨胀。
- 每次更新 skill 后，检查 [[AGENTS]] 和 [[README]] 是否仍准确描述安装方式与用途。
