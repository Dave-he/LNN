---
title: LNN 每日研究追踪 - 2026-09-27
date: 2026-09-27
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-09-27

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：0 篇
- GitHub 候选仓库：41 个
- Hugging Face 候选模型：19 个
- 已下载 PDF：0 个

## 数据源状态
- `arXiv fetch failed: HTTP Error 406: Not Acceptable`
- 若当天已有历史结果，脚本会保留上一轮成功获取的数据，避免 transient API 错误清空候选池。

## arXiv 候选论文

> 备注：今日 `scripts/daily_lnn_research.py` 的 arXiv fetch 在 urllib 路径上仍触发 `HTTP Error 406: Not Acceptable`（即便 9-26 已将 `request_text` 的 Accept 头改为 `application/atom+xml`，本日 cron 进程仍出现 406，疑似与 urllib 在 9-term OR 查询上偶尔触发的 Varnish 节流相关，**根因待复盘**，本日内已通过 curl 旁路把 50 条原始候选拉到 `/tmp/arxiv_2026-09-27.xml`，经 `_parse_arxiv_fallback.py` 解析后保留 48 条 LNN/CfC/LTC/NCP/closed-form-continuous-time 相关候选。下表 = 经过 keyword_score>0 + STRONG_KEYWORDS 过滤后被人工复核、且在过去 30 天内提交/更新的命中论文。前两条同时**已被 2026-09-27 cron 阶段消化**（前者已存在于 `docs/reports/`，后者为本轮新增研读）。

| 更新 | 论文 | 作者 | 摘要 (truncated) |
|---|---|---|---|
| 2026-09-23 | [2609.28716v1](https://arxiv.org/abs/2609.28716) — Temporal Learning for End-Effector Position Estimation under Aerodynamic Disturbances in Aerial Continuum Manipulation | Niloufar Amiri, Houman Masnavi, Farrokh Janabi-Sharifi | 在 free-hovering UAV 条件下，首次用 Closed-form Continuous-time (CfC) 神经网络代替/补偿 strain-parameterized Cosserat rod 名义模型，对 3D 末端位置残差做估计；MLP/GRU/CfC 同基准五 seed 对比，CfC 取得 22.00±1.70 mm vs MLP 36.38±3.58 / GRU 27.72±2.92 mm。 |
| 2026-08-27 | [2608.28702v1](https://arxiv.org/abs/2608.28702) — Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-Form Continuous-Time Cells for Dynamic 3D Gaussian Splatting | Mingzhao Li, Arghya Pal | 在 Dynamic 3D Gaussian Splatting 框架中把 CfC 形变场推广为带 Gaussian 噪声的 SDE 形式，仅在训练期使用噪声、推理免求解器；D-NeRF 合成场景与确定性 CfC 持平并胜 MLP，NeRF-DS 真实场景下噪声不增益。**已被 2026-09-02 研读（见 [[Stochastic_Liquid_Deformation_Fields_SDE_CfC_2608.28702_研读报告]]），并已被 [[docs/research/2026-09-24_technical_route_landscape]] §B "SDE 概率论重解释" 路线标为诚实负结果 (r306)** |

> 完整 48 条候选见 `papers/daily/_arxiv_fallback_2026-09-27.json`（由 `_parse_arxiv_fallback.py` 生成）。本日不再单条列出，避免冗长。

## GitHub 候选仓库
| 更新 | 仓库 | Star | 语言 | 说明 |
|---|---|---:|---|---|
| 2026-09-26 | [tobert/lfm2d](https://github.com/tobert/lfm2d) | 0 | Rust | A System 1 service around LiquidAI's LFM2.5 suite on candle: encoder heads plus an opinion engine |
| 2026-09-26 | [0kqnet/LFM.mcfunction](https://github.com/0kqnet/LFM.mcfunction) | 0 | Python | Running LiquidAI/LFM2.5-1.2B-JP-202606 on pure vanilla minecraft |
| 2026-09-25 | [kds1123001/liquid-time-constant](https://github.com/kds1123001/liquid-time-constant) | 2 |  | Mojo-native Liquid Time-Constant neural network for edge robotics. Hand-built SIMD RK4 adaptive ODE solver, cache-tiled Struct-of-Arrays state, and lock-free p… |
| 2026-09-25 | [Isobel2026/liquid-minds](https://github.com/Isobel2026/liquid-minds) | 0 |  | Research into liquid neural networks, weight plasticity, and continuous-time architectures |
| 2026-09-25 | [PrithiveenKumaarRamkumar/lfm_case_study](https://github.com/PrithiveenKumaarRamkumar/lfm_case_study) | 0 | Python | LiquidAI LFM2.5 D-Spark Models case study on edge deployment |
| 2026-09-24 | [QilinLi147/LiquidFocus](https://github.com/QilinLi147/LiquidFocus) | 0 | Python | Liquid neural networks for EEG emotion recognition with dual-timescale fusion and prediction-decoupled hierarchical evidence localisation. |
| 2026-09-23 | [api-evangelist/liquid-ai](https://github.com/api-evangelist/liquid-ai) | 1 |  | Liquid AI — independent third-party profile of a public API surface, by API Evangelist. Liquid AI is an MIT spinoff developing Liquid Foundation Models (LFMs)… |
| 2026-09-23 | [NovaResearch9022/Fuzzy-LNN-Speech-Emotion-Recognition](https://github.com/NovaResearch9022/Fuzzy-LNN-Speech-Emotion-Recognition) | 0 | Python | Implementation and experiments for the Fuzzy Liquid Neural Network framework for Speech Emotion Recognition |
| 2026-09-22 | [Think520change/gb-lnn](https://github.com/Think520change/gb-lnn) | 0 |  | To address these issues, a Multi-Scale Granular-Ball Liquid Neural Network (GB-LNN) is proposed as a common representation and temporal-modelling framework. |
| 2026-09-20 | [AlexanderRumyantcev/LNN-LowLight](https://github.com/AlexanderRumyantcev/LNN-LowLight) | 0 | Python | Liquid Neural Networks (CfC) for low-light video enhancement on top of a RetinexFormer pipeline. |
| 2026-09-20 | [m-swetanjali/Edge-AI-in-LNN](https://github.com/m-swetanjali/Edge-AI-in-LNN) | 0 | Jupyter Notebook | Edge AI for Cardiovascular Disease using Liquid Neural Network |
| 2026-09-18 | [AwareLiquid/M1](https://github.com/AwareLiquid/M1) | 1 | Python | MT-LNN: Microtubule-inspired liquid neural network — bio-inspired LLM architecture with O(1) working memory |

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-09-26 | [ryugyosoft/LFM2-8B-A1B-onw](https://huggingface.co/ryugyosoft/LFM2-8B-A1B-onw) | 1315 | 0 | text-generation |
| 2026-09-26 | [blaj/LFM2.5-8B-A1B-heretic-int4-ov](https://huggingface.co/blaj/LFM2.5-8B-A1B-heretic-int4-ov) | 103 | 0 | text-generation |
| 2026-09-26 | [blaj/LFM2.5-8B-A1B-heretic-int8-ov](https://huggingface.co/blaj/LFM2.5-8B-A1B-heretic-int8-ov) | 98 | 0 | text-generation |
| 2026-09-26 | [blaj/LFM2.5-2.6B-heretic-int4-ov](https://huggingface.co/blaj/LFM2.5-2.6B-heretic-int4-ov) | 92 | 0 | text-generation |
| 2026-09-26 | [blaj/LFM2.5-2.6B-heretic-int8-ov](https://huggingface.co/blaj/LFM2.5-2.6B-heretic-int8-ov) | 81 | 0 | text-generation |
| 2026-09-26 | [Panga-Azazia/LFM2.5-350M-TTS-v3](https://huggingface.co/Panga-Azazia/LFM2.5-350M-TTS-v3) | 0 | 0 | text-generation |
| 2026-09-26 | [thealper2/lfm2-350m-2048-lora](https://huggingface.co/thealper2/lfm2-350m-2048-lora) | 0 | 0 | text-generation |
| 2026-09-26 | [nodcai/turn-LFM2.5-2.6B-GGUF](https://huggingface.co/nodcai/turn-LFM2.5-2.6B-GGUF) | 0 | 0 |  |
| 2026-09-26 | [1bit-MONSTER/LFM2.5-8B-A1B-GGUF](https://huggingface.co/1bit-MONSTER/LFM2.5-8B-A1B-GGUF) | 0 | 0 |  |
| 2026-09-26 | [1bit-MONSTER/LFM2.5-2.6B-GGUF](https://huggingface.co/1bit-MONSTER/LFM2.5-2.6B-GGUF) | 0 | 0 |  |
| 2026-09-24 | [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) | 26286 | 213 | image-text-to-text |
| 2026-09-24 | [LiquidAI/LFM2.5-VL-3B-DSpark-GGUF](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark-GGUF) | 433 | 9 | image-text-to-text |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
