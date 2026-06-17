---
title: LNN 每日研究追踪 - 2026-06-17
date: 2026-06-17
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-06-17

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：25 篇
- GitHub 候选仓库：49 个
- Hugging Face 候选模型：23 个
- 已下载 PDF：0 个

## 数据源状态
- `arXiv fetch failed: The read operation timed out`
- 若当天已有历史结果，脚本会保留上一轮成功获取的数据，避免 transient API 错误清空候选池。

## arXiv 候选论文
| 日期 | 论文 | 作者 | 摘要 |
|---|---|---|---|
| 2026-06-14 | [Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liquid Time-Constant Networks](https://arxiv.org/abs/2606.15807v1) | Jinrong Xiang, Ming Xu | Traffic state prediction is a fundamental task in intelligent transportation systems. In practical applications, some regions suffer from limited traffic observations due to insufficient sensing infrastructure, making cross-domain knowledge transfer an important solution for dat… |
| 2026-06-14 | [Liquid Random Feature Methods for Time-Dependent Partial Differential Equations](https://arxiv.org/abs/2606.15571v1) | Jiale Linghu, Yangshuai Wang | A central challenge in mesh-free space--time approximation for time-dependent partial differential equations is to represent evolving temporal scales while keeping residual minimization computationally tractable. Random feature methods simplify this algebraic problem by freezing… |
| 2026-06-10 | [Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training](https://arxiv.org/abs/2606.12240v1) | Shilong Zong, Almuatazbellah Boker, Hoda Eldardiry | Multivariate time-series data often exhibit complex temporal dependencies, irregular sampling, and heterogeneous dynamics across multiple time scales, making accurate sequence modeling particularly challenging. Traditional recurrent neural networks (RNNs), such as Long Short-Ter… |
| 2026-06-04 | [Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting](https://arxiv.org/abs/2606.07670v1) | Mingzhao Li, Arghya Pal, Guan Yuan Tan | Deformable 3D Gaussian Splatting (D-3DGS) re-constructs dynamic scenes from monocular video by deforming a canonical set of 3D Gaussians through a positional-encoded MLP of frame time t. Although fitted to a continuous variable, the MLP couples no two values of t in its architec… |
| 2026-05-26 | [Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility](https://arxiv.org/abs/2605.27467v1) | Ye Kyaw Thu, Thazin Myint Oo, Thepchai Supnithi | Traditional Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) units operate on discrete time steps, often failing to capture the fluid temporal dynamics of real-world physical processes. Liquid Neural Networks (LNNs), specifically Closed-form Continuous-time (Cf… |
| 2026-05-21 | [EMMA: Extracting Multiple physical parameters from Multimodal Data](https://arxiv.org/abs/2605.24047v1) | Farhat Shaikh, Ayan Banerjee, Sandeep Gupta | We introduce EMMA, a physics-informed multimodal framework that recovers all identifiable dynamical parameters of a system directly from raw video, audio, and image-based time-series observations. Unlike prior video-only approaches that struggle with occluded states, hidden actu… |
| 2026-05-05 | [Physics-Modeled Neural Networks](https://arxiv.org/abs/2605.08176v1) | Raul Felipe-Sosa, Angel Martin del Rey, Maria Flores Ceballos | We introduce \emph{Dynamical Physics-Modeled Neural Networks} (DynPMNNs), a continuous-time deep learning architecture in which each hidden layer is defined as the solution of an ordinary differential equation. Unlike classical feed-forward networks, this approach replaces stati… |
| 2026-04-24 | [Liquid Neural Network Models for Natural Gas Spot Price Time-Series Forecasting](https://arxiv.org/abs/2604.24788v1) | Yiqian Liu, Jiayi Niu, Adam Kelleher 等 | Natural gas is undoubtedly an essential component of the global energy system. Accurate short-term forecasting of natural gas price is challenging due to pronounced volatility driven by seasonal demand patterns, geopolitical developments, and shifting macroeconomic conditions. T… |
| 2026-04-20 | [LiquidTAD: Efficient Temporal Action Detection via Parallel Liquid-Inspired Temporal Relaxation](https://arxiv.org/abs/2604.18274v2) | Zepeng Sun, Naichuan Zheng, Hailun Xia 等 | Temporal Action Detection (TAD) requires precise localization of action boundaries within long, untrimmed video sequences. While current high-performing methods achieve strong accuracy, they are often characterized by excessive parameter counts, substantial computational overhea… |
| 2026-04-15 | [A Nonasymptotic Theory of Gain-Dependent Error Dynamics in Behavior Cloning](https://arxiv.org/abs/2604.14484v2) | Junghoon Seo | Behavior cloning (BC) policies on position-controlled robots inherit the closed-loop response of the underlying PD controller, yet the nonasymptotic finite-horizon consequences of controller gains for BC failure remain open. We show that independent sub-Gaussian action errors pr… |
| 2026-04-12 | [MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation](https://arxiv.org/abs/2604.10815v2) | Hongwei Xu | MeloTune is an iPhone-deployed music agent that instantiates the Mesh Memory Protocol (MMP) and Symbolic-Vector Attention Fusion (SVAF) as a production system for affect-aware music curation with peer-to-peer mood coupling. Each device runs two closed-form continuous-time (CfC)… |
| 2026-04-08 | [Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks](https://arxiv.org/abs/2604.07219v1) | Xinquan Wang, Mingjun Ying, Hongren Chen 等 | Sub-terahertz (sub-THz) multi-user multiple-input multiple-output (MU-MIMO) systems unlock immense bandwidth for 6G wireless communications. However, practical deployment of wireless systems in sub-THz bands faces critical challenges such as increased atmospheric absorption, red… |

## GitHub 候选仓库
| 更新 | 仓库 | Star | 语言 | 说明 |
|---|---|---:|---|---|
| 2026-06-16 | [maximecb/bebelm](https://github.com/maximecb/bebelm) | 55 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-06-16 | [Sum-Outman/Self-LNN](https://github.com/Sum-Outman/Self-LNN) | 1 | C | Self AGI System（Self AGI robot System）.自主通用人工智能系统（自主通用人工智能机器人系统）。 AI capable of perceiving the real world。能够感知真实世界的人工智能。 Liquid Neural Network, C language vers… |
| 2026-06-16 | [KaiserDna23/RL_Trading](https://github.com/KaiserDna23/RL_Trading) | 1 | Python | Forex trading using Reinforcement Learning with Meta Trader 5 environment. This will use algorithms such as Actor Critic(A2C) Network with Proximal Policy Opti… |
| 2026-06-16 | [Gonzablanmar/Liquid-NN](https://github.com/Gonzablanmar/Liquid-NN) | 0 | Jupyter Notebook | Liquid neural network, where the Adam optimizer is implanted by hand. |
| 2026-06-16 | [heimdilon/sncp-ppo-crowdnav](https://github.com/heimdilon/sncp-ppo-crowdnav) | 0 | Python | PPO + LTC (Liquid Time Constant) crowd-aware navigation for TurtleBot3 Waffle. 5-phase curriculum, multi-scenario holdout, clipped value loss. Includes Colab n… |
| 2026-06-15 | [404reese/XWormNet](https://github.com/404reese/XWormNet) | 0 | Python | Explainable Liquid Neural Network Framework for Real-Time Zero-Day Worm Detection in IoT and Enterprise Networks |
| 2026-06-15 | [aliobaidbt/Liquid-neural-networks-for-adaptive-perception-and-control-in-autonomous-drone-navigation](https://github.com/aliobaidbt/Liquid-neural-networks-for-adaptive-perception-and-control-in-autonomous-drone-navigation) | 0 |  |  |
| 2026-06-14 | [raminmh/CfC](https://github.com/raminmh/CfC) | 1048 | Python | Closed-form Continuous-time Neural Networks |
| 2026-06-10 | [Dhivya-DD17/DLNet](https://github.com/Dhivya-DD17/DLNet) | 1 | Jupyter Notebook | This repo is the official implementation of the paper "When Smaller Wins: Dual-Stage Distillation and Pareto-Guided Compression of Liquid Neural Networks for E… |
| 2026-06-10 | [ochigenuka/FYP](https://github.com/ochigenuka/FYP) | 1 | Jupyter Notebook | Solar Energy Forecasting using Liquid Time-Constant Network |
| 2026-06-10 | [ochigenuka/FYP2](https://github.com/ochigenuka/FYP2) | 0 |  | Solar Energy Forecasting using Liquid Time-Constant Networks |
| 2026-06-08 | [g023/cuda_inf](https://github.com/g023/cuda_inf) | 1 | Cuda | A self-contained CUDA inference engine for LiquidAI/LFM2.5-8B-A1B (hybrid conv + GQA-attention MoE, 8.5B params, 1B active) targeting a single RTX 3060 (12 GB)… |

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-06-17 | [LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters](https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters) | 0 | 1 |  |
| 2026-06-17 | [LLM-OS-Models/LFM2.5-8B-A1B-Raw-ECHO-RLVR-GRPO-Adapters](https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-Raw-ECHO-RLVR-GRPO-Adapters) | 0 | 1 |  |
| 2026-06-16 | [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 113327 | 608 | text-generation |
| 2026-06-16 | [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 45406 | 139 | sentence-similarity |
| 2026-06-16 | [reaperdoesntknow/LFM2.5-1.2B-Distilled-SFT](https://huggingface.co/reaperdoesntknow/LFM2.5-1.2B-Distilled-SFT) | 383 | 0 | text-generation |
| 2026-06-16 | [reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil](https://huggingface.co/reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil) | 176 | 5 | text-generation |
| 2026-06-16 | [mkurman/LiquidAI-LFM2.5-350M-SYNTH](https://huggingface.co/mkurman/LiquidAI-LFM2.5-350M-SYNTH) | 29 | 1 | text-generation |
| 2026-06-16 | [Cactus-Compute/LFM2-VL-450M](https://huggingface.co/Cactus-Compute/LFM2-VL-450M) | 10 | 1 | image-text-to-text |
| 2026-06-16 | [mkurman/LiquidAI-LFM2.5-350M-SYNTH-GGUF](https://huggingface.co/mkurman/LiquidAI-LFM2.5-350M-SYNTH-GGUF) | 0 | 0 |  |
| 2026-06-16 | [skylord/lfm2-1.2b-kcc-sample100k-gguf](https://huggingface.co/skylord/lfm2-1.2b-kcc-sample100k-gguf) | 0 | 0 |  |
| 2026-06-16 | [skylord/lfm2-1.2b-kcc-sample100k](https://huggingface.co/skylord/lfm2-1.2b-kcc-sample100k) | 0 | 0 |  |
| 2026-06-16 | [aaronrockmenezes/lfm25-1.2b-k2tools-lora](https://huggingface.co/aaronrockmenezes/lfm25-1.2b-k2tools-lora) | 0 | 0 | text-generation |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 人工研判更新

复核时间：2026-06-17 11:19 北京时间 (UTC+8)。

数据口径：
- 论文池沿用 2026-06-17 06:32 北京时间 (UTC+8) 成功抓取的 arXiv 快照；11:18-11:19 北京时间 (UTC+8) 连续两次重试 arXiv API 均超时，已保留在 `papers/daily/2026-06-17_lnn_research.json` 的 `errors` 字段中。
- GitHub / Hugging Face 已在 11:19 北京时间 (UTC+8) 刷新，候选规模从早间的 41 / 21 更新为 49 / 23。
- 原始机器数据：`papers/daily/2026-06-17_lnn_research.json`。
- 开源观察表：`analysis/repo_watchlist/2026-06-17_lnn_open_source_watchlist.md`。

### 今日结论

1. **MA-GLTC 是今日最强 `read_now` 项**：它把 LTC 的 adaptive time constant 扩展到图结构，核心不只是 message passing，而是把邻居反馈注入 $\tau_{\text{eff}}$，对本仓的 `LTCNetwork` / `CfCCell` / `moe_ecology` 都有直接扩展价值。
2. **L-RFM 是相关但不应归入 LNN 主线的 `watch` 项**：论文使用 liquid temporal response / relaxation scales 做 PDE 随机特征，不是 Liquid Neural Network / LTC / CfC。可作为“连续时间基函数/解析导数/least-squares surrogate”的相邻数学线索保留。
3. **开源生态今天的重点从论文转向 LFM2.5 部署**：`maximecb/bebelm`、`g023/cuda_inf`、LFM2.5 GGUF / ONNX / MLX / ColBERT 生态都指向一个共同趋势：Liquid AI 路线正在从模型发布进入边缘推理与本地 RAG 工程化。
4. **高 Star 旧仓库仍要保留作基准**：`raminmh/CfC` 今天被搜索刷新捕获，虽然不是新项目，但 Star 最高，应该继续作为 CfC 复现与 API 对照的参考基线，而不是按“旧”过滤掉。

### 候选分级

| 候选 | 类型 | 状态 | 理由 | 下一步 |
|---|---|---|---|---|
| [MA-GLTC](https://arxiv.org/abs/2606.15807) | paper | `read_now` / `experiment` | 直接提出 Graph LTC + MTS，主题相关性高；arXiv 摘要报告 5 个公开交通数据集均优于代表性域内/跨域基线。 | 已生成 [[docs/reports/MA-GLTC_Graph_Liquid_Time_Constant_Cross_Domain_Traffic_2606.15807_研读报告.md]]；下一步读 PDF 全文补 ablation 与延迟。 |
| [Liquid Random Feature Methods](https://arxiv.org/abs/2606.15571) | paper | `watch` | 命中 liquid，但主体是 PDE 随机特征与松弛尺度，不是 LNN/LTC/CfC；可为连续时间 surrogate 提供数学借鉴。 | 暂不生成 LNN 研读报告；周度整理时放入“相邻方法”小节。 |
| [Multi-Rate MoE for LNN Training](https://arxiv.org/abs/2606.12240) | paper | `experiment` | 直指多时间尺度 LNN 训练加速，和本仓 MoE / router 资产强相关。 | 复查 [[docs/reports/Multi-Rate_MoE_Accelerating_LNN_Training_2606.12240_研读报告.md]] 后设计 MR-MoE vs CfC / LTC 微基准。 |
| [LNN as 3DGS Deformation Field](https://arxiv.org/abs/2606.07670) | paper | `watch` / `experiment` | 连续时间形变场是 LNN 的新应用面，但本仓当前缺 3DGS 数据管线。 | 保留报告链接；暂不进入近期代码实现。 |
| [maximecb/bebelm](https://github.com/maximecb/bebelm) | repo | `repo_analyze` | Rust CPU-only LFM2.5-8B-A1B Q4_K_M，本地推理链路清晰，适合作边缘推理工程调研。 | 建议输出 `analysis/repo_watchlist/2026-06-17_bebelm_repo_notes.md`，记录构建、权重、吞吐、内存。 |
| [g023/cuda_inf](https://github.com/g023/cuda_inf) | repo | `repo_analyze` | 单 `.cu` LFM2.5-8B-A1B 推理实现，适合和 Rust CPU 线形成 GPU/CPU 对照。 | 若有 CUDA 环境，测 tokens/s、显存峰值、prefill/decode 分离。 |
| [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | model | `experiment` | 1.2B 指令模型，模型卡标注 32k 上下文、GGUF/ONNX/MLX 等部署格式，符合边缘验证路线。 | 复用 `analysis/llm_micro_eval/` 模板，补 2026-06-17 版本地吞吐与 RAG/agentic 小测。 |
| [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | model | `experiment` | 350M late-interaction retriever，适合把 `docs/reports/` 做本地多语 RAG 索引。 | 建议新增 `analysis/lfm_retrieval/2026-06-17_colbert_docs_rag.md`，指标用 NDCG@10、索引大小、查询延迟。 |

### 已完成深读与索引

- 已完成 MA-GLTC 深读：[[docs/reports/MA-GLTC_Graph_Liquid_Time_Constant_Cross_Domain_Traffic_2606.15807_研读报告.md]]。
- 已完成 SVAF 补充深读：[[docs/reports/SVAF_Symbolic_Vector_Attention_Fusion_Collective_Intelligence_2604.03955_研读报告.md]]。
- 两篇已追加到全局索引：[[docs/LNN_深度研读报告.md]]。
- 今日自动追踪条目也已追加到 `docs/Liquid_Neural_Networks_Latest_Papers_Summary.md` 与 [[docs/LNN_深度研读报告.md]] 的自动化队列区。

### 下一步实验队列

| 优先级 | 实验 | 目标 | 指标 | 输出路径 |
|---|---|---|---|---|
| P0 | Graph-LTC synthetic benchmark | 验证 graph-coupled $\tau$ 是否在跨图/稀疏观测下优于 GCN-LSTM、DCRNN-lite、普通 LTC。 | MSE、MAE、推理延迟、参数量、$\tau_{\text{eff}}$ 分布。 | `analysis/graph_ltc/2026-06-17_gltc_synthetic_benchmark.md` |
| P1 | LFM2.5 本地推理对照 | 对比 `bebelm` CPU、GGUF llama.cpp、CUDA 单文件路线的部署成本。 | tokens/s、prefill/decode 拆分、内存/显存峰值、构建复杂度。 | `analysis/lfm_edge/2026-06-17_lfm25_edge_inference.md` |
| P1 | LFM2-ColBERT 文档检索 | 用 350M retriever 索引本仓 `docs/reports/`，验证 LNN 知识库 RAG 可用性。 | NDCG@10、Recall@5、查询延迟、索引大小。 | `analysis/lfm_retrieval/2026-06-17_colbert_docs_rag.md` |
| P2 | SVAF band-pass 4-outcome | 把 per-field gate 与 CfC Layer 6 接到多源时序融合任务。 | aligned / guarded / rejected precision-recall、OOD 拒绝率、误拒率。 | `analysis/svaf/2026-06-17_band_pass_gate.md` |

### 外部复核来源

- arXiv: [MA-GLTC / arXiv:2606.15807](https://arxiv.org/abs/2606.15807)。
- arXiv: [Liquid Random Feature Methods / arXiv:2606.15571](https://arxiv.org/abs/2606.15571)。
- GitHub: [maximecb/bebelm](https://github.com/maximecb/bebelm)。
- Hugging Face: [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)。
- Hugging Face: [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M)。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
