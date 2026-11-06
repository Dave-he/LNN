---
title: LNN 每日研究追踪 - 2026-11-06
date: 2026-11-06
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-11-06

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：94 篇
- GitHub 候选仓库：24 个 ⚠️（5 条查询失败，本次覆盖不全，计数偏少）
- Hugging Face 候选模型：16 个
- 已下载 PDF：0 个
- 下表每类仅渲染前 12 条；全量候选池见 `papers/daily/2026-11-06_lnn_research.json`，研读选择器以该 JSON 为准。

## 数据源状态
- ⚠️ 以下是**抓取失败**记录。对应来源的候选池本次**不完整**，其空结果或偏少的计数**不得读作「领域没有新条目」**（AGENTS.md 约束 #7）。
- `GitHub query failed for '"liquid neural network"': Remote end closed connection without response`
- `GitHub query failed for '"liquid time constant"': <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1082)>`
- `GitHub query failed for '"neural circuit policy"': <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1082)>`
- `GitHub query failed for 'LTC CfC neural network': IncompleteRead(0 bytes read)`
- `GitHub query failed for 'LiquidAI LFM2': Remote end closed connection without response`
- 若当天已有历史结果，脚本会保留上一轮成功获取的数据，避免 transient API 错误清空候选池。

## arXiv 候选论文
| 日期 | 论文 | 作者 | 摘要 |
|---|---|---|---|
| 2026-11-03 | [LINDA: Low Resource Foot Ulcer Segmentation using Local Similarity and Iterative Neural Dynamics](https://arxiv.org/abs/2611.04233v1) | Trishaani Acharjee, Sudip Chakrabarty, Rajdeep Chatterjee 等 | Foot ulcers are a leading cause of lower limb amputation, and the highest burden often falls on low and middle income regions where specialists and computing hardware are both scarce. Wound monitoring tools are most useful there when they are accurate yet small enough to run on… |
| 2026-10-22 | [Query-Conditioned Liquid Temporal Grounding for Long-Form Audio](https://arxiv.org/abs/2610.31252v1) | Yuelan Cheng, Yonggang Zhu, Yongqiang Chen 等 | Long-form temporal audio grounding (TAG) is challenging because target events often occupy only a small portion of a recording, while queries may impose multiple semantic constraints over a large temporal search space. We propose a query-conditioned liquid temporal grounding fra… |
| 2026-10-08 | [ContiLNN: Mitigating Slice Sampling Discontinuity with Liquid Neural Networks for Medical Image Restoration](https://arxiv.org/abs/2610.12337v1) | Jialei He, Enhe Liu, Sifan Song 等 | Anatomical continuity provides complementary information for medical image restoration, but its use requires accounting for local anatomy and variations in slice sampling. We introduce ContiLNN, which augments two-dimensional restoration backbones with bidirectional closed-form… |
| 2026-09-27 | [MTLiquid: Enabling Efficient Multi-Task Learning using Liquid Neural Networks for Lightweight Healthcare Monitoring Systems](https://arxiv.org/abs/2609.33232v1) | Rachmad Vidya Wicaksana Putra, Fahad Abdul Rauf, Muhammad Shafique | Continuous-time sensing and monitoring with timely and accurate decision-making are critical for many real-world applications. In healthcare monitoring systems, physiological signals are often available or sampled at irregular time intervals, hence requiring continuous-time proc… |
| 2026-08-27 | [Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-Form Continuous-Time Cells for Dynamic 3D Gaussian Splatting](https://arxiv.org/abs/2608.28702v1) | Mingzhao Li, Arghya Pal | Deformable 3D Gaussian Splatting (D-3DGS) reconstructs dynamic scenes by deforming a canonical set of 3D Gaussians through a deformation field of frame time. Replacing its MLP with a stack of Closed-form Continuous-time (CfC) cells-a Liquid Neural Network that solves the Liquid… |
| 2026-08-13 | [History-informed Lagrangian Neural Networks](https://arxiv.org/abs/2608.13215v1) | Tianshuo Zhang, Xianglei Xing, Wenzhe Zhai 等 | Forecasting the long-horizon evolution of mechanical systems from position-only observations is a pivotal yet difficult task, as hidden velocities and trajectory-specific physical properties must be inferred simultaneously. Although physics-guided neural networks like Lagrangian… |
| 2026-08-04 | [PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling](https://arxiv.org/abs/2608.03041v1) | Dhivya Dharshini Kannan, Wei Zhang, Jieyi Bi 等 | Deep reinforcement learning (DRL) approaches for flexible job shop scheduling (FJSP) heavily rely on attention-centric architectures to achieve state-of-the-art performance. However, these models suffer from excessive parameter counts and prohibitive inference latency as problem… |
| 2026-07-16 | [In-Place Tokenizer Expansion for Pre-trained LLMs](https://arxiv.org/abs/2607.15232v1) | Jimmy T. H. Smith, Tarek Dakhran, Alberto Cabrera 等 | A tokenizer fixed at the start of pre-training allocates vocabulary in proportion to the pre-training corpus, reflecting the deployment priorities at that time. When those priorities shift, languages added later are split into many more tokens per word, which can raise latency,… |
| 2026-07-14 | [Real-time fall detection based on vision for low-power edge platforms](https://arxiv.org/abs/2607.12909v1) | Wenjun Xia, Zhicheng Peng, Haopeng Li 等 | Falling detection is vital for elderly care and intelligent surveillance; however, prevailing vision-based approaches predominantly frame it as static pose classification or discrete temporal pattern matching, fundamentally overlooking the instability dynamics of the human suppo… |
| 2026-07-09 | [TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning](https://arxiv.org/abs/2607.08283v3) | Yushen Liang, Yue Peng, Baosheng Jin 等 | Vision--Language--Action (VLA) policies such as $π_{0.5}$ and OpenVLA perform well on many manipulation tasks, but they are often reactive: the next action is predicted from the current observation, instruction, and proprioceptive state. This assumption breaks down in stage-depe… |
| 2026-07-02 | [Liquid Latent State Dynamics for Interpretable Turbofan Degradation Modeling](https://arxiv.org/abs/2607.01986v1) | Weizhi Nie, Weijie Wang, Yuting Su | Multivariate time-series models for prognostics are often evaluated by point prediction accuracy, yet their internal states rarely expose a coherent degradation process. We study liquid neural networks as latent dynamics models for aircraft engine health monitoring on the C-MAPS… |
| 2026-06-27 | [LNN-Fly: Continuous-Time UAV Navigation for Robust Obstacle Avoidance under Timing Mismatch](https://arxiv.org/abs/2606.28827v1) | Yulin Huang, Shaojie Chen, Di Feng 等 | End-to-end unmanned aerial vehicle (UAV) navigation can achieve impressive agility in simulation, yet its obstacle-avoidance behavior often degrades after deployment because the policy must tolerate simulator mismatch, sensing irregularity, and variable-rate control. These effec… |

## GitHub 候选仓库
| 更新 | 仓库 | Star | Fork | 语言 | 说明 |
|---|---|---:|---:|---|---|
| 2026-10-17 | [Ipsedo/LiquidNetworks](https://github.com/Ipsedo/LiquidNetworks) | 29 | 6 | Python | Liquid Time-constant Networks implementation with PyTorch |
| 2026-10-21 | [blacklike02/Liquid-Network-Model](https://github.com/blacklike02/Liquid-Network-Model) | 4 | 0 | Python | Liquid Neural Network language model with LTC and CfC dynamics |
| 2026-10-25 | [AwareLiquid/M1](https://github.com/AwareLiquid/M1) | 2 | 0 | Python | MT-LNN: Microtubule-inspired liquid neural network — bio-inspired LLM architecture with O(1) working memory |
| 2026-10-21 | [julschmidtJJL/lnn-vs-classical-forecasting](https://github.com/julschmidtJJL/lnn-vs-classical-forecasting) | 2 | 0 | Python | Bachelor's thesis: do liquid neural networks beat classical models on daily financial returns? No: 0 of 120 corrected comparisons. 11 LTC/CfC variants vs 8 bas… |
| 2026-10-30 | [api-evangelist/liquid-ai](https://github.com/api-evangelist/liquid-ai) | 1 | 0 |  | Liquid AI — independent third-party profile of a public API surface, by API Evangelist. Liquid AI is an MIT spinoff developing Liquid Foundation Models (LFMs)… |
| 2026-10-22 | [niharmaddina/Liquidity](https://github.com/niharmaddina/Liquidity) | 1 | 0 | Jupyter Notebook | Next-day stock price prediction with a liquid neural network and a neural ODE (torchdiffeq), served by FastAPI over a websocket to a React chart. Research prot… |
| 2026-10-13 | [AlanL182/Closed-Form-Continuous-Time-Network-Test-On-DB2-sEMG-data](https://github.com/AlanL182/Closed-Form-Continuous-Time-Network-Test-On-DB2-sEMG-data) | 1 | 0 | C |  |
| 2026-10-10 | [kds1123001/liquid-time-constant](https://github.com/kds1123001/liquid-time-constant) | 1 | 0 |  | Mojo-native Liquid Time-Constant neural network for edge robotics. Hand-built SIMD RK4 adaptive ODE solver, cache-tiled Struct-of-Arrays state, and lock-free p… |
| 2026-10-29 | [Farhan-Rashid2024/Liquid-Neural-Networks-for-AI-Based-CPU-Scheduling](https://github.com/Farhan-Rashid2024/Liquid-Neural-Networks-for-AI-Based-CPU-Scheduling) | 0 | 0 | Python |  |
| 2026-10-25 | [ertakay10/DoctorLNN](https://github.com/ertakay10/DoctorLNN) | 0 | 0 | Python | Patient-independent EEG seizure prediction with Liquid Neural Networks on CHB-MIT. LOSO evaluation with few-shot per-patient calibration: AUC 0.74 vs 0.63 (Log… |
| 2026-10-24 | [Abdulrahiim/liquid-lab](https://github.com/Abdulrahiim/liquid-lab) | 0 | 0 | HTML | Liquid Lab: live demo for our AI flipped classroom on Liquid Neural Networks |
| 2026-10-21 | [BuczynskiRafal/ltc-hydrology-flood-prediction](https://github.com/BuczynskiRafal/ltc-hydrology-flood-prediction) | 0 | 0 | Python | Official PyTorch implementation & benchmark of Liquid Time-Constant Networks (LNN) for urban drainage water level forecasting (MDPI Water 2026). |

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-11-06 | [LiquidAI/LFM2.5-24B-A2B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) | 256903 | 249 | text-generation |
| 2026-11-06 | [wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF](https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF) | 1249 | 0 | text-generation |
| 2026-11-06 | [blaj/LFM2.5-24B-A2B-Uncensored-int8-ov](https://huggingface.co/blaj/LFM2.5-24B-A2B-Uncensored-int8-ov) | 10 | 0 | text-generation |
| 2026-11-06 | [vvsotnikov/LFM2.5-2.6B-junie-harbor-rl-v0.2](https://huggingface.co/vvsotnikov/LFM2.5-2.6B-junie-harbor-rl-v0.2) | 0 | 0 | reinforcement-learning |
| 2026-11-05 | [mradermacher/LFM2.5-24B-A2B-Uncensored-BF16-i1-GGUF](https://huggingface.co/mradermacher/LFM2.5-24B-A2B-Uncensored-BF16-i1-GGUF) | 5080 | 1 |  |
| 2026-11-05 | [LiquidAI/LFM3-1.7B-NVFP4](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4) | 3633 | 17 | text-generation |
| 2026-11-05 | [hermitdave/LFM2.5-24B-A2B-MLX-DWQ-3bit](https://huggingface.co/hermitdave/LFM2.5-24B-A2B-MLX-DWQ-3bit) | 2371 | 0 | text-generation |
| 2026-11-05 | [LiquidAI/LFM3-1.7B-FP8](https://huggingface.co/LiquidAI/LFM3-1.7B-FP8) | 2265 | 12 | text-generation |
| 2026-11-05 | [LiquidAI/LFM3-VL-2B-FP8](https://huggingface.co/LiquidAI/LFM3-VL-2B-FP8) | 965 | 10 | image-text-to-text |
| 2026-11-05 | [LiquidAI/d1-Omni-2B-MLX-8bit](https://huggingface.co/LiquidAI/d1-Omni-2B-MLX-8bit) | 861 | 10 | audio-text-to-text |
| 2026-11-05 | [LiquidAI/LFM3-VL-2B-NVFP4](https://huggingface.co/LiquidAI/LFM3-VL-2B-NVFP4) | 851 | 15 | image-text-to-text |
| 2026-11-05 | [LiquidAI/d1-Omni-2B-MLX-fp16](https://huggingface.co/LiquidAI/d1-Omni-2B-MLX-fp16) | 763 | 9 | audio-text-to-text |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
