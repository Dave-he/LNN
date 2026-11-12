---
title: LNN 每日研究追踪 - 2026-11-13
date: 2026-11-13
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-11-13

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：94 篇
- GitHub 候选仓库：45 个
- Hugging Face 候选模型：16 个
- 已下载 PDF：0 个
- 下表每类仅渲染前 12 条；全量候选池见 `papers/daily/2026-11-13_lnn_research.json`，研读选择器以该 JSON 为准。

## arXiv 候选论文
| 日期 | 论文 | 作者 | 摘要 |
|---|---|---|---|
| 2026-11-07 | [Locating and Repairing Domain Shift in VLM Trajectory Planning](https://arxiv.org/abs/2611.10720v1) | Zhihong Cui, Hengyu Liu, Michael Alexander Riegler 等 | Vision-Language Models (VLMs) are increasingly adopted as end-to-end driving planners, yet they suffer from cross-city domain shift. Existing methods treat this shift as a single quantity addressed by uniform alignment or single-site editing. We challenge this view on two counts… |
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

## GitHub 候选仓库
| 更新 | 仓库 | Star | Fork | 语言 | 说明 |
|---|---|---:|---:|---|---|
| 2026-10-17 | [Ipsedo/LiquidNetworks](https://github.com/Ipsedo/LiquidNetworks) | 29 | 6 | Python | Liquid Time-constant Networks implementation with PyTorch |
| 2026-11-11 | [zhangshuibai/molten](https://github.com/zhangshuibai/molten) | 27 | 2 | Metal | Run a coding model at 300+ tok/s on your Mac. Native Rust + Metal inference engine for LiquidAI LFM2.5-24B-A2B (MLX 4-bit) on Apple Silicon. |
| 2026-11-07 | [daivarsi/TyphoonLLM](https://github.com/daivarsi/TyphoonLLM) | 9 | 1 | C++ | Fully local LLM inference on PlayStation Vita using LiquidAI LFM2.5-230M. |
| 2026-10-28 | [paragontasx/ferro](https://github.com/paragontasx/ferro) | 9 | 0 | C++ | Native OpenCL inference engine for Intel N100-class SoCs: EmbeddingGemma-2, LiquidAI d1-omni and LFM2.5 chat in one C++ binary, on the iGPU. Measured ~2x llama… |
| 2026-10-21 | [RESMP-DEV/lfm25-embedding-trainer](https://github.com/RESMP-DEV/lfm25-embedding-trainer) | 7 | 1 | Python | Fine-tune LiquidAI LFM2.5 Embedding models for dense retrieval using your own data |
| 2026-11-11 | [flaviagiammarino/lnn-sagemaker](https://github.com/flaviagiammarino/lnn-sagemaker) | 5 | 0 | Jupyter Notebook | An Amazon SageMaker algorithm for time series forecasting with liquid neural networks (LNNs). |
| 2026-10-21 | [blacklike02/Liquid-Network-Model](https://github.com/blacklike02/Liquid-Network-Model) | 4 | 0 | Python | Liquid Neural Network language model with LTC and CfC dynamics |
| 2025-11-04 | [siddharthk27/Neural-Circuit-Policies](https://github.com/siddharthk27/Neural-Circuit-Policies) | 4 | 0 | Jupyter Notebook | This contains my experiments with Liquid Time Constant Networks (LTCs) and Closed Form Continuous Network (CFCs) architectures. |
| 2026-11-11 | [AwareLiquid/O1](https://github.com/AwareLiquid/O1) | 3 | 2 | Python | MT-LNN — clean prototype of a brain-inspired liquid neural network (constant-memory recurrent state, O(1) generation cache). Early baseline; active development… |
| 2026-10-25 | [AwareLiquid/M1](https://github.com/AwareLiquid/M1) | 2 | 0 | Python | MT-LNN: Microtubule-inspired liquid neural network — bio-inspired LLM architecture with O(1) working memory |
| 2026-10-21 | [julschmidtJJL/lnn-vs-classical-forecasting](https://github.com/julschmidtJJL/lnn-vs-classical-forecasting) | 2 | 0 | Python | Bachelor's thesis: do liquid neural networks beat classical models on daily financial returns? No: 0 of 120 corrected comparisons. 11 LTC/CfC variants vs 8 bas… |
| 2026-07-31 | [middesurya/daily-webapp-2026-05-04-liquidlab](https://github.com/middesurya/daily-webapp-2026-05-04-liquidlab) | 2 | 0 | HTML | LiquidLab — Interactive Liquid Neural Networks & Continuous-Time Models Laboratory. 6 modules: LTC dynamics, Neural ODE, C. elegans connectome, CfC networks, a… |

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-11-12 | [LiquidAI/Liquid-Codec](https://huggingface.co/LiquidAI/Liquid-Codec) | 6227 | 38 | feature-extraction |
| 2026-11-12 | [keitokei1994/LiquidAI-d1-2B-fp16-ov](https://huggingface.co/keitokei1994/LiquidAI-d1-2B-fp16-ov) | 0 | 0 |  |
| 2026-11-12 | [lugman/LFM2.5-350M-Math-Haiku-5.5-Rewrite-SFT](https://huggingface.co/lugman/LFM2.5-350M-Math-Haiku-5.5-Rewrite-SFT) | 0 | 0 | text-generation |
| 2026-11-12 | [lugman/LFM2.5-350M-Math-Haiku-5.5-Rewrite-SFT-adapter](https://huggingface.co/lugman/LFM2.5-350M-Math-Haiku-5.5-Rewrite-SFT-adapter) | 0 | 0 |  |
| 2026-11-12 | [lugman/LFM2.5-350M-Math-SFT](https://huggingface.co/lugman/LFM2.5-350M-Math-SFT) | 0 | 0 | text-generation |
| 2026-11-12 | [lugman/LFM2.5-350M-Math-SFT-adapter](https://huggingface.co/lugman/LFM2.5-350M-Math-SFT-adapter) | 0 | 0 |  |
| 2026-11-12 | [lugman/LFM2.5-350M-Math-Self-Rewrite-SFT](https://huggingface.co/lugman/LFM2.5-350M-Math-Self-Rewrite-SFT) | 0 | 0 | text-generation |
| 2026-11-12 | [lugman/LFM2.5-350M-Math-Self-Rewrite-SFT-adapter](https://huggingface.co/lugman/LFM2.5-350M-Math-Self-Rewrite-SFT-adapter) | 0 | 0 |  |
| 2026-11-12 | [ldjflsajr3/LFM2.5-350M-docqa-GGUF](https://huggingface.co/ldjflsajr3/LFM2.5-350M-docqa-GGUF) | 0 | 0 |  |
| 2026-11-11 | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 26858 | 504 | image-text-to-text |
| 2026-11-11 | [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | 3707 | 359 | text-generation |
| 2026-11-11 | [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) | 3659 | 477 | audio-to-audio |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
