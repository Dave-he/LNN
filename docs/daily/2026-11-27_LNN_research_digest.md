---
title: LNN 每日研究追踪 - 2026-11-27
date: 2026-11-27
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-11-27

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：98 篇
- GitHub 候选仓库：46 个
- Hugging Face 候选模型：7 个
- 已下载 PDF：0 个
- 下表每类仅渲染前 12 条；全量候选池见 `papers/daily/2026-11-27_lnn_research.json`，研读选择器以该 JSON 为准。

## arXiv 候选论文
| 日期 | 论文 | 作者 | 摘要 |
|---|---|---|---|
| 2026-11-25 | [DiCALM: Converting Autoregressive Convolution-Attention Language Models into Discrete Diffusion Language Models](https://arxiv.org/abs/2611.33919v1) | Phong Le, Ngoc-Quan Pham, Dat Quoc Nguyen | Existing methods for adapting autoregressive (AR) language models to diffusion generation focus mainly on full-attention architectures. More efficient hybrid architectures remain underexplored. We introduce DiCALM, a recipe for converting the hybrid convolution-attention models… |
| 2026-11-20 | [Decision models for UAVBench and the viability of edge deployment](https://arxiv.org/abs/2611.28331v1) | Matthew Michelson, Goran Muric, Adrien Bibal 等 | Small language models are often proposed for reasoning onboard a drone. However, a good benchmark score does not tell us whether a model will actually run, and run usefully, on an edge computer. In this paper we study three related questions using the UAV decision benchmark, UAV… |
| 2026-11-17 | [Weight Space Programming for Vision Language Model Repair](https://arxiv.org/abs/2611.23615v2) | Avinash Goutham Aluguvelly | In weight-space programming, a language model edits a network's weights directly, without gradients. We ask what a successful repair of this kind relies on. We plant faults in the weights of two vision-language models, SmolVLM-256M and LFM2-VL-450M, and check each edit made by a… |
| 2026-11-12 | [Parsed, Plausible, Wrong: Output Format Controls Silent Errors in Small-Model Semantic Parsing](https://arxiv.org/abs/2611.17169v1) | Umar Ahmed, Maxim Dunaev, Mohammad Nour Al Awad 等 | A small fine-tuned parser that emits a well-formed but wrong tree fails silently: nothing downstream can tell. We ask whether the output format, a free design choice, controls how often this happens. On the PIZZA ordering benchmark we fine-tune two sub-billion models (Qwen3.5-0.… |
| 2026-11-07 | [Locating and Repairing Domain Shift in VLM Trajectory Planning](https://arxiv.org/abs/2611.10720v1) | Zhihong Cui, Hengyu Liu, Michael Alexander Riegler 等 | Vision-Language Models (VLMs) are increasingly adopted as end-to-end driving planners, yet they suffer from cross-city domain shift. Existing methods treat this shift as a single quantity addressed by uniform alignment or single-site editing. We challenge this view on two counts… |
| 2026-11-03 | [LINDA: Low Resource Foot Ulcer Segmentation using Local Similarity and Iterative Neural Dynamics](https://arxiv.org/abs/2611.04233v1) | Trishaani Acharjee, Sudip Chakrabarty, Rajdeep Chatterjee 等 | Foot ulcers are a leading cause of lower limb amputation, and the highest burden often falls on low and middle income regions where specialists and computing hardware are both scarce. Wound monitoring tools are most useful there when they are accurate yet small enough to run on… |
| 2026-10-22 | [Query-Conditioned Liquid Temporal Grounding for Long-Form Audio](https://arxiv.org/abs/2610.31252v1) | Yuelan Cheng, Yonggang Zhu, Yongqiang Chen 等 | Long-form temporal audio grounding (TAG) is challenging because target events often occupy only a small portion of a recording, while queries may impose multiple semantic constraints over a large temporal search space. We propose a query-conditioned liquid temporal grounding fra… |
| 2026-10-08 | [ContiLNN: Mitigating Slice Sampling Discontinuity with Liquid Neural Networks for Medical Image Restoration](https://arxiv.org/abs/2610.12337v1) | Jialei He, Enhe Liu, Sifan Song 等 | Anatomical continuity provides complementary information for medical image restoration, but its use requires accounting for local anatomy and variations in slice sampling. We introduce ContiLNN, which augments two-dimensional restoration backbones with bidirectional closed-form… |
| 2026-09-27 | [MTLiquid: Enabling Efficient Multi-Task Learning using Liquid Neural Networks for Lightweight Healthcare Monitoring Systems](https://arxiv.org/abs/2609.33232v1) | Rachmad Vidya Wicaksana Putra, Fahad Abdul Rauf, Muhammad Shafique | Continuous-time sensing and monitoring with timely and accurate decision-making are critical for many real-world applications. In healthcare monitoring systems, physiological signals are often available or sampled at irregular time intervals, hence requiring continuous-time proc… |
| 2026-08-27 | [Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-Form Continuous-Time Cells for Dynamic 3D Gaussian Splatting](https://arxiv.org/abs/2608.28702v1) | Mingzhao Li, Arghya Pal | Deformable 3D Gaussian Splatting (D-3DGS) reconstructs dynamic scenes by deforming a canonical set of 3D Gaussians through a deformation field of frame time. Replacing its MLP with a stack of Closed-form Continuous-time (CfC) cells-a Liquid Neural Network that solves the Liquid… |
| 2026-08-13 | [History-informed Lagrangian Neural Networks](https://arxiv.org/abs/2608.13215v1) | Tianshuo Zhang, Xianglei Xing, Wenzhe Zhai 等 | Forecasting the long-horizon evolution of mechanical systems from position-only observations is a pivotal yet difficult task, as hidden velocities and trajectory-specific physical properties must be inferred simultaneously. Although physics-guided neural networks like Lagrangian… |
| 2026-08-04 | [PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling](https://arxiv.org/abs/2608.03041v1) | Dhivya Dharshini Kannan, Wei Zhang, Jieyi Bi 等 | Deep reinforcement learning (DRL) approaches for flexible job shop scheduling (FJSP) heavily rely on attention-centric architectures to achieve state-of-the-art performance. However, these models suffer from excessive parameter counts and prohibitive inference latency as problem… |

## GitHub 候选仓库
| 更新 | 仓库 | Star | Fork | 语言 | 说明 |
|---|---|---:|---:|---|---|
| 2026-11-20 | [Ipsedo/LiquidNetworks](https://github.com/Ipsedo/LiquidNetworks) | 30 | 6 | Python | Liquid Time-constant Networks implementation with PyTorch |
| 2026-11-19 | [zhangshuibai/molten](https://github.com/zhangshuibai/molten) | 29 | 2 | Metal | Run a coding model at 300+ tok/s on your Mac. Native Rust + Metal inference engine for LiquidAI LFM2.5-24B-A2B (MLX 4-bit) on Apple Silicon. |
| 2026-11-16 | [paragontasx/ferro](https://github.com/paragontasx/ferro) | 10 | 0 | C++ | Native OpenCL inference engine for Intel N100-class SoCs: EmbeddingGemma-2, LiquidAI d1-omni and LFM2.5 chat in one C++ binary, on the iGPU. Measured ~2x llama… |
| 2026-11-07 | [daivarsi/TyphoonLLM](https://github.com/daivarsi/TyphoonLLM) | 9 | 1 | C++ | Fully local LLM inference on PlayStation Vita using LiquidAI LFM2.5-230M. |
| 2026-10-21 | [RESMP-DEV/lfm25-embedding-trainer](https://github.com/RESMP-DEV/lfm25-embedding-trainer) | 7 | 1 | Python | Fine-tune LiquidAI LFM2.5 Embedding models for dense retrieval using your own data |
| 2026-10-21 | [blacklike02/Liquid-Network-Model](https://github.com/blacklike02/Liquid-Network-Model) | 4 | 0 | Python | Liquid Neural Network language model with LTC and CfC dynamics |
| 2025-11-04 | [siddharthk27/Neural-Circuit-Policies](https://github.com/siddharthk27/Neural-Circuit-Policies) | 4 | 0 | Jupyter Notebook | This contains my experiments with Liquid Time Constant Networks (LTCs) and Closed Form Continuous Network (CFCs) architectures. |
| 2026-11-11 | [AwareLiquid/O1](https://github.com/AwareLiquid/O1) | 3 | 2 | Python | MT-LNN — clean prototype of a brain-inspired liquid neural network (constant-memory recurrent state, O(1) generation cache). Early baseline; active development… |
| 2026-10-21 | [julschmidtJJL/lnn-vs-classical-forecasting](https://github.com/julschmidtJJL/lnn-vs-classical-forecasting) | 2 | 0 | Python | Bachelor's thesis: do liquid neural networks beat classical models on daily financial returns? No: 0 of 120 corrected comparisons. 11 LTC/CfC variants vs 8 bas… |
| 2026-07-31 | [middesurya/daily-webapp-2026-05-04-liquidlab](https://github.com/middesurya/daily-webapp-2026-05-04-liquidlab) | 2 | 0 | HTML | LiquidLab — Interactive Liquid Neural Networks & Continuous-Time Models Laboratory. 6 modules: LTC dynamics, Neural ODE, C. elegans connectome, CfC networks, a… |
| 2026-11-15 | [uipo9243-svg/Fly-CfC](https://github.com/uipo9243-svg/Fly-CfC) | 1 | 0 | Python | Whole-brain Drosophila connectome-constrained Closed-Form Continuous-Time (CfC) neural network for ultra-low-latency autonomous flight |
| 2026-11-11 | [jhacksman/Surprise-Minimization-Liquid-Neural-Network](https://github.com/jhacksman/Surprise-Minimization-Liquid-Neural-Network) | 1 | 0 |  | Implementation of a real-time learning AI model combining Liquid Time Constant Neural Networks with surprise minimization-based learning rules |
- 同名 fork 已折叠 1 条（见 `papers/daily/2026-11-27_lnn_research.json` 的 `fork_variants` / `fork_of` 字段）

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-11-26 | [reaperdoesntknow/LFM2.5-2.6B-CyberSec](https://huggingface.co/reaperdoesntknow/LFM2.5-2.6B-CyberSec) | 2108 | 0 | text-generation |
| 2026-11-26 | [reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil](https://huggingface.co/reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil) | 1743 | 12 | text-generation |
| 2026-11-26 | [TimesheetMe/timesheetme-lfm2.5-1.2b-release](https://huggingface.co/TimesheetMe/timesheetme-lfm2.5-1.2b-release) | 479 | 0 |  |
| 2026-11-26 | [experimentalmachines/LFM2.5-VL-450M-MobileAgent-Dev-AndroidControlOnly-GGUF](https://huggingface.co/experimentalmachines/LFM2.5-VL-450M-MobileAgent-Dev-AndroidControlOnly-GGUF) | 117 | 1 | image-text-to-text |
| 2026-11-26 | [experimentalmachines/LFM2.5-VL-450M-MobileAgent-Dev-AndroidControlOnly](https://huggingface.co/experimentalmachines/LFM2.5-VL-450M-MobileAgent-Dev-AndroidControlOnly) | 26 | 0 | image-text-to-text |
| 2026-11-26 | [Pratham804/Pratham-LFM2.5-350M-V25-Polisher](https://huggingface.co/Pratham804/Pratham-LFM2.5-350M-V25-Polisher) | 0 | 1 |  |
| 2026-05-06 | [P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE](https://huggingface.co/P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE) | 0 | 0 |  |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
