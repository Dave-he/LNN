---
title: LNN 每日研究追踪 - 2026-10-19
date: 2026-10-19
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-10-19

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：24 篇
- GitHub 候选仓库：42 个
- Hugging Face 候选模型：18 个
- 已下载 PDF：0 个
- 下表每类仅渲染前 12 条；全量候选池见 `papers/daily/2026-10-19_lnn_research.json`，研读选择器以该 JSON 为准。

## arXiv 候选论文
| 日期 | 论文 | 作者 | 摘要 |
|---|---|---|---|
| 2026-10-08 | [ContiLNN: Mitigating Slice Sampling Discontinuity with Liquid Neural Networks for Medical Image Restoration](https://arxiv.org/abs/2610.12337v1) | Jialei He, Enhe Liu, Sifan Song 等 | Anatomical continuity provides complementary information for medical image restoration, but its use requires accounting for local anatomy and variations in slice sampling. We introduce ContiLNN, which augments two-dimensional restoration backbones with bidirectional closed-form… |
| 2026-09-27 | [MTLiquid: Enabling Efficient Multi-Task Learning using Liquid Neural Networks for Lightweight Healthcare Monitoring Systems](https://arxiv.org/abs/2609.33232v1) | Rachmad Vidya Wicaksana Putra, Fahad Abdul Rauf, Muhammad Shafique | Continuous-time sensing and monitoring with timely and accurate decision-making are critical for many real-world applications. In healthcare monitoring systems, physiological signals are often available or sampled at irregular time intervals, hence requiring continuous-time proc… |
| 2026-08-27 | [Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-Form Continuous-Time Cells for Dynamic 3D Gaussian Splatting](https://arxiv.org/abs/2608.28702v1) | Mingzhao Li, Arghya Pal | Deformable 3D Gaussian Splatting (D-3DGS) reconstructs dynamic scenes by deforming a canonical set of 3D Gaussians through a deformation field of frame time. Replacing its MLP with a stack of Closed-form Continuous-time (CfC) cells-a Liquid Neural Network that solves the Liquid… |
| 2026-08-04 | [PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling](https://arxiv.org/abs/2608.03041v1) | Dhivya Dharshini Kannan, Wei Zhang, Jieyi Bi 等 | Deep reinforcement learning (DRL) approaches for flexible job shop scheduling (FJSP) heavily rely on attention-centric architectures to achieve state-of-the-art performance. However, these models suffer from excessive parameter counts and prohibitive inference latency as problem… |
| 2026-07-14 | [Real-time fall detection based on vision for low-power edge platforms](https://arxiv.org/abs/2607.12909v1) | Wenjun Xia, Zhicheng Peng, Haopeng Li 等 | Falling detection is vital for elderly care and intelligent surveillance; however, prevailing vision-based approaches predominantly frame it as static pose classification or discrete temporal pattern matching, fundamentally overlooking the instability dynamics of the human suppo… |
| 2026-07-09 | [TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning](https://arxiv.org/abs/2607.08283v3) | Yushen Liang, Yue Peng, Baosheng Jin 等 | Vision--Language--Action (VLA) policies such as $π_{0.5}$ and OpenVLA perform well on many manipulation tasks, but they are often reactive: the next action is predicted from the current observation, instruction, and proprioceptive state. This assumption breaks down in stage-depe… |
| 2026-07-02 | [Liquid Latent State Dynamics for Interpretable Turbofan Degradation Modeling](https://arxiv.org/abs/2607.01986v1) | Weizhi Nie, Weijie Wang, Yuting Su | Multivariate time-series models for prognostics are often evaluated by point prediction accuracy, yet their internal states rarely expose a coherent degradation process. We study liquid neural networks as latent dynamics models for aircraft engine health monitoring on the C-MAPS… |
| 2026-06-25 | [Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection](https://arxiv.org/abs/2606.26849v1) | Ke Chen, Ling Zhou, Guangqi Jiang 等 | General Salient Object Detection (SOD) aims to identify and segment visually interesting objects from uni-modality or multi-modality scenes, recently advanced by cutting-edge State Space Models (SSMs). However, a critical limitation of current approaches is their neglect of the… |
| 2026-06-19 | [Topological Neural Dynamics: A Neuron-wise Framework for Sequence Modeling](https://arxiv.org/abs/2606.21295v6) | Borui Cai, Yao Zhao | Existing sequence models, including RNNs, LSTMs, continuous-time networks, and Transformers, share a common structural principle: layer-wise dynamics, where all neurons in the same layer co-evolve through a shared parameterized operator, leaving individual neurons no freedom to… |
| 2026-06-18 | [Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation](https://arxiv.org/abs/2606.20491v1) | Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis | Human visual attention relies on structured scanpaths to efficiently process scenes, yet instilling this behavior into robot autonomy is in its infancy and hindered by the high,computational costs of existing predictive models. To address this, we introduce GazeLNN, a computatio… |
| 2026-06-17 | [FlowFake: Liquid Networks for Audio Deepfake Detection](https://arxiv.org/abs/2606.19579v1) | Shivaay Dhondiyal, Divyansh Sharma, Dinesh Kumar Vishwakarma | Audio deepfakes generated by neural text-to-speech and voice-cloning systems threaten speaker verification and public discourse at scale. The core challenge is cross-dataset generalization: detectors trained on one synthesis pipeline collapse on unseen forgeries. We argue that t… |
| 2026-06-14 | [Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liquid Time-Constant Networks](https://arxiv.org/abs/2606.15807v1) | Jinrong Xiang, Ming Xu | Traffic state prediction is a fundamental task in intelligent transportation systems. In practical applications, some regions suffer from limited traffic observations due to insufficient sensing infrastructure, making cross-domain knowledge transfer an important solution for dat… |

## GitHub 候选仓库
| 更新 | 仓库 | Star | 语言 | 说明 |
|---|---|---:|---|---|
| 2026-10-19 | [di-omics/liquid-policy-robustness](https://github.com/di-omics/liquid-policy-robustness) | 0 | Python | Do liquid neural networks control a robot arm better under sensor noise than matched-size MLPs and LSTMs? Behavior cloning on Reacher-v5. |
| 2026-10-19 | [kerstinbrouwer1119/bebelm](https://github.com/kerstinbrouwer1119/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-19 | [dabdullah1589/bebelm](https://github.com/dabdullah1589/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-19 | [Renatah8659/bebelm](https://github.com/Renatah8659/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-19 | [Nicolas44345/bebelm](https://github.com/Nicolas44345/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-19 | [Mitchell98738/bebelm](https://github.com/Mitchell98738/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-19 | [daniellebaran22544/bebelm](https://github.com/daniellebaran22544/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-18 | [BuczynskiRafal/ltc-hydrology-flood-prediction](https://github.com/BuczynskiRafal/ltc-hydrology-flood-prediction) | 0 | Python | Official PyTorch implementation & benchmark of Liquid Time-Constant Networks (LNN) for urban drainage water level forecasting (MDPI Water 2026). |
| 2026-10-18 | [elizabeth2756/bebelm](https://github.com/elizabeth2756/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-18 | [Naoki09383/bebelm](https://github.com/Naoki09383/bebelm) | 0 | Rust | CPU-only, pure-Rust implementation of LiquidAI's LFM2.5-8B-A1B LLM |
| 2026-10-17 | [Ipsedo/LiquidNetworks](https://github.com/Ipsedo/LiquidNetworks) | 29 | Python | Liquid Time-constant Networks implementation with PyTorch |
| 2026-10-16 | [api-evangelist/liquid-ai](https://github.com/api-evangelist/liquid-ai) | 1 |  | Liquid AI — independent third-party profile of a public API surface, by API Evangelist. Liquid AI is an MIT spinoff developing Liquid Foundation Models (LFMs)… |

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-10-19 | [Ninnix96/LFM2.5-24B-A2B-gguf](https://huggingface.co/Ninnix96/LFM2.5-24B-A2B-gguf) | 82 | 0 | text-generation |
| 2026-10-19 | [PollardWeights/LFM2.5-24B-A2B-Pollard](https://huggingface.co/PollardWeights/LFM2.5-24B-A2B-Pollard) | 0 | 0 | text-generation |
| 2026-10-19 | [Rcarvalo/lfm2a-fr-front-desk](https://huggingface.co/Rcarvalo/lfm2a-fr-front-desk) | 0 | 0 | audio-to-audio |
| 2026-10-19 | [electroglyph/LFM2.5-24B-A2B-uncensored](https://huggingface.co/electroglyph/LFM2.5-24B-A2B-uncensored) | 0 | 0 |  |
| 2026-10-18 | [reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil](https://huggingface.co/reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil) | 2535 | 12 | text-generation |
| 2026-10-18 | [mfgcvguy/LFM2.5-1.2B-Kids-Tutor-GGUF](https://huggingface.co/mfgcvguy/LFM2.5-1.2B-Kids-Tutor-GGUF) | 12 | 0 |  |
| 2026-10-18 | [randmaru/LFM2.5-24B-A2B-mlx-mxfp4](https://huggingface.co/randmaru/LFM2.5-24B-A2B-mlx-mxfp4) | 0 | 0 | text-generation |
| 2026-10-18 | [LLlamaAPI/LFM2.5-24B-A2B-RAMPool-e2g128r4](https://huggingface.co/LLlamaAPI/LFM2.5-24B-A2B-RAMPool-e2g128r4) | 0 | 0 | text-generation |
| 2026-10-18 | [janihal/LFM2.5-230M-Chess-oQ8](https://huggingface.co/janihal/LFM2.5-230M-Chess-oQ8) | 0 | 0 | text-generation |
| 2026-10-17 | [LiquidAI/d1-omni-600M-GGUF](https://huggingface.co/LiquidAI/d1-omni-600M-GGUF) | 8808 | 45 | image-text-to-text |
| 2026-10-15 | [prithivMLmods/LiquidAI-LFM2.5-24B-A2B-GGUF](https://huggingface.co/prithivMLmods/LiquidAI-LFM2.5-24B-A2B-GGUF) | 442 | 2 | text-generation |
| 2026-10-14 | [bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF](https://huggingface.co/bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF) | 4849 | 2 | text-generation |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
