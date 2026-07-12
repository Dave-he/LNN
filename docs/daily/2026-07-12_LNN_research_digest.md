---
title: LNN 每日研究追踪 - 2026-07-12
date: 2026-07-12
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-07-12

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：21 篇（自上次 digest 2026-07-11 起无新增）
- GitHub 候选仓库：6 个
- Hugging Face 候选模型：8 个
- 已下载 PDF：0 个
- **当日研究重点**：延续 round 284 pulse-augmented gated liquid τ 报告中的"predictability-gated pulse amplitude"建议，作为 round 285 候选。

## arXiv 候选论文（与 2026-07-11 digest 相同的最新子集）
| 日期 | 论文 | 作者 | 摘要要点 |
|---|---|---|---|
| 2026-07-09 | [TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning](https://arxiv.org/abs/2607.08283v1) | Liang et al. | VLA 策略 + 时序记忆融合，用于 stage-dependent 操作任务 |
| 2026-07-02 | [Liquid Latent State Dynamics for Interpretable Turbofan Degradation Modeling](https://arxiv.org/abs/2607.01986v1) | Nie, Wang, Su | C-MAPSS 涡扇退化建模，LNN 作为潜变量动力学 |
| 2026-06-25 | [Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection](https://arxiv.org/abs/2606.26849v1) | Chen et al. | 通用 SOD 多模态融合，Liquid + SSM |
| 2026-06-19 | [Topological Neural Dynamics: A Neuron-wise Framework for Sequence Modeling](https://arxiv.org/abs/2606.21295v6) | Cai, Zhao | 神经元级别 ODE 在可学习有向图上 |
| 2026-06-18 | [GazeLNN](https://arxiv.org/abs/2606.20491v1) | Mohammed et al. | 自主导航注视预测，计算高效 |
| 2026-06-17 | [FlowFake: Liquid Networks for Audio Deepfake Detection](https://arxiv.org/abs/2606.19579v1) | Dhondiyal et al. | 跨数据集音频深伪检测，LNN |
| 2026-06-14 | [Memory-Augmented Graph Liquid Time-Constant Networks](https://arxiv.org/abs/2606.15807v1) | Xiang, Xu | 跨域交通状态预测，记忆增强图 LTC |
| 2026-06-14 | [Liquid Random Feature Methods for Time-Dependent PDEs](https://arxiv.org/abs/2606.15571v1) | Linghu, Wang | 无网格时空逼近 PDE，Liquid 随机特征 |
| 2026-06-10 | [Multi-Rate MoE for Accelerating LNN Training](https://arxiv.org/abs/2606.12240v1) | Zong et al. | 多速率 MoE 加速 LNN 训练（已在 round 77 MR-MoE 复现） |
| 2026-06-04 | [Liquid NN as Drop-in Continuous-Time Deformation Field for D-3DGS](https://arxiv.org/abs/2606.07670v1) | Li, Pal, Tan | 动态 3D Gaussian Splatting 形变场 |
| 2026-05-26 | [Comparative Analysis of LNN and LSTM for Sequential Pattern Recognition](https://arxiv.org/abs/2605.27467v1) | Thu, Oo, Supnithi | 鲁棒性/效率/临床应用比较（round 92-93 复现结论：LSTM > CfC 在 1D） |
| 2026-05-21 | [EMMA: Multimodal Physical Parameter Extraction](https://arxiv.org/abs/2605.24047v1) | Shaikh et al. | 多模态物理参数提取（已在 EMMA rover benchmark 集成） |

**arXiv 新增**：自 2026-07-11 起 arXiv 无新增 LNN 相关论文（窗口 2026-07-09 → 2026-07-12）。搜索关键词覆盖：`liquid neural network`, `CfC`, `liquid time-constant`, `pulse-driven neural`, `neural ODE`, `closed-form continuous-time`, `liquid network`。

## GitHub 候选仓库
| 更新 | 仓库 | Star | 语言 | 说明 |
|---|---|---:|---|---|
| 2026-07-12 | [Dave-he/LNN](https://github.com/Dave-he/LNN) |  | Python | （本仓）round 284 pulse-gated liquid τ 落地，待 r285 |
| 2026-07-11 | [AlexanderRumyantcev/LNN-LowLight](https://github.com/AlexanderRumyantcev/LNN-LowLight) | 0 | Python | CfC for low-light video enhancement (RetinexFormer pipeline) |
| 2026-07-11 | [liuyhoo/F-CfC](https://github.com/liuyhoo/F-CfC) | 0 |  | F-CfC: Fractional Closed-form Continuous-time Networks |
| 2026-07-11 | [lajosbencz/lfm-train-image](https://github.com/lajosbencz/lfm-train-image) | 0 | Dockerfile | Optimized base image for LiquidAI LFM2.5 |
| 2026-07-10 | [Sum-Outman/Self-LNN](https://github.com/Sum-Outman/Self-LNN) | 1 | C | Self AGI System，LNN C 语言版本 |
| 2026-07-10 | [kydaong/lnn_prediction](https://github.com/kydaong/lnn_prediction) | 0 |  | LNN for turbomachinery baseline detection |
| 2026-07-09 | [kakopappa/proxy-kd-lfm2](https://github.com/kakopappa/proxy-kd-lfm2) | 0 | Python | Proxy-KD distillation of Claude into LFM2.5-350M |

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-07-12 | [LiquidAI/LFM2.5-VL-450M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF) | 7900+ | 56 | image-text-to-text |
| 2026-07-10 | [FadedRedStar/LFM2.5-8B-A1B-heretic-imatrix-GGUF](https://huggingface.co/FadedRedStar/LFM2.5-8B-A1B-heretic-imatrix-GGUF) | 2127 | 0 | text-generation |
| 2026-07-10 | [MuXodious/LFM2.5-8B-A1B-SOMPOA-heresy](https://huggingface.co/MuXodious/LFM2.5-8B-A1B-SOMPOA-heresy) | 880 | 0 | text-generation |
| 2026-07-10 | [Synaptics/liquidAI-LFM2p5-230M-LLM](https://huggingface.co/Synaptics/liquidAI-LFM2p5-230M-LLM) | 570 | 0 | text-generation |
| 2026-07-10 | [FadedRedStar/LFM2.5-8B-A1B-heretic-GGUF](https://huggingface.co/FadedRedStar/LFM2.5-8B-A1B-heretic-GGUF) | 558 | 0 | text-generation |
| 2026-07-10 | [FadedRedStar/LFM2.5-VL-1.6B-heretic-GGUF](https://huggingface.co/FadedRedStar/LFM2.5-VL-1.6B-heretic-GGUF) | 486 | 0 | image-text-to-text |
| 2026-07-10 | [FadedRedStar/LFM2.5-350M-heretic-imatrix-GGUF](https://huggingface.co/FadedRedStar/LFM2.5-350M-heretic-imatrix-GGUF) | 257 | 0 | text-generation |
| 2026-07-10 | [FadedRedStar/LFM2.5-350M-heretic-GGUF](https://huggingface.co/FadedRedStar/LFM2.5-350M-heretic-GGUF) | 246 | 0 | text-generation |

## 建议动作（ranked）
1. **round 285 — predictability-gated pulse amplitude**（直接接续 r284）：
   用 r280 blend gate 抑制 r284 pulse 在噪声上的振幅 `pulse = g_t · A · sin(...)`，目标是同时保留 structured 的 gap-robustness 并恢复 noise safety，把 r284 从 target-dependent 提升为 strict-positive default。
2. **decorrelation loss (arXiv:2607.01986)**：turbofan 退化建模的 degradation/condition decorrelation loss，r100 SNNL 之外的另一条 disentanglement 轴，可在 toy_sin / structured / random 上复现。
3. **neuron-wise topological dynamics (arXiv:2606.21295)**：作为新 backbone 候选，r76 n_tau 的扩展版。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api