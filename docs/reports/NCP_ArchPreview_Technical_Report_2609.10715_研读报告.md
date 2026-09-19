---
title: NCP-ArchPreview Technical Report — 研读报告
date: 2026-09-20
tags: [LNN, paper, NCP, latent-space-LM, JEPA, vector-quantization, arxiv:2609.10715]
arxiv_id: 2609.10715
source: docs/daily/2026-09-20_LNN_research_digest.md
---

# NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction

> **核心一句话**：把"Next Concept Prediction (NCP)"从单论文探索扩展到 **8.9B 参数 / 5.73T token / 与 OLMo-3-7B 严格头对头**的 frontier scale——证实 **latent-space prediction 可作为 token-level 训练的一等公民目标**，51.3% 的训练 token 即达到 OLMo-3-7B 最终 loss，下游 macro-average +2.45 pts（含 GSM8K +5.99）。NCP 这个缩写正好也是 Hasani 等人 2020 Neural Circuit Policies 的简称，本文沿用此命名但探索方向已迁出 LNN 体系。

## 元数据

| 项 | 内容 |
|---|---|
| 论文标题 | NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction |
| arXiv ID | [2609.10715v1](https://arxiv.org/abs/2609.10715) (cs.CL) |
| 发表时间 | 2026-09-09（v1），2026-09-11（v1 公开日期） |
| 作者 | The Intern-NCP Team（Shanghai AI Lab; LUMIA Lab, Shanghai Jiao Tong University）—— Jiaqi Cao、Chiyu Chen、Shuang Cheng、… 等 |
| PDF | [`papers/daily/2026-09-20/2609.10715.pdf`](../../papers/daily/2026-09-20/2609.10715.pdf) |
| 资源 | [HF Collection](https://huggingface.co/collections/ArchSpace-Collection/ncp-archpreview)、[lmdeploy 推理代码](https://github.com/InternLM/lmdeploy)、[OLMo 评测代码](https://github.com/LUMIA-Group/ncp_olmo_eval) |
| 关键词 | NCP、latent-space LM、product quantization、JEPA、OLMo-3、multiway residual、concept module、speculative decoding、DFlash2 |
| 与 LNN 关联度 | **中（命名同源，方法学异路）**——Hasani 2020 Neural Circuit Policies 用 "NCP" 命名生物启发的液态电路策略；本文借用同缩写但指代 "Next Concept Prediction"，与 LNN 没有直接架构连续性**，但代表 LLM 主流正在认真借鉴 latent-prediction 思路（JEPA 范式） |

## 核心问题

当前主流 LLM 严格遵循 **Next Token Prediction (NTP)**：监督信号只覆盖粒度最细的 surface token，模型必须**通过梯度间接**学到 sentence-/paragraph-level 抽象。这带来三个问题：

1. **抽象作为副产品**：高层语义结构（如论元、概念、意图）只能从 token-level cross-entropy 中"挤压"出来，效率受限；
2. **难以独立操作 latent 空间**：domain adaptation、speculative decoding 等任务需要 latent 表示，但当前模型没有"暴露"可分离的概念层；
3. **JEPA 范式尚未在 LM 主流被证明可扩展**：I-JEPA / V-JEPA 2 在 vision/video 上验证了 latent-space prediction 的优势，但 language 侧一直缺 frontier-scale 实证。

NCP-ArchPreview 的核心问题是：**在 8.9B 参数、5.73T token 规模下，joint token-and-concept pretraining 能否在严格控制变量的对比中带来一致的、不可被容量解释的效率/能力提升？**

## 方法论与核心思路

### 1. 三模块架构（基于 OLMo-3-7B backbone）

```
Token Encoder (16 layers) → Concept Module (8 layers) → Token Decoder (16 layers)
                ↓                    ↓                    ↓
              h1:T            VQ-c 离散化预测            x̂_{t+1}
```

- **Token Encoder**：标准 Transformer，输出 token-level hidden states $h_t \in \mathbb{R}^d$；
- **Concept Module**：对 $k$ 个连续 token states 做 mean pooling，得到 $c_m \in \mathbb{R}^d$；用 **Product Quantization**（多码本分段）把连续 $c_m$ 映射为 $S$ 段的离散 codebook indices；用一个 8 层 Transformer 在 concept 序列上预测下一 concept；
- **Token Decoder**：把 predicted concept sequence 上采样回 token 分辨率并 shift（causal），与 token-level hidden states 联合生成下一 token。

**Hierarchical residual (multiway)**：参考 MUDDFormer，每个 block 内有 FULL（完整 attention）/ SWA（sliding window attention）/ IRC（intra residual）/ CRC（cross-module residual）四条路径，目标条件（concept）决定哪条激活。

### 2. NCP 训练目标（joint with NTP）

- **NTP loss**：标准的 $\mathcal{L}_{\text{NTP}} = -\sum_t \log p_\theta(x_{t+1} | x_{1:t})$；
- **NCP loss**：预测下一 concept（codebook index），在 $S$ 个码本段上各做一个分类 $\mathcal{L}_{\text{NCP}}^{(s)} = -\sum_m \log p_\theta(c_{m+1}^{(s)} | c_{1:m}, x_{1:T})$；
- **VQ commitment loss**：straight-through estimator 逼近 $c_m$ 与最近 codebook entry 之间的 $\ell_2$，避免 codebook collapse；
- **总目标**：$\mathcal{L} = \mathcal{L}_{\text{NTP}} + \sum_{s=1}^S \mathcal{L}_{\text{NCP}}^{(s)} + \lambda \mathcal{L}_{\text{VQ}}$。

### 3. 三组严格对照

- **Compute-aligned**：相同 token 数预算，比较 loss 下游曲线；
- **Parameter-aligned**：相同 8.9B 参数，与 strict 40 层 Transformer baseline 对比；
- **Ablation**：拆 NCP objective、latent hierarchy、hierarchical routing 三组件，isolate each 的贡献。

### 4. 部署侧利用学到的 concept 空间

- **17M 参数 domain adaptation**：只更新 VQ 模块，无需全模型 finetune；
- **Speculative decoding**：把 concept representations 注入 DFlash2 drafter，mean accepted length +4.17%。

## 核心公式

1. **Token Encoder 输出**：$h_{1:T} = \text{TokenEncoder}_{\theta_e}(x_{1:T})$，$h_t \in \mathbb{R}^d$。
2. **Concept 压缩（mean pooling）**：$c_m = f_c(h_{(m-1)k+1:mk}) = \frac{1}{k} \sum_{i=1}^{k} h_{(m-1)k+i}$，$c_m \in \mathbb{R}^d$，$M = \lfloor T/k \rfloor$。
3. **Product Quantization**：$c_m \in \mathbb{R}^d$ 拆为 $S$ 段 $c_m^{(s)} \in \mathbb{R}^{d/S}$，每段用一个 codebook $\mathcal{C}^{(s)} = \{e_j^{(s)}\}_{j=1}^{K}$；assignment $a_m^{(s)} = \arg\min_j \|c_m^{(s)} - e_j^{(s)}\|_2$，quantized value $\hat{c}_m^{(s)} = e_{a_m^{(s)}}^{(s)}$。
4. **Concept Module**：在 $(\hat{c}_{1:M})$ 上做 autoregressive Transformer，输出 predicted concept distribution $p_\theta(c_{m+1} | c_{1:m}, x_{1:T})$。
5. **Joint 训练目标**：$\mathcal{L} = \mathcal{L}_{\text{NTP}}(x_{t+1}|h_t, \hat{c}_{\text{shifted}}) + \sum_s \mathcal{L}_{\text{NCP}}^{(s)}(c_{m+1}^{(s)} | c_{1:m}^{(s)}) + \lambda \cdot \|sg[c_m] - \hat{c}_m\|_2^2$（VQ commitment，stop-gradient）。

## 关键成果与贡献

### 1. Pretraining 效率（Stage-1，5.73T tokens，相同数据）

| 模型 | Token 数达到 OLMo-3-7B 最终 loss | 最终 loss delta |
|---|:---:|:---:|
| OLMo-3-7B | 5.73T (full) | 0 |
| **NCP-ArchPreview** | **51.3% (2.94T)** | **−0.091** |
| 收敛速度 | — | **1.95×** |

**Stage-2 mid-training (100B tokens)**: NCP-ArchPreview 1.51× 收敛速度，loss 更低且波动更小。

### 2. 下游能力（macro-average，27 benchmarks）

- 总体 **+2.45 pts**（NCP-ArchPreview vs OLMo-3-7B，全 pretraining 后）；
- **GSM8K: +5.99 pts**（最显著单项）；
- ARC、BBH、HellaSwag、MMLU 等亦普遍正增益。

### 3. 严格对照（Pareto 计算效率）

- **Parameter-aligned**：8.9B NCP-ArchPreview 用 85% 的标准 computation 接近 strict 40-layer Transformer baseline；
- **Compute-aligned**：相同 compute 预算下 NCP 始终领先；
- **Ablation**：单独加 NCP objective / latent hierarchy / hierarchical routing，每一项都贡献正增益；NCP objective 单独贡献最大（isolation study）。

### 4. 部署侧 concept 空间利用

- **17M VQ 参数 domain adaptation**：无需全模型 finetune，可低成本换 domain；
- **DFlash2 speculative drafter**：注入 concept representations 后 **mean accepted length +4.17%**，overhead 可忽略；
- **Scaling law 拟合**：1.74× compute efficiency 改善。

### 5. 关键贡献

1. **首次 trillion-token scale 的 latent-space LM 完整训练实证**（5.73T tokens，8.9B params）—— JEPA 在 LM 主流的可扩展性首次获得 frontier-scale 支持；
2. **Joint token-and-concept pretraining 配方开源**：checkpoint / 中间评估 / 推理代码（lmdeploy）/ 评测代码（ncp_olmo_eval）全部发布；
3. **严格 ablations 排除容量/算术 confound**：1.95× 收敛不是来自"参数多一点"或"算一点"；
4. **可重用 concept vocabulary**：17M VQ 参数做 domain adaptation、DFlash2 drafter 提升，证明 latent 层不仅对训练阶段有用，也对下游有独立价值。

### 与 LNN / NCP (Neural Circuit Policy) 命名同源的辨析

- **同源命名**："NCP" 这个缩写最早属于 Hasani 等 2020 *Neural Circuit Policies for Auditable Autonomous Driving* 提出的生物启发液态电路策略；
- **方法学断裂**：本文 NCP 是 **离散 concept-level prediction**（product quantization + Transformer），与 LNN 的 ODE-driven continuous-time 状态动力学**没有架构继承关系**；
- **联系点**：两者都相信 **latent / intermediate 层应被显式建模**——LNN 用 $\dot{x} = f(x, I, t; \theta)$，NCP-Arch 用 $c_{m+1} = g(c_{1:m}, x_{1:T}; \theta)$；这是更大的 "intermediate representation engineering" 趋势的两个不同方向。

## 局限性与未来展望

### 1. 长期上下文训练未包含
本报告只覆盖 standard context length 的 5.73T pretrain + 100B mid-train；long-context training 是后续 full recipe 的一部分。concept-level pathway 的 compressed sequence 在长上下文场景可能特别有利——本文尚未验证。

### 2. Loss → 下游能力的转化不均匀
- pretraining 完成后下游 macro-average **+2.45 pts**，
- mid-training 完成后下游增益降到 **+0.59 pts** 且 **跨 benchmark 方差更大**；
- 跨 benchmark 表现参差，且 cross-entropy loss 优势与下游能力提升之间的关系**因 stage / 任务而异**。

### 3. 概念码本的语义解释性
codebook entry 是否对应人类可理解的"概念"未做系统研究；17M VQ 参数 domain adaptation 有效可能仅因为"统计学上抓住了 domain 分布偏移"，而非真正"概念层迁移"。

### 4. 多模态扩展未覆盖
目前 NCP-ArchPreview 是 text-only；v-JEPA 2 已经在 video 上验证 latent prediction 的价值，多模态 latent space 的统一是未来方向。

### 5. 未来工作
- long-context training + concept-level pathway 的 scaling 验证；
- capability-aware mid-training data curation；
- 把 token-loss + concept-loss 联合 checkpoint selection 形式化；
- 把 latent 概念空间外推到 multimodal foundation model。

## 元评估

| 维度 | 评分 | 备注 |
|---|:---:|---|
| 清晰度 | ★★★★ | Technical report 风格，结构清晰；数学推导略简 |
| 可复现性 | ★★★★★ | 全部 checkpoint、推理 / 评测代码、训练数据链接都公开 |
| 与 LNN 主线关联 | ★★★ | 命名同源、方法学异路；但代表 latent-prediction 在 frontier 已被认真采用 |
| 工程可落地性 | ★★★★ | lmdeploy 部署可直接用；DFlash2 集成改动小 |
| 总评 | **值得跟踪**：所有做"中间层显式建模"的 LNN 衍生研究应监测 NCP-Arch 系列后续；命名巧合容易引起误读，需明确区分两类 NCP |