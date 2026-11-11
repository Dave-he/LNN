---
title: "MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery"
date: 2026-03
tags: [LNN, LFM2, foundation-model, drug-discovery, molecular-LLM, domain-adaptation, SFT-RFT, ADMET, retrosynthesis]
---

# 研读报告：MMAI Gym for Science — 用"数据 + 训练 + 评测"三件套把 2.6B 的 LFM2 训成近专家级的药物发现模型

## 1. 元数据

- **论文标题**：MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery
- **作者**：Maksim Kuznetsov, Zulfat Miftahutdinov, Rim Shayakhmetov, Mikolaj Mizera, Roman Schutski, Bogdan Zagribelnyy, Ivan Ilin, Nikita Bondarev, Thomas MacDougall, Mathieu Reymond, Mihir Bafna, Kaeli Kaymak-Loveless, Eugene Babin, Maxim Malkov, **Mathias Lechner**, **Ramin Hasani**, Alexander Amini, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov
- **机构**：Insilico Medicine（多数作者）+ **Liquid AI**（Lechner / Hasani / Amini / Aladinskiy）
- **发表时间**：2026-03-03 (arXiv: 2603.03517v2, cs.LG)
- **代码**：文中未给出公开仓库；MMAI Gym 有官网（LFM2-MMAI checkpoint 作者称"计划发布"或可经 LFM2-MMAI AC 申请）
- **本地 PDF**：[papers/2603.03517/2603.03517.pdf](../../papers/2603.03517/2603.03517.pdf)
- **本地文本**：[papers/2603.03517/2603.03517.txt](../../papers/2603.03517/2603.03517.txt)
- **关联概念**：LFM2 (Liquid Foundation Model) / 领域适配 / SFT + RFT (强化微调) / 分子基础模型 / ADMET 预测 / 逆合成 / 3D 口袋条件生成 / 评测去污染
- **本仓定位**：**"Liquid" 命名陷阱的重点案例** —— 这是 **Liquid AI 公司出品的 LFM2（Conv+GQA 混合架构）**，**不是**本仓关注的 Hasani 谱系的连续时间液态网络。见 §7 的口径纠偏。

## 2. 核心问题

通用大语言模型依靠 in-context learning，在药物发现这类需要**领域理解**的任务上表现不可靠。论文指出两条常见补救路线都收益有限：

- **单纯增大模型规模**：不带来显著提升；
- **引入 reasoning tokens**：同样不带来显著提升。

作者的诊断是：药物发现的问题**不是"知识量不够"，而是"表达形式不对"** —— 药物化学家的实际工作流使用的是 SMILES / SELFIES / IUPAC 等分子表示、化学反应链、3D 结构与蛋白口袋，而不是自然语言续写。把药物发现问题当作"领域特定的推理任务（domain-specific reasoning）"而非"纯文本预测"，是本文的核心立场。

同时，行业缺乏**统一的数据格式 + 训练配方 + 评测协议**的成套设施：每个团队重复造轮子、评测口径不一、且普遍存在**训练/测试数据泄漏**风险。

因此论文把问题重新表述为：

> **能否用一套"数据 + 训练 + 评测"的一站式环境（MMAI Gym），把一个高效但通用的小型架构（LFM2-2.6B）适配成在药物发现多任务上达到近专家级水平、且比大 10 倍的通用模型更有效的单一模型？**

## 3. 方法论与核心思路

### 3.1 总体架构（MMAI Gym 三件套）

```
┌─ MMAI Gym for Science ────────────────────────────────────────────┐
│ (i) 数据层                                                       │
│    · 分子多模态格式: SMILES / SELFIES / IUPAC / 分子图 / 反应 / 3D │
│    · 领域推理数据集: 分子优化链、合成与反应推理、DMPK、毒性、几何任务 │
│    · 多任务数据平衡 (balancing)                                    │
│ (ii) 训练层                                                      │
│    · 阶段1: 监督微调 SFT (multi-task 或 single-task)              │
│    · 阶段2: 强化微调 RFT (奖励: 快速 ground-truth + 预测式奖励)     │
│    · 推理时可选 reasoning 模式                                     │
│ (iii) 评测层                                                      │
│    · 自动数据去污染 (train/test overlap 检查)                      │
│    · 公开基准 + 内部 OOD 基准 (分布偏移评估)                        │
└───────────────────────────────────────────────────────────────────┘
        │ 领域适配
        ▼
   LFM2-2.6B (稠密) / LFM2-24B-A2B (MoE)
```

### 3.2 LFM2 架构（Appendix B）—— **这是与本仓最相关的部分**

LFM2 是 **Liquid AI** 的混合架构，由两种序列算子构成：

- **主力：门控短卷积 (ShortConv)** —— 负责序列特征化；
- **少量：分组查询 softmax 注意力 (GQA)** —— 负责全局序列混合。

每个 block 由两段 **Norm-Operator-Residual** 组成：先 ShortConv 或 GQA，再 SwiGLU 前馈；使用 **pre-RMSNorm**（省略 LayerNorm 的中心化步骤以提升效率）。

**门控短卷积算子的核心公式**（对隐藏序列 $h \in \mathbb{R}^{L \times d}$）：

$$(B, C, \tilde{h}) = h W_{in} \tag{Eq.1}$$

$$o = \left( C \odot \mathrm{Conv}_k(B \odot \tilde{h}) \right) W_{out} \tag{Eq.2}$$

其中 $W_{in} \in \mathbb{R}^{d \times 3d}$、$W_{out} \in \mathbb{R}^{d \times d}$ 为可学习线性投影，$\mathrm{Conv}_k$ 为 **depthwise 1D 卷积，$k=3$**，$\odot$ 为 Hadamard 积。局部算子编码短程"motif"，长程上下文由少量全局注意力算子整合。

### 3.3 训练与评测协议

- **两阶段**：SFT → RFT；两种训练设置：**MT**（multi-domain multi-task）与 **ST**（single-category，注意 ST 仍可含多任务，如 ADMET 类含 22 个属性预测任务）。
- **评测**：每个测试样本**多次重复评测**，每次从**改写指令模板池**随机采样组装 prompt 并配 few-shot 示例（从训练集采样）；连续量取**中位数**、分类量取**多数投票**。
- **基线口径（作者自陈的不对称性）**：LFM2-MMAI **在相关任务分布上训练过**，而专有基线（GPT-5.5、Claude 4.6 Opus 等）仅用 few-shot prompting、**未经训练**。未按本方指令格式微调的开源基线则使用作者提供的 prompt。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq.1 | $$(B, C, \tilde{h}) = h W_{in}$$ | ShortConv 输入投影：$W_{in} \in \mathbb{R}^{d\times 3d}$，生成 3 份（B、C、h̃） |
| Eq.2 | $$o = \left(C \odot \mathrm{Conv}_k(B \odot \tilde{h})\right)W_{out}$$ | **门控 depthwise 短卷积**：$C$ 为门控信号，$k=3$，$W_{out}\in\mathbb{R}^{d\times d}$ |
| LFM2 block | Norm → (ShortConv \| GQA) → Residual → Norm → SwiGLU → Residual | 混合算子结构；pre-RMSNorm 变体 |

## 5. 关键成果与贡献

### 5.1 主结果

**(a) 多目标分子优化 — Success Rate（Table 1, MuMO-Instruct）**

| 模型 | BDP | BDQ | BPQ | DPQ | BDPQ |
|---|---|---|---|---|---|
| GPT-5.5 | 83.2 | 91.6 | 92.6 | 72.4 | 77.0 |
| Claude 4.6 Opus | 69.4 | 62.2 | 98.6 | 73.2 | 56.4 |
| GeLLM3O-3Mistral（任务专用） | 84.8 | 87.0 | 93.0 | 62.8 | — |
| GeLLM3O-3Llama（任务专用） | 86.8 | 90.0 | 94.0 | 60.6 | — |
| **LFM2-2.6B-MMAI, MT** | **86.0** | 91.6 | **98.8** | 71.2 | 56.4 |
| **LFM2-2.6B-MMAI, ST** | 86.0 | 91.2 | **98.8** | 69.8 | 57.8 |
| LFM2-24B-A2B-MMAI, ST | 68.2 | 78.8 | 97.2 | 60.2 | 60.4 |

> **口径要点**：**最难的四属性设定 BDPQ 上，LFM2-2.6B-MMAI（56.4/57.8）低于 GPT-5.5（77.0）与 GeLLM3O-3Llama（60.6）**；2.6B 稠密模型在多数设置上超过 24B-A2B MoE。作者的正面表述限定在"最难的四属性设定上 MoE 架构处理得更好"。

**(b) 功能团推理 — FGBench（Table 2，Acc / RMSE）**

| 模型 | Single Acc | Single RMSE | Interaction Acc | Interaction RMSE | Comparison Acc | Comparison RMSE |
|---|---|---|---|---|---|---|
| GPT-5-5 | 0.772 | 187.4 | 0.777 | 56.0 | 0.789 | 77.7 |
| Opus-4-5 | 0.761 | 105.0 | 0.729 | 51.9 | 0.777 | 71.6 |
| Llama-3.1 70B | 0.683 | 84.1 | 0.530 | 38.6 | 0.456 | 64.8 |
| DeepSeek-3-2 | 0.629 | 68.8 | 0.476 | 44.6 | 0.494 | 55.5 |
| **LFM2-2.6B-MMAI, MT** | **0.840** | 58.8 | 0.815 | 27.1 | 0.801 | 48.3 |
| **LFM2-2.6B-MMAI, ST** | **0.840** | **55.9** | **0.819** | **25.0** | **0.810** | **50.6** |
| LFM2-24B-A2B-MMAI, ST | 0.820 | 50.2 | 0.780 | 31.0 | 0.790 | 47.3 |

> 2.6B 在此任务上**全面超过 GPT-5.5 / Opus-4-5 / Llama-3.1 70B / DeepSeek-3-2**，是本文最干净的正面证据。

**(c) 单步逆合成 — URSA-expert-2026（Table 3）**

| 模型 | Unique % | Max Av. PT | @3 | @5 | @10 |
|---|---|---|---|---|---|
| Gemini 3 Flash preview | 40% | 1.80 | 1.21 | 0.83 | 0.42 |
| Grok-4.1 | 43% | 1.75 | 1.29 | 0.93 | 0.49 |
| NatureLM（化学专家） | 27% | 1.57 | 0.97 | 0.61 | 0.30 |
| C3LM（化学专家） | 42% | 1.69 | 1.14 | 0.78 | 0.40 |
| **LFM2-2.6B-MMAI, MT** | **94%** | **1.45** | **0.84** | 0.55 | 0.28 |
| **LFM2-2.6B-MMAI, ST** | 93% | **1.43** | 0.90 | **0.60** | **0.30** |
| LFM2-24B-A2B-MMAI, ST | 70% | 1.73 | 1.38 | 1.09 | 0.63 |

> **口径要点**：唯一性（94% vs 27–43%）是**碾压级**的，且平均合理性（Av. PT 1.43–1.45）**最低即最好**，即 LFM2-MMAI 在"平均"上更合理，但**最高合理性（Max Av. PT）低于 Gemini/Grok**。作者自陈：专有基线"只在很窄的一组预测上取得高最大合理性"，而 MMAI 模型"在多个不同候选上维持强平均合理性"。另需注意 **24B-A2B 的 MT→ST 表现反而更差（Unique 70% vs 94%）**。

**(d) ADMET/PK/毒性 — TDC（Table 4，节选）**

| 任务 | 指标 | LFM2-MMAI 2.6B | TxGemma-27B | TDC SOTA |
|---|---|---|---|---|
| Caco2 Wang | MAE ↓ | **0.347** | 0.36 | 0.401 |
| Lipophilicity AZ | MAE ↓ | **0.434** | 0.57 | 0.538 |
| Solubility AqSolDB | MAE ↓ | 0.812 | 0.84 | **0.741** |
| PPBR AZ | MAE ↓ | 8.00 | 8.62 | **7.44** |
| Pgp Broccatelli | AUROC ↑ | 0.863 | 0.91 | **0.937** |
| HIA Hou | AUROC ↑ | 0.987 | 0.99 | 0.988 |
| BBB Martins | AUROC ↑ | 0.924 | 0.96 | 0.908 |
| Bioavailability Ma | AUROC ↑ | 0.751 | 0.74 | **0.694** |
| CYP3A4 Substrate | AUROC ↑ | 0.681 | 0.7 | **0.667** |
| LD50 Zhu | MAE ↓ | 0.703 | 0.67 | **0.627** |
| hERG | AUROC ↑ | 0.787 | 0.83 | **0.885** |
| AMES | AUROC ↑ | 0.782 | 0.81 | **0.871** |

> **口径要点**：**LFM2-2.6B 在多数端点上优于大 10 倍的 TxGemma-27B，但在多个端点上仍不及 TDC 排行榜专有模型**（作者自陈"specialist leaderboard models still retain an advantage on some difficult regression endpoints"）。**不得**表述为"2.6B 全面超越专有模型"。

**(e) 3D 分子分布学习（Table 5，JS 散度，越小越好）**

| 指标 | MolDiff | OpenLLaMA | Anach | **LFM2-2.6B-MMAI** | LFM2-24B-A2B-MMAI |
|---|---|---|---|---|---|
| JS. bond lengths | 0.436 | 0.257 | 0.148 | **0.116** | 0.122 |
| JS. bond angles | 0.181 | 0.136 | 0.096 | **0.094** | 0.098 |
| JS. dihedral angles | 0.198 | 0.110 | 0.110 | **0.101** | 0.139 |
| JS. freq. bond types | 0.170 | 0.031 | 0.046 | **0.040** | 0.066 |
| JS. # rings | 0.079 | 0.033 | 0.56 | **0.081** | 0.111 |

> **口径要点**：LFM2-MMAI 在**几何分布指标**上最好，但在**离散的键/环分布**上仍落后于 OpenLLaMA（0.031 vs 0.040、0.033 vs 0.081）。作者自陈"3D 分子生成仍是困难且异质化的任务"。

**(f) 口袋条件 3D 配体生成**：两个 LFM2-MMAI 变体在所有语言模型中取得**最好的 UniDock 结合亲和力分数**，显著超过专有与开源 LLM 基线，接近专家扩散模型水平。作者的对比观察：GPT-5.5 生成**有效性好、几何合理**但亲和力中等；MMAI 模型更直接优化亲和力目标。

### 5.2 消融结论（Appendix F）

作者总结四条：**MMAI Gym 训练是提升的主要来源**；**多任务训练通常带来更好的鲁棒性**，而单任务在个别端点受益；**reasoning 对组合性强的任务最有帮助**，对布尔型问题则非 reasoning 是更省的选项；**扩大规模带来的是任务相关的、非均匀的增益**。

### 5.3 贡献清单

- **Claim 1**：**MMAI Gym** —— 一套覆盖数据格式/多模态、训练配方（SFT+RFT）、评测（含去污染与 OOD）的**一体化**药物发现环境。
- **Claim 2**：经适配的 **2.6B LFM2 在多个药物发现基准上达到近专家级表现**，并在多数设置下超过更大的通用模型与开源模型（同时在部分端点不及 TDC 专有模型）。
- **Claim 3**：**领域适配后训练可以补偿模型规模的巨大差距**（2.6B vs TxGemma-27B 约小 10 倍）。
- **Claim 4**：**多任务训练在若干场景下反超强单任务专家模型**，说明更广的训练可以提升准确率与鲁棒性而不稀释性能。

## 6. 局限性与未来展望

### 6.1 作者自陈局限（§6 "Ethical Considerations and Limitations"）

- **安全与双用风险**：分子语言模型可能被滥用于生成有害/毒性/双用化合物；尽管 MMAI Gym 聚焦类药空间，**风险未被消除**（毒性任务虽设计为预测而非最大化，但目标可被反转）。模型**未经过对抗红队或安全对齐**。
- **评测局限**：依赖既有计算基准，基准可能含**分布伪影**、**边缘案例覆盖有限**，基准上的成功**不保证**在专有工业数据或训练分布外的新化学空间中同样成立。
- **无前瞻实验验证**：**没有**通过合成与生物测定对生成的分子做前瞻性实验验证，仅依赖既有的公开 ground truth。
- **checkpoint 未完全开放**：计划发布 LFM2-MMAI checkpoints，或经 LFM2-MMAI AC 申请获取。

### 6.2 隐含局限（本仓视角）

- **"Liquid" 是同名不同物（关键）**：本文的 LFM2 是 **Liquid AI 公司的混合架构（ShortConv + GQA）**，**与本仓关注的 Hasani 谱系连续时间液态网络（LTC/CfC）无任何架构关系**。本仓既有知识 NR-05 已判定"LFM2.5 是纯液态模型"为**事实错误**；**本文进一步说明 LFM2 的"液态"是品牌命名而非架构属性**。**引用本文时不得用它作为"液态架构有效"的证据**。
- **基线对比存在训练/未训练的不对称**（作者自陈）：LFM2-MMAI 训练过、GPT-5.5/Opus 只做 few-shot。因此"2.6B 打败 GPT-5.5"的正确解读是"**2.6B 在此域经领域适配后胜过通用前沿模型的 zero/few-shot 使用**"，**不是"小模型能力强于大模型"**。
- **内部 OOD 基准不可复现**：论文用了内部基准与内部数据，外部读者无法核验这部分。
- **去污染范围仅限其使用的公开集**，与"新分子空间"无关。
- **数字精度**：表格为 3 位有效数字（如 0.847），未见重复实验的方差报告。

### 6.3 未来方向

- 前瞻性合成 + 生物测定验证（作者明确列为缺失项）。
- 对分子模型做安全对齐与红队。
- 开放 checkpoint 与完整评测协议以支持独立复现。
- 探索"效率优先的混合架构"（线性注意力 / SSM 为主的层 + 少量全局注意力层）的更优配比。

## 7. 对本仓的意义

- **可作 `lnn/core/variants.py` 的什么变体**：**不是变体，而是一条警示线**。本文展示的"高效混合架构 + 领域适配 + 成套评测"是 LFM2 路线，与本仓的连续时间路线**正交**。对本仓的直接价值是**评测工程**：其"去污染 + 公开/内部 OOD 双轨 + 多次重复取中位数/多数票 + 多提示模板池"的评测协议，是本仓 toy benchmark 目前**缺失**的严谨性来源，可直接移植。
- **可嫁接到哪个 bench 脚本**：可把 MMAI 的评测协议移植为 `scripts/` 下的 `bench_protocol.py` —— 特别是**训练/测试重叠自动检查**（对应本仓 AGENTS.md 中负结果台账"grounding 路径必须真实存在"的纪律精神），以及**多提示模板 + 多次重复取中位数**的稳健统计口径。
- **提供哪个新 ablation axis**：**(a) "适配 vs 未适配"的规模差轴**（小模型经领域适配能否补上规模差）；**(b) MT vs ST 训练轴**（本文给出 MT 往往更鲁棒、ST 偶尔更好，本仓 toy bench 可加同款对照）；**(c) reason on/off 轴**；**(d) "架构效率 vs 领域适配"的正交消融**。
- **对本仓既有结论的影响**：**无直接支持或反驳**。本文**不涉及** LTC/CfC/NCP 的任何比较，**不可用于**支持或反驳 NR-01/NR-02/NR-06。其价值在于两点：(1) 提供"**高效架构 + 领域适配**"这条**非连续时间**路线的强证据，提醒本仓"LNN 路线"的比较必须带上"同预算的非 LNN 混合架构"这一基线，否则"LNN 更高效"的结论会被质疑；(2) 提供"**评测严谨性**"的工程模板。

**Verdict**：
- 对 **"领域适配可以让小模型在垂直任务上匹敌大模型"**：**TARGET-POSITIVE** —— 在 FGBench、逆合成、3D 几何等多项任务上证据充分且数字明确；但**必须带"基线未经训练"的口径限定**。
- 对 **"LFM2 是液态/连续时间模型"**：**TARGET-NEGATIVE-WITH-NUANCE** —— 事实层面**否证**：LFM2 是 ShortConv+GQA 混合架构，无 ODE/闭式连续时间成分；与本仓 NR-05 一致并进一步强化。
- 对 **"LFM2-MMAI 全面超越专有模型"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 在 FGBench / 逆合成唯一性 / 口袋亲和力上成立；在**最难的四属性优化（56.4 vs GPT-5.5 的 77.0）**、**多个 ADMET 端点对 TDC 专有模型**、**离散键/环分布**上**不成立**。引用时须逐表标注。
- 对 **"本仓 LNN/LTC 路线"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 无直接证据；但它设定了一个**必须面对的对照基线**（同参数量级的非液态高效架构 + 领域适配）。
