---
title: LFM2 Technical Report (arXiv:2511.23404) 研读报告
date: 2026-10-22
tags: [LNN, LFM, LiquidAI, LFM2, Hardware-in-the-Loop, Architecture-Search, Gated-Short-Conv, GQA, MoE, Top-K-KD, Edge, On-Device, NR-05, NR-06]
---

# 研读报告：LFM2 Technical Report (arXiv:2511.23404)

> 本报告是 `scripts/select_papers_for_report.py --date 2026-10-22` 给出的
> **全场最高分候选 (41 分)**，也是 [[docs/LNN_深度研读报告]] §0 覆盖度审计
> (2026-10-20) 明确点名"下一步应优先研读、本仓至今无独立研读"的那一篇。
> 本次首次独立研读，**并纠正了本仓 NR-05 的一处版本 off-by-one**（见 §7）。

## 1. 元数据

- **标题**：LFM2 Technical Report
- **作者**：Liquid AI Team（署名 "Liquid AI (2025)"；§10 Authors 列出完整贡献者，
  核心为 Ramin Hasani / Alexander Amini / Mathias Lechner / Maxime Labonne 等）
- **arXiv**：[`2511.23404v1`](https://arxiv.org/abs/2511.23404) · cs.LG · **2025-11-28**
- **License**：论文 CC BY 4.0；**模型权重 LFM 1.0 license**（基于 Apache 2.0，
  对营收 < $10M 的公司开放研究/商用，营收更大的需商业授权）
- **权重**：[`LiquidAI/LFM2-*`](https://huggingface.co/LiquidAI) — 350M / 700M / 1.2B /
  2.6B 稠密 + 8B-A1B MoE（8.3B 总参 / 1.5B 激活），32K 上下文
- **部署栈**：ExecuTorch、llama.cpp、vLLM
- **关联**（本仓已有）：[[docs/reports/LFM2.5_Parallel_CfC_Integration_r304_2026-08-07]]
  （r304 把 CfC 与 LFM2.5 集成）、[[docs/reports/LiquidAI_LFM2.5_Encoder_Family_350M_研读报告]]

## 2. 核心问题

**在边缘设备（手机 / 笔记本 CPU）的延迟与内存硬预算下，什么样的序列建模
backbone 能同时拿到最好的质量与最好的部署指标？**

这个问题对 LNN 社区尤其尖锐，因为 LiquidAI 本身就是"液态"路线的提出方
（Hasani / Lechner 即 LTC 原始作者）。任何"液态即最优"的叙事，都必须先通过
**Liquid AI 自己设的边缘预算关**才能成立。§2.1 正是全文的戏剧性所在：
他们真的把液态算子放进了候选池，然后**没选它**。

## 3. 方法论与核心思路

### 3.1 硬件在环（hardware-in-the-loop）架构搜索 — 本文真正的贡献

不是又一个"我调了超参"，而是在**目标硬件上直接编译并测量**每一个候选架构。

**目标函数（三轴）**：
1. **质量**：内部 50+ 项评测套件（知识回忆 / 多跳推理 / 指令遵循 / 多语言鲁棒性 /
   工具使用 / 数学 / 长上下文），每个候选架构都在**参考数据集上真实训练**后打分
2. **延迟**：TTFT、p50/p95 decode latency (ms/token)、prefill 吞吐 (tokens/s)，batch=1
3. **峰值内存**：prefill + decode 期间的最大 RSS，目标上下文 4K 与 32K

违反任一设备侧预算的候选**直接丢弃**；剩余候选按**质量–延迟–内存 Pareto 前沿的
超体积改进（hypervolume improvement）**排序。

**搜索空间**（关键，全部是"液态/亚二次"家族）：
- **局部上下文块**：深度可分离 1D 短卷积；**LTC 网络（如 CfC, Hasani et al. 2022）**；
  线性注意力；**状态空间变体（S4 / Liquid-S4 / S5 / RTF / Mamba / Mamba2）**
- **全局上下文块**：GQA（组数与 head dim 随搜索变化）+ QK-Norm
- **逐位置块**：SwiGLU 前馈（扩张比随搜索）
- **布局**：局部/全局/逐位置块的交错模式与总块数（含共享权重、cache 复用选项）
- **MoE 选项**：逐层稀疏 FFN（宽度与专家粒度随搜索）

**片上 profiling**：每个候选以**完全相同的设置**（batch=1、固定 context、同一 tokenizer）
编译进部署栈 —
- CPU 路径：ExecuTorch (8da4w) 与 llama.cpp (Q4_0)，跑在 Samsung Galaxy S24 Ultra
  (Snapdragon 8 Gen 3) 与 AMD Ryzen HX 370
- 加速器路径：vLLM（单请求 / online batching，仅作 sanity check，**主目标是设备侧 CPU**）

### 3.2 搜索结果 = 液态算子被自己的厂商否决

> **"Under identical on-device performance budgets, augmenting these stacks
> (as in recent hybrid variants) with linear-attention, state-space, or additional
> convolution operators does not improve aggregate quality on the evaluation suite
> and typically worsens device metrics."**
> — §2.1 Outcomes

搜索在所有尺寸上反复收敛到**极简混合架构**：绝大多数是廉价的 gated short convolution
块，**交错少量 GQA 块**。

原文的因果解释（§8.2）值得逐字记住：

> **"While linear attention and SSM variants can perform global processing of the input,
> we consider them to be in the same class as local context blocks due to their limitations
> in retrieval-intensive tasks."**

以及 §8.2 对 LTC 的定位：

> **"Our work draws inspiration from Liquid Time-Constant (LTC) networks… The liquid
> formulation's ability to dynamically adjust its effective temporal receptive field
> based on input characteristics inspired LFM2's adaptive processing of local and global
> context."**

**注意这里的措辞分寸**：LTC 被放在 **"inspired"（启发）** 位置，不是架构组件位置。
被测的块里含 CfC / Liquid-S4 / S4 / Mamba2 / 线性注意力，**获胜的是纯 conv + GQA**。

### 3.3 Gated short convolution 块（最终选中的算子）

输入 hidden 序列 $\mathbf{h}\in\mathbb{R}^{L\times d}$，块内做输入相关的乘性门控
包裹一个 depthwise 短卷积：

$$
(\mathbf{B},\,\mathbf{C},\,\tilde{\mathbf{h}})=\mathrm{Linear}(\mathbf{h}),\qquad
\mathbf{y}=\mathbf{B}\odot\tilde{\mathbf{h}},\qquad
\mathbf{z}=\mathrm{Conv}_{k}(\mathbf{y}),\qquad
\mathbf{o}=\mathrm{Linear}_{\text{out}}(\mathbf{C}\odot\mathbf{z})
$$

其中 $\mathrm{Linear}:\mathbb{R}^d\to\mathbb{R}^{3d}$ 逐位置作用，输出通道沿特征维切成
$(\mathbf{B},\mathbf{C},\tilde{\mathbf{h}})\in\mathbb{R}^{L\times d}$；
$\mathrm{Conv}_k$ 是沿序列的 depthwise 1D 卷积（核大小 $k$）；$\odot$ 为逐元素乘。

原文对这一算子的定位（§2.2）：
> **"This operator provides fast local mixing with excellent cache behavior on CPUs."**

并附了一句对本仓非常关键的话：
> **"The search results from Section 2.1 can be viewed as an ablation of these hybrids
> in the on-device setting. Once a handful of GQA blocks are available to handle
> long-range retrieval, the inexpensive gated short convolution alone is sufficient
> to reach the best quality–latency–memory trade-off we observe, without additional
> linear attention/SSM/long convolution branches."**

### 3.4 训练侧三个可迁移设计

- **Decoupled Top-K KD**（§3.3 + 附录 A）：用 **LFM1-7B 作 teacher**，但只蒸馏
  Top-K=32 的 teacher logits。对 Top-K 截断分布直接做 KL + 温度缩放会产生
  **support mismatch**（截断后 teacher 分布的支撑集与 student 分布不一致），
  朴素做法会导致损失不稳定。解法是**沿 chain rule 分解 KL**（附录 A.1 给出
  forward KL 的链式恒等式与下界），把"Top-K 截断"与"温度"两个因素**解耦**。
- **三阶段 post-training**（§4）：SFT → length-normalized DPO → **模型合并**
  （附录 B：Model Soup / Task Arithmetic / TIES-Merging / DARE / DELLA）。
- **MoE 细节**（§2.3）：MoE 专家本身是 SwiGLU MLP；每层 **32 个专家、Top-k=4**，
  归一化 sigmoid router + adaptive routing bias 做负载均衡。（注：**24B-A2B 那一代
  改成了 64 专家 Top-4**，见 §6。）

## 4. 关键成果

| 维度 | 结果 |
|---|---|
| 架构 | 极简混合：gated short conv 为主 + 少量 GQA；**液态/SSM 算子被实测否决** |
| 速度 | CPU 上 prefill 与 decode 相对同尺寸基线**最高 ~2× 加速**（Snapdragon 8 Elite / Galaxy S25、Ryzen HX 370，llama.cpp，batch=1） |
| 规模 | 350M / 700M / 1.2B / 2.6B 稠密 + 8B-A1B MoE（8.3B 总 / 1.5B 激活），全系 32K 上下文 |
| 质量 | LFM2-2.6B：IFEval **79.56%**、GSM8K **82.41%**；LFM2-8B-A1B 以 ~1.5B 激活达到 3–4B 级质量 |
| 预训练 | 10–12T tokens；数据配比 ~75% 英文 / 20% 多语 / 5% 代码（MoE 变体代码占比更高：60/25/…） |
| 衍生 | LFM2-VL（视觉-语言，SigLIP2 NaFlex 塔）、LFM2-Audio（语音，实时 speech-to-speech）、LFM2-ColBERT-350M（晚交互检索） |

## 5. 局限性与未来展望（作者自述，§9.2）

- **硬件与部署覆盖有限**：仅在两类 x86/ARM **CPU**（手机 SoC、笔记本 CPU）上做过
  片上测量；**NPU / GPU 路径未覆盖**，加速器仅作 sanity check。
- **容量与任务覆盖有限**：最大 8.3B（MoE）；内部 50+ 评测套件**未公开**，
  外部无法复现"质量"这一轴的排序。
- **多模态范围有限**：VL / Audio / ColBERT 各自独立训练，非统一多模态模型。
- **Late Interaction Models**：ColBERT 路线单独讨论，与主干解耦。

## 6. 对本仓的直接影响（LFM2.5-24B-A2B，2026-10-21 新发布）

本仓 §0 记的是"LFM2 → LFM2.5 把 ODE 换成 double-gated conv + GQA"。
今天 `LiquidAI/LFM2.5-24B-A2B` 上线，**其 `config.json` 给出了最硬的证据**
（2026-10-22 直接拉取核对）：

```json
"architectures": ["Lfm2MoeForCausalLM"],
"model_type": "lfm2_moe",
"num_hidden_layers": 40,
"layer_types": ["conv","conv","full_attention", ... 40 项 ... ],
"num_experts": 64, "num_experts_per_tok": 4,
"hidden_size": 2048, "vocab_size": 128000, "max_position_embeddings": 131072
```

**40 层里 30 层是 `conv`、10 层是 `full_attention` —— 没有一层是 ODE / LTC / CfC。**
且模型卡自述："LFM2.5-24B-A2B keeps the LFM2-24B-A2B architecture with two changes:
the context window grows from 32K to 131K tokens, and the vocabulary doubles from 65K to 128K."

即：**"液态"在 LFM 系里的存在方式，是 2025-11 的那次架构搜索里被当作候选测过、
然后被自己否决掉的东西**；LFM2.5 只是把 conv 换成 double-gated 变体并放大到 24B。
这比"厂商自己降级液态"的说法更精确，也更有杀伤力。

> **本报告未做的事（诚实边界）**：未 clone 权重、未在 Jetson 上跑过 24B-A2B、
> 未复现 §2.1 架构搜索本身。24B-A2B 的参数量与显存需求决定了它
> **不适合 Orin Nano**（与本仓 [[analysis/jetson/]] 记录的 LFM2.5-1.2B Q4_0
> ~696MB / 30-100 tok/s 档位不是一个量级）。本仓 Jetson 路线仍锚定 350M–2.6B 档。

## 7. ⚠️ 对本仓 NR-05 的 off-by-one 纠正（重要）

本仓 NR-05 / §0 原文写作：

> "LFM2 → LFM2.5 把『完整 ODE + RK4/Euler solver』替换为『double-gated conv + GQA』"

**今天逐字核对 LFM2 Technical Report 原文后，这个版本归属是错的：**

| 项 | 本仓原表述 | 核对后的事实 |
|---|---|---|
| 何时换掉 ODE | LFM2 → LFM2.5 | **LFM1 → LFM2**。LFM2 (2025-11-28) 自己的 abstract 就已经是 "gated short convolutions with a small number of grouped query attention blocks"，**LFM2 里就没有 ODE** |
| 换成了什么 | double-gated conv + GQA | LFM2 = **gated short conv**（单门控，§3.3 公式）；"double-gated" 是 LFM2.5 的 conv 变体 |
| 换掉的方式 | 暗示"悄悄替换" | **不是悄悄替换，是架构搜索里正面测过并否决**。§2.1 搜索空间显式包含 CfC / Liquid-S4 / S4 / S5 / Mamba / Mamba2 / 线性注意力，结论是它们 "does not improve aggregate quality … and typically worsens device metrics" |

**纠正后的表述（建议替换本仓原文）**：

> **LiquidAI 在 LFM2 的硬件在环架构搜索里，把 CfC / Liquid-S4 / SSM / 线性注意力
> 显式放进候选空间实测，并在边缘延迟–内存预算下选择了纯 conv + GQA；
> "液态"从此只是 LFM 的灵感来源（"inspired"），不再是 backbone 组件。
> LFM2.5-24B-A2B（2026-10-21）的 `config.json` 确认 40 层中 30 层 `conv` +
> 10 层 `full_attention`，无任何 ODE/LTC/CfC 层。**

**为什么这条纠正比原文更重要**：原文说"厂商降级液态"，读者容易理解成
"工程妥协 / 无奈退让"。核对后的事实是**液态算子被正式放进对照实验并输掉了**，
且输在**质量**轴（不是只输在延迟轴）。这直接加强 [[analysis/negative_results/README]]
NR-05 的判定，且与 NR-06（"LNN 在标准主基准上从未显著超过 transformer"）
互为佐证 —— **连液态的原创团队在自家主线上也没能让液态赢过纯 conv**。

## 8. 对本仓路线图的影响

- **路线 A（Parallel-CfC 边缘推理化）不受影响**：本仓的 CfC 用法是
  "**作为 LFM2.5 之上的一个时间序列 head / adapter**"，不是"替换 LFM backbone"。
  §2.1 的结论恰恰支持这种用法：液态块作为**局部时序组件**（而非全局 backbone）
  才可能有价值 —— 原文自己把 SSM 与 conv 归为"同一类局部上下文块"。
  → **本仓路线与厂商证据不冲突，但"液态必须做 backbone"的读法被否掉了**。
- **r304（LFM2.5 + CfC 集成）需要重新定位**：不能再表述为"LFM2.5 是液态模型所以
  我们在液态模型上加 CfC"，而应表述为"**在生产级边缘 backbone 之上挂载一个
  连续时间时序 head**"。建议在 r304 报告里加一句交叉引用。
- **可借鉴的训练技术**：`Decoupled Top-K KD`（附录 A 的 chain-rule KL 分解）
  与本仓蒸馏路线（DLNet 双阶段 KD、r304 路线）方法论同族，可作为
  teacher=大模型的**蒸馏目标函数**候选。
- **可立即复用的部署事实**：llama.cpp (Q4_0) + ExecuTorch 双栈、batch=1、
  手机 SoC 上做片上 profiling —— 与本仓 Jetson 验证脚本的"目标硬件上实测"方法论一致。

## 9. Verdict

**TARGET-POSITIVE（对厂商工程方法论）/ NEGATIVE-POSITIVE（对"液态必须做 backbone"）**

- 架构搜索方法论（硬件在环 + 三轴超体积 Pareto）**值得本仓借鉴**，且其
  搜索空间显式纳入液态家族使得否证结论**没有幸存者偏差**。
- 对"液态"的结论是**上游一手证据**（作者自述，非我方复现），但**证据等级高**：
  出自原始论文、含搜索空间描述、含被测家族清单、含量化结论。
- ⚠️ **未验证项**：内部 50+ 评测套件未公开 → "质量"轴无法外部复核；
  我方未复现 §2.1 搜索；24B-A2B 未在本仓 Jetson 上实测（也不建议测，见 §6）。

---

**关联**

- 全局研读索引：[[docs/LNN_深度研读报告]] §0
- 负结果台账：[[analysis/negative_results/README]] NR-05 / NR-06 / NR-09
- 当日研究笔记：[[analysis/research/2026-10-22/2026-10-22_lnn_research_note]]
- 论文：https://arxiv.org/abs/2511.23404 · 权重：https://huggingface.co/LiquidAI
