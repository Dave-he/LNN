---
title: "Liquid Structural State-Space Models (Liquid-S4)"
date: 2026-10-26
tags: [LNN, LTC, 状态空间模型, S4, 结构性SSM, 长序列, 自相关, 复算, 论文-代码不一致]
arxiv_id: 2209.12951v1
---

# Liquid Structural State-Space Models —— 研读报告

> **本报告定位**：对 arXiv:2209.12951v1 (Liquid-S4) 的独立研读。
> 证据分三层，逐层标注：
> - **【读层复算】**：只用论文自身给出的公式与表格数字做的算术/代数核验，结论写在 §4。
> - **【代码层核验】**：在第三方公开实现 (`github.com/raminmh/liquid-s4`) 上实跑，核验"实现是否做了论文声称的事"，结论写在 §4.6 与 §5.3。
> - **【作者自报】**：论文自己声称的数字，未经我方复现，标注在 §5.1。
>
> **诚实约束前置（三条，读者请先读）**：
> 1. **本报告不构成"我方复现失败"**。我**没有**跑 LRA 六任务的全量训练，也没有复现论文的任何主实验数字。本报告的所有"不一致"都是 (i) 论文自身数字的算术核验，或 (ii) 第三方开源实现的**代码级行为**核验。按 `AGENTS.md` 约束 #4 与 `analysis/negative_results/README.md` §1 的收录门槛（"我方复现失败"），**本篇不新增台账条目，NR-01~NR-09 不变**。
> 2. **本报告的所有"LRA SOTA"数字都是作者自报**。本仓目前**没有**任何本地 grounding 支持这批数字（见 §7.3），因此**不得**用本篇去支持或反驳台账 **NR-06**（"LNN 在标准主基准上 SOTA 横扫 transformer"）—— 本篇既不构成对 NR-06 的否证，也不构成支持。
> 3. **本篇不是"液态神经网络"论文**。它是**线性化 LTC**（`ẋ=(A+Bu)x+Bu`）在结构性 SSM 核空间里的一次实现尝试，不含 nonlinearity、不含可学习时间常数、不含门控 —— 与本仓 `lnn/core/ltc.py` 的完整 LTC 不是同一个东西（对照见 §7.2）。
>
> 复算脚本为本次研读的一次性产物，置于 `/tmp`（`/tmp/ls4_boot.py`, `/tmp/ls4_test{,2,3,4,5,6,7}.py`），**未纳入本仓 `scripts/`**；所有数字均在本报告里写全，便于独立复核。

---

## 1. 元数据

| 项 | 内容 |
|---|---|
| 标题 | Liquid Structural State-Space Models |
| arXiv | **2209.12951v1** [cs.LG]，提交日期 **2022-09-26** |
| 作者 | Ramin Hasani\*, Mathias Lechner\*（共同一作）, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, Daniela Rus |
| 单位 | Computer Science and Artificial Intelligence Lab (CSAIL), Massachusetts Institute of Technology (MIT), Cambridge, MA |
| 通讯 | rhasani@mit.edu |
| 官方代码 | `https://github.com/raminmh/liquid-s4`（本仓核验版本：`main` @ `06031dc` "Update README.md", 2023-06-15；`src/models/sequence/ss/s4.py` 最后一次改动为 `d3a72bd9` "Bbar instead of B bugfix", 2022-08-30） |
| 本仓 PDF | `papers/2209.12951/2209.12951.pdf`（595,210 B），抽取文本 `papers/2209.12951/2209.12951.txt`（77,178 字符） |
| 会议 | 社区普遍引用为 ICLR 2023。**本次未从 PDF 正文逐字确认会议信息**（v1 稿头只有 arXiv 戳），引用时请以 ICLR 2023 proceedings 为准并自行核对 |
| 关键词 | 状态空间模型 / S4 / HiPPO / LTC / 长序列 / LRA / 语音命令 / BIDMC |

**标签**：LNN, LTC, 状态空间模型, S4, 结构性SSM, 长序列, 自相关, 复算, 论文-代码不一致

---

## 2. 核心问题

### 2.1 S4 的软肋：状态转移矩阵是"死"的

S4 家族（Gu et al. 2022a/b/c）的骨架是**线性时不变**（LTI）连续系统：

$$\dot{x}(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t) \tag{Eq. 1}$$

其中 $A$ 走 HiPPO / LegS 路线（为了记住信号历史），$B, C$ 可学，$D$ 是 skip connection。
关键点：**$A$ 是学出来的常数**。它对每一条输入序列都一模一样，因此：

- 系统是 **input-independent** 的 —— 换个分布、换个序列长度，"记住什么"的机制不变；
- 好处是线性，可以走 Cauchy/FFT 卷积核，复杂度 $\tilde{O}(N+L)$，序列长到 16k 也不掉速。

### 2.2 液态网络（LTC）的卖点：状态转移**随输入变**

液态网络把 $A$ 换成 $A + B u(t)$（线性化版本）：

$$\dot{x}(t) = (A + B u(t))\,x(t) + B u(t), \qquad y(t) = C x(t) \tag{Eq. 8}$$

作者在 §1 的说法是：LTC "**are causal continuous-time neural networks with an input-dependent state transition module**"，因此是 "**dynamic causal models**"。也就是说，模型可以对每条输入**在线调整自己的时间常数/状态演化速率** —— 这正是 §2.1 里 S4 缺的东西。

### 2.3 但 LTC 走不远

LTC 是 ODE 形式，需要数值求解器（RK4 / Euler）。求解器带来三重代价：训练慢、长序列梯度不稳、batch 内样本长度必须对齐。§1 的原话：LTC 的 "complexity … is bottlenecked by their differential equation numerical solver that limits their scalability to longer-term sequences"。

### 2.4 本文要回答的问题

> **能不能既保留 LTC "状态转移随输入变"的能力，又保留 S4 的线性卷积核效率？**

作者的答案：**能，但不是把 ODE 求解器留着，而是把"输入依赖"搬到核空间去**。
具体做法：S4 的卷积核 $K$ 只负责"各个时刻独立"的那部分权重；再额外做一个 **liquid kernel** $K_{liquid}$，负责"输入各阶自相关"的那部分权重（Eq. 10 里作者用黑色/紫色区分这两类权重）。最终

$$y_k = (K * u)_k + (K_{liquid} * u_{correlations})_k \tag{Eq. 11}$$

作者自报的卖点是：LRA 六任务**全部第一**，平均 **87.32%**（对 S4-LegS 的 86.09%）；Speech Command full-labels 16kHz **96.78%** 且参数量 **224K**（S4-LegS 307K）。

### 2.5 本文**没有**回答（我方识别，且对 §5 的解读至关重要）

1. **增益到底来自"液态"还是来自"多加了一条输入高阶乘积通道"？** 论文**没有**做这个消融（见 §4.5 的分析：PB 模式下 $A$ 被退化为 Identity，液体核退化为与 lag 无关的常核，"输入依赖"完全落在 $u_{correlations}$ 上）。
2. **论文自己宣称的 "causality capabilities" 在公开实现里不成立**（§4.6，实测非因果）。
3. **PB 模式下 $A\to I$ 意味着 HiPPO/LegS 的"结构化记忆"在液体核里被丢掉了** —— 作者没说这意味着什么。

---

## 3. 方法论与核心思路

### 3.1 S4 基座：HiPPO → NPLR → DPLR → Cauchy 核

论文 §3 沿用 S4 的三件套：

- **HiPPO 矩阵**（LegS 测度，Eq. 14）：$A_{nk}$ 的显式闭式；
- **NPLR 分解**（Eq. 15）：$A = V(\Lambda - (V^*P)(V^*Q)^*)V^*$，把 $A$ 写成"对角 + 低秩"；
- **Cauchy 核 + Woodbury 恒等式**（Algorithm 1 行 1–5）：把生成函数在单位根上求值，逆 FFT 得到卷积核 $K$。

Algorithm 1 行 1–5 是**黑**的（S4 原样），行 6–15 是**紫**的（Liquid-S4 新增）。

### 3.2 线性液态 SSM 及其离散化

Eq. 8（连续）→ Eq. 9（离散，作者称"双线性变换 / 梯形法"）：

$$x_k = (A + B u_k)\,x_{k-1} + B u_k, \qquad y_k = C x_k \tag{Eq. 9}$$

**本报告对 Eq. 9 的读层复算见 §4.1：Eq. 9 并不是梯形法的结果。**

### 3.3 展开：哪些权重属于"自相关"

假设 $x_{-1}=0$，把 Eq. 9 递推展开（Eq. 10）：

```
x0 = B u0 ,                       y0 = C B u0
x1 = A B u0 + B u1 + B² u0 u1 ,   y1 = C A B u0 + C B u1 + C B² u0 u1
x2 = A² B u0 + A B u1 + B u2
     + A B² u0 u1 + A B² u0 u2 + B² u1 u2 + B³ u0 u1 u2
```

**本报告复算：Eq. 10 的展开与 Eq. 9 的递推是一致的**（逐项代入 $(A+Bu_k)x_{k-1}+Bu_k$ 即可对上，包括 $B^2u_0u_1$、$B^3u_0u_1u_2$ 这些高阶项）。**注意 Eq. 10 里的二阶项包含 $u_0u_2$ 这类非相邻对** —— 这一点后面会变成 §4.3 的一处不自洽。

作者据此把权重分成两类：
1. **黑**：各时刻独立映射的权重 → 就是 S4 的卷积核（Eq. 5/6）；
2. **紫**：输入各阶**自相关（auto-correlation）**的权重 → 就是新增的 liquid kernel。

### 3.4 液体核的两种模式

**通用形式（Eq. 13）**：

$$K_{liquid} \in \mathbb{R}^{\tilde L} := K_L(C,A,B) := \Big\{ C A^{\tilde L - i - p} B^{p} \Big\}_{i \in [\tilde L],\, p \in [P]}
= \big\{ C A^{\tilde L-2} B^2, \dots, C B^{p} \big\} \tag{Eq. 13}$$

其中 $\tilde L$ 是相关向量的长度，$P$ 是液体阶数上限。

**模式 A —— KB（Kernel × B，Prop. 1）**：不必重算核，直接用已算好的 S4 核乘上 $B$ 的幂：

$$K_{liquid}^{p} = K(L-\tilde L, L)\, B^{(L-\tilde L, L)} * J_{\tilde L} \tag{Alg. 1 行 8}$$

（$J_{\tilde L}$ 是"backward identity matrix"，反序矩阵。）
**模式 B —— PB（powers of B，Eq. 16）**：把 $A$ 退化为 Identity：

$$K_{liquid}^{p} \in \mathbb{R}^{\tilde L} := K_L(C,B) := C B^{p} \tag{Eq. 16}$$

作者明说 "Liquid-S4 - PB, as it is obtained by powers of the vector $B$"，且 Algorithm 1 行 11–15 注明 "Liquid-S4 Kernel of Eq. 13 **with A reduced to Identity**"。

**论文全部 LRA 主结果用的都是 PB 核**（§4.1 原文："Liquid-S4 instances (**all use the PB kernel** with a scaled Legendre (LegS) configuration)"），各任务的液体阶数逐任务给定：

| 任务 | ListOps | IMDB | AAN | CIFAR | Pathfinder | Path-X |
|---|---|---|---|---|---|---|
| $p$ | 5 | 6 | 2 | 3 | 2 | 2 |

**PB 模式的关键推论（本报告分析）**：$A \to I$ ⇒ $A^{\tilde L-i-p} = I$ ⇒ **Eq. 16 的液体核与 lag 无关，是一个常核**。也就是说，PB 模式下"输入依赖"完全落在 $u_{correlations}$ 上，液体核只提供一个常数滤波器。

### 3.5 复杂度

$$\tilde{O}(N + L + p_{max}\tilde L)$$

作者补充：$p$ 一般取 <10（典型 $p_{max}=3$），$\tilde L$ 比序列长度小两个数量级。

### 3.6 实验设置

- **LRA**（Tay et al. 2020b）六任务：ListOps(2048) / IMDB(2048) / AAN(4000) / CIFAR(1024) / Pathfinder(1024) / Path-X(16384)；
- **BIDMC** 生命体征：HR / RR / SpO2，序列长 4000，报 RMSE；
- **sCIFAR**：1024 长度的展平图像 10 分类；
- **RAW Speech Command（35 类全量）**：训练/测试 16kHz；另有"用 16kHz 训练、在 8kHz 上零样本测试"的降频实验。

---

## 4. 核心公式提取与复算

> 本节所有结论都标注了证据层级。凡标 **【读层复算】** 的，读者只需论文本身即可复核；凡标 **【代码层核验】** 的，需在 `raminmh/liquid-s4` 的 `main@06031dc` 上复跑（脚本见 §8）。

### 4.1 复算 ①：Eq. 9 不是梯形法 【读层复算】

**论文表述**（§3 "Discretization of Liquid-SSMs"）："Liquid-SSMs can also be discretized by a **bilinear transform (trapezoidal rule)** to construct a sequence-to-sequence model as follows: (Eq. 9)"。

**我的推导**：梯形法（bilinear transform）对 $\dot x = (A + Bu_k)x + Bu_k$ 的离散化应为

$$x_k = \Big(I - \tfrac{\Delta}{2}(A + Bu_k)\Big)^{-1}\Big[\Big(I + \tfrac{\Delta}{2}(A+Bu_k)\Big)x_{k-1} + \Delta B u_k\Big]$$

即：必须出现 $\Delta$ 因子、$\Delta/2$ 的**加性**项、$\Delta/2$ 的**减性**项、以及一个 $2\times2$ 块矩阵的逆。

而论文 Eq. 9 写的是 $x_k = (A+Bu_k)x_{k-1} + Bu_k$：**没有 $\Delta$**、**没有 $\Delta/2$ 项**、**没有 $(I-\Delta A/2)^{-1}$**。它是一个**一阶前向步（forward-Euler 型）**，不是梯形法。

**旁证（代码层）**：公开实现里确实出现了梯形法的那一半 —— `src/models/sequence/ss/s4.py` 中

```python
dB = torch.diag_embed(1.0 / (1.0 - 0.5 * dt[:, None] * w))   # (H,N,N)  ← (I - Δ/2 A)^{-1} 的对角
dB = dt[:, None] * contract("dab,db->da", dB, B)              # ← Δ · B
```

即 $\Delta (I-\Delta A/2)^{-1} B$ —— 梯形法的 $\Delta$ 与 $(I-\Delta A/2)^{-1}$ 都在，**但作用在参数 $B$ 上、不是作用在 $u$ 上**。而且 $u$ 在这条支路里**完全不出现**（详见 §4.6）。

**结论**：Eq. 9 的文字表述（"trapezoidal rule"）与其自身公式不符；实际是前向欧拉型一步。**这不影响论文的主张**（因为论文的工作全部在核空间做，Eq. 9/10 只是推导装置），但它意味着"梯形法"这个说法是**未兑现的**。

### 4.2 复算 ②：Eq. 10 的展开与 Eq. 9 一致 【读层复算】

逐项代入 Eq. 9 的递推：

$$x_1 = (A+Bu_1)\,(Bu_0) + Bu_1 = ABu_0 + B^2u_0u_1 + Bu_1 \quad ✓\ \text{与 Eq. 10 一致}$$

$$x_2 = (A+Bu_2)x_1 + Bu_2 = A^2Bu_0 + AB^2u_0u_1 + ABu_1 + AB^2u_0u_2 + B^3u_0u_1u_2 + B^2u_1u_2 + Bu_2 \quad ✓$$

**结论：Eq. 10 本身没有代数错误。**（本报告早期草稿曾疑心 Eq. 10 漏项，复算后撤回该疑心。）

### 4.3 复算 ③：Eq. 10 与 Eq. 11 互不自洽 【读层复算】

- **Eq. 10 的 $x_2$** 里，二阶项有 **三个**：$u_0u_1$、$u_0u_2$、$u_1u_2$ —— 包含 $u_0u_2$ 这种**非相邻**对（以及全部 $\binom{3}{2}=3$ 个组合）。
- **Eq. 11 的求和**只写了 $u_i u_{i+1}\dots u_p$ —— **只含相邻的滑动窗口乘积**。

我按 Eq. 11 的权重公式 $C A^{k+1-p-i}B^p$ 回代 Eq. 10 的项：

| Eq. 10 的项 | $p$ | $i$ | Eq. 11 公式给出的权重 | 是否被 Eq. 11 的求和覆盖 |
|---|---|---|---|---|
| $CAB^2u_0u_1$ | 2 | 0 | $CA^{3-2-0}B^2 = CAB^2$ | ✓ |
| $CAB^2u_0u_2$ | 2 | 0 | $CA^{3-2-0}B^2 = CAB^2$ | ✗ **Eq. 11 的 $u_i u_{i+1}$ 写不到它** |
| $CB^2u_1u_2$ | 2 | 1 | $CA^{3-2-1}B^2 = CB^2$ | ✓ |
| $CB^3u_0u_1u_2$ | 3 | 0 | $CA^{3-3-0}B^3 = CAB^3$ | ✓ |

**结论**：Eq. 11 的权重公式与 Eq. 10 一致，但 Eq. 11 的**求和范围**（只写相邻乘积）比 Eq. 10 的实际内容（全部组合）**窄**。论文正文对自相关的文字描述（"2 order (C(L+1,2)) … $u_i u_j$"）又是**全部组合**（§3 里 "1st order (1) $u_i$; 2nd order (2) $u_i u_j$; 3rd order (3) $u_i u_j u_k$"，并明确给出组合数 $\binom{L+1}{2},\binom{L+1}{3},\binom{L+1}{4}$）。

**即：论文自身对"自相关"的定义在 Eq. 10（全部组合）、Eq. 11（相邻滑窗）、正文（全部组合）三处之间摇摆。** §4.6 会看到，公开实现的两个分支恰好分别对应这两种定义 —— 但**都不是论文的任一版本**。

### 4.4 复算 ④：Table 1 的 Avg 列有 9/26 行不可复算 【读层复算】

按表内逐任务数字重算 Avg（Path-X 为 `x` 的行按 5 项平均）：

| 模型行 | 表内 Avg | 重算 Avg | 一致? |
|---|---|---|---|
| Random | 36.67 | 36.667 | ✓ |
| Transformer | 54.39 | 54.388 | ✓ |
| Local Att. | 46.06 | 46.056 | ✓ |
| Sparse Transformer | 51.24 | 51.238 | ✓ |
| Longformer | 53.46 | 53.460 | ✓ |
| Linformer | 50.55 | 50.552 | ✓ |
| **Reformer** | 50.56 | **50.668** | ✗ |
| **Sinkhorn Trans.** | 51.23 | **51.476** | ✗ |
| BigBird | 55.01 | 55.012 | ✓ |
| **Linear Trans.** | 50.46 | **50.552** | ✗ |
| **Performer** | 51.18 | **51.410** | ✗ |
| **FNet** | 54.42 | **55.304** | ✗ |
| **Nyströmformer** | 57.46 | **58.950** | ✗ |
| **Luna-256** | 59.37 | **61.242** | ✗ |
| H-Transformer-1D | 61.41 | 61.408 | ✓ |
| CDIL | 74.96 | 74.960 | ✓ |
| DSS | 79.45 | 79.450 | ✓ |
| S4 (original) | 80.48 | 80.478 | ✓ |
| S4-LegS | 86.09 | 86.087 | ✓ |
| **S4-FouT** | 77.90 | **83.482** | ✗ |
| **S4-LegS/FouT** | 78.50 | **84.194** | ✗ |
| S4D-LegS | 84.89 | 84.885 | ✓ |
| S4D-Inv | 85.50 | 85.503 | ✓ |
| **S4D-Lin** | 78.39 | **84.068** | ✗ |
| S5 | 82.46 | 82.455 | ✓ |
| **Liquid-S4 (ours)** | **87.32** | **87.3217** | ✓ |

**结论（重要且要分开说）**：
- **Liquid-S4 自己的 87.32 是自洽的**：$(62.75+89.02+91.20+89.50+94.80+96.66)/6 = 523.93/6 = 87.3217$。而且**与"SOTA"主张直接相关的 6 个基线（S4 80.48 / S4-LegS 86.09 / S4D-LegS 84.89 / S4D-Inv 85.50 / S5 82.46 / DSS 79.45）全部自洽**。→ **"87.32 > 86.09" 这个比较本身是站得住的**。
- 但 **9 行（几乎全是 Transformer 系与 Path-X 缺项行）的 Avg 与表内数字对不上**，最大偏差 5.6 pp（FNet 54.42 vs 55.30）。这说明这张表的 Avg 列对不同来源的行用了**不同的口径**（大概率是从各自的来源论文直接抄的 6 任务平均，而表内只列了 5 个任务）。→ **表本身不能当作自洽证据引用；引用具体数字时请逐项引，不要引 Avg。**

### 4.5 复算 ⑤：正文与表格的三处口径不一致 【读层复算】

| 论文正文说法 | 表内数字 | 重算 | 差异 |
|---|---|---|---|
| "on ListOps, Liquid-S4 improves S4-LegS performance by more than 3%" | 62.75 vs 59.60 | +3.15 pp | ✓ 成立 |
| "on character-level IMDB by 2.2%" | 89.02 vs 86.82 | +2.20 pp | ✓ 成立 |
| "on 1-D pixel-level classification (CIFAR) by **0.65%**" | 89.50 vs 88.65 | **+0.85 pp** | ✗ 差 0.20 pp |
| "establishing the SOTA on the hardest LRA task by gaining **96.54%** accuracy" | Table 1 Path-X = **96.66** | — | ✗ 差 0.12 pp |
| "96.78% accuracy with **30% reduction in parameter counts** compared to S4" | Table 4：Liquid-S4 **224K** vs S4-LegS **307K** | $1-224/307 = 27.04\%$ | ✗ 差约 3 pp |
| "In RR, Liquid-S4 outperforms S4-LegS by a significant margin of **36%**" | 0.158 vs 0.247 | $(0.247-0.158)/0.247 = 36.03\%$ | ✓ 成立 |
| "On SpO2, Liquid-S4 performs **26.67%** better than S4-Legs" | 0.066 vs 0.090 | $(0.090-0.066)/0.090 = 26.67\%$ | ✓ 成立 |
| "in HR, Liquid-S4 outperforms S4-Legs by **8.7%**" | 0.303 vs 0.332 | $(0.332-0.303)/0.332 = 8.73\%$ | ✓ 成立 |

**结论**：BIDMC 三个百分比**全部复算通过**（作者是按相对误差降幅算的，口径自洽）；但 **CIFAR 增益、Path-X 精度、参数量降幅**三处正文与表格对不上。

**另：Table 4 的 8kHz 列在本仓 PDF 抽取文本中，四个 CNN 基线读作 `05.18 / 08.74 / 07.72 / 07.26`**，均 <10 且丢失一位有效数字（若补一位，恰好保持与同表 16kHz 列相同的排序：25.18 < 38.74 < 47.72 < 57.26 对应 61.24 < 77.86 < 83.01 < 95.51）。**这是文本抽取层的疑点还是论文排版错误，需人工核对原始 PDF**；本报告**不**据此判定论文有错，只标记为"本仓抽取文本不可直接引用此列"。

### 4.6 复算 ⑥：公开实现与论文公式的六处偏离 【代码层核验】

核验对象：`raminmh/liquid-s4` @ `main/06031dc`，文件 `src/models/sequence/ss/s4.py`。
以下每条都在该文件上实跑验证过（脚本见 §8）。

#### 4.6.1 实现里**从未出现过** $A + B u(t)$（输入依赖的状态转移）

论文的立论是"状态转移矩阵随输入变"（Eq. 8）。但在实现里：

- KB 支路（`s4.py` L284–310）与 PB 支路（L311–340）里的 `dB` 计算为
  `dB = dt[:,None] * contract("dab,db->da", diag_embed(1/(1-0.5*dt*w)), B)`，
  **只含 $\Delta$、$A$ 的对角 $w$、参数 $B$ —— 不含 $u$**；
- 整条支路中 $u$ 唯一出现的地方，是被当作**特征**（乘积）喂进去，而不是被当作**状态转移的扰动**。

**结论**：实现是"**核空间**"的近似（把输入依赖折进 $u_{correlations}$ 这条额外通道），不是"**状态空间**"的近似。这与论文的推导思路（Eq. 11 的 $y = K*u + K_{liquid}*u_{corr}$）**是一致的**——但它意味着 **Eq. 8 的"输入依赖状态转移"从未被实例化**。

#### 4.6.2 PB 支路是**逐点加**，没有卷积，也没有 lag 结构

论文 Eq. 16 / Alg. 1 行 11–15 给的是**一个长度 $\tilde L$ 的液体核** $K_{liquid}^p = CB^p$，要与 $u_{correlations}$ **做卷积**。实现里 PB 支路实际是：

```python
us_corr = torch.flip(u_corr, [-1])
dCB = contract("abc,bc->ab", dC, dB).unsqueeze(2)      # (H,1) —— 每个 head 一个标量
y = y + (us_corr * dCB).unsqueeze(1).float()            # 逐点乘加，无 FFT、无 lag
```

**没有卷积、没有 lag 结构**：`dCB` 只是"每个 head 一个标量"（$\sum_n C_n \cdot dB_n$），乘到特征序列上就完事。论文的常核 $CB^p$ 至少还带 $\tilde L$ 长的滑动平均。→ **实现的 PB 支路不是 Eq. 16。**

#### 4.6.3 PB 支路的 $B$ 幂律是**双指数增长**，不是 $B^p$

同一段代码里 `dB` 的递推是

```python
dB1 = dB.unsqueeze(2); dB2 = dB.unsqueeze(1)
dB  = (dB1 * dB2).sum(2)        # dB_new[n] = dB[n] * Σ_m dB[m]
```

我把它写成闭式并实跑验证（head 0，$\|dB\|$）：设 $dB_0 = \Delta\bar B$，$T_0 = \sum_n dB_0[n]$，则

$$dB_p = dB_0 \cdot T_0^{\,2^{p-1}-1}$$

- $p=2$：$e=1$ → 实测 `allclose` 成立，max err 1.04e-09；
- $p=3$：$e=3$ → 实测 `allclose` 成立，max err 4.37e-11（注意 $e=2$ **不**成立）。

对比论文 Eq. 16 的 $C B^p$（把 $B$ 逐元素取 $p$ 次幂再与 $C$ 收缩）：

| | 论文 Eq. 16 | 实现（$p=3$，head 0） |
|---|---|---|
| $C\cdot(\cdot)$ | $-23.37 - 273.43i$ | $0.0000326 + 0.0000100i$ |

**相差约 7 个数量级**。而且实现里 $B$ 的**逐元素 $p$ 次幂从来没有发生** —— 状态维上的 $B$ 结构只被整体乘了一个标量 $T_0^{2^{p-1}-1}$。→ **"powers of the vector B" 这个命名在实现里没有对应物。**

#### 4.6.4 论文的 $J_{\tilde L}$（反序矩阵）被搬到了**特征侧** → **实现是非因果的**

论文里 $J_{\tilde L}$ 作用在**核**上（Alg. 1 行 8 / 行 13：`K_liquid^p = … * J_{L̃}`）。实现里反序作用在**特征**上：

```python
us_corr = torch.flip(u_corr, [-1])      # ← 反序的是"输入乘积特征"
y = y + (us_corr * dCB)                 # ← 然后逐点加回去
```

**实测因果性（决定性实验）**：构造 $u$，只改**最后一个**输入样本（其余完全相同，且最后两个样本非零以便激活二阶乘积）：

| 模型 | $\Delta y_0$（输出第一个时刻的变化） | 说明 |
|---|---|---|
| plain S4（`liquid_kernel=None`） | **0.0**（精确） | 严格因果 ✓ |
| Liquid-S4（PB, `allcombs=True`） | **4.905753e-03** | **非因果** ✗ |
| Liquid-S4（PB, `allcombs=False`） | **4.905753e-03** | **非因果** ✗ |
| Liquid-S4（KB 支路） | **−3.946044e-03** | **非因果** ✗ |

（该配置的 $|y|_{\max} \approx 0.265$，故 $\Delta y_0$ 相对量级约 1.8%。）

对照实验：把输入换成"**只有最后一个样本非零**"时，Liquid-S4 与 plain S4 的输出差为 **0.0** —— 证明二阶乘积特征需要两个非零样本才激活，上述 Δ 不是数值噪声。

**结论**：同一个模型里，S4 主干是因果的、液体支路是**反因果**的，输出 $y_k$ 因此依赖**未来**的输入样本。
**这与论文标题与摘要里的 "causality capabilities"、"dynamic causal models" 直接冲突。** 若把该实现用于自回归/在线场景，结果不可用；用于分类场景（整段看全）时，等价于模型看到了未来。

#### 4.6.5 `allcombs=True`（**默认**）构造的是**固定不滑动**的乘积下标集

`get_combs_cache()` 的逻辑：取末尾一段窗口，对窗口内取 $p$ 元组合，**数量凑够 `seq_len` 就停**，再把结果**平铺**成 `seq_len` 个下标元组。于是输出时刻 $t$ 对应的输入下标是**固定**的、与 $t$ 无关的一组对。

实测（`seq_len=32`, $p=2$）：组合窗口 = $\{23,\dots,31\}$，共 36 个二元组，取最后 32 个：

```
t=0 → (23,28)   t=1 → (23,29)   t=2 → (23,30)   t=3 → (23,31)
t=4 → (24,25)   t=5 → (24,26)  ...              t=31 → (30,31)
```

$p=3$ 同理（窗口 $\{25..31\}$，元组如 $(25,26,30),(25,26,31),(25,27,28),\dots$）。

**这不是论文说的自相关**：论文的 $u_{correlations}$ 是**滑动**的（每个时刻取该时刻及其之前的 $p$ 个样本的乘积，Eq. 11 的 $u_i u_{i+1}\dots u_p$ 明确是相邻滑窗）。实现默认路径给出的是**一组固定的、非滑动的乘积特征**，且它们全部落在**序列尾部**。

#### 4.6.6 `allcombs=False` 走的是"相邻滑窗"，但会**累乘**出重复样本幂次

`upgrade_degree()` 在 `allcombs=False` 时是

```python
us_shift = F.pad(us[..., :-1], (1, 0), "constant", 0)
us = us * us_shift
```

即 `us[t] *= us[t-1]`。于是 $p=3$ 时的特征不是 $u_t u_{t-1} u_{t-2}$，而是

$$us[t] = us^{(1)}[t]\cdot us^{(1)}[t-1] = (u_t u_{t-1})(u_{t-1}u_{t-2}) = u_t\,u_{t-1}^{2}\,u_{t-2}$$

**实测验证**（head0, channel0, 位置 4..9，输入 $u_t = t+1$）：

```
iter0 (u[t]u[t-1])  : [ 20,   30,   42,   56,   72,   90]
iter1 实际           : [240,  600, 1260, 2352, 4032, 6480]
iter1 期望 u_t·u_{t-1}·u_{t-2} : [126, 240, 420, 672, 1008, 1440]   ← 不符
iter1 解析 u_t·u_{t-1}²·u_{t-2} : [240, 600, 1260, 2352, 4032, 6480] ← 完全吻合
```

**结论**：两个分支**分别**近似了论文的两种说法（`allcombs=True` ≈ Eq. 10 的"全部组合"，`allcombs=False` ≈ Eq. 11 的"相邻滑窗"），**但都不是论文的任一版本**：前者不滑动、后者带重复样本幂次。**这是"论文↔代码"的一处真实缺口，不是"我方复现失败"。**

#### 4.6.7 复现配置：默认配置**根本不是** Liquid-S4

- `configs/model/layer/s4.yaml`：`liquid_kernel: null`、`liquid_degree: 2`、`allcombs: true`、`lcontract: null`（恒等）；
- 六个 `configs/experiment/lra/s4-lra-*.yaml` **均未设置** `liquid_kernel`；
- README 给出的启用方式是命令行覆盖：`python3 -m train wandb=null experiment=lra/s4-lra-imdb model.layer.liquid_kernel=polyb model.layer.liquid_degree=3`。

**⇒ 任何人直接跑仓库默认配置，拿到的是普通 S4，不是 Liquid-S4。** 论文 LRA 用的还是 **PB** 核（README 也推荐 polyb）。

#### 4.6.8 其它代码层事实

- 实现的 `dB` 用的是 $\bar B$（"B-bar"）而论文 Eq. 12/13/16 写的是 $B^p$；`s4.py` 最后一次改动 `d3a72bd9`（2022-08-30）就叫 **"Bbar instead of B bugfix"** —— 说明这一处**论文与代码在版本上就已经分叉**，而论文的 v1 早于该修复（2022-09-26 > 2022-08-30，v1 在修复之后，但正文没写 $\bar B$）。
- `liquid_kernel="kb"` 与 `shift=True` 同时开启会直接 `raise NotImplementedError()`（`s4.py` L276–278）。
- `lcontract` 默认 `null` → 恒等映射；论文的 $\mathcal{L}$ 契约算子（Identity/Tanh/LeCun）在实现里可选但默认关闭。

### 4.7 复算小结表

| # | 复算对象 | 层级 | 结论 |
|---|---|---|---|
| ① | Eq. 9 vs "trapezoidal rule" | 读层 | ✗ 实为前向欧拉型一步 |
| ② | Eq. 10 展开 vs Eq. 9 递推 | 读层 | ✓ 一致（早期疑心撤回） |
| ③ | Eq. 10 vs Eq. 11 的自相关定义 | 读层 | ✗ 三处口径不一（全部组合 / 相邻滑窗 / 正文组合数） |
| ④ | Table 1 Avg 列 | 读层 | Liquid-S4 行与 6 个相关基线 ✓；全表 9/26 行 ✗ |
| ⑤ | 正文 vs 表格（CIFAR 0.65% / Path-X 96.54 / 参数量 30%） | 读层 | ✗ 三处对不上；BIDMC 三个百分比 ✓ |
| ⑥ | 论文公式 vs 公开实现（6 项） | 代码层 | ✗ 6 项偏离，含**非因果** |

---

## 5. 关键成果与贡献

### 5.1 作者自报（未经我方复现）

| 基准 | 数据集 | Liquid-S4 | 对照 | 出处 |
|---|---|---|---|---|
| LRA 平均 | 6 任务 | **87.32** | S4-LegS 86.09 / S4D-Inv 85.50 / S5 82.46 / DSS 79.45 | Table 1 |
| ListOps | 2048 | 62.75 (0.2), $p=5$ | S4-LegS 59.60 | Table 1 |
| IMDB | 2048 | 89.02 (0.04), $p=6$ | S4-LegS 86.82 | Table 1 |
| AAN | 4000 | 91.20 (0.01), $p=2$ | S4-LegS 90.90 | Table 1 |
| CIFAR | 1024 | 89.50 (0.4), $p=3$ | S4-LegS 88.65 | Table 1 |
| Pathfinder | 1024 | 94.8 (0.2), $p=2$ | S4-LegS 94.20 | Table 1 |
| Path-X | 16384 | 96.66 (0.001), $p=2$ | S4-LegS 96.35 | Table 1 |
| BIDMC HR | RMSE | **0.303** (0.002) | S4-LegS 0.332 | Table 2 |
| BIDMC RR | RMSE | **0.158** (0.001) | S4-LegS 0.247 | Table 2 |
| BIDMC SpO2 | RMSE | **0.066** (0.002) | S4-LegS 0.090 | Table 2 |
| sCIFAR | 10 类 | **92.02** (0.14) | S4-LegS 91.80 | Table 3 |
| SC full 35 类 | 16kHz | **96.78** (0.05) | S4D-Lin 96.25 / S4-LegS 96.08 | Table 4 |
| SC full 35 类 | 8kHz 零样本 | 90.00 (0.25) | S4D-Inv 91.80 / S4-LegS 91.32 | Table 4 |
| 参数量 | — | **224K** | S4-LegS 307K | Table 4 |
| SC10（附录） | 16kHz | 98.51 | — | Table S2 |

**逐任务幅度（我方重算 vs S4-LegS）**：ListOps **+3.15**、IMDB **+2.20**、AAN **+0.30**、CIFAR **+0.85**、Pathfinder **+0.60**、Path-X **+0.31**，平均 +1.23。
**按论文自报标准差看**：ListOps（0.2）与 IMDB（0.04）幅度明确；但 **AAN(+0.30, σ=0.01)** 看似显著、**CIFAR(+0.85, σ=0.4, n=3)** 约合 1.9σ、**Pathfinder(+0.60, σ=0.2)** 约合 3σ、**Path-X(+0.31, σ=0.001)** 显著。即"六任务全部第一"成立，但**"全面显著领先"不成立** —— 六项里两项的差距在 1 个百分点以内。

### 5.2 论文自己承认的不足

- **8kHz 零样本不是最好**（90.00 vs S4D-Inv 91.80）；作者归因为"liquid kernel accounts for covariance terms … might influence the learned representations in a way that hurts performance"；
- $p < 10$（典型 $p_{max}=3$），$\tilde L$ 需比序列长度小两个数量级；
- 复杂度从 $\tilde O(N+L)$ 增到 $\tilde O(N+L+p_{max}\tilde L)$。

### 5.3 我方的折扣（本报告的核心结论）

> **D1｜"液态"这一层贡献与"多加一条输入高阶乘积通道"这一层贡献，论文没有分开。**
> 结合 §3.4：PB 模式下 $A\to I$ ⇒ 液体核是**与 lag 无关的常核** ⇒ 液体支路在数学上就是
> $$y \mathrel{+}= \alpha \cdot \big(\text{输入的 } p \text{ 阶乘积特征}\big)$$
> 即"给 S4 额外接了一条**二阶/高阶乘积特征**的可学习通道"，而**不是**"给系统装了随输入变化的时间常数"。论文**没有**做这个消融（"S4 + 同样阶数的乘积特征，但不含 $B$ 幂/液体核"）。**没有这个消融，§5.1 的 LRA SOTA 就无法归因给"液态"。**

> **D2｜公开实现与论文公式有 6 处可核验的偏离，其中"非因果"是硬伤。**
> 见 §4.6。**非因果**意味着：用该实现报告的任何分类数字，模型在预测时刻 $t$ 时**看到了未来的输入样本**。这对 Path-X（16384）、ListOps（2048）这类需要长程依赖的任务，其数字的**可比性存疑**（不能与严格因果的 S4-LegS 直接比）。论文没有讨论这一点。

> **D3｜"87.32" 本身可复算，但那张表整体不可作为自洽证据。**
> 9/26 行的 Avg 与表内数字对不上（§4.4）。Liquid-S4 行与直接相关的 6 个基线行自洽，**所以"SOTA"比较可用；但"全表可信"的说法不可用**。

> **D4｜三处正文数字与表格对不上**（CIFAR 增益、Path-X 精度、参数量降幅）（§4.5）。这些属于**表述层**问题，不推翻主结论，但引用具体数字时必须以**表格**为准。

> **D5｜本仓目前没有本地 grounding。**
> 本仓 `analysis/pdna_lra/2026-06-08_pdna_lra_summary.md` 里，三种变体在 LRA 规模的 Pathfinder 上都是 **48.75%**（二分类随机水平，n=1 seed、1 epoch、hidden 32、200 条训练样本）——**本仓自己的 LRA 复现能力仍在随机水平**，因此本篇的 LRA SOTA 在本仓内部**没有任何可对照的复现证据**。这与台账 **NR-06**（"LNN 在标准主基准上 SOTA 横扫 transformer"）的判定方向一致：本篇的 SOTA 属于"第三方自报 + 未消融 + 实现与公式不符"，**不足以推翻也不足以支撑 NR-06**。

---

## 6. 局限性与未来展望

### 6.1 论文自述
见 §5.2（8kHz 零样本掉点、复杂度增量、$p$ 的取值范围）。

### 6.2 本报告独立识别的局限

**表述与公式层**
1. Eq. 9 被称为梯形法但不是（§4.1）；实现里梯形法只作用在参数 $B$ 上、$u$ 不参与。
2. "自相关"的三处定义互不一致：Eq. 10（全部组合）/ Eq. 11（相邻滑窗）/ 正文（组合数 $\binom{L+1}{p}$）（§4.3）。
3. Eq. 8 的 $A + B u(t)$ 存在**维数歧义**：$A\in\mathbb{R}^{N\times N}$、$Bu(t)\in\mathbb{R}^{N\times 1}$，按字面无法相加；作者说"element-wise added to A"，在**对角参数化**下才成立（实现里正是把它加在 $A$ 的对角 $w$ 上）—— 即"输入依赖的状态转移"被限制为**只扰动对角/实部**。
4. 正文三处数字与表格不一致（§4.5）。

**方法论层**
5. **缺关键消融**（§5.3 D1）：无法把增益归因给"液态"。
6. **PB 模式丢弃 HiPPO/LegS 结构记忆**：$A\to I$ 意味着液体核不含 lag 结构，S4 的"结构性记忆"优势在液体核里不存在；论文未讨论这一取舍。
7. **逐任务调 $p$**（2/3/5/6 四个不同取值），与基线单一超参的对比口径不完全对等。
8. Table 1 的 Avg 列口径不统一（§4.4）。

**实现层**
9. **非因果**（§4.6.4）—— 与论文"causality capabilities"的自我定位冲突。
10. PB 支路是逐点标量加，不是 Eq. 16 的常核卷积；$B$ 的逐元素 $p$ 次幂从未发生，幂律实为 $T_0^{2^{p-1}-1}$（§4.6.2/§4.6.3）。
11. `allcombs=True`（默认）用固定不滑动下标集，`allcombs=False` 产生重复样本幂次（§4.6.5/§4.6.6）。
12. 默认配置 `liquid_kernel: null` ⇒ 直接跑仓库拿到的是普通 S4（§4.6.7）。
13. $\bar B$ vs $B^p$ 的版本分叉（`d3a72bd9` "Bbar instead of B bugfix"），正文未反映。

### 6.3 未来方向（本报告按论文自身逻辑补出，非作者陈述）

- **补上那个消融**：$y \mathrel{+}= \alpha \cdot \text{乘积特征}$ vs $y \mathrel{+}= (CB^p) * \text{乘积特征}$ vs 完整液体核 —— 三个版本在 LRA 上对拍，才能回答"液态有没有用"。
- **改成因果实现**：把 $J_{\tilde L}$ 放回**核侧**（论文 Eq. 16 的写法），特征侧不反序；或直接用滑动窗口 $u_{t-p+1}\dots u_t$。
- **用 KB 模式重跑**（KB 保留了 $A^{k+1-p-i}$ 的 lag 结构，理论上才是"液态"味道更足的模式），但作者的主结果全是 PB。
- **报告"液态支路 vs 参数量"**：作者声称 224K vs 307K 少 27%，但液体支路本身加了多少参数没说。

---

## 7. 对本仓的意义

### 7.1 与本仓已有资产的对应关系

| 本仓资产 | 与本篇的关系 |
|---|---|
| `lnn/core/dss_cell.py` | 本仓已实现 **DSS**（Gupta/Gu/Berant 2022, arXiv:2203.14343）—— 正是 Table 1 里 Avg 79.45 的那一行。**结构化 SSM 这条线本仓已经在做**，但 Table 1 显示 S4 系（80.48~86.09）整体高于 DSS（79.45）。 |
| `analysis/pdna_lra/2026-06-08_pdna_lra_summary.md` | 本仓自己的 LRA 规模 Pathfinder 复现 = **48.75%（随机水平）**，三种变体无差别。**这是本篇数字在本仓的"落地缺口"**（§5.3 D5）。 |
| `lnn/core/ltc.py` | 本仓实现的是**完整 LTC**（可学习时间常数、门控、非线性），而本篇是**线性化** LTC（$A+Bu$，无 $\tau$、无门控）。**不可混谈。** |

### 7.2 若本仓要吸收"液体核"这一设计，必须改掉的三件事

1. **必须因果**：特征侧取 $u_{t-p+1},\dots,u_t$ 的**滑动**乘积，**不要**反序、不要固定下标集。这是本篇实现最硬的坑（§4.6.4/§4.6.5）。
2. **必须归一化**：$p$ 阶乘积**不是**尺度等变的（$u \mapsto cu$ 时乘积项按 $c^p$ 缩放，而 S4 主干按 $c$ 缩放）→ 乘积特征通道前必须做幅值归一化（例如各自除以 $u$ 的滑动 RMS），否则模型会因输入幅值不同而行为漂移。这与本仓 `ltc.py` 的时间常数参数化无关，但与 `cfc.py`/`ltc.py` 里的输入缩放策略直接相关。
3. **必须显式命名**：若本仓新增此类模块，建议命名为**"因果输入自相关通道（causal input-correlation channel）"**之类的描述性名字，**不要**直接叫 "liquid kernel" —— 本篇的实现是非因果的且与公式不符，照搬名字会污染本仓的语义（且容易与 `lnn/core/ltc.py` 的真实 LTC 混淆）。

### 7.3 引用纪律（写 LNN 报告 / 答辩时）

- 引用本篇的 **LRA 87.32 / SOTA** 时，必须带上三个限定：**① 作者自报，本仓未复现；② 论文缺"液态 vs 乘积特征通道"的消融；③ 公开实现与公式有 6 处偏离且非因果。**
- **不得**用本篇去支持台账 **NR-06**（"LNN 在标准主基准 SOTA 横扫 transformer"）—— 本篇既无 transformer 强对照的完整复现、也未消融"液态"贡献，**不构成对 NR-06 的否证，也不构成支持**。
- **不得**把本篇的 $p$ 阶乘积机制包装成"液态神经网络"的一般性结论 —— 本篇是**核空间的线性化近似**（§5.3 D1），与本仓 `ltc.py` 的"随输入变的时间常数"在机制上不是一回事。
- 可安全引用的部分：**Eq. 10/11 的展开结构**（自相关的来源）、**Prop. 1 的 KB 构造**（用 S4 核乘 $B$ 幂得到液体核）、**复杂度增量 $\tilde O(p_{max}\tilde L)$**、**BIDMC 三个相对降幅百分比**（本报告复算通过）。

---

## 8. 复算脚本与复现方式

本次研读的复算脚本为**一次性产物**，放在 `/tmp`，**未纳入本仓 `scripts/`**（与本仓其它研读一致：验证脚本不长期留在仓内）。所有结论的复现路径：

| 结论 | 复现方式 |
|---|---|
| §4.1 梯形法推导 | 纯代数，读者可直接推 |
| §4.2 Eq. 10 展开 | 纯代数，逐步代入 |
| §4.3 Eq. 10 vs Eq. 11 | 纯代数，逐项回代 |
| §4.4 Table 1 Avg | 表内数字逐行平均（数据全部抄在本报告表内） |
| §4.5 正文 vs 表格 | 同上 |
| §4.6 全部 6 项 | `git clone https://github.com/raminmh/liquid-s4` @ `06031dc`；用 `opt_einsum→torch.einsum` + `contract_expression` 垫片、`hydra/omegaconf/pytorch_lightning` 用 `MagicMock` 打桩后即可 import `src.models.sequence.ss.s4`；脚本见 `/tmp/ls4_boot.py` 与 `/tmp/ls4_test{,2,3,4,5,6,7}.py` |

**未做（诚实声明）**：**没有**跑 LRA/BIDMC/sCIFAR/SC 的任何训练或评测；**没有**复现论文的任何一个主实验数字；**没有**做 §6.3 建议的消融。

---

## 9. 与负结果台账的关系

按 `AGENTS.md` 约束 #4 与 `analysis/negative_results/README.md` §1 的收录门槛（"有可证伪主张 + 有实测或可核查证据 + **我方复现失败**"）：

- 本报告的复算属于 **① 论文自身数字的算术核验**、**② 第三方开源实现的代码级行为核验**，**不构成"我方复现失败"**（我方并未尝试复现主实验）；
- 因此 **不新增台账条目**，`analysis/negative_results/README.md` 的 **NR-01 ~ NR-09 保持不变**；
- `python3 scripts/check_negative_results.py` 仍应通过。

---

## 10. 一句话总结

**Liquid-S4 的想法是好的（把"输入依赖"从 ODE 求解器搬到核空间，用输入高阶乘积当额外通道），但论文缺了那一个决定性的消融（液态 vs 乘积特征通道），公开实现又与自身公式有 6 处可核验的偏离——包括把论文核侧的反序矩阵搬到了特征侧、导致模型非因果——因此它的 "LRA SOTA 87.32" 在本仓只能作为"第三方自报且未消融"的材料引用，不能当作"液态机制被证明有效"的证据。**
