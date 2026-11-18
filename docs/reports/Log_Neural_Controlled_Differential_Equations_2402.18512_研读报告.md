---
title: Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference
date: 2026-11
tags: [LNN, Continuous-Time, Neural-CDE, Log-ODE, Lie-Bracket, Log-Signature, Rough-Path, Irregular-Sampling, StateSpace, PMLR]
---

# 研读报告：Log-NCDE — 用 Lie 括号把「路径的高阶几何」接回神经向量场，换来 SOTA 精度 + 3× 加速

> **本篇不在 digest 候选池里** —— 它的标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**。

## 1. 元数据

- **论文标题**：Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference
- **作者**：Benjamin Walker¹, Andrew D. McLeod¹, Tiexin Qin², Yichuan Cheng², Haoliang Li², Terry Lyons¹
  - ¹ Mathematical Institute, University of Oxford, UK
  - ² Department of Electrical Engineering, City University of Hong Kong
- **发表时间**：2024-02-28 (v1) / 2025-10-23 (v4)；**ICML 2024**（PMLR 235, Vienna）；23 页，5 图
- **来源**：arXiv:2402.18512v4 [cs.LG]
- **代码**：https://github.com/Benjamin-Walker/Log-Neural-CDEs
- **本地 PDF**：[papers/2402.18512/2402.18512v4.pdf](../../papers/2402.18512/2402.18512v4.pdf) ／ 文本 [papers/2402.18512/2402.18512v4.txt](../../papers/2402.18512/2402.18512v4.txt)
- **数据集**：Toy（自造，10 万条 6 通道 × 100 采样点）；UEA-MTSCA 6 个最长序列数据集（EigenWorms 17984 点 / EthanolConcentration / Heartbeat / MotorImagery / SelfRegulationSCP1 / SCP2）；PPG-DaLiA（腕带 128Hz 生理回归，窗口长度 49920）
- **关联概念**：Neural CDE / Log-ODE / NRDE / log-signature / Lie 括号 / Lip(γ) 正则性 / 粗糙路径 / S5 / Mamba / LRU / LTC / CfC（**对照组，非本方法族**）

## 2. 核心问题

Neural CDE 把时序数据当作**受控路径的观测**，用向量场 $f_\theta$ 沿路径增量演化隐藏状态 $h_t$。它天然适配不规则采样，理论上是连续实值函数的通用逼近器（Kidger 2022, Thm 3.9），但存在**两个同时发作的性能与效率缺口**：

1. **性能缺口**：论文开篇即承认 "there exists a gap in performance between NCDEs and current state-of-the-art approaches for time series modelling, such as S5, the linear recurrent unit (LRU), and MAMBA"。也就是说，**理论上最优雅的连续时间模型，在实证上打不过朴素的结构化线性递推**。
2. **效率缺口**：NCDE 的前向传播必须在每个观测区间**递归地解一个非线性微分方程**，完全无法并行；观测点越多，代价越高。

既有的补救是 **NRDE**（Morrill et al. 2021）：用 Log-ODE 方法把 CDE 在每个区间 $[r_i, r_{i+1}]$ 上近似成一个**自治 ODE**，驱动量换成该区间的截断 log-signature：

$$g_{\theta,X}(\tilde h_s)=\bar f_\theta(\tilde h_s)\frac{\log(S^N(X)[r_i,r_{i+1}])}{r_{i+1}-r_i}$$

其中 $\bar f_\theta$ 被**当作一个自由神经网络**来学（$\bar f_\theta:\mathbb{R}^u\to\mathbb{R}^{u\times\beta(v,N)}$），从而在每个区间内变成自治的、可并行推进的。

但 NRDE 的代价是：**向量场的输出维度从 $u\times v$ 膨胀到 $u\times\beta(v,N)$**（$\beta(v,N)$ 是 $v$ 维路径深度-$N$ 截断 log-signature 的维度），当 $v$ 增大时急剧膨胀。论文的判断是：NRDE **放弃了向量场的结构**（"By neglecting the structure of $\bar f_\theta$"），因此它**学不会也用不上 Lie 括号结构**——而 Lie 括号恰恰是 CDE 中"路径几何 ↔ 动力学"耦合的数学载体。

问题因此被重述为一个形式化的问题：**能否让 NRDE 的 log-ODE 近似保留 NCDE 向量场的完整结构（用真正的 Lie 括号构造 $\bar f_\theta$），同时不付出输出维度膨胀的代价？** 若能，则"路径的高阶几何"就直接进入了动力学，而不是被丢给一个自由的 MLP 去猜。

## 3. 方法论与核心思路

### 3.1 总体架构

| 部件 | NCDE | NRDE | **Log-NCDE（本文）** |
|---|---|---|---|
| 隐藏状态演化 | $h_t=h_{t_0}+\int f_\theta(h_s)\,dX_s$ | 同左，但驱动换成 log-sig | 同左，驱动换成 log-sig |
| 区间内向量场 | 非自治（依赖 $X_s$） | **自治**，$\bar f_\theta$ 是自由 MLP | **自治**，$\bar f_\theta$ 由 $f_\theta$ 的 Lie 括号**结构化构造** |
| $\bar f_\theta$ 来源 | — | 自由神经网络（丢弃结构） | $f_\theta$ 的迭代 Lie 括号（保留结构） |
| 网络输出维度 | $u\times v$ | $u\times\beta(v,N)$（**膨胀**） | $u\times v$（**不膨胀**） |
| 正则性要求 | Lipschitz 即可 | 无 | $f_\theta\in\mathrm{Lip}(\gamma),\ \gamma\in(N-1,N]$ |

**核心机制一句话**：Log-NCDE 把 $\bar f_\theta$ 从"另一个自由网络"改成"**由 $f_\theta$ 经 Lie 代数同态 $\bar f$ 递推构造**"，于是 (a) 输出维度退回 $u\times v$；(b) 动力学天然含 Lie 括号，与 CDE 理论结构对齐。

### 3.2 Log-ODE 的向量场构造（Eq. 19–21）

$C^\infty(\mathbb{R}^u,\mathbb{R}^u)$ 在 Lie 括号 $[a,b](x)=J_b(x)a(x)-J_a(x)b(x)$ 下构成 Lie 代数。$f_\theta(\cdot)dX$ 是从 $\mathbb{R}^v$ 到该 Lie 代数的线性映射，故由**自由 Lie 代数的定义**存在唯一线性映射 $\bar f$，且 $\bar f$ 是 Lie 代数同态，可**递归定义**：

$$\bar f(\cdot)a=f(\cdot)a,\quad a\in\mathbb{R}^v \qquad\qquad \bar f(\cdot)[a,b]=[\bar f(\cdot)a,\bar f(\cdot)b]$$

这就是本文的全部"结构"。实现上取 **Hall 基** $\{\hat e_k\}_{k=1}^{\beta(v,N)}$（Lie 括号的一个特定子集），把 log-sig 展开：

$$\bar f_\theta(h_s)\frac{\log(S^N(X)[r_i,r_{i+1}])}{r_{i+1}-r_i}=\sum_{k=1}^{\beta(v,N)}\lambda_k\,\bar f_\theta(h_s)\hat e_k$$

由于每个 $\hat e_k$ 都是 $\{e_j\}_{j=1}^v$ 的迭代 Lie 括号，$\bar f_\theta(\cdot)\hat e_k$ 可以**完全由 $f_\theta$ 的 Jacobian-向量积（JVP）**算出：

$$[f_\theta(\cdot)a,\ f_\theta(\cdot)b]=J_{f_\theta(\cdot)a}f_\theta(\cdot)b-J_{f_\theta(\cdot)b}f_\theta(\cdot)a$$

因为 $f_\theta(\cdot)e_j$ 就是网络输出的第 $j$ 列（$f_\theta:\mathbb{R}^u\to\mathbb{R}^{u\times v}$），所以 $v$ 个 JVP 就够算出 Eq. 27 的全部项。

### 3.3 Lip(γ) 正则性：让 Lie 括号可算（本篇的第二个理论贡献）

计算 Lie 括号要求 $f_\theta$ 有 $\mathrm{Lip}(\gamma)$ 正则性（$\gamma\in(N-1,N]$）。论文给出一个**新的显式界**：

**Theorem 3.1**：设 $f_\theta$ 是输入维 $n_{in}$、隐层维 $n_h$、深度 $m$、激活函数为 SiLU 的全连接网络（FCNN）。若输入满足 $|x_j|\le 1$，则 $f_\theta\in\mathrm{Lip}(2)$，且

$$\|f_\theta\|_{\mathrm{Lip}(2)}\le C\,P_{m!}\Big(\max_{1\le i\le m}\{\|W^i\|_2,\ \|b^i\|_2\}\Big)$$

其中 $C$ 依赖 $n_{in},n_h,m$，$P_{m!}$ 是 $m!$ 阶多项式。

**这个界的实际后果非常尖锐**：若每层的 $\|L^i\|_{\mathrm{Lip}(2)}=1$，显式求值得 $\|f_\theta\|_{\mathrm{Lip}(2)}\le 5\cdot 2^{m-1}-1$。论文指出——**对深度 7 的 FCNN，该上界已经超过单精度浮点数的最大值**。因此**必须在训练中显式控制** $\mathrm{Lip}(2)$ 范数，做法是往损失里加一项（谱范数正则化的一个变体）：

$$\mathcal L \to \mathcal L+\lambda\Big(\sum_{i=1}^{m}\big(\|W^i\|_2+\|b^i\|_2\big)\Big)!$$

### 3.4 计算成本（Eq. 28）

设 $F_x$ 为评估模型 $x$ 向量场的 FLOPs，$m,n_h$ 为 FCNN 深度与隐宽，$u,v$ 为状态维与路径维：

$$F_{NCDE}=2u n_h+2(m-1)n_h^2+2uvn_h,\qquad F_{NRDE}=2u n_h+2(m-1)n_h^2+u(v^2-v)n_h,\qquad F_{\text{Log-NCDE}}=3v\,F_{NCDE}$$

即：Log-NCDE 与 NRDE 的**渐近复杂度相同**（都随 $v$ 线性/二次增长），但 Log-NCDE 的常数因子是 $3v$ 倍 JVP 代价。论文给出的**关键工程细节**：所有 JVP 都在**同一点** $h_s$ 上评估，因此可以用 JAX 的 `vmap` 批量化，把高维时序上的实际负担降下来。

### 3.5 与 LTC / CfC / GRU 的关系（关键澄清）

🔴 **Log-NCDE 与本仓主线的 LTC / CfC 不是同一族**，不可混引：

| | 驱动源 | 时间常数 | 是否连续时间 | 路径几何是否进入动力学 |
|---|---|---|---|---|
| **LTC** (1811.00321) | 无（自治） | 可学习 $\tau$ | 是（Euler 解 ODE） | **否** |
| **CfC** (2106.13898) | 无（自治） | 可学习 $\tau$ | 是（**闭式解**） | **否** |
| **NCDE / Log-NCDE** | **路径增量 $dX_s$** | 无显式 $\tau$ | 是 | **是**（log-signature） |
| **GRU** | 观测值 $x_t$ | 固定 1 | 否 | 否 |

按本仓 11-18 建立的"自主 / 受控 / 形状编码"三档分类，Log-NCDE 属于**第 ② 档（受控）**且已推进到**第 ③ 档（形状编码）**。LTC/CfC 停在**第 ① 档（自主）**。⇒ **本篇不能用来支持"液态网络有效"，只能用来说明"路径级驱动 + Lie 括号结构"这一档的收益与代价**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$h_{t_0}=\xi_\phi(t_0,x_0),\quad h_t=h_{t_0}+\int_{t_0}^t f_\theta(h_s)\,dX_s,\quad z_t=l_\psi(h_t)$$ | NCDE 定义：状态沿**路径增量**演化 |
| Eq. 2–3 | $$\tilde h_t=\tilde h_{t_0}+\int g_{\theta,X}(\tilde h_s)ds,\quad g_{\theta,X}(\tilde h_s)=f_\theta(\tilde h_s)\frac{dX}{ds}$$ | NCDE 改写成普通 ODE（可导插值） |
| Eq. 4 | $$g_{\theta,X}(\tilde h_s)=\bar f_\theta(\tilde h_s)\frac{\log(S^N(X)[r_i,r_{i+1}])}{r_{i+1}-r_i}$$ | **Log-ODE 方法**：区间自治化的驱动 |
| Eq. 19–20 | $$\bar f(\cdot)a=f(\cdot)a,\qquad \bar f(\cdot)[a,b]=[\bar f(\cdot)a,\bar f(\cdot)b]$$ | **本篇核心**：$\bar f$ 是 Lie 代数同态，递归构造 |
| Eq. 21 | $$h_t=h_{t_0}+\int_{t_0}^t g_{\theta,X}(h_s)ds,\quad g_{\theta,X}(h_s)=\bar f_\theta(h_s)\frac{\log(S^N(X)[r_i,r_{i+1}])}{r_{i+1}-r_i}$$ | **Log-NCDE 定义式** |
| Eq. 22 | $$g_{\theta,X}(h_s)=f_\theta(h_s)\frac{X_{r_{i+1}}-X_{r_i}}{r_{i+1}-r_i}$$ | $N=1$ 时退化为 NRDE 形式（仅差正则化） |
| Eq. 23 | $$\|f_\theta\|_{\mathrm{Lip}(2)}\le C\,P_{m!}\Big(\max_i\{\|W^i\|_2,\|b^i\|_2\}\Big)$$ | **Theorem 3.1**：FCNN 的 Lip(2) 显式界 |
| Eq. 25 | $$\mathcal L\to\mathcal L+\lambda\Big(\sum_{i=1}^{m}(\|W^i\|_2+\|b^i\|_2)\Big)!$$ | 训练中显式控制 Lip(2) 范数 |
| Eq. 26 | $$[f_\theta(\cdot)a,f_\theta(\cdot)b]=J_{f_\theta(\cdot)a}f_\theta(\cdot)b-J_{f_\theta(\cdot)b}f_\theta(\cdot)a$$ | Lie 括号的 Jacobian 实现 |
| Eq. 27 | $$\bar f_\theta(h_s)\frac{\log(S^N(X)[r_i,r_{i+1}])}{r_{i+1}-r_i}=\sum_{k=1}^{\beta(v,N)}\lambda_k\bar f_\theta(h_s)\hat e_k$$ | Hall 基展开（可用 JVP 实现） |
| Eq. 28 | $$F_{NCDE}=2un_h+2(m-1)n_h^2+2uvn_h;\quad F_{NRDE}=2un_h+2(m-1)n_h^2+u(v^2-v)n_h;\quad F_{\text{Log-NCDE}}=3vF_{NCDE}$$ | FLOPs 对比 |

## 5. 关键成果与贡献

### 5.1 主结果：UEA-MTSCA（Table 1，5 个随机划分的测试集准确率 ± 标准差，%）

| Dataset | LRU | S5 | S6 | MAMBA | NCDE | NRDE | **Log-NCDE** |
|---|---|---|---|---|---|---|---|
| EigenWorms | 87.8±2.8 | 81.1±3.7 | 85.0±16.1 | 70.9±15.8 | 75.0±3.9 | 83.9±7.3 | **85.6±5.1** |
| EthanolConcentration | 21.5±2.1 | 24.1±4.3 | 26.4±6.4 | 27.9±4.5 | 29.9±6.5 | 25.3±1.8 | **34.4±6.4** |
| Heartbeat | 78.4±6.7 | 77.7±5.5 | 76.5±8.3 | 76.2±3.8 | 73.9±2.6 | 72.9±4.8 | **75.2±4.6** |
| MotorImagery | 48.4±5.0 | 47.7±5.5 | 51.3±4.7 | 47.7±4.5 | 49.5±2.8 | 47.0±5.7 | **53.7±5.3** |
| SelfRegulationSCP1 | 82.6±3.4 | **89.9±4.6** | 82.8±2.7 | 80.7±1.4 | 79.8±5.6 | 80.9±2.5 | 83.1±2.8 |
| SelfRegulationSCP2 | 51.2±3.6 | 50.5±2.6 | 49.9±9.5 | 48.2±3.9 | 53.0±2.8 | 53.7±6.9 | **53.7±4.1** |
| **平均准确率** | 61.7 | 61.8 | 62.0 | 58.6 | 60.2 | 60.6 | **64.3** |
| **平均排名** | 3.5 | 4.0 | 3.5 | 5.5 | 4.5 | 4.9 | **2.1** |

**Log-NCDE 在 6 个数据集全部 ≥ NRDE**，在 4 个数据集上标准差更低；平均准确率 64.3%（次优的 S6 62.0%，+2.3pp）。

### 5.2 效率（Table 2，6 数据集平均）

| Model | Av. GPU Mem. (MB) | Av. run time (s) |
|---|---|---|
| LRU | 4121.67 | 466.09 |
| S5 | 2815.00 | 244.78 |
| S6 | 2608.00 | 578.15 |
| MAMBA | 4450.33 | 1553.83 |
| NCDE | **1759.67** | 6649.91 |
| NRDE | 2676.33 | 7284.20 |
| **Log-NCDE** | 1999.67 | **2128.32** |

Log-NCDE 比 NCDE 快 **3.1×**、比 NRDE 快 **3.4×**，显存也低于 NRDE。🔴 **但四个堆叠循环模型全部比 Log-NCDE 快**（S5 的 244.78s vs 2128.32s，差 8.7 倍）—— 论文自己承认 "all four stacked recurrent models have lower average run times than Log-NCDEs"。

### 5.3 PPG-DaLiA 生理回归（Table 3，测试集 MSE ×10⁻²）

| LRU | S5 | S6 | MAMBA | NCDE | NRDE | **Log-NCDE** |
|---|---|---|---|---|---|---|
| 12.17±0.49 | 12.63±1.25 | 12.88±2.05 | 10.65±2.20 | 13.54±0.69 | 9.90±0.97 | **9.56±0.59** |

NCDE 在这个任务上**是最差**的（13.54），Log-NCDE 把它拉到最优。🔴 论文自己给出一个更保守的机制解释："the best GRU-$\Delta t$, GRU-D and ODE-RNN models did match the performance"（见 11-13 轮 CT-GRU 研读）。

### 5.4 Toy 数据集：一个机制诊断实验（§4.2 / §5.1）

自造 10 万条 6 通道 × 100 采样点序列；标签取路径 signature 的不同阶数项（1 通道变号 / 2 通道面积 / 3 通道体积 / 4 通道 4D 体积）。**结果极有信息量**：

- **NCDE 是 toy 上最好的模型** —— 因为这些分类问题**本身就是 CDE 的解**。这是论文唯一一个"连续时间机制在构造上就是正确答案"的场景。
- **NRDE 与 Log-NCDE 都被固定在 $r_{i+1}-r_i=4$、$N=2$**，因此都是 CDE 的近似；**Log-NCDE 一致优于 NRDE** ⇒ 论文据此给出实证论断："providing empirical evidence that **NRDEs do not always accurately learn the Lie bracket structure of $\bar f_\theta$**"。
- **堆叠循环模型随通道数增加而退化**：1–2 通道都好，3 通道开始下降，4 通道只有 MAMBA 还行。
- 论文用 Cirone et al. (2024) 的理论解释：非选择性 SSM（S5）的单个 block **无法捕获依赖多于一个通道的 signature 项**；堆叠选择性 SSM（MAMBA）可以用**纯线性混合层**捕获高阶项。

### 5.5 贡献清单

1. **方法**：Log-NCDE —— 用 $f_\theta$ 的迭代 Lie 括号结构化构造 $\bar f_\theta$，在不膨胀输出维度的前提下保留 CDE 的向量场结构；在 6 个 UEA 数据集 + PPG-DaLiA 上超过 NCDE / NRDE / LRU / S5 / S6 / MAMBA。
2. **理论**：Theorem 3.1 —— FCNN 的 Lip($\gamma$)（$1<\gamma\le2$）显式界，以及配套的训练期正则化项；并说明深度 7 就会超出单精度数值上限。
3. **工程**：用 `vmap` 批量化 JVP 评估，使 Log-NCDE 在高维时序上的实际运行时间**低于** NRDE（尽管 FLOPs 常数因子是 $3v$）。
4. **诊断**：toy 实验给出"NRDE 丢掉了 Lie 括号结构"的可证伪证据。

## 6. 局限性与未来展望

### 6.1 作者自陈局限（§3.5）

1. **只做到深度 $N\le 2$**：两个原因 —— (a) **$\gamma>2$ 时 FCNN 的 Lip($\gamma$) 界没有理论结果**；(b) 评估 $g_{\theta,X}$ 的计算成本随 $N$ **快速增长**，$N>2$ 对高维时序**在计算上不可行**。
2. **NCDE 的递归求解本质阻碍并行化** —— 与非选择性结构化 SSM（S4/S5）"微分方程可 time-parallel 求解"形成对照（论文明确引用 Gu et al. 2022）。
3. **对深度 $N>2$ 的未来工作**：需要为 $\gamma>2$ 证明等价结果；需要用**廉价 JVP 的结构化网络**当向量场；以及整合 Bayer et al. (2023) 的自适应 Log-ODE。

### 6.2 隐含局限（本仓视角）

- 🔴 **(a) 所有主结果都在「规则采样」数据上**。论文自己在结论里写："despite achieving state-of-the-art performance on the **regularly sampled datasets** considered in this paper, Log-NCDEs maintain the ability of NCDEs to naturally handle irregular sampling" —— **不规则采样只是"保留的能力"，不是"验证过的收益"**。本仓 11-16 轮已确立的选论文纪律（"必须先核对采样是否真·不等间隔"）在本篇上**恰好反向适用**：本篇**不能**用来支持"连续时间机制在真·不等间隔下有结构性收益"。这与 11-18 的 Neural CDE（2005.08926）形成互补：后者有 PhysioNet 10.3% 观测率，但精度优势在 std 内。
- 🔴 **(b) 与本仓 11-16 的"结构还原"结论一致**：连续时间机制的收益主要出现在**路径几何确实承载标签**的任务上（toy 实验）。UEA 六个数据集是等间隔的，路径几何信息基本是冗余的 ⇒ **Log-NCDE 的增益更可能来自"更强的向量场表达 + 训练稳定性"，而非"连续时间"本身**。论文自己对 NCDE→Linear NCDE 的解释也是 "removing the differential equation solver improves training stability"（见 2607.05280 研读报告）。
- 🔴 **(c) 与本仓主线的适用性问题**：本仓所有 bench 都是**自主式**的（LTC/CfC），Log-NCDE 属于受控档 ⇒ **不能直接嫁接**，需要先改驱动源。
- 🔴 **(d) 效率主张被自己的表格削弱**：Log-NCDE 虽然比 NCDE 快 3×，但比 S5 慢 8.7 倍、比 LRU 慢 4.6 倍 ⇒ "scalable" 是相对连续时间家族而言的。
- 🔴 **(e) 复现成本高**：需要 `signatory`/`iisignature` + JAX 的 `vmap` JVP + 手写 Hall 基展开，**不是现成 pip 包**；本仓 11-13/11-18 记录了"3rd-party 大 port 必死循环"的教训 ⇒ 移植前需先评估。

### 6.3 未来方向

- 深度 $N>2$ 的 Log-ODE（需 $\gamma>2$ 的正则性理论 + 廉价 JVP 结构化向量场）；
- 整合 Bayer et al. (2023) 的自适应区间选择；
- 把 Log-ODE 的区间自治化与结构化线性向量场结合（→ 2607.05280 的 Log-Linear NCDE，见同轮研读报告）。

## 7. 对本仓的意义

🔴 **(1) 本篇是本仓"ControlledCfCCell" / "LogSigDrivenCfCCell" 两个变体候选的第一份权威理论依据。** 11-18 轮已提出这两个候选（把 CfC 闭式解的驱动源从 $\Delta t$ 换成 $dX_s$ 调制 / 把 $D=2$ 的 log-signature 作额外输入），但当时**只有直觉没有理论支撑**。本篇补上了两块：(a) **机制层** —— 驱动源换成形签名之后，**Lie 括号是让动力学真正"看见"路径几何的通道**（toy 实验就是靠它证明的）；(b) **正则性层** —— 一旦向量场要算 Lie 括号，就必须显式控制 $\mathrm{Lip}(\gamma)$ 范数，否则深度 7 就超出单精度 ⇒ **本仓若做 `LogSigDrivenCfCCell`，必须同时加一条 Lip(2) 正则项**，否则训练会先炸在数值上。

🔴 **(2) 一条新的、消融成本极低的对照轴：$\bar f_\theta$ 是"自由网络"还是"结构化 Lie 括号"。** 论文的 toy 实验证明这二者**不是同一件事**（Log-NCDE 一致优于 NRDE，差距最大在 4 通道的 4D 体积分类）。本仓可做**最小版本**：在 toy signature 分类任务上比较 (a) 本仓现有 LTC/CfC；(b) 加 log-signature 驱动但 $\bar f$ 自由；(c) 加 log-signature 驱动且 $\bar f$ 由 Lie 括号构造。成本可控（不需要 JAX vmap，可用 `torch.func.jvp` 或 `torch.autograd.functional.jvp`）。

🔴 **(3) 本篇强化了本仓的一条既有纪律（"必须核对采样是否真·不等间隔"），并且给出了"何时连续时间才可能赢"的结构性判据。** 本篇最强的实验证据（toy）出现在**标签本身就是路径 signature 项**的场景 ⇒ **选论文/做消融时应优先构造"标签依赖路径几何"的任务**，而不是等间隔的通用时序基准（后者连本领域最强作者都只能做到 +2.3pp）。

🔴 **(4) 一条可直接落地的效率对照基线。** 本仓 bench 至今**只在液态家族内部互比**（LTC vs CfC vs GRU，11-18 轮已连续两次把"把 S4/Mamba 加入对照组"列为最便宜的高价值新轴）。本篇给出了**同一批 UEA 数据集上 NCDE / NRDE / Log-NCDE / LRU / S5 / S6 / MAMBA 的完整准确率 + 显存 + 运行时间三方对照表** ⇒ 可以直接抄成一份"本仓 bench 的参照基准表"，避免每次重新跑。

**Verdict**：
- **"路径级驱动 + Lie 括号结构在连续时间家族内部能带来真实增益" → TARGET-POSITIVE**（UEA 6/6 数据集 ≥ NRDE，平均 +3.7pp；PPG-DaLiA 由最差变最优；toy 上有明确的机制级证据）。**但引用时必须标注"主结果在规则采样数据上"，不可外推到不规则采样收益。**
- **"连续时间机制在通用时序任务上优于结构化线性递推" → TARGET-DEPENDENT-WITH-NUANCE**（Log-NCDE 64.3% vs S6 62.0%，差距 +2.3pp 且**在 std 量级附近**；且运行时间慢 8.7× ⇒ 精度优势与效率劣势同表并存，不能只引精度）。
- **"液态网络（LTC/CfC）在时序建模上有效" → TARGET-NEGATIVE（不适用）**。本篇**不是**液态模型（LTC/CfC 是自主式、$\Delta t$ 只作积分长度；Log-NCDE 是受控式、路径几何进入动力学）⇒ **本篇不能作为液态命题的任何一侧证据**，只能作为"连续时间谱系的第 ②/③ 档"的证据。引用时必须显式标注档位差异。
- **"NRDE 因为丢弃 Lie 括号结构而有性能损失" → TARGET-POSITIVE（机制级，已证伪性较强）**（toy 实验直接对比 + UEA 6/6 一致性）。
