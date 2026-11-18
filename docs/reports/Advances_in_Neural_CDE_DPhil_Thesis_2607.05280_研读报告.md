---
title: Advances in Neural Controlled Differential Equations (DPhil thesis, Benjamin Walker)
date: 2026-11
tags: [LNN, Continuous-Time, Neural-CDE, Linear-NCDE, Closed-Form, Parallel-Scan, Log-ODE, SLiCE, StateSpace, DPhil]
---

# 研读报告：Advances in Neural CDEs（牛津 DPhil 学位论文）— 把向量场线性化，于是 ODE 求解器彻底消失

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**。
> 🔴 **本篇对本仓最尖锐的一点**：**"闭式连续时间"不是液态家族的独有主张** —— CfC 的闭式解与 Linear NCDE 的闭式 flow 是**两条不同的路**，但它们在**同一个维度（连续时间计算成本）上正面相遇**。

## 1. 元数据

- **论文标题**：Advances in Neural Controlled Differential Equations
- **作者**：Benjamin Walker（Mathematical Institute, University of Oxford；DPhil 学位论文）
- **发表时间**：2026-07-06 (v1)；**DPhil thesis, University of Oxford**；188 页，17 图
- **来源**：arXiv:2607.05280v1 [cs.LG]
- **代码**：https://github.com/Benjamin-Walker/Log-Neural-CDEs（Log-NCDE 部分）；本篇三章的组合工作未单独给仓库
- **本地 PDF**：[papers/2607.05280/2607.05280v1.pdf](../../papers/2607.05280/2607.05280v1.pdf) ／ 文本 [papers/2607.05280/2607.05280v1.txt](../../papers/2607.05280/2607.05280v1.txt)
- **数据集**：UEA-MTSCA 6 个数据集（EigenWorms 17984 点 / EthanolConcentration / Heartbeat / MotorImagery / SCP1 / SCP2）、PPG-DaLiA（128Hz 生理回归）
- **关联概念**：Neural CDE / Log-ODE / **Linear NCDE** / **Structured Linear CDE (SLiCE)** / 平行结合扫描（parallel associative scan）/ 状态转移矩阵 / SSM 作为线性 NCDE / Mamba / S5 / Log-NCDE / LTC / CfC

## 2. 核心问题

NCDE 的"通用逼近 + 不规则采样"两个优点都来自同一个机制：**向量场是非线性的**（$f_\theta(h_s)dX_s$）。但这个机制同时带来一个**无法回避的工程代价**：

> 前向传播必须在每个观测区间**递归地解一个非线性微分方程**，因此**完全无法并行**。

论文把这条代价量化到令人不安的程度：在 EigenWorms（17984 个观测点）上，**NCDE 每 1000 个训练步需要 26020 秒**（§5.2.5）⇒ **10000 步训练约需 3 天**。对比之下，本仓最关心的边缘/资源受限场景根本不可能承受这个量级。

同时，SSM 家族（尤其 Mamba）已经把"线性扩展 + 训练高效"做成了行业标准。**如果连续时间模型连训练成本都比不过对角 SSM，它在工程上就没有立足点** —— 除非它能在保持连续时间表达力的同时把求解器消掉。

论文提出的路线是**结构性的**：**把向量场从非线性改成线性**。一旦向量场线性，受控微分方程

$$h_t=h_{t_0}+\int_{t_0}^t A_\theta h_s\,d\omega^X_s$$

就退化成一个**常系数线性系统**，其解**有闭式表达**（矩阵指数），并且在每个区间上是**常微分动力学的精确流映射**。这意味着：

1. **不需要 ODE 求解器** —— 不是"更快的求解器"，而是**没有求解器**；
2. **区间内的更新是矩阵指数**，可以**沿时间并行**（结合结合律的 prefix scan，把递归深度从 $n$ 降到 $O(\log n)$）；
3. **Log-ODE 方法的近似流也有闭式解** —— 于是可以同时拿到"区间自治化"和"闭式"两个好处（**Log-Linear NCDE**）。

问题因此被重述为：**把向量场线性化会损失多少表达力？** 如果答案是"在隐藏维度可变时不损失"，那么连续时间模型就能同时拥有"最大表达力"与"SSM 级的效率"，从而彻底解决第 1 段提出的工程困境。

## 3. 方法论与核心思路

### 3.1 三项贡献的分工

本学位论文合并三份工作，构成一个**互补的三层结构**：

| 层 | 方法 | 解决的问题 | 是否消除 ODE 求解器 |
|---|---|---|---|
| ① | **Log-NCDE**（Ch.4，= arXiv:2402.18512） | 区间自治化 + Lie 括号结构 | ❌ 仍需解非线性 ODE |
| ② | **Linear NCDE**（Ch.5.2） | 向量场线性化 ⇒ 闭式 flow | ✅ **是** |
| ③ | **Structured Linear NCDE / SLiCE**（Ch.5.4） | 稠密矩阵 ⇒ 结构化矩阵 | ✅ **是** |

论文自陈："**Log-ODE method provides an efficient path-based approximation, Linear NCDEs provide a scalable model architecture, and SLiCEs improve the computational efficiency.**"

### 3.2 Linear NCDE 的定义与闭式 flow（Eq. 5.1–5.4）

**Definition 5.1**：设 $\mathcal X^{(d)}$ 为有界变差 $d$ 维路径空间（含时间通道），$\omega:\mathcal X^{(d_X)}\to\mathcal X^{(d_\omega)}$ 为路径上的连续函数。Linear NCDE 定义为

$$h_{t_0}=L^1_\theta X_{t_0},\qquad h_t=h_{t_0}+\int_{t_0}^t A_\theta h_s\,d\omega^X_s,\qquad y_t=L^2_\theta h_t$$

与 NCDE 的两个本质差异：(a) 驱动路径是**插值数据的函数** $\omega^X=\omega(X)$ 而非插值本身；(b) **向量场对隐藏状态是线性的**：

$$h_t=h_{t_0}+\int_{t_0}^t A_\theta h_s\,d\omega^X_s=h_{t_0}+\sum_{i=1}^{d_\omega}\int_{t_0}^t A^i_\theta h_s\,d\omega^{X,i}_s$$

**闭式 flow**：当 $\omega^X$ 在网格上分段线性时，每个子区间 $[t_j,t_{j+1}]$ 上动力学为常的，于是更新**精确**可算：

$$\tilde h_{t_{j+1}}=\exp\Big(\sum_{i=1}^{d_\omega}\big(\omega^{X,i}_{t_{j+1}}-\omega^{X,i}_{t_j}\big)A^i\Big)\tilde h_{t_j}$$

论文强调这**不只是"去掉通用 ODE 求解器"**，而是**视角的改变**：

> "Rather than tracking a trajectory $t\mapsto h_t$, we compute the state-transition map from $t_j$ to $t_{j+1}$."

这个"状态转移映射"正是**可以并行组合的对象**。因为矩阵指数满足结合律，任意区间上的流可以用 **$O(\log n)$ 步平行结合扫描**（parallel associative scan）算出。

### 3.3 Log-Linear NCDE：闭式 + 区间自治化

在 Log-ODE 框架下，$\bar f_\theta$ 换成矩阵 $A^i_\theta$，Lie 括号变成**矩阵的交换子**：

$$\bar A^k_\theta=A^k_\theta,\qquad \bar A^k_\theta=[\bar A^i_\theta,\bar A^j_\theta]=\bar A^i_\theta\bar A^j_\theta-\bar A^j_\theta\bar A^i_\theta$$

于是 **Log-ODE 仍然产生区间上的线性流**，平行扫描依然适用。🔴 **论文指出的关键工程收益**：

> "the Lie brackets $\bar A^k_\theta$ are computed from **products and commutators of the matrices $A^i_\theta$**, rather than from the forward-mode auto-differentiated Jacobian-vector products required in Log-NCDEs, which substantially reduces the computational cost."

且只需在 GPU 显存中materialise $m<n$ 个状态转移矩阵而非 $n$ 个 ⇒ **扫描的显存与 I/O 成本大幅下降**。

### 3.4 表达力：线性会不会白丢？

**Theorem 5.2（最大表达力的 Linear NCDE）**：设 $F$ 是 Linear NCDE 诱导的路径→点函数类，则 $F$ 在 $\mathcal X$ 上**最大表达**（maximally expressive）。

**论文给出的机制解释非常清晰**：

> "Although a Linear CDE is linear in the hidden state, its evolution is driven by the **multiplicative interaction** between the hidden state and the increments of the driving path. This is precisely the mechanism that generates iterated integrals, since each tensor level of the truncated signature is obtained by integrating the previous level against the path. Therefore, just as repeated application of the map $(x,y)\mapsto xy$ generates monomials, **repeated multiplication of the hidden state by path increments generates the tensor levels of the truncated signature**. Hence, this single multiplicative interaction is sufficient for maximal expressivity."

**Theorem 5.4（最大概率表达力）**：即使 $A^i_\theta$ 的元素从 $\mathcal N(0,1/d_h)$ 独立随机采样，Linear NCDE 仍具**最大概率表达力**（与 random-tensor RNN 同一族）。

🔴 **重要限定**：论文明确写了 —— "**For a fixed hidden dimension**, Linear NCDEs are **less expressive** than NCDEs, since the former are a strict subclass of the latter. **However, once the hidden dimension is allowed to vary**, both model classes have the same theoretical expressivity."

### 3.5 SSM 是 Linear NCDE 的子类（§5.3.2）

论文把 **Mamba 重写为仿射 Linear NCDE**，从而把整个 SSM 谱系纳入 CDE 框架。这对本仓的意义在 2610.05051 的研读报告里已体现（LogSig-SSM 正是基于这个视角）。本篇进一步指出：

> "The main difference between SSMs and the general Linear NCDE is that SSMs process the hidden state **for each channel individually**, whereas a Linear NCDE **mixes the hidden state and the path**."

### 3.6 SLiCE：结构化线性向量场

四种结构：**Diagonal (D)** / **Block-Diagonal (BD)** / **Diagonal-plus-Low-Rank (DPLR)** / **Sparse (S)** / **Walsh–Hadamard (WH)**。**Table 5.2 的结构对比（本篇最有实用价值的一张表）**：

| Structure | Parameters | Recurrent Cost | Parallel Cost | Efficient Impl. | Maximally Expressive |
|---|---|---|---|---|---|
| Dense | $O(d^3h)$ | $O(nd^3h)$ | $O(\log n, d^3h)$ | Yes | Yes |
| **Diagonal** | $O(d^2h)$ | $O(nd^2h)$ | $O(\log n, d^2h)$ | Yes | **No** |
| DPLR | $O(rd^2h)$ | $O(nrd^2h)$ | $O(\log n, d^3h)$ | Yes | Yes |
| Sparse | $O(d^2+\epsilon h)$ | $O(nd^2+\epsilon h)$ | $O(\log n, d^3h)$ | **No** | Yes |
| WH | $O(d^2h)$ | $O(nd^2h)$ | $O(\log n, d^3h)$ | Yes | Yes |
| BD | $O(\sum_j b_j^2 h)$ | $O(n\sum_j b_j^2h)$ | $O(\log n, \sum_j b_j^3h)$ | Yes | Yes |

🔴 **一个容易被忽略的坑，论文自己指出**：**只有对角与块对角矩阵在乘法下封闭** ⇒ 其余结构（DPLR / Sparse / WH）的**平行扫描每次组合的成本退化到与稠密相同的 $O(d^3h)$**。即"结构化"在参数与递归成本上的节省**不一定**转化为平行扫描上的节省。

**另一条被论文明确指出的否定性结论**：

> "current deep-learning frameworks, such as JAX and PyTorch, are **not optimised for unstructured sparsity**, so the sparse structure **does not lead to practical speed-ups** in our implementations."

### 3.7 与 LTC / CfC 的关系（本篇对本仓最尖锐的一点）

🔴 **CfC (Hasani et al. 2021) 的核心卖点是"闭式连续时间"** —— 本仓也一直把 CfC 的闭式解当作液态家族的独特优势。**Linear NCDE 证明闭式解与"连续时间"没有绑定关系**：

| | 连续时间？ | 闭式？ | 驱动源 | 路径几何（Lévy area / signature 项）是否进入 |
|---|---|---|---|---|
| **LTC** (1811.00321) | 是（Euler 数值积分） | **否** | 无（自治） | 否 |
| **CfC** (2106.13898) | 是 | **是** | 无（自治） | 否 |
| **Linear NCDE** | 是 | **是**（矩阵指数） | **路径 $\omega^X$** | **是**（乘法交互生成 signature 各级） |
| **GRU / LRU / S5 / Mamba** | 视作线性 CDE | 是 | 观测值 | 视结构而定（对角 block 不含） |

⇒ **本篇与 11-18 的 Neural CDE 研读形成闭环**：CfC 的闭式解**不引入**路径几何（因为 CfC 是自主式，没有 $X$）；Linear NCDE 的闭式解**引入**路径几何（因为路径乘进隐藏状态）。**"闭式"和"路径级驱动"是两个独立的轴，CfC 只占了前者。**

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 5.1 | $$h_{t_0}=L^1_\theta X_{t_0},\quad h_t=h_{t_0}+\int_{t_0}^t A_\theta h_s\,d\omega^X_s,\quad y_t=L^2_\theta h_t$$ | **Linear NCDE 定义** |
| Eq. 5.2 | $$h_t=h_{t_0}+\int_{t_0}^t A_\theta h_s\,d\omega^X_s=h_{t_0}+\sum_{i=1}^{d_\omega}\int_{t_0}^t A^i_\theta h_s\,d\omega^{X,i}_s$$ | 向量场的线性分解（按驱动通道） |
| Eq. 5.3 | $$\tilde h_{t_{j+1}}=\exp\Big(\sum_{i=1}^{d_\omega}\big(\omega^{X,i}_{t_{j+1}}-\omega^{X,i}_{t_j}\big)A^i\Big)\tilde h_{t_j}$$ | **闭式 flow（区间更新）** |
| Eq. 5.4 | $$\exp\Big(\sum_{i=1}^{d_\omega}\big(\omega^{X,i}_{t_{j+1}}-\omega^{X,i}_{t_j}\big)A^i\Big)$$ | **状态转移映射（可并行组合的对象）** |
| Eq. 5.6 | $$(a\otimes b)\otimes c=a\otimes(b\otimes c)$$ | 结合律 ⇒ prefix scan 可并行 |
| Eq. 5.11–5.12 | $$\bar A^k_\theta=A^k_\theta;\qquad \bar A^k_\theta=[\bar A^i_\theta,\bar A^j_\theta]=\bar A^i_\theta\bar A^j_\theta-\bar A^j_\theta\bar A^i_\theta$$ | **Log-Linear 的 Lie 括号 = 矩阵交换子**（无需 JVP） |
| Eq. 5.20 | $$\frac{\omega^X_{t_{j+1}}-\omega^X_{t_j}}{t_{j+1}-t_j}=(1,X_{t_j}),\qquad \xi^X_t=0\ \Rightarrow\ d_\omega=7$$ | 实验的驱动构造 |
| Thm 5.2 | Linear NCDE 诱导的函数类在 $\mathcal X$ 上**最大表达** | **线性不丢表达力**（隐藏维可变时） |
| Thm 5.4 | $A^i_\theta$ 元素 i.i.d. $\mathcal N(0,1/d_h)$ 时仍**最大概率表达** | 与 random-tensor RNN 同族 |
| Table 5.1 | 见 §5.2 | 精度 + 时间 + 显存三元对照 |

## 5. 关键成果与贡献

### 5.1 主结果：EigenWorms 上的四模型三元对照（Table 5.1，RTX 4090，batch=1，$d_h=128$，17984 观测点）

| | NCDE | Log-NCDE | Linear NCDE | **Log-Linear NCDE** |
|---|---|---|---|---|
| **测试准确率 (%)** | 75.0±3.9 | 85.6±5.1 | 87.2±5.2 | **87.8±5.7** |
| 递归：时间/1000 步 (s) | 26020 | 2263 | 299.4 | **29.9** |
| 递归：显存 (MB) | 3484 | 3494 | 9624 | 3486 |
| 平行：时间/1000 步 (s) | — | — | 100.4 | **17.5** |
| 平行：显存 (MB) | — | — | 13730 | 3492 |

🔴 **论文自己的结论（§5.2.5）**：
- 线性化把准确率从 75.0% **提到 87.2%** —— **不是降级，是提升**。论文给出一个诚实的可能解释："removing the differential equation solver **improves training stability**, since the Linear NCDE is solved exactly over each interval"。
- 递归时间降 **80×**（26020 → 299.4 s）；**平行扫描后降 250×**（26020 → 100.4 s）。
- 加 Log-ODE 后进一步降至 **17.5 s** ⇒ **相对 NCDE 约 1500×**："10000 training steps take **under 3 minutes** rather than around 3 days"，且**准确率 +12.8pp**，显存只多 8MB。

### 5.2 SLiCE 的 UEA 全基准（Table 5.5，5 个划分平均准确率 ± 标准差，%）

| Model | EW | EC | HB | MI | SCP1 | SCP2 |
|---|---|---|---|---|---|---|
| **BD-SLiCE** | 86.1±3.6 | 28.6±6.4 | 77.4±5.6 | **53.0±2.1** | 84.9±1.9 | **54.0±7.4** |
| Log-NCDE | 85.6±5.1 | **34.4±6.4** | 75.2±4.7 | 53.7±5.3 | 83.1±2.9 | 53.7±4.1 |
| D-DE-SLiCE | 85.6±6.0 | 27.3±6.9 | 73.9±3.8 | 54.7±3.5 | 84.7±3.7 | 51.6±5.0 |
| WH-SLiCE | 85.0±6.0 | 30.1±4.8 | 76.1±5.9 | 49.5±6.8 | 82.4±2.2 | 51.9±5.4 |
| DPLR-SLiCE | 84.4±5.2 | 27.6±4.7 | 74.2±5.4 | 51.6±5.8 | 83.5±2.2 | 50.9±5.8 |
| D-SLiCE | 79.4±5.8 | 27.1±4.6 | 72.9±5.1 | 54.4±6.3 | 83.5±2.2 | 53.0±5.9 |
| LRU | 87.8±2.9 | 21.5±2.2 | 78.4±6.7 | 48.4±5.1 | 82.6±3.5 | 51.2±3.6 |
| S6 | 85.0±16.1 | 26.4±5.6 | 76.5±8.3 | 51.3±4.8 | 82.8±2.8 | 49.9±9.5 |
| S5 | 81.1±3.7 | 24.1±4.4 | 77.7±5.6 | 47.7±5.5 | **89.9±4.7** | 50.5±2.6 |
| S-SLiCE | 87.8±4.2 | 30.4±6.7 | 72.6±5.5 | 47.7±2.4 | 82.8±1.6 | 49.5±4.0 |
| DE-LNCDE | 88.3±3.7 | 25.8±4.1 | 74.2±4.5 | 49.8±5.3 | 81.9±4.0 | 49.8±2.4 |
| NCDE | 75.0±4.0 | 29.9±6.6 | 73.9±2.6 | 49.5±2.9 | 79.8±5.7 | 53.0±2.9 |
| NRDE | 83.9±7.3 | 25.3±1.8 | 72.9±4.9 | 47.0±5.8 | 80.9±2.6 | 53.7±6.9 |
| Mamba | 70.9±15.9 | 27.9±4.6 | 76.2±3.9 | 47.7±4.5 | 80.7±1.4 | 48.2±4.0 |

🔴 **统计检验的诚实报告**：对 14 个模型 × 6 个数据集做 **tie-corrected Friedman 检验：$\chi^2=19.89$, $df=13$, $p=0.098$** ⇒ **在 5% 显著性水平下没有检测到显著差异**。论文的措辞是 "Nevertheless, the results show that..." ⇒ **作者没有把统计显著性不足隐藏掉，但结论句仍带上了它**。引用时必须写明 $p=0.098$。

**Table 5.6（效率汇总，H100）**：BD-SLiCE 平均准确率 **64.0** / 平均排名 **3.2** / 时间 **68.1 s per 1000 steps** / 显存 2344 MB；Log-NCDE 64.3 / 4.0 / **1321.7 s** / 2177 MB；D-SLiCE 61.7 / 7.2 / **11.0 s** / 1875 MB；LRU 61.7 / 7.3 / 26.9 s / 4308 MB。

⇒ **BD-SLiCE 用 Log-NCDE 1/19 的时间达到几乎相同的准确率**；论文的解释："**the block-diagonal structure achieves very similar performance to the non-linear model, whilst reducing the average time per training step by a factor of nearly 20**"，并提出一个有趣假设：**BD 结构与多头注意力同构**（每个 block 像一个独立的 head，在自己的子空间里处理输入路径），这种模块化"可能防止单一稠密模型的过拟合"。

### 5.3 平行扫描的 chunk 效应（Table 5.7，EigenWorms，H100，batch=1）

| Parallel Steps | D-SLiCE | BD-SLiCE | DE-LNCDE |
|---|---|---|---|
| None | 311.0 | 374.89 | 444.68 |
| 4 | 134.2 | 326.75 | 439.51 |
| 16 | 57.50 | 161.74 | 257.56 |
| 64 | 31.01 | 68.59 | 126.01 |
| 256 | **21.10** | **30.53** | **71.54** |

**Table 5.8（Log-ODE + 平行扫描）**：D-SLiCE 317.49 → **10.96** s/1000 步（**约 30×**），显存 2.69 GB 不变；DE-LNCDE 465.94 → 29.37 s（16×），显存 **35.45 → 6.79 GB（>5×）**。

### 5.4 贡献清单

1. **Linear NCDE**：把向量场线性化，得到**闭式 flow**（矩阵指数）+ **可并行组合的状态转移映射**，**消除 ODE 求解器**；在隐藏维度可变时**保持最大表达力**（Thm 5.2），甚至在随机初始化下也具最大概率表达力（Thm 5.4）。
2. **Log-Linear NCDE**：Log-ODE + 线性向量场 ⇒ 区间自治化与闭式**兼得**；Lie 括号退化为**矩阵交换子**（无需 JVP）⇒ 计算与显存 I/O 双降。
3. **SLiCE**：五种结构化线性向量场（D / BD / DPLR / Sparse / WH），**给出参数、递归成本、平行成本、是否有高效实现、是否最大表达力**的完整对照表（Table 5.2）；BD 结构在 UEA 上以 **1/19 的时间**达到与 Log-NCDE 相当的准确率。
4. **把 SSM 纳入 CDE 框架**（Mamba 重写为仿射 Linear NCDE），为整个结构化线性谱系提供统一的连续时间解释。
5. **端到端效率结论**：Log-Linear NCDE + 平行扫描相对 NCDE **约 1500× 加速**，同时准确率 **+12.8pp**，显存仅多 8MB。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **Ch.5.4.8（稳定性）**：**没有对矩阵 $A^i_\theta$ 施加稳定性约束** ⇒ 虽然在所考虑的长序列上未观察到不稳定，但"**不应指望当前参数化在序列长度任意增大时保持一致稳定**"。🔴 **这与本仓 11-13 轮 Lipschitz RNN 研读（2006.12070，全局指数稳定性）形成鲜明对照** —— 那篇的"稳定性"是**理论保证**，本篇的是**未观察到**。
- **§5.4.8（结构化实现的现实）**：**当前深度学习框架（JAX / PyTorch）未针对无结构稀疏优化** ⇒ sparse 结构**没有实际加速**。
- **平行扫描的封闭性限制**：只有对角/块对角在矩阵乘法下封闭，其余结构每次组合成本退化到稠密的 $O(d^3h)$。
- **平行扫描的 I/O 成本**：每个状态转移矩阵都必须materialise在 GPU 显存中（引用 Yang et al. 2024b）。
- **Log-NCDE 的深度限制**（继承自 2402.18512）：$\gamma>2$ 无 Lip 界、$N>2$ 计算不可行。

### 6.2 隐含局限（本仓视角）

- 🔴 **(a) 线性化带来的"准确率提升"可能主要来自训练稳定性，而非表达力。** 论文自己的解释是 "removing the differential equation solver improves training stability"。⇒ **"Linear NCDE 比 NCDE 准 12.8pp" 这个数字不能被读作"线性表达力更强"**，只能读作"**求解器带来的优化困难占了 NCDE 的很大一部分劣势**"。这与本仓 11-13 轮 CT-GRU 的负结果（"连续时间建模做成完整模型后无优势"）方向一致：**连续时间的"正统实现"（数值求解 ODE）在本就是自己的弱项，而非其表达力的弱项**。
- 🔴 **(b) 训练时间是 RTX 4090 / H100 单卡数字，无边缘设备实测** ⇒ 本仓最关心的 Jetson Orin Nano / 端侧场景**完全未覆盖**。注意 Table 5.1 的 batch size = 1、Table 5.8 的 batch size = 4，且 DE-LNCDE 在无 Log-ODE 时显存高达 35.45 GB（batch=4）⇒ **在端侧不可行**。
- 🔴 **(c) 与本仓主线的可迁移性有限** —— Linear NCDE 是**受控式**（$\omega^X$ 是插值数据的函数），本仓 `ltc.py` / `cfc.py` 是**自主式**。要移植需要先改驱动源与状态演化形式 ⇒ 不是"把 A 换成 I"的事。
- 🔴 **(d) 学位论文体裁 ⇒ 章节间有重复，结论分散** —— 188 页里 Ch.4（Log-NCDE）与已单独发表的 2402.18512 大量重叠。**引用时应优先引 2402.18512（ICML 2024 正式版），本篇用于引用 Linear NCDE / SLiCE 的独有部分**（Ch.5）。
- 🔴 **(e) 统计检验未达显著（$p=0.098$）** ⇒ Table 5.5 的 14 模型排序**不能**当作可靠排名使用；只有 Table 5.1（EigenWorms 单点，+12.8pp 与 1500× 加速）是大效应量、可靠的。

### 6.3 未来方向

- 为 $A^i_\theta$ 施加稳定性约束（论文自陈的第一条局限）；
- 解决平行扫描的显存 I/O 瓶颈（引用 Yang et al. 2024b 的替代方案）；
- 把结构化线性向量场与更高阶路径几何结合。

## 7. 对本仓的意义

🔴 **(1) 一条对本仓"液态路线定位"的重大校准：**闭式连续时间不是液态的护城河。** CfC 的核心卖点是"闭式连续时间"；本篇证明，**把向量场线性化同样能得到闭式连续时间，而且比 CfC 走得更远**（矩阵指数 + 平行扫描 ⇒ 相对 NCDE 约 1500× 加速，且路径几何**进入**了动力学）。⇒ **本仓索引里"CfC 的闭式解是液态家族的关键优势"这一表述需要加限定** —— 更准确的表述是：**"闭式"是连续时间计算的一档工程属性（可由线性化或特定参数化获得），它本身不证明任何表达力优势；真正稀缺的是"路径级驱动 + 路径几何进入动力学"，而 CfC 不具备后者。** 这是本仓今天最重要的一次口径修正。

🔴 **(2) 一个本仓可以立刻做、且成本极低的对照轴："把 $\tau$ 换成 $A$，把 ODE 数值积分换成矩阵指数"。** 本仓 `lnn/core/cfc.py` 的闭式解是**逐元素**的（每个神经元一个标量指数），而 Linear NCDE 的闭式解是**矩阵指数**（带通道间混合）。⇒ **一个具体的新变体候选：`MatrixExpCfCCell` / `LinearLiquidCell`** —— 把本仓 CfC 的自治 ODE 写成线性形式 $h_t = \exp(A\Delta t)h_0 + \dots$，用**矩阵指数**代替逐元素指数，从而**免费获得通道间耦合**。这是本仓 `lnn/core/variants.py` 至今**没有**的维度（本仓所有变体都是逐元素的）。**成本**：一个 `torch.linalg.matrix_exp` 调用 + 一个 $d_h\times d_h$ 参数矩阵，**不需要 ODE 求解器**。**验证方式**：在现有 toy benchmark 上与 CfC 对比；若通道数少（toy 常用 16/32）则矩阵指数开销可忽略。

🔴 **(3) 一条对本仓 bench 设计的具体建议（承接 11-13 与 11-18 的两条纪律）：** 本篇 Table 5.1 给出了"**精度 / 时间 / 显存**"三元对照的一个现成模板 —— 这是本仓 bench 脚本至今**只有精度**的一个明显缺口。本仓的 `bench_mdl_cfc.py` 等脚本若要支持"液态是否值得"的决策，**必须同时报告训练时间与显存**，因为本篇的数据说明：Log-NCDE 与 BD-SLiCE 精度几乎相同（64.3 vs 64.0），**时间却差 19 倍**。

🔴 **(4) 一条稳定性纪律：** 本篇自陈"**没有对 $A^i_\theta$ 施加稳定性约束，不应指望在序列长度任意增大时保持一致稳定**" ⇒ 结合本仓 11-13 的 Lipschitz RNN（2006.12070，**有**全局指数稳定性保证）研读，**本仓可以开始积累一条"连续时间模型的稳定性保证"对照表**：(a) 本仓 LTC/CfC（可学习 $\tau$，**结构上隐含有界**，因为 $\exp(-\Delta t/\tau)$ 的收缩性是构造性的）；(b) Linear NCDE / SLiCE（**无保证，作者自陈**）；(c) Lipschitz RNN（**有保证**）。⇒ 这是一条**本仓可以立刻写进索引的新消融轴：长序列下的状态发散 vs 有界性**。

🔴 **(5) 一条对 11-16 轮"把 S4/Mamba 加入本仓对照组"这条待办的强力支持，且给出了正确的对照对象。** 本篇不是简单地说"Mamba 也不错"，而是给出了**统一解释**：Mamba/S5 是 **Linear NCDE 的受限子类**（按通道独立处理），而**Linear NCDE 本身（带通道混合）是最大表达力的**。⇒ **本仓要加的对照组不应止于 S4/Mamba，而应包含"带通道混合的线性连续时间模型"** —— 因为这才是与本仓 LTC/CfC 真正同构、且能隔离"连续时间"与"通道混合"两个因素的对照物。**这是一个方法论级升级，不是加几个基线。**

**Verdict**：
- **"把连续时间模型的向量场线性化，可以在保持（隐藏维可变时）最大表达力的同时消除 ODE 求解器，带来约 1500× 的端到端训练加速" → TARGET-POSITIVE（本篇最强的一条，EigenWorms 上 26020 s → 17.5 s per 1000 steps，同时准确率 +12.8pp、显存仅多 8MB）**。
- **"闭式连续时间是液态家族的独特优势" → TARGET-NEGATIVE-WITH-NUANCE（本轮最重要的口径修正）**。**闭式解可由线性化获得，且该路线同时具备"路径几何进入动力学"（CfC 不具备）** ⇒ **本仓引用 CfC 时必须把"闭式"与"路径级驱动"分开陈述，不得把"闭式"当作液态的表达力证据。**
- **"SLiCE（结构化线性向量场）在 UEA 上不劣于非线性模型" → TARGET-DEPENDENT-WITH-NUANCE**（BD-SLiCE 时间少 19×、准确率 64.0 vs 64.3，**但 tie-corrected Friedman $p=0.098$ 未达 5% 显著** ⇒ **只能引用为"未观察到显著劣势"，不能引用为"等效"**）。
- **"连续时间模型在长序列上工程可行" → TARGET-POSITIVE（但仅限大算力）**。**所有数字来自 RTX 4090 / H100，无端侧实测；DE-LNCDE 在 batch=4 时显存达 35.45 GB ⇒ 对本仓边缘部署目标目前不可用。**
- **"Linear NCDE / SLiCE 与液态命题（LTC/CfC）的关系" → TARGET-NEGATIVE（不适用）**。二者都是**受控式**（有驱动路径），而 LTC/CfC 是**自主式** ⇒ **本篇不能作为液态命题的任何一侧证据**；它的价值是**为液态范式提供外部对照压力，并校准"闭式"这一属性的真实稀缺性**。
