---
title: Faithful Embeddings of Irregular and Asynchronous Data for Online Log-NCDEs
date: 2026-11
tags: [LNN, Continuous-Time, Log-NCDE, Log-SLiCE, Signature-Method, Faithful-Embedding, Irregular-Sampling, Universal-Approximation, Input-Representation-Ablation]
---

# 研读报告：Log-NCDE 的输入侧革命 —— 不插值，直接把离散观测做成 log-signature

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**（`2605.30213` 在 171 篇报告中零命中）。
> 🔴 **本篇对本仓的意义是"划输入侧的边界"** —— 它第一次把"插值方案"从**建模选择**降格为**可证明多余的表示步骤**，并给出一条本仓至今完全没有的消融轴：**固定架构、只换输入表示**。

## 1. 元数据

- **论文标题**：Faithful Embeddings of Irregular and Asynchronous Data for Online Log-NCDEs
- **作者**：Benjamin Walker†, Alexandre Bloch†, Lingyi Yang, Sam Morley, Terry Lyons（†共同一作）
- **机构**：University of Oxford / Continuous Intelligence Limited / University of Nottingham / Imperial College London
- **发表时间**：2026-05-28 (v1)，v2 更新 2026-11-07（arXiv:2605.30213v2）
- **发表渠道**：**NeurIPS 2026**（40th Conference on Neural Information Processing Systems）；**已评审**
- **来源**：arXiv:2605.30213v2 [cs.LG]
- **代码**：N/A（论文未给出代码链接；作者团队另有 `Roughrax` JAX 包见 11-21 同轮 `2606.05272`）
- **本地 PDF**：[papers/2605.30213/2605.30213v1.pdf](../../papers/2605.30213/2605.30213v1.pdf) ／ 文本 [papers/2605.30213/2605.30213v1.txt](../../papers/2605.30213/2605.30213v1.txt)
- **页数**：36 页
- **关联概念**：Log-NCDE / Log-SLiCE / NCDE / path signature / log-signature / tensor algebra / **faithful embedding（连续 + 单射）** / universality transfer / rectilinear control path / irregular & asynchronous sampling / input dropping / UEA 多变量时序分类 / S5 / LRU / S6 / Mamba

## 2. 核心问题

连续时间模型（NCDE / Log-NCDE / ContiFormer）在处理**不规则、异步、部分观测**的时序数据时，都必须先回答一个前置问题：**离散观测如何嵌入连续时间？**

主流做法是**先补全、再建模**：

- **插值（interpolation）**：用三次 Hermite / 样条 / GP 把离散点连成一条光滑路径。问题是**插值器引入了数据里没有的动态**，且可能**造出落在观测值域之外的数值**。
- **插补（imputation）**：用全局均值 / 前值 / 模型填充缺失值。问题是**填进了从未观测过的值**。
- **ODE-RNN / GRU-ODE-Bayes 路线**：连续潜动态 + 离散更新，避开显式插值，但**不提供任意时刻查询**的连续状态语义。

这些做法的共同代价是：**模型学到的输入包含了"补全假设"，而不是"观测事实"**。论文把这一点表述为一个**结构性缺陷**：

> "The central principle of this paper is that a continuous-time representation of discrete observations should preserve only the information that was observed, rather than guess the unobserved behaviour between measurements. The latter should form part of the modelling as opposed to part of the representation."

此前的工作（Morrill et al. 2022 的 rectilinear control path）研究过 NCDE 控制路径嵌入的 online / smooth / bounded / unique（等价于单射）性质，但**只对 NCDE 成立**——Log-NCDE 用的是区间 log-signature 而非路径逐点值，"光滑性"不再内蕴。论文于是提出两个此前没人问的问题：

1. **理论上**：如果**不插值**，连续时间模型还能保持**通用逼近性**吗？换句话说，**表示方式的"忠实性"是否可证明地等价于一个可控条件**？
2. **实践上**：能不能构造一个**同时满足 faithful / online / 可高效算任意区间 log-signature** 的嵌入？以及——**在固定架构下，光换输入表示能带来多少收益**？

问题因此被重述为：**"嵌入的忠实性（faithfulness）是通用性转移的充分条件"** —— 一个拓扑命题，而非工程调参问题。

## 3. 方法论与核心思路

### 3.1 总体架构

论文的逻辑链是"**定理 → 构造 → 实证**"三段：

1. **定理（Lemma 1）**：若数据空间 $X$ 与模型输入空间 $P$ 都是 Hausdorff，且嵌入 $\varphi:X\to P$ **连续且单射**，则模型输入空间上紧集上的**通用逼近性**可以**转移到数据空间**上。
2. **构造（§2.4）**：把每个观测事件转成张量代数中的一个增量（增量 + 计数），跨任意查询区间组合成区间 log-signature，全程**不插值**。
3. **实证（§3）**：EigenWorms（加速/显存）、合成耦合系统（跨采样 regime 泛化）、Brownian 受控系统（高阶信息增益）、UEA 六数据集 input dropping（鲁棒性）。

⚠️ **一个容易读漏的关键点**：Lemma 1 说的是"**紧集上**的通用性"转移，**不涉及样本效率或优化保证**（Appendix B 明确划界）。也就是说——**faithful embedding 保证"不会系统性少掉可表示的函数"，但不保证"更容易学"**。本仓引用时必须带上这条限定，否则会把它误读成"faithful 嵌入 = 更好训"。

### 3.2 核心模块 1：faithful embedding 的定义与通用性转移（Lemma 1）

**定义**：嵌入 $\varphi:X\to P$ 称为 **faithful**，若它**连续**且**单射**。

- **连续** ⇒ 相邻数据在嵌入后仍相邻（不会把小的输入差异放大成大的表示差异）；
- **单射** ⇒ 不同的数据保持可区分（反之，若 $\varphi$ 把两个数据映成同一个点，**任何下游模型都无法区分它们** —— 论文在 Appendix A 明确写出这个"反向障碍"）。

$$\text{Lemma 1 (Universality Transfer)}\quad \forall K_0\subseteq X\ \text{紧}:\quad \overline{(F\circ\varphi)|_{K_0}}^{\|\cdot\|_\infty}=C(K_0;\mathbb{R})\ \text{若}\ \overline{F|_K}^{\|\cdot\|_\infty}=C(K;\mathbb{R})\ \forall K\subseteq P\ \text{紧}$$

由于 NCDE 与 Log-NCDE 在路径空间上是通用的（Kidger et al. 2020；Walker et al. 2024），该结论对**任意 faithful 观测嵌入**都成立 —— 这就把"选哪个插值器"从**必须回答的问题**降级为**可选的风格**。

### 3.3 核心模块 2：从观测流到区间 log-signature（§2.4）

**观测流的形式化**（Eq. 8）：

$$x=\big((t_0,s_0,x_0),\ldots,(t_n,s_n,x_n)\big),\quad 0\le t_0<\cdots<t_n\le T$$

其中 $s_i\subseteq\{1,\ldots,d_{\text{disc}}\}$ 是时刻 $t_i$ **被观测到的通道子集**（非空），$x_i^k\in\mathbb{R}$ 是通道 $k\in s_i$ 的观测值。**间隔 $t_{i+1}-t_i$ 不必相等，不同时刻的 $s_i$ 可以不同** ⇒ 不规则 + 异步 + 部分观测三者统一表达。

**坐标设计**（Eq. 9）—— 这是"忠实"的关键：

$$d_X = 2d_{\text{disc}} + d_{\text{cont}}$$

- 前 $d_{\text{disc}}$ 个坐标：各离散通道**当前记录值**；
- 后 $d_{\text{disc}}$ 个坐标：各通道的**观测计数**；
- 最后 $d_{\text{cont}}$ 个坐标：**连续观测变量**（时间是标准例子）。

> **计数坐标为什么必要**：如果只有"当前值"坐标，则"通道 $k$ 这次没被观测"与"通道 $k$ 被观测但值恰好等于上次"**在表示上不可区分** ⇒ 不单射。计数坐标把"是否发生了新观测"显式编码进来。
>
> 🔴 **代价必须写明**：加入计数坐标后，该嵌入**有意**把"仅在插入/删除保值观测上不同"的观测流**识别为同一个**。论文明确称之为"**a modelling choice**"（不是 bug），并在 Appendix A.5/B 讨论这个不变性与计算量之间的取舍。**引用时不可写成"完全无损"。**

**事件增量**（Eq. 10）：对 $k\in s_i$，令 $\text{prev}(i,k)$ 为 $k$ 上一次被观测的索引（若无则基点增广为 $x^{\text{prev}(i,k)}_k=0$）：

$$\delta_i=\sum_{k\in s_i}\left|x^k_i-x^k_{\text{prev}(i,k)}\right| e_k + e_{d_{\text{disc}}+k}$$

即每个被观测通道贡献**自上次观测以来的变化量**（注意是**绝对值**，这是增量而非增量向量）+ 其计数坐标上的一个单位增量。

**Lie 代数嵌入与区间组合**（Eq. 11–12）：默认把 $\delta_i$ 放在**一次项**，高阶分量置零：

$$\Delta_i=(0,\delta_i,0,\ldots,0)\in L_N(\mathbb{R}^{d_X}),\qquad E_i=\exp(\Delta_i)\in T^N(\mathbb{R}^{d_X})$$

对查询区间 $[\alpha,\beta)$（含事件 $\alpha\le t_{i_1}<\cdots<t_{i_m}<\beta$）：

$$G_{\alpha,\beta}(x,C)=B_C\otimes\Gamma_{\alpha,t_{i_1}}\otimes E_{i_1}\otimes\Gamma_{t_{i_1},t_{i_2}}\otimes E_{i_2}\otimes\cdots\otimes E_{i_m}\otimes\Gamma_{t_{i_m},\beta}$$

$$\Phi_{\alpha,\beta}(x,C)=\log G_{\alpha,\beta}(x,C)\in L_N(\mathbb{R}^{d_X})$$

其中 $\Gamma_{u,v}=S^N_{u,v}(\tilde C)$ 是连续观测通道（时间）在事件间隙上的签名因子，$B_C=\exp(\tilde C_0)$ 是基点增广。**事件因子与间隙因子可并行组合（associative scan）**，且**只依赖当前区间之前的信息 ⇒ 在线**。

🔴 **高阶信息的入口**：$\Delta_i$ 的高阶分量**不为零**时可直接塞入"预计算的带符号面积 / 局部迭代积分"。论文 §3.3 用 Brownian 路径的 Lévy area 演示了这条通路（见下文 §5.2）。

### 3.4 核心模块 3：Log-ODE 方法与 Log-SLiCE（§2.3）

**Log-ODE** 把区间上的 CDE 动力学近似为自治 ODE：给定截断 log-signature $\log S^N_{r_i,r_{i+1}}(X)$ 与向量场的 Lie 括号迭代，构造

$$\frac{d\bar h_s}{ds}=\bar f_\theta(\bar h_s)\,\log S^N_{r_i,r_{i+1}}(X)\,(r_{i+1}-r_i),\quad s\in[r_i,r_{i+1}],\quad \bar h_{r_i}=h_{r_i}$$

其中 $\bar f_\theta$ 是 Lie 代数同态扩展（Eq. 6）：$\bar f_\theta(\cdot)[a,b]=[\bar f_\theta(\cdot)a,\bar f_\theta(\cdot)b]$，要求 $f_\theta$ 为 $\mathrm{Lip}(\gamma)$，$\gamma\ge N$。

**Log-SLiCE** 是把 SLiCE 的结构化矩阵保持性质带进 Log-ODE 的变体：区间流有闭式

$$h_{r_{i+1}}\approx\exp\!\Big(\bar A_\theta\,\log S^N_{r_i,r_{i+1}}(X)\Big)h_{r_i}$$

其中 $\bar A_\theta$ 由 $A_\theta^i$ 通过同样的 Lie 括号构造得到（向量场括号换成**符号翻转的矩阵对易子**）。对**块对角** SLiCE，这个性质保留（块对角矩阵在乘积、对易子、矩阵指数下封闭），因此**可并行 scan**。

> **本仓必须记住的一条口径**：**Log-SLiCE 与 CfC 不是同一族**。CfC 的闭式解来自"标量可塑性时间常数"这一**特殊参数化**；Log-SLiCE 的闭式解来自"**线性**向量场 + 结构化矩阵"。两者虽然都"闭式"，但**表达力来源完全不同**。这与 11-19 的发现（`2607.05280` 证明"闭式 ≠ 连续时间的护城河"）同源，**再次说明"闭式"是工程属性而非家族标识**。

### 3.5 与 LTC (Hasani 2021) / CfC / NCP / GRU 的关系

| 维度 | 本篇 (Log-NCDE / Log-SLiCE) | LTC (Hasani 2021) | CfC | NCP (Lechner 2020) | GRU |
|---|---|---|---|---|---|
| 驱动 | 观测流的**区间 log-signature**（路径几何） | 连续时间的输入 $x(t)$（外生） | 连续时间的输入（外生） | 固定时间常数 + 稀疏接线 | 离散 token |
| 输入插值 | **不需要**（本文核心） | 需要（ODE 积分要求） | 需要（闭式解要求连续 $x$） | 不适用（离散） | 不适用 |
| 求解 | Log-ODE（区间级）/ 线性时闭式 | `odeint` 数值积分 | 闭式（逐元素） | 离散 | 离散 |
| 状态更新 | 路径级（区间流） | 逐步 ODE | 逐步闭式 | 逐步离散 | 逐步离散 |
| 顺序敏感项 | **是**（Lie 括号 / 二阶项） | 否（线性） | 否（逐元素闭式） | 否 | 否 |
| 表达力 | 路径空间通用（faithful 嵌入下） | 通用逼近器 | 通用逼近器 | 有限 | 有限 |

🔴 **对 LTC/CfC 最重要的一条否证性含义**：本篇证明**"不插值"在连续时间模型里是可行的、而且可证明保持通用性** ⇒ **"插值"不是连续时间的必要代价**。这对 LTC/CfC 是**利好**（它们的插值负担可以被证明为"建模选择"而非"机制必然"），但**同时**意味着：LTC/CfC 现有实验里插值方案带来的差异（Morrill et al. 2022 系统研究过）**是表示层的噪声源，而不是机制层的信号** ⇒ **本仓做 LTC/CfC 对照时，插值方案必须作为受控变量固定下来或单独报告**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$S^s,t(X)_I=\int_{s<u_1<\cdots<u_k<t} dX^{i_1}_{u_1}\cdots dX^{i_k}_{u_k}$$ | 路径区间上的签名：所有迭代积分 |
| Eq. 3 | $$S^s,t(X)=S^s,u(X)\otimes S^u,t(X)$$ | **Chen 恒等式**：签名在区间拼接下可分解 ⇒ 并行 scan 的代数基础 |
| Eq. 4 | $$\log S^s,t(X)=\sum_{m\ge1}\frac{(-1)^{m+1}}{m}\,(S^s,t(X)-1)^{\otimes m}$$ | log-signature：把签名映到自由 Lie 代数，去掉 shuffle 冗余 |
| Eq. 5 | $$dh_t=f_\theta(h_t)\,dX_t$$ | NCDE 主方程（$f_\theta:\mathbb{R}^{d_h}\to\mathbb{R}^{d_h\times d_X}$） |
| Eq. 6 | $$\bar f_\theta(\cdot)[a,b]=[\bar f_\theta(\cdot)a,\bar f_\theta(\cdot)b]=J\bar f_\theta(\cdot)b\,\bar f_\theta(\cdot)a-J\bar f_\theta(\cdot)a\,\bar f_\theta(\cdot)b$$ | 向量场的 Lie 代数同态扩展（需 $f_\theta\in\mathrm{Lip}(\gamma),\gamma\ge N$） |
| Eq. 7 | $$\frac{d\bar h_s}{ds}=\bar f_\theta(\bar h_s)\log S^N_{r_i,r_{i+1}}(X)(r_{i+1}-r_i)$$ | **Log-ODE 区间更新**：动力学只通过区间 log-signature 看到驱动 |
| Eq. 9 | $$d_X=2d_{\text{disc}}+d_{\text{cont}}$$ | **坐标设计**：值坐标 + 计数坐标 + 连续坐标（计数保证单射） |
| Eq. 10 | $$\delta_i=\sum_{k\in s_i}\left|x^k_i-x^k_{\text{prev}(i,k)}\right|e_k+e_{d_{\text{disc}}+k}$$ | **事件增量**：变化量 + 计数单位增量 |
| Eq. 11 | $$G_{\alpha,\beta}=B_C\otimes\Gamma_{\alpha,t_{i_1}}\otimes E_{i_1}\otimes\cdots\otimes E_{i_m}\otimes\Gamma_{t_{i_m},\beta}$$ | **区间签名组合**：事件因子 × 间隙因子 |
| Eq. 12 | $$\Phi_{\alpha,\beta}=\log G_{\alpha,\beta}$$ | 区间 log-signature（模型实际吃的东西） |
| Lemma 1 | $$\overline{(F\circ\varphi)|_{K_0}}=C(K_0;\mathbb{R})\ \text{若}\ \overline{F|_K}=C(K;\mathbb{R})$$ | **通用性转移**：faithful（连续+单射）嵌入把通用性从输入空间搬回数据空间 |
| Eq. 14 | $$dX_t=V_1X_t\circ dW^1_t+V_2X_t\circ dW^2_t$$ | Brownian Stratonovich 受控线性系统（§3.3，验证高阶信息增益） |
| Eq. 13 | $$x_i(t)=A_i\sin\big(\lvert z(t)+\delta_i\rvert\big),\quad z(t)=\omega t+\varphi$$ | 合成耦合系统（§3.2，共享潜相位） |

## 5. 关键成果与贡献

### 5.1 主结果：EigenWorms 上的精度 / 时间 / 显存三元对照

| Model family | Log-ODE | Test Acc. (%) | Time / step (s) | GPU Mem. (MB) |
|---|:---:|---:|---:|---:|
| NCDE | ✗ | 75.0 ± 3.9 † | 26.02 | 3484 |
| NCDE | ✓ | 85.6 ± 5.1 † | 2.263 | 3494 |
| LNCDE | ✗ | 87.2 ± 5.2 | 0.100 | 13730 |
| LNCDE | ✓ | 88.3 ± 3.7 ‡ | 0.018 | 3492 |
| BD-SLiCE | ✗ | 81.1 ± 6.7 | 0.024 | 3494 |
| BD-SLiCE | ✓ | **86.1 ± 3.5** ‡ | **0.009** | 3494 |

†来自 Walker et al. 2024；‡来自 Walker et al. 2025。EigenWorms 每条序列 **17,984** 个观测。

**关键读数**：
- Log-ODE + SLiCE 相对标准 NCDE：**精度 +11.1 个百分点**，每步时间**降约 3000×**（10,000 步训练从 **3 天 → 1.5 分钟**）。
- 族内加速比：NCDE ≈ **11.5×**，LNCDE ≈ **5.6×**，BD-SLiCE ≈ **2.7×**。
- ⚠️ **显存口径要读准**：LNCDE 无 Log-ODE 时 13730 MB，加 Log-ODE 后降到 3492 MB（**降 3.9×**）；BD-SLiCE 两种情况都是 3494 MB。**"Log-ODE 省显存"只对 LNCDE 成立，不是普遍结论。**

### 5.2 主结果：UEA 六数据集 + input dropping（Table 2 / Table 3）

**Table 2（宏平均准确率 % / 平均数据集排名，越低越好）**

| Model | No drop Acc. | 30% drop | 70% drop | 95% drop | Av. Rank (no drop) |
|---|---:|---:|---:|---:|---:|
| S5 | 61.8 | 63.4 | 60.1 | 56.7 | 4.67 |
| LRU | 61.7 | 60.9 | 61.0 | 60.8 | 4.50 |
| S6 | 62.0 | 58.4 | 61.4 | 58.9 | 5.00 |
| Mamba | 58.6 | 59.5 | 60.6 | 57.6 | 6.67 |
| NCDE | 60.2 | 59.6 | 58.9 | 58.7 | 5.33 |
| **Log-NCDE** | **64.3** | **64.2** | **62.7** | 60.9 | **3.00** |
| D-SLiCE | 61.7 | 61.7 | 62.2 | 62.2 | 4.50 |
| **BD-SLiCE** | 64.0 | 63.7 | **62.5** | **62.5** | **2.33** |

🔴 **对连续时间叙事最重要的一行**：**95% drop 下 S5 从 61.8 掉到 56.7（−5.1pp），而 Log-NCDE 64.3→60.9（−3.4pp）、BD-SLiCE 64.0→62.5（−1.5pp）** ⇒ **在极端稀疏下，连续时间路径模型的退化明显小于结构化 SSM**。这是**本仓迄今最接近"连续时间机制本身有增益"的一条外部证据**，但**注意它测的是"稀疏 + 连续时间路径表示"的组合，不是"液态闭式解"**。

**Table 3 —— 本篇最有价值的消融（固定架构，只换输入表示）**

| Input representation | 30% drop | 70% drop | 95% drop | Avg. |
|---|---:|---:|---:|---:|
| **Event-increment embedding（本文）** | **3.00 (1)** | **3.25 (1)** | **2.83 (= 1)** | **3.03** |
| Cubic Hermite interpolation | 3.58 (3) | 3.00 (1) | 3.58 (2) | 3.39 |
| Gaussian-process imputation | 4.33 (4) | 4.50 (5) | 3.58 (2) | 4.14 |
| Global-mean imputation | 5.17 (5) | 3.75 (2) | 3.33 (2) | 4.08 |

**这是本篇对本仓最有直接价值的一张表** —— 架构、超参、数据划分、随机种子**全部固定**，**只换输入表示**，event-increment 嵌入是**唯一在每个 drop 水平都排第一（或并列第一）**的表示，平均排名 3.03 优于 Hermite 的 3.39。⚠️ 但**差距不大**（平均排名差 0.36），**不能写成"大幅领先"**；且插值方案**只换了三种**，未覆盖全部方案族。

### 5.3 机制分析

- **§3.2 跨采样 regime 泛化矩阵（Figure 4，5 seeds）**：在训练于 sync-regular 的模型上，跨到 async-sparse 的 MSE 是 **7.335×10⁻²**，而训练于 async-sparse 的模型在 sync-regular 上只有 **2.985×10⁻²** ⇒ **模型会在"只见过规则采样"时学捷径**。论文的解释是：这与 faithful 嵌入的观点一致 —— sync-regular 与 sync-irregular **包含的信息量相近**，所以两者表现相近，而 async-sparse 最难因此最鲁棒。
- **§3.3 高阶信息增益（Figure 5）**：Brownian 受控系统上，用 level-2 log-signature（带 Lévy area）比 level-1 的测试 MSE 低一个数量级（$10^{-6}$ 量级 vs 更高），且**区间数越少差距越大** ⇒ 因为区间少时 level-1 的截断误差更大。论文同时指出：**区间数增加 ⇒ level-1 的截断误差下降 ⇒ level-1 的 MSE 也在下降** ⇒ **"用多长的区间"和"用多高的阶"是两条可分离的轴**。
- **§3.3 高阶信息的入口**：论文演示了在 $\Delta_i$ 里直接放入预计算的 Lévy area（level-2），这说明**这套表示不只支持"一阶观测"，也能容纳预计算的高阶路径特征**。

### 5.4 贡献清单

1. **【理论】** 形式化 **faithful embedding**（连续 + 单射）并证明 **Lemma 1（通用性转移）**：紧集上的通用逼近性可从模型输入空间搬回数据空间 —— **"不插值"在原则上不损失表达力**。
2. **【构造】** 给出**面向 Log-NCDE 的忠实控制路径嵌入**：观测 → Lie 代数增量（值增量 + 计数增量）→ 跨任意查询区间组合成区间 log-signature，**全程不插值、在线、可并行 scan**。
3. **【实证】** ① Log-ODE + SLiCE 在 EigenWorms 上 **+11.1pp 精度 / ~3000× 加速**；② 跨采样 regime 的泛化分析揭示"规则采样训练的模型会学捷径"；③ **固定架构只换输入表示**的消融显示 event-increment 嵌入在全部 drop 水平排名第一；④ 95% drop 下 BD-SLiCE 只掉 1.5pp，而 S5 掉 5.1pp。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **§5.1 附录 B**：通用性陈述的**范围**（只说紧集上的通用性，**与样本效率、优化保证无关**）以及它与二者的**区别**。
- **§2.4 + Appendix A.5**：计数坐标引入的**不变性–计算量取舍** —— 插入/删除保值观测会被视为同一流；这是**建模选择**。
- **§6 结论**：表示"避免了插值、支持在线预测、解耦输入观测时刻与输出查询时刻、区间内与区间间并行"。

⚠️ **本仓需要主动补上的、作者未强调的三条隐含局限**：

1. **未与 CfC / LTC / NCP 做直接对照** —— 全文的实验对照轴是 NCDE / Log-NCDE / LNCDE / SLiCE / S5 / LRU / S6 / Mamba，**没有一行提到 CfC 或 LTC** ⇒ **本篇不能被引用为"液态（closed-form）路线更优"的证据**，它证的是"路径几何表示 + 忠实嵌入"这条线的价值。
2. **与 11-19 研读的 `2607.05280` 有明显重叠但方向相反** —— `2607.05280` 的路线是"**线性化向量场以消除 ODE 求解器**"（相对 NCDE 约 1500× 加速）；本篇的路线是"**用 log-signature 把路径几何接回非线性向量场**"（仍需解 ODE，但区间级 ⇒ 步数少）。**两者不是竞争关系，而是"要不要保留非线性表达力"的取舍**：本篇保留了非线性（代价是仍要解 ODE），`2607.05280` 放弃了非线性（代价是表达力受限）。**引用时必须分开陈述，不可混为"都是 log-signature 方案"。**
3. **"计数坐标"使表示对"重复保值观测"不敏感** —— 若某通道有真实的物理"保持不变但被测量"的事件（如"已到位"标志位连续为真），该事件会被表示为"计数+1、值不变" ⇒ **不丢信息**；但若下游关心"被观测的频次"而非"事件序列"，则**冗余观测与真空会被合并**。这一条对本仓做**事件序列**建模（11-13 的 CT-GRU 轴）有直接参照价值。

### 6.2 隐含局限（本仓视角）

- **实验规模**：UEA 六数据集、batch size 1、H100；**EigenWorms 的 3000× 加速依赖极长序列**（17,984 点），在**短序列**上收益会显著缩水。**本仓 bench 多为中短序列 ⇒ 不能直接套用该加速比。**
- **"faithful ⇒ 通用性"是存在性结论**：它保证"某个最优模型存在"，**不保证 SGD 能找到、不保证样本效率**。Table 3 的经验优势（约 0.36 个平均排名）**远小于**这个理论保证可能让人以为的量级 ⇒ **不要把 Lemma 1 当作 Table 3 差距的解释**。
- **复现成本高**：需要实现 log-signature（可用 `signatory`，本篇用 PyTorch 原生重实现）、Lie 代数基、associative scan、SLiCE 结构化矩阵 —— **本仓若要复现，起点应是 `2602.02157`（本仓已研读，MVC-CDE）而非从零实现**。

### 6.3 未来方向

- **坐标侧 primitive 投影**（与 `2606.05272` §5.2 同向）可进一步省内存。
- **可插值的 faithful 嵌入族**：本文只给了一种构造，忠实性证明允许**任何**连续单射嵌入 ⇒ 存在**一族**解，包括"已插值但对插值不敏感"的设计。
- **Appendix B 提到**：通用性陈述与**样本效率 / 优化**的关系尚待厘清。

## 7. 对本仓的意义

**① 可作 `lnn/core/variants.py` 的什么变体？**
本篇**不提供新的 cell 变体**，它提供的是**输入表示层**。对本仓最直接的可落地物是 **`EventIncrementEmbedding`**（一个输入侧工具类，不是 cell）：
- 输入：不规则/异步的观测流 $\{(t_i, s_i, x_i)\}$；
- 输出：可直接喂给任何**区间级**模型的 $\Phi_{\alpha,\beta}$（含计数坐标）。
- **但前提是模型必须是"区间级"的** —— 本仓 `cfc.py` / `ltc.py` 是**逐步**更新的（per-step），**没有区间这一层** ⇒ **不能直接套用**。要套用需要先引入"区间流"抽象（把 $\Delta t$ 内的多步聚成一次区间更新），这是一个**结构性改造**，不是加参数。
⇒ **诚实结论**：本篇对本仓的直接可落地价值**低于** 11-19 的 `2607.05280`（矩阵指数可直接换 `cfc.py` 的逐元素指数），**主要价值在"输入表示是独立变量"这个方法论**。

**② 可嫁接到哪个 bench 脚本？**
- **`bench_cfc_temporal_dropout.py` / `bench_cfc_input_dropout.py`（本仓已有 input dropout 类 bench）** —— 这是**最直接的嫁接点**：本篇 Table 2/3 就是在 input dropping 下比较输入表示 ⇒ 本仓可以直接在**自己的 bench 上做同样的"固定架构、只换输入表示"消融**，并与本篇的 UEA 数字对照。
- **`bench_cfc_temporal_smoothness.py`** —— 本篇的核心命题是"**路径几何不该被插值伪造**" ⇒ 该 bench 的"平滑度"度量可与本篇的"忠实性"度量并置。
- **新建 `bench_input_representation_ablation.py`** —— 四种输入表示（event-increment / cubic Hermite / GP imputation / global-mean）× 本仓 LtC/CfC/GRU ⇒ **这是本仓至今完全没有的一类实验**，且直接服务于"液态机制 vs 表示层"这个本仓核心问题。

**③ 提供哪个新 ablation axis？**
🔴 **轴 A（最便宜、最高价值）：固定架构、只换输入表示。** 本仓的 `cfc.py`（闭式、逐元素）与 `ltc.py`（odeint、逐元素）**都隐含了"输入被插值/连续化"的假设**（LTC 显式、**CfC 隐式**——CfC 的闭式解要求 $x(t)$ 连续可查）。本仓至今**从未测过"把输入表示换掉会怎样"** ⇒ 这条轴一旦补上，本仓对"液态机制"的每一个既有结论都需要重新审视。

🔴 **轴 B：路径几何的可辨识性 vs 表征敏感度（与 11-19 `2611.03887` 呼应）。** `2611.03887` 证明稀疏观测下 CDE 预测有 $O(\delta^2)$ 的**不可辨识地板**；本篇给了一个**逃出该地板的方法**（event-increment 嵌入不伪造几何）⇒ 合起来是一个可执行的实验命题：**在稀疏观测下，本仓 LTC/CfC 的输出对"同一份观测的不同插值补全"的敏感度是多少？** 若敏感度高，本仓的连续时间结论里有相当一部分是**补全假设的产物**；若敏感度低，则 11-13 CT-GRU 的负结果得到加强。

🔴 **轴 C：level-1 vs level-2 的可分离性。** 本篇 §3.3 显示：**区间数**与**路径阶数**是两条独立轴 ⇒ 本仓应能构造一个 2×2 实验（{短区间, 长区间} × {一阶, 二阶}）⇒ 判定"本仓的 $\tau$ 变体是否其实在做二阶效应"。

🔴 **轴 D：计数字段的敏感性。** 本篇的计数坐标是一个**免费的、可解释的诊断量**：观测事件的频率/稀疏度 ⇒ 本仓可在 bench 中**直接把计数坐标当作输入稀疏度的代理指标**来报告（比"drop 率"更细）。

**④ 对本仓既有结论的直接修正要求**：
- 本仓索引中任何**"LTC/CfC 在不规则采样上更好"**的表述，**必须补上"输入插值方案"这一受控变量** —— 本篇证明插值层的差异可以**独立地**贡献可观收益（Table 3：Hermite 3.39 vs event-increment 3.03；且插补更差），**因此旧结论里"连续时间机制"的归因可能是错的**。
- 本仓至今**所有** LTC/CfC 与 GRU 的对比，**隐含地共享同一个插值/离散化** ⇒ **这不构成偏袒任何一方**（对双方一样），但**意味着"连续时间机制"的效应量被系统性地混在了"表示层差异"里** ⇒ **必须补做轴 A 才能把两者分开**。

**Verdict**：
- **"faithful（连续+单射）嵌入把通用性从输入空间转移到数据空间" → TARGET-POSITIVE（理论层面）**：Lemma 1 表述清晰、假设（Hausdorff + 紧集）明确、且给出了反向障碍；**但须标注**：这是**存在性**结论，**不涉及样本效率与优化**，不可直接读作"更好训"。
- **"不插值的输入表示在连续时间模型上是可行且更优的" → TARGET-POSITIVE-WITH-NUANCE**：Table 3 在**固定架构下**证明 event-increment 嵌入在全部 drop 水平排第一；但**平均排名只领先 0.36**，**插值方案只比较了三种**，**且对照轴全是 NCDE 家族而完全不含 CfC / LTC / NCP** ⇒ **不得引用为"液态闭式路线更优"的证据**。
- **"Log-ODE + SLiCE 是连续时间的高效路径" → TARGET-POSITIVE-WITH-NUANCE**：EigenWorms 上 **+11.1pp / ~3000×** 加速、95% drop 下退化仅 1.5pp（对比 S5 的 5.1pp）—— 这是**本仓迄今最接近"连续时间机制本身有增益"的外部证据**；**但须标注**：依赖**极长序列**（17,984 点）、**依赖"路径级表示"这一前提**（本仓的逐步 CfC/LTC 不满足）、**且不与 CfC/LTC 对照**。
- **"连续时间模型在极端稀疏下优于结构化 SSM" → TARGET-DEPENDENT-WITH-NUANCE**：Table 2 的 drop 趋势支持该方向；**但须标注**这是 **Log-NCDE/Log-SLiCE 特有的**（区间级 + 路径几何），不能推广到"任何连续时间模型"，更不能推广到"液态闭式细胞"。
- **对液态路线（LtC/CfC）本身 → 不适用（族外，但有强方法论含义）**：本篇**不是**液态家族的证据，但它**给本仓一条必须补做的消融**（轴 A）—— 在补上"只换输入表示"的对照之前，本仓任何"液态机制带来收益"的归因都**无法排除表示层贡献**。
