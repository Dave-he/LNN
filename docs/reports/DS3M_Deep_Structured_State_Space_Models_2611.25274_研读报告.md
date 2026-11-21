---
title: "DS3M: Deep structured state space models for nonlinear system identification on modern hardware"
date: 2026-11
tags: [LNN, Continuous-Time, System-Identification, SUBNET, State-Space-Model, Port-Hamiltonian, Triton, GPU-Kernel, Mamba-Comparison]
---

# 研读报告：DS3M — 连续时间系统辨识的瓶颈**不在表达力而在 GPU 调度**：把"小而多的串行递推"塞进一个融合 Triton kernel

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**（`2611.25274` 在 174 篇报告中零命中）。
> 🔴 **本篇对本仓的意义是"给液态路线的一个长期假设做了一次硬体检"** —— 它把 CT-SUBNET 与 ResNet50/ImageNet 放在同一张图上比较，发现**连续时间模型在 GPU 上的劣势来自"串行模型求值次数 × 每次的工作量太小"，而不是来自连续时间本身**；并用 kernel fusion 把它**加速 28–160×**，同时**精度持平甚至更好**（WH 0.24 mV 打平 SOTA、EMPS 1.8 mm 显著优于全部基线）。**这直接冲击本仓"液态模型慢"这一常见归因。**

## 1. 元数据

- **论文标题**：DS3M: Deep structured state space models for nonlinear system identification on modern hardware
- **作者**：Marco Roschkowski, Hannes Gernandt
- **机构**：University of Wuppertal（Wuppertal, Germany）；DFG CRC 1701 资助
- **发表时间**：2026-11-18 (v1)
- **发表渠道**：**preprint，已投稿 IFAC**（作者自陈"submitted to IFAC for possible publication"）⇒ **未评审**
- **来源**：arXiv:2611.25274v1 [cs.LG]
- **代码**：论文未给出仓库链接；仅说明 Triton kernel 实现（作者保留）
- **本地 PDF**：[papers/2611.25274/2611.25274.pdf](../../papers/2611.25274/2611.25274.pdf) ／ 文本 [papers/2611.25274/2611.25274.txt](../../papers/2611.25274/2611.25274.txt)
- **页数**：10 页
- **关联概念**：连续时间状态空间 / **CT-SUBNET**（Beintema 2023, ICLR）/ SUBNET（Beintema 2021, L4DC）/ 子空间编码器 / **port-Hamiltonian 神经状态方程** / **Triton kernel fusion** / 数值积分（RK4 / Heun）/ 显式 Euler / Mamba / S4 / 状态空间模型 / Wiener–Hammerstein / Silverbox / EMPS / 平行 WH

## 2. 核心问题

**系统辨识（system identification）**：给定测得的输入 $u(t)$ 与输出 $y(t)$，构造一个模型

$$\dot x(t) = f(x(t), u(t)), \qquad y(t) = g(x(t), u(t)), \tag{1}$$

其中状态 $x(t)$ 通常不可观测。SUBNET（Beintema et al., 2021）把这件事与一个"**子空间编码器**"结合：编码器 $\mathcal{E}_\phi$ 从一小段测量历史（长度 $W$ 的输入 + 输出窗口）估计出模拟轨迹的初始状态 $\hat x_0$；然后用记录的输入把状态推演下去，把模拟输出与测量输出比对。编码器、状态方程、输出映射**联合训练**。这个构造的价值在于**不需要测量状态，也不把每段子序列的初始状态当成独立优化变量**。

**连续时间版本（CT-SUBNET, Beintema et al., ICLR 2023）**把状态方程参数化成一个神经网络，$\dot x = \tau^{-1} f_\theta(x, u)$，再由数值积分器产生采样点上的状态转移：

$$x_{k+1} = I(f_\theta, x_k, u_k), \qquad \dot x = \tau^{-1} f_\theta(x, u). \tag{2}$$

**论文指出的结构性缺陷**：这条路线**训练极慢**。原因不是模型表达力不足，而是**硬件不匹配** —— 连续时间模型在一次训练步里需要**成百上千次串行的模型求值**（$sT$ 次，$s$ = 积分级数，$T$ = 窗口长度），而**每次求值的工作量都很小**（状态方程小）。论文把 CT-SUBNET 在 Wiener–Hammerstein 上的训练负载与 ResNet50 在 ImageNet 上的训练负载放在**同一张图**上比较（Figure 1），结论是：

| 训练参数 | ResNet-50 / ImageNet | CT-SUBNET / WH |
|---|---|---|
| 串行模型求值次数 | $1\times$ | $400\times$ |
| 每次串行求值的工作量 | $1.6\times10^{-2}$ | $1.6\times10^{-7}$ |
| 可用的 batch 并行度 | $2.8\times10^{-5}$ | $6.4\times10^{-5}$ |
| 训练算力 | $8.3\times10^{-7}$ | $2.8\times10^{-5}$ |

⇒ **两次方根的规模都小了两个数量级**。论文的判断是：**ImageNet 规模的训练能从现代加速器获益更多，而 CT-SUBNET 不能** —— 原因是"rollout 期间的串行模型求值"。

**因此问题被重述为**：*能不能在保留"连续时间 + 非线性状态方程 + 子空间编码器"这三点的前提下，把这些小的、依赖性强的串行计算组织成 GPU 友好的形状？* 论文给出的答案是 **"深度结构化状态空间模型"（DS3M）**：**用 kernel fusion 消掉调度开销 + 用并行/串联的多个小动力学块换取并行度**。

## 3. 方法论与核心思路

### 3.1 总体架构

DS3M = **CT-SUBNET 的并行/串联重构 + Triton 融合 kernel**。整体流水线（Figure 2A）：

```
过去测量 (窗口长 W):  u_{-W+1..0}, y_{-W+1..0}
        │
   子空间编码器 E_φ : R^{W(M+O)} → R^{N×nH}     ← 一次性给出所有 N·H 个分量的初始状态
        │
   嵌入 P_ψ : R^M → R^D ,   (z_0)_k = P_ψ(u_k)
        │
   ┌──────────── N 个 DS3M block 串联 ────────────┐
   │  z_i = B_{θ_i}( z_{i-1}, x_{0,i} ),  i=1..N     │
   │  每个 block 内 H 个 CT-SUBNET 并行 → 状态混合器 → 特征映射  │
   └───────────────────────────────────────────────┘
        │
   读出 Y_φ : R^D → R^O ,   ŷ_k = Y_φ((z_N)_k)
```

### 3.2 核心模块 1：DS3M block（H 个并行 CT-SUBNET + 状态混合 + 残差）

令 $z=(z_0,\dots,z_{T-1})$，$z_k\in\mathbb{R}^D$。一个 block 含 $H$ 个动力学分量，第 $j$ 个有状态 $x^{(j)}_k\in\mathbb{R}^n$、内部输入 $u^{(j)}_k\in\mathbb{R}^m$。其网络维度为

$$h^{(j)}_\theta:\mathbb{R}^D\to\mathbb{R}^m,\quad f^{(j)}_\theta:\mathbb{R}^{n+m}\to\mathbb{R}^n,\quad k_\theta:\mathbb{R}^{nH}\to\mathbb{R}^D,\quad g_\theta:\mathbb{R}^D\to\mathbb{R}^D .$$

block 的三个操作（Eq. 3）：

$$u^{(j)}_k = h^{(j)}_\theta(z_k),$$
$$x^{(j)}_{k+1} = I\big(f^{(j)}_\theta,\ x^{(j)}_k,\ u^{(j)}_k\big),$$
$$v_k = k_\theta\big(x^{(1)}_k,\dots,x^{(H)}_k\big) + z_k, \qquad z^{+}_k = g_\theta(v_k) + v_k .$$

🔴 **关键点**：最后两式是**带 skip connection 的**（论文明确引用 ResNet）⇒ **状态混合器 + 特征映射是恒等捷径**，block 的输出不会因为动力学学坏而崩塌。

**并行结构的来源**：$H$ 个 CT-SUBNET **在同一个 Triton kernel 内并行**（论文类比 Transformer 的 multi-head），$N$ 个 block **串联**（类比 deep stacking）。隐状态总维度为 $NnH$。

### 3.3 核心模块 2：分量动力学 = 残差 MLP（Eq. Figure 3）

把状态与内部输入拼成 $v=(x\ u)\in\mathbb{R}^{n+m}$，分量向量场用一个**残差 MLP** 参数化：

$$a = \sigma(L_1 v), \qquad f^{(j)}_\theta(x,u) = L_3\big(a + \sigma(L_2 a)\big).$$

$L_1\in\mathbb{R}^{(n+m)\times h}$、$L_2\in\mathbb{R}^{h\times h}$、$L_3\in\mathbb{R}^{h\times n}$，$\sigma$ 逐元素。**hidden 层的 skip connection（$a + \sigma(L_2a)$）是参数量最大的一层，论文说它"通常能稳定训练"**。

### 3.4 核心模块 3：Triton 融合 kernel —— 加速的真正来源

常规 PyTorch 实现里，矩阵乘、激活、积分级数会在**每个采样点**产生许多次小 kernel launch；当状态方程很小时，**launch 开销与显存搬运开销主导**。DS3M 的融合 kernel 把"**神经状态方程求值 + 积分步**"合并到一个 kernel 里：时间循环在**单个 program instance 内部**执行，**独立的训练窗口与 $H$ 个分量提供并行工作**，中间量留在**寄存器/片上快存**里。

🔴 **代价（论文自陈）**：融合迫使**权重常驻寄存器**，因此**参数量受 SRAM 容量限制** ⇒ "limits future applications to larger systems"。论文的缓解方式是**把模型拆小**（$H$ 个小分量并行、$N$ 个小 block 串联）而不是把模型做大。论文另提了 **CUDA + persistent RNN** 路线（Diamos et al., 2016）作为"更大的 kernel、program instance 之间可通信"的替代设计。

### 3.5 核心模块 4：port-Hamiltonian 变体（物理结构先验）

每个分量可以替换成 **port-Hamiltonian 结构**的状态方程：

$$f^{(j)}_\theta(x,u) = \big(J^{(j)}_\theta(x) - R^{(j)}_\theta(x)\big)\nabla H^{(j)}_\theta(x) + B^{(j)}_\theta(x)\,u,$$

其中 $J^{(j)}_\theta(x)$ **反对称**、$R^{(j)}_\theta(x)$ **半正定**、$H^{(j)}_\theta:\mathbb{R}^n\to[0,\infty)$。论文为紧凑 kernel 取

$$H^{(j)}_\theta(x) = \tfrac12\|x\|^2, \qquad R^{(j)}_\theta(x) = \mathrm{diag}\big(r^{(j)}_{\theta,1}(x),\dots,r^{(j)}_{\theta,n}(x)\big),$$

$\mathrm{softplus}$ 保证对角非负。由此得到连续时间分量的能量方程：

$$\tau\frac{d}{dt}H(x) = -\nabla H(x)^\top R(x)\nabla H(x) + u^\top B(x)^\top\nabla H(x).$$

⇒ **零输入时能量 $t\mapsto H(x(t))$ 单调不增**（耗散性被结构性地保证，而不是学出来的）。

🔴 **论文自陈的两处简化**：(a) 取 $H=\frac12\|x\|^2$ 使 forward 实现**不需要对神经网络反传**，简化 Triton kernel；(b) $R$ 取**对角**而非 Cherifi et al. (2026) 的 **Cholesky 分解** ⇒ "进一步稳定训练"的折中。

### 3.6 与 LTC / CfC / Mamba 的关系（明确写清）

| | 时间常数机制 | 状态依赖性 | 积分方式 | 与本文差异 |
|---|---|---|---|---|
| **DS3M / CT-SUBNET** | **全局标量 $\tau$**（对 $\dot x$ 整体缩放，Eq. 2） | $f_\theta$ 对状态非线性 | 数值积分（RK4 / Heun），**每步 $s$ 次向量场求值** | 本文基座 |
| **LTC**（Hasani 2021） | **输入依赖的可变 $\tau$**（per-channel） | 非线性 | 数值积分 | DS3M 的 $\tau$ 是**全局标量**，**不是** LTC 那种输入依赖时间常数 |
| **CfC** | 闭式解 | 非线性 | **无求解器**（闭式） | DS3M **没有**闭式路径 ⇒ 成本结构与 CfC **正交** |
| **Mamba / S4** | 结构性线性 | $x_{k+1}=A(u_k)x_k+B(u_k)$，**线性** | 闭式/并行 scan | 论文明确对比：Mamba 用"仿射线性变换在复合下的封闭性"提效；**DS3M 保留对状态的非线性依赖，改用 fusion 降低串行积分成本** |
| **NCP** | 固定时间常数 + 稀疏离散接线 | — | 离散 | 与本文无关 |

🔴 **一处关键口径**：论文称 "DS3M 的动力学**显著更有表达力**，因此能用**显著更少的状态**识别系统"（相对线性 SSM）—— 这是**对 Mamba/S4 的对比**，**不是**对 LTC/CfC 的对比。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $\dot x = f(x,u),\ y = g(x,u)$ | 系统辨识的标准状态空间形式 |
| Eq. 2 | $x_{k+1}=I(f_\theta,x_k,u_k),\ \dot x=\tau^{-1}f_\theta(x,u)$ | CT-SUBNET 的连续时间 + 数值积分；显式 Euler 下 $x_{k+1}=x_k+(\Delta t/\tau)f_\theta(x_k,u_k)$，**$\Delta t/\tau$ 同时决定状态更新幅度与训练数值条件** |
| Eq. 3 | $u^{(j)}_k=h^{(j)}_\theta(z_k)$；$x^{(j)}_{k+1}=I(f^{(j)}_\theta,x^{(j)}_k,u^{(j)}_k)$；$v_k=k_\theta(\cdot)+z_k$；$z^+_k=g_\theta(v_k)+v_k$ | DS3M block 的四个操作；后两式带 skip connection |
| Fig. 3 | $a=\sigma(L_1v),\ f^{(j)}_\theta(x,u)=L_3(a+\sigma(L_2a))$ | 分量向量场的残差 MLP 参数化 |
| pH | $f^{(j)}_\theta=(J^{(j)}_\theta-R^{(j)}_\theta)\nabla H^{(j)}_\theta + B^{(j)}_\theta u$ | port-Hamiltonian 分量：反对称 $J$、半正定 $R$ |
| 能量 | $\tau\frac{d}{dt}H(x)=-\nabla H^\top R(x)\nabla H + u^\top B(x)^\top\nabla H$ | 零输入 ⇒ 能量单调不增（耗散性由结构保证） |
| 损失 | $\mathcal{L}(\hat y,y)=\frac{1}{TO}\sum_{k=0}^{T}\|\hat y_k-y_k\|^2$ | 训练目标：子序列上的输出 MSE |

## 5. 关键成果与贡献

### 5.1 主结果一：单次 CT-SUBNET 前向+反向的加速（Table 1，RTX 3090，PyTorch 2.11.0+cu126 / Triton 3.6.0，fp32）

设定：$n=6$、$m=3$、$h=32$、batch 128、SiLU、**RK4**；时间含 forward + backward。

| 集成级数 | 基线 | Base [ms] | Triton [ms] | 加速比 |
|---|---|---|---|---|
| 100 | CPU eager | 154.651 | 1.850 | **83.61×** |
| 100 | CPU compiled | 52.854 | 1.850 | **28.58×** |
| 100 | CUDA eager | 280.963 | 1.850 | **151.91×** |
| 100 | CUDA compiled | 79.145 | 1.850 | **42.79×** |
| 200 | CPU eager | 370.954 | 3.779 | **98.16×** |
| 200 | CPU compiled | 117.881 | 3.779 | **31.19×** |
| 200 | CUDA eager | 603.999 | 3.779 | **159.82×** |
| 200 | CUDA compiled | 153.942 | 3.779 | **40.73×** |

🔴 **注意读法**：Triton 那一列**与集成级数无关**（100 步和 200 步都是 1.850 / 3.779 ms）—— 因为时间循环在 program instance 内部执行，**级数只影响寄存器内的算术量，不影响 launch 开销**。这正是论文论点最干净的一处证据。

### 5.2 主结果二：辨识基准（Tables 3–6）

| 基准 | 方法 | 训练时间 | RMSE |
|---|---|---|---|
| **Wiener–Hammerstein** | SUBNET | > 200 h | 0.24 mV |
| | PNLSS | n.a. | 0.42 mV |
| | **DS3M** | **1.5 h** | **0.24 mV** |
| | DS3M (port-Hamiltonian) | 2.4 h | 0.28 mV |
| **EMPS** | CT-SUBNET | 1–24 h | 4.6 mm |
| | dynoNet (original) | n.a. | 2.6 mm |
| | PNARX | 1–60 min | 2.6 mm |
| | **DS3M** | **0.5 h** | **1.8 mm** |
| **平行 WH** | NL-LFR | 1–60 min | [0.20, 0.40, 0.41, 0.47, 0.83] |
| | **DS3M** | **2.7 h** | **[0.20, 0.38, 0.26, 0.30, 0.49]** |
| **Silverbox** | NL-LFR | 1–60 s | [0.29, 0.33, 0.26] |
| | SUBNET | 1–4 d | [0.36, 1.4, 0.32] |
| | **DS3M** | **0.5 h** | **[0.23, 0.74, 0.23]** |

🔴 **口径必须带**：(a) **训练时间跨硬件不可比** —— 论文明确写了 "training times in the table which are numbers reported in the literature on **non-identical hardware**"；只有 DS3M 的时间是本文自测（RTX 3090）。(b) **WH 上是"打平"不是"超越"** —— 0.24 mV 与 SUBNET 完全相同，论文的措辞是 "reaches the test error of 0.24 mV with significantly less training time"。(c) **EMPS 上的 1.8 mm 是真超越**（全部基线最好 2.6 mm），且论文归因于"能做 $T=2500$ 的更长训练序列"—— 即**加速换来的收益**。(d) **平行 WH / Silverbox 上第 2 条轨迹更差**（Silverbox 0.74 vs NL-LFR 0.33；平行 WH 0.38 vs 0.40 但 0.26 vs 0.41），论文自陈"这是 SUBNET 类方法的局限，本文没有解决"。

### 5.3 机制分析

- **Figure 1（负载对比）**：CT-SUBNET 与 ResNet50/ImageNet 的四个轴对比，是全文的**动机核心**，也是最容易被引用的一张图。
- **加速的来源被拆开**：不是"更少的算力"，而是"**更少的 launch 与显存搬运**"。论文用"state updates remain sequential"明确承认**依赖性没有被消除，只被隐藏进 kernel**。
- **Mamba 对比**：论文明确指出 Mamba 用"仿射线性变换在复合下的封闭性"提效，而 **DS3M 保留对状态的非线性依赖** ⇒ 二者是"表达力 vs 调度"的不同取舍，**不是同一问题的两个解**。

### 5.4 贡献清单

1. **一条量化诊断**：连续时间模型在 GPU 上的劣势来自"串行求值次数 × 单次工作量太小"的**乘积**，并给出与 ImageNet 训练的同图对比（400× vs 1×，$1.6\times10^{-7}$ vs $1.6\times10^{-2}$）。
2. **一个可复用的工程结论**：**kernel fusion 把"小而多的串行递推"变成 GPU 友好形状，实测 28–160×**，且融合后**时间与积分级数几乎无关**。
3. **一个结构调整范式**：用 **$H$ 个小分量并行 + $N$ 个小 block 串联**来换取并行度，**用结构（skip connection）而不是用参数量来保表达力**。
4. **一个物理先验变体**：port-Hamiltonian 分量把**耗散性写进结构**（能量单调不增），而非靠 loss 逼出来。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **权重常驻寄存器 ⇒ 参数量受 SRAM 限制** ⇒ "limits future applications to identifying larger systems"；论文称"原则上可扩到任意容量"，但**未给 scaling law 实验**（明确引用了 Roschkowski et al. 2026 的 scaling law 工作但**没有在本文复现**）。
- **架构必须预先已知**（"the neural network architecture is known in advance"）—— 融合 kernel 无法处理动态结构。
- **port-Hamiltonian 变体的两处简化**（$H=\frac12\|x\|^2$、$R$ 对角而非 Cholesky）**牺牲了一部分表达力** ⇒ WH 上 0.28 vs 0.24 mV。
- **第二条轨迹上仍不如 NL-LFR** ⇒ "our method does not seem to resolve this limitation of SUBNET-type methods"。
- **未与 CfC / LTC 家族做任何对比** —— 全文没有出现 CfC / closed-form / LTC。

### 6.2 隐含局限（本仓视角）

- 🔴 **"连续时间"的代价论断只针对"数值积分路线"** —— 论文的成本模型 $sT$ 次向量场求值**只对 ODE 求解器成立**。**CfC 的闭式路径成本结构完全不同**（一次矩阵/逐元素指数，无串行级数）⇒ **本仓不可引用本文来支持"液态模型慢是因为连续时间"**；恰恰相反，本文**支持"液态模型慢不是因为连续时间，而是因为实现没有针对调度优化"**。
- 🔴 **没有与可并行扫描的连续时间模型对比** —— 本文也没有拿 CfC / S4 / Mamba 的并行 scan 路线做同硬件对比 ⇒ **"连续时间 vs 线性 SSM" 的表达力-成本权衡在本篇里没有被公平摆出**。
- 🔴 **benchmark 全是低维（$n\le 8$）的经典系统辨识基准**（WH / Silverbox / EMPS），**不是时序预测任务** ⇒ **本仓的工业/EEG/金融任务上无直接证据**。
- 🔴 **只报单次训练的最终精度，无多种子误差条**（与本仓已研读的多篇论文不同）⇒ 0.24 vs 0.24 的"打平"没有方差支撑。
- ⚠️ **AI 生成声明**：论文明确写了多头的实现与 port-Hamiltonian 变体由 **GPT-6 Astra pro** 生成 ⇒ **该变体的正确性只由作者自审背书**。
- ⚠️ **未评审**（submitted to IFAC）。

### 6.3 未来方向

- 扩展到更大系统（需要绕开寄存器容量限制，或走 CUDA persistent-kernel 路线）。
- 给出神经 scaling law（论文引用了但未做）。
- port-Hamiltonian 变体的更丰富 $H$ / $R$ 参数化。

## 7. 对本仓的意义

**🔴 本篇最重要的可落地物不是"DS3M"这个架构，而是一条"**本仓液态模型的慢可能来自哪里**"的诊断清单。** 本仓此前（11-13 / 11-19 / 11-20 / 11-21）反复记录的 bench 纪律是"必须报告精度 / 时间 / 显存三元组"，本篇给这条纪律补上了**第四个必须分离的归因轴**：

> **"连续时间" ≠ "串行递推"。一个连续时间模型的训练成本 ≈ (串行求值次数) × (单次求值工作量) + (调度/搬运开销)。本仓过去把"液态模型慢"归因给"连续时间"，而本文证明：**在单次工作量足够大时，串行本身不是问题；在单次工作量很小时，调度开销压倒一切。** ⇒ 本仓的 $\tau$ 变体扫描（几十个变体 × 多次前向）**很可能测的是调度开销，而不是"液态机制"的差异** —— 这是一个此前从未被写下来的混淆变量。

**四条可直接执行的动作**：

1. 🔴 **新消融轴（最高优先级，成本最低）："批量化 vs 串行化"轴** —— 本仓所有 bench 脚本每次跑几十个变体，**每个变体单独前向** ⇒ **变体之间的加速比可能被调度开销主导**。可做的最小实验：把 $N$ 个 $\tau$ 变体**批成一个大 batch**（一次前向）vs **逐个前向**，比较总墙钟时间与**相对**误差排序。**若排序不变但绝对时间差一个量级 ⇒ 本仓过去多轮"变体间差异"的表述全部需要加"绝对时间不可比"的限定。**
2. 🔴 **新消融轴："单次工作量"轴** —— 本仓 LTC/CfC 的状态维度与隐藏维都远小于 ImageNet 规模 ⇒ 按本文 Figure 1 的逻辑，本仓处于"**串行求值次数多 × 单次工作量极小**"的最坏象限 ⇒ **任何"液态更慢"的结论必须先报告"单次前向的 wall time 与 GPU 利用率"**，否则无法区分机制成本与调度成本。
3. **新变体候选：`PortHamiltonianCfCCell` / `PortHamiltonianLTCCell`** —— 本文给了一个**把耗散性写进结构**的模板（$J$ 反对称、$R$ 半正定、能量单调不增）。本仓 `lnn/core/variants.py` 至今**没有**任何能量/耗散结构先验的变体 ⇒ 这是**一条与本仓"多 $\tau$ 变体"正交的新维度**（不是又一个 $\tau$ 的写法）。⚠️ 但注意：**本仓的 CfC 是逐元素闭式解**，要套 port-Hamiltonian 结构需先改成"矩阵指数 + 反对称/半正定分解"，这与 11-19 提出的 `MatrixExpCfCCell` 候选**恰好是同一件事的两面**（那个候选加通道耦合，这个候选加物理结构）⇒ **建议合并为同一个变体族**。
4. **新 bench 纪律**：本仓的 bench 报告应新增一列"**每步串行模型求值次数**"（LTC/CfC 的 NFE），并与"单次前向 wall time"并列 ⇒ **这是 11-20 建立的"NFE 是一维"纪律的进一步细化**：本篇说明 **NFE × 单次工作量才是成本**，单报 NFE 会误导。

**一条对既有结论的口径提醒**：本篇**不能**被引用为"连续时间模型在现代硬件上不划算" —— 它的结论恰好相反：**连续时间 + 非线性 + 融合 kernel = 28–160× 加速且精度不降**。⇒ 本仓若在任何地方写过"液态路线在工程上不占优（因为慢）"，**本文提供了一条不必放弃液态机制的解法**（改调度，而不是改机制）。

**Verdict**：
- **"连续时间系统辨识的 GPU 劣势来自调度/搬运开销而非表达力" → TARGET-POSITIVE**（Table 1 的 28–160× 与"融合后时间与级数无关"是直接、干净、可核对的证据；Figure 1 的同图对比给出量化画像）。
- **"DS3M 在辨识基准上达到或超过 SOTA" → TARGET-POSITIVE-WITH-NUANCE**（EMPS 1.8 mm 确为超越；但 WH 是**打平**而非超越；跨硬件训练时间不可比；无多种子误差条；平行 WH/Silverbox 的第 2 条轨迹仍不及 NL-LFR；论文未评审）。
- **"port-Hamiltonian 结构能提升稳定性建模" → TARGET-DEPENDENT-WITH-NUANCE**（WH 上 0.28 vs 0.24 mV，**在本篇自己的实验里就是更差**；$H=\frac12\|x\|^2$ 与对角 $R$ 是两次表达力简化 ⇒ **不得引用为"物理先验提升精度"**，只可引用为"物理先验提供结构性耗散保证"）。
- **对液态路线（LTC / CfC）本身 → 不适用（族外，但有强工程含义）** —— 本篇**不是**液态家族的证据（全文无 CfC / LTC 对比）；**但它对本仓有一条必须纠正的归因**：**"液态模型慢"不能笼统归给"连续时间"**，因为本文证明连续时间的成本可以在不改机制的前提下被消掉一个到两个数量级。
