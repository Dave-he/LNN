---
title: LogSig-SSM: Time-Series Modelling with Multi-Scale Log-Signature Compression for State-Space Models
date: 2026-11
tags: [LNN, Continuous-Time, StateSpace, Log-Signature, Mamba, Selective-SSM, Irregular-Sampling, NeurIPS2026, Efficiency]
---

# 研读报告：LogSig-SSM — 把「路径的高阶几何」当成 token 喂给 Mamba，用 30× 加速换掉 SSM 的跨通道盲区

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**。
> 🔴 **但它对本仓意义极大**：它是本仓 11-16 轮已两次列为"最便宜、价值最高的新轴"（**把 S4/Mamba 加入本仓对照组**）的**第一个带完整数字的外部证据**。

## 1. 元数据

- **论文标题**：LogSig-SSM: Time-Series Modelling with Multi-Scale Log-Signature Compression for State-Space Models
- **作者**：Felix Oury, Nicolas Calvo Peiro, Reiko J. Tanaka（Department of Computing / Department of Bioengineering, Imperial College London）
- **发表时间**：2026-10-04 (v1)；**NeurIPS 2026 已接收**（arXiv comment 明示）；23 页，2 图，15 表
- **来源**：arXiv:2610.05051v1 [cs.LG]
- **代码**：https://github.com/felix-oury/LogSigSSM
- **本地 PDF**：[papers/2610.05051/2610.05051v1.pdf](../../papers/2610.05051/2610.05051v1.pdf) ／ 文本 [papers/2610.05051/2610.05051v1.txt](../../papers/2610.05051/2610.05051v1.txt)
- **数据集**：UEA 长序列分类（6 个数据集，沿用 Log-NCDE 协议）、PPG-DaLiA（128Hz 生理回归，窗口 49920）、Weather（720→720 小时长程预测）、PhysioNet 2019 Sepsis（40335 条 ICU，**观测率仅 10.3%**）
- **关联概念**：Log-signature / 粗糙路径 / Selective SSM / Mamba / S5 / LinOSS / D-LinOSS / NCDE / NRDE / Log-NCDE / Lévy area / 内部时间重标定 / LTC / CfC（**仅作对照，非本方法族**）

## 2. 核心问题

长时序建模面对一个**结构性二选一**：

- **连续时间模型**（Neural ODE / NCDE / NRDE）天然处理不规则采样，但**在长序列上扩展性差** —— 数值求解器在长序列上需要更多积分步，前向传播**顺序不可并行**。
- **结构化 SSM**（S4/S5/Mamba）**随序列长度线性扩展**，但有两条被本文点名的硬伤：

**(a) 长序列退化。** Mamba 类模型在极长上下文上会退化，可能低于注意力基线；已有缓解手段是 token pruning 或 decimation。

**(b) 🔴 对角状态转移没有 block 内跨维混合 —— 这是本文的靶心。** 论文引用 Cirone et al. (2024) 的结论：**单个 Mamba block 的闭包可分解为逐通道泛函之和**，因此像 **Lévy area** 这种依赖两个通道联合行为的**真正多变量路径泛函，落在该闭包之外**。换句话说：**Mamba 的对角扫描在原理上看不见"通道 A 和通道 B 交叉出来的面积"。**

已有两种补救都有代价：
- **堆叠 block + 层间线性混合** → 能恢复完整表达力，但**只��深度和隐状态规模无界增长时才恢复**（cross-channel 结构**用模型容量买单**）；而且 Cirone et al. 证明要恢复深度-$m$ 的跨通道特征，**Mamba 链的深度必须随 $m$ 增长**。
- **换更丰富的状态转移结构**（PD-SSM 等）→ 避免了深度代价，但有**非平凡的效率开销**（PD-SSM 报告约 6× 慢于 Mamba）。

本文的策略是一条**互补路线**：**在输入侧注入显式的高阶跨通道信息，同时压缩有效序列长度，把高效的对角扫描原封不动保留**。

问题因此被重述为：**能否用 log-signature 作为"区间描述子"（它恰好编码迭代积分与 Lévy area，且对单调时间重参数化不变 ⇒ 天然抗不规则采样），把长多变量序列压缩成短 token 流，再交给选择性 SSM 处理长程依赖？** 若能，就同时得到 (i) 不规则采样的鲁棒性、(ii) 单 block 内的跨通道表达力、(iii) 线性扩展性。

## 3. 方法论与核心思路

### 3.1 总体架构（两阶段）

| 阶段 | 做什么 | 关键点 |
|---|---|---|
| **① 离线 tokeniser** | 用**多尺度窗口 log-signature**把长度 $T$ 的多变量序列压缩成长度 $T'\approx T/s$ 的 token 流 | 一次性离线预处理，**不在训练反向图里**；不规则采样由分段线性插值 + 重参数化不变性保证 |
| **② 在线骨干** | 对 token 流跑**选择性 SSM**（Mamba block） | 保留对角扫描的高效性；selectivity 提供输入依赖的内部时间分配 |

**token 构造**：固定 stride $s$、锚点 $\tilde t_\tau := t_{\tau s}$、$K$ 个尺度（窗口长 $L_k$ + 截断深度 $m_k$）。第 $k$ 个尺度取窗口内样本的分段线性插值路径 $W_k(x)_\tau$，算其截断 log-signature，在固定 Lie 基下的坐标向量：

$$u_{\tau,k}:=\phi_{m_k}\Big(\mathrm{LogSig}_{m_k}\big(W_k(x)_\tau\big)\Big)\in\mathbb{R}^{D_k}$$

跨尺度拼接成 token：

$$U_\tau:=\mathrm{concat}\big(u_{\tau,1},\dots,u_{\tau,K}\big)\in\mathbb{R}^{D_{\mathrm{tot}}},\qquad D_{\mathrm{tot}}:=\sum_{k=1}^{K}D_k$$

再过一个学习到的线性嵌入 $W_{in}\in\mathbb{R}^{d\times D_{\mathrm{tot}}}$ 得到扫描宽度 $d$ 的输入 $eU_\tau$。

### 3.2 选择性 SSM 骨干与"内部时间重标定"

通道级线性 SSM 的连续时间形式：

$$\frac{dH_i(t)}{dt}=A_iH_i(t)+B_iX_i(t),\qquad y_i(t)=C_iH_i(t)$$

Mamba 的**选择性**体现在步长是输入依赖的：$\Delta_{i,\ell}=\mathrm{softplus}(\alpha_i^\top x_\ell+\beta_i)$，于是

$$A_{i,\ell}=\exp(\Delta_{i,\ell}A_i),\qquad B_{i,\ell}=(\Delta_{i,\ell}A_i)^{-1}\big(\exp(\Delta_{i,\ell}A_i)-I\big)\Delta_{i,\ell}s_B(x_\ell)$$

**小步长近似**下 $B_{i,\ell}\approx\Delta_{i,\ell}s_B(x_\ell)$ ⇒ **selectivity 本质上是状态更新的输入依赖缩放**。论文的解读：这是"**输入依赖的自适应时间重标定**"—— 模型按输入特征动态调整隐状态演化的快慢。

### 3.3 连续时间解读（本文的理论贡献之一）

把 token 流视为分段常数控制路径 $U:[0,1]\to\mathbb{R}^{D_{\mathrm{tot}}}$，有效驱动 $eU(t)=W_{in}U(t)$，则 LogSig-SSM **恰好是一个线性 CDE**：

$$dH_i(t)=\sum_j A_{ij}H_i(t)\,d\omega_{ij}(t)+W_B\,d\xi_i(t),\qquad H_i(0)=h_{i,0}$$

驱动路径为

$$\omega_i(t):=\int_0^t\sigma(\tilde\alpha_i^\top eU_s+\tilde\beta_i)\,ds,\qquad \xi_i(t):=\int_0^t eU_s\,\sigma(\tilde\alpha_i^\top eU_s+\tilde\beta_i)\,ds$$

在 ZOH 扩展下，每个区间的驱动增量 $\Lambda_{i,\tau}:=(\Delta\omega_{i,\tau},\Delta\xi_{i,\tau})^\top$ 使 (13) 退化为 **NRDE 风格的区间驱动积分形式**：

$$H_i(t)=H_i(r_{\tau-1})+\int_{r_{\tau-1}}^{t}\big(A_iH_i(s),\,W_B\big)\frac{1}{\Delta r_\tau}\Lambda_{i,\tau}\,ds$$

其中 $\Delta\omega_{i,\tau}=\sigma(\tilde\alpha_i^\top eU_\tau+\tilde\beta_i)\Delta r_\tau$ 起"**内容依赖的内部时间分配**"作用。

🔴 **与 NRDE 的两个关键差异（论文自陈）**：
1. **NRDE 用原始路径在单个窗口上的 log-signature；LogSig-SSM 用 $K$ 个不同窗口长度的多尺度 log-signature 拼成一个 token** ⇒ 同一 token 内含互补分辨率的描述子。
2. **NRDE 的驱动 $\Lambda$ 是固定的区间参数化；LogSig-SSM 的驱动是内容依赖的** ⇒ 模型学会在压缩后的序列上**选择性地分配内部时间**。

### 3.4 深度–截断权衡（Proposition 4.1）

**命题**：固定紧集 $\mathcal{X}\subset C_b^1$（$d_x\ge2$）、窗口尺度 $(L_k,m_k)$、锚点 $\tau$、$2\le m\le m_k$。设连续泛函 $F$ 可分解为 $F(X)=f(\mathrm{LogSig}_{m_k}(W_k(X)_\tau))$。则：

(i) **单个** LogSig-SSM block + 后置 MLP 可在 $\mathcal{X}$ 上一致逼近 $F$ 至任意 $\varepsilon$；
(ii) 存在这样的 $F\notin\mathcal{F}^{(1)}_{\text{Mamba}}[\mathcal{X}]$ —— 即**单个对角 Mamba block 的均匀闭包算不出它**（引 Cirone et al. 的 Cor. 4.4）；
(iii) 用**原始输入**的 Mamba 逼近这类 $F$ **需要链式对角 block**，且恢复深度-$m$ 跨通道特征所需的链深度**随 $m$ 增长**（Cirone et al. Sec. C.2.1, Prop. 4.5）；而 LogSig-SSM **在 $m_k\ge m$ 时单 block 就能实现 $F$**。

**Remark 4.2**：这**不是表达力缺口，而是深度–截断权衡** —— 代价只是把跨通道几何从"在线选择性扫描"挪到"离线并行 tokeniser"。在 $m_k\to\infty$ 的极限下，由 signature 的普适性，单个 LogSig-SSM block 可逼近窗口路径的任意连续泛函，与链式对角 Mamba 的渐近普适性一致。

### 3.5 与 LTC / CfC 的关系（关键澄清）

🔴 **LogSig-SSM 与本仓主线不是同一族，且它对本仓主线的定位有直接纠偏作用**：

| | 驱动源 | 时间常数 $\tau$ | 跨通道高阶几何（Lévy area / signature 项）是否进入模型 |
|---|---|---|---|
| **LTC** (1811.00321) | 无（自治） | 可学习 $\tau$ | **否** |
| **CfC** (2106.13898) | 无（自治） | 可学习 $\tau$（闭式解） | **否** |
| **GRU** | 观测值 $x_t$ | 固定 1 | 否 |
| **LogSig-SSM** | **log-signature token** | 无显式 $\tau$（selectivity 充当输入依赖时间重标定） | **是（depth-2 二阶项）** |

⇒ 按 11-18 建立的"自主/受控/形状编码"三档，LogSig-SSM 属于**第 ③ 档（形状编码）**，且**刻意用第 ① 档（自主 + SSM）+ 第 ③ 档 token 的混合来换取效率**。**本篇不是液态模型，不能作为液态命题的任何一侧证据。**

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$\frac{dH_i(t)}{dt}=A_iH_i(t)+B_iX_i(t),\qquad y_i(t)=C_iH_i(t)$$ | 通道级线性 SSM 的连续时间形式 |
| Eq. 2 | $$H_{i,\ell}=A_iH_{i,\ell-1}+B_ix_{i,\ell},\quad \bar A_i=e^{\Delta_iA_i},\quad \bar B_i=(\Delta_iA_i)^{-1}(e^{\Delta_iA_i}-I)\Delta_iB_i$$ | ZOH 离散化 |
| Eq. 3–4 | $$\Delta_{i,\ell}=\mathrm{softplus}(\alpha_i^\top x_\ell+\beta_i);\quad A_{i,\ell}=\exp(\Delta_{i,\ell}A_i),\ B_{i,\ell}=(\Delta_{i,\ell}A_i)^{-1}\big(\exp(\Delta_{i,\ell}A_i)-I\big)\Delta_{i,\ell}s_B(x_\ell)$$ | **Mamba 的输入依赖步长（选择性）** |
| Eq. 5 | $$dH_i=\sum_{j=1}^d A_{ij}H_i\,d\omega_{ij,t}+W_B\,d\xi_t,\qquad H_0\in\mathbb{R}^N$$ | **SSM 的线性 CDE 解读** |
| Eq. 6–8 | $$\mathrm{Sig}^m(X)_{a,b}:=(1,X^1_{a,b},\dots,X^m_{a,b}),\quad X^m_{a,b}=\int_{a<u_1<\cdots<u_m<b}dXu_1\otimes\cdots\otimes dXu_m$$ $$\mathrm{Sig}(X)_{a,b}=\mathrm{Sig}(X\circ\gamma)_{0,1}\quad(\gamma\ \text{单调非降、连续、满射})$$ $$\mathrm{LogSig}^m(X)_{a,b}:=\log(\mathrm{Sig}(X)_{a,b})$$ | **路径 signature / log-signature + 单调重参数化不变性（抗不规则采样的来源）** |
| Eq. 9 | $$U_\tau:=\mathrm{concat}(u_{\tau,1},\dots,u_{\tau,K})\in\mathbb{R}^{D_{\mathrm{tot}}},\quad D_{\mathrm{tot}}=\sum_k D_k$$ | **多尺度 log-signature token** |
| Eq. 10 | $$eU_\tau:=W_{in}U_\tau\in\mathbb{R}^d,\qquad W_{in}\in\mathbb{R}^{d\times D_{\mathrm{tot}}}$$ | token → 扫描宽度 |
| Eq. 13 | $$dH_i(t)=\sum_j A_{ij}H_i(t)\,d\omega_{ij}(t)+W_B\,d\xi_i(t)$$ | **LogSig-SSM 的线性 CDE 形式** |
| Eq. 14 | $$\omega_i(t)=\int_0^t\sigma(\tilde\alpha_i^\top eU_s+\tilde\beta_i)ds,\qquad \xi_i(t)=\int_0^t eU_s\,\sigma(\tilde\alpha_i^\top eU_s+\tilde\beta_i)ds$$ | 驱动路径（选择性 = 输入依赖时间重标定） |
| Eq. 15 | $$H_i(t)=H_i(r_{\tau-1})+\int_{r_{\tau-1}}^t\big(A_iH_i(s),W_B\big)\frac{1}{\Delta r_\tau}\Lambda_{i,\tau}\,ds$$ | **NRDE 风格的区间驱动积分形式** |
| Prop. 4.1 | 单 block + MLP 可逼近深度-$m$ 连续泛函，而单对角 Mamba block 的闭包**不能**；原始输入的 Mamba 需深度随 $m$ 增长的链 | **深度–截断权衡（本篇核心定理）** |
| Prop. 4.3 | 若 $W_k(\hat X')_\tau(t)=W_k(\hat X)_\tau(\gamma_{\tau,k}(t))$（单调重参数化），则 $U_\tau(\hat X)=U_\tau(\hat X')$ 且 $\mathrm{LogSigSSM}(\hat X)=\mathrm{LogSigSSM}(\hat X')$ | **窗口内重参数化不变性** |

## 5. 关键成果与贡献

### 5.1 UEA 长序列分类（Table 1，16 个模型，5 个随机划分平均准确率 ± 标准差，%）

| Model | EigenWorms | SCP1 | SCP2 | Ethanol | Heartbeat | Motor | **Avg** | **Rank** |
|---|---|---|---|---|---|---|---|---|
| NRDE | 83.9±7.3 | 80.9±2.5 | 53.7±6.9 | 25.3±1.8 | 72.9±4.8 | 47.0±5.7 | 60.6 | 11.5 |
| NCDE | 75.0±3.9 | 79.8±5.6 | 53.0±2.8 | 29.9±6.5 | 73.9±2.6 | 49.5±2.8 | 60.2 | 11.3 |
| Log-NCDE | 85.6±5.1 | 83.1±2.8 | 53.7±4.1 | 34.4±6.4 | 75.2±4.6 | 53.7±5.3 | 64.3 | 7.7 |
| Transformer | OOM | 84.3±6.3 | 49.1±2.5 | 40.5±6.3 | 70.5±0.1 | 50.5±3.0 | 59.0 | 10.3 |
| RFormer | 90.3±0.0 | 81.2±2.0 | 52.3±3.0 | 34.7±4.0 | 72.5±0.0 | 55.8±6.0 | 64.5 | 8.0 |
| LRU | 87.8±2.8 | 82.6±3.4 | 51.2±3.6 | 21.5±2.1 | 78.4±6.7 | 48.4±5.0 | 61.7 | 9.7 |
| S5 | 81.1±3.7 | **89.9±4.6** | 50.5±2.6 | 24.1±4.3 | 77.7±5.5 | 47.7±5.5 | 61.8 | 9.3 |
| Mamba | 70.9±15.8 | 80.7±1.4 | 48.2±3.9 | 27.9±4.5 | 76.2±3.8 | 47.7±4.5 | 58.6 | 12.5 |
| LinOSS-IMEX | 80.0±2.7 | 87.5±4.0 | 58.9±8.1 | 29.9±1.0 | 75.5±4.3 | 57.9±5.3 | 65.0 | 6.7 |
| LinOSS-IM | 95.0±4.4 | 87.8±2.6 | 58.2±6.9 | 29.9±0.6 | 75.8±3.7 | 60.0±7.5 | 67.8 | 4.3 |
| D-LinOSS | 93.9±3.2 | 88.9±3.0 | 58.6±2.3 | 29.9±0.6 | 75.8±4.9 | 61.1±2.0 | 68.0 | 3.7 |
| PD-SSM | 90.0±5.7 | 80.9±2.0 | 56.1±8.6 | 34.7±4.0 | **80.0±2.6** | 60.0±3.7 | 67.0 | 5.2 |
| **LogSig-SSM** | **95.3±3.6** | 88.4±1.9 | 58.4±2.9 | **51.4±6.9** | 76.9±9.7 | 59.9±3.4 | **71.7** | **2.7** |

🔴 **最突出的一格是 EthanolConcentration：51.4% vs 次优 RFormer 的 34.7%，+16.7pp**。论文的解释：这是"**吸收曲线**"，类别差异体现在"**长区间的平滑峰形位移**"而非高频变化 ⇒ 正是"**全局轨迹几何**"，多尺度 log-signature token 能直接捕捉，而对角 SSM 骨干自己解析不出来。

### 5.2 效率（Table 2，EigenWorms，T=17984，单张 RTX 4090）

| | NCDE | NRDE | Log-NCDE | Mamba | S5 | LinOSS-IM | D-LinOSS | **LogSig-SSM** |
|---|---|---|---|---|---|---|---|---|
| 显存 (MB) | 2484 | 2506 | 2510 | 13486 | 6646 | 3488 | 3488 | **362** |
| 时间 (s/1000 步) | 24595 | 5386 | 1956 | 122 | 31 | 14 | 14 | **4** |
| 准确率 (%) | 75.0 | 83.9 | 85.6 | 70.9 | 81.1 | 95.0 | 93.9 | **95.3** |

**相对 Mamba：训练快 30×、显存少 37×，且准确率还更高（95.3 vs 70.9）**。tokenisation 是一次性离线步骤：EigenWorms 全数据集 **1.6s**，六个 UEA 数据集合计 **6s**（约等于单次训练的 1%）。

### 5.3 其余三个基准（Table 3）

| 任务 | 指标 | 最佳非 LogSig-SSM | **LogSig-SSM** | 相对增益 |
|---|---|---|---|---|
| PPG-DaLiA（回归，窗口 49920） | MSE ×10⁻² | D-LinOSS 6.16 | **4.46** | **−28%** |
| PPG-DaLiA vs Mamba | MSE ×10⁻² | Mamba 10.65 | **4.46** | **−58%** |
| Weather（720→720 小时） | MAE | D-LinOSS 0.486 | **0.432** | **−11%** |
| PhysioNet Sepsis（有 OI） | AUROC | Neural LNSDE 91.1 | 90.4 | 略低 |
| PhysioNet Sepsis（无 OI） | AUROC | Neural GSDE 88.4 | 87.4 | 略低 |

🔴 **诚实记录的负面点**：在 **PhysioNet Sepsis（真·不规则，观测率 10.3%）** 上，LogSig-SSM（90.4/87.4）**略低于**专为不规则时间设计的 Neural LNSDE（91.1/88.1）和 Neural GSDE（90.9/88.4），只显著优于 Neural CDE（88.0/77.6）与 ANCDE（90.0/82.3）。**这与本仓 11-13/11-18 的纪律完全一致**：在真·不等间隔场景下，本仓连续时间方法的收益并没有被证实，**只在"路径几何确实承载标签"的场景里才出现结构性收益**。

### 5.4 消融（Table 4，UEA 六数据集平均，Mamba 骨干固定，逐级叠加）

| 配置 | Avg. acc. (%) | Δ vs 原始 Mamba | 增加了什么 |
|---|---|---|---|
| Mamba（原始输入） | 58.6 | – | 无压缩、无通道混合 |
| Catch24 统计量 | 58.9 | +0.3 | **丢弃路径**，只有汇总统计 |
| Depthwise conv | 60.1 | +1.5 | 各通道分开保留 |
| 标准卷积 | 62.9 | +4.3 | 一阶（线性）通道混合 |
| Wavelet scattering | 67.6 | +9.0 | 多尺度谱内容，**无跨通道几何** |
| Log-signature，深度 1 | 69.5 | +10.9 | 多尺度增量，**无跨通道内容** |
| **Log-signature，深度 2（LogSig-SSM）** | **71.7** | **+13.1** | **二阶项：跨通道几何、Lévy area（Prop. 4.1）** |

🔴 **这张表是全篇最有价值的一张**：13.1pp 的总增益被逐级分解，**最后一步（深度 1 → 深度 2）独占 +2.2pp，且这一步唯一引入的就是"跨通道几何"**。论文用 wavelet scattering 做了正面对照：在这三个少通道数据集（EigenWorms/SCP1/SCP2）上，**深度 1 的 log-sig 与最好的替代描述子 Wst-Mamba 基本打平（−1.0 / −1.8 / +2.5），而深度 2 领先 6.2 / 2.5 / 5.3** ⇒ **差距恰好就是 Prop 4.1 的跨通道几何**。另：同深度的 **signature token（不用 log）平均损失 3.1pp** 且 tokenisation 时间约翻倍 ⇒ **log 优于 signature 不是琐碎细节**。

### 5.5 其他两个消融

- **骨干对照（tokeniser 固定）**：LogSig-S5 **63.0**（+4.4）、LogSig-Transformer **65.4**（+6.8）—— 论文结论："**the tokens are useful regardless of backbone**"，但**选择性递归在每个数据集上仍领先**。
- **免搜索 tokeniser**：把搜出来的 tokeniser 超参换成一条纯规则（$s=T/100$，窗口长 $[T/25,T/100]$，深度 $[2,1]$），LogSig-SSM 仍达 **64.0%**（> S5 61.8 / LRU 61.7 / Mamba 58.6），搜索再加 7.7pp。⇒ **tokeniser 的收益不依赖精调**。

## 6. 局限性与未来展望

### 6.1 作者自陈局限（§6 Conclusion）

1. **真·单通道路径下 log-signature token 退化为路径增量**，高阶跨通道结构失效；补救是"把时间作为额外通道增广路径"。
2. **token 宽度按 $O(d_x^{m}/m)$ 增长** ⇒ 通道数增加时，tokenisation 时间与输入嵌入尺寸**迅速膨胀**。
3. **窗口长度与 stride 仍需按数据集选择**；作者提出"学习式或自适应窗口选择"作为自然方向。

### 6.2 隐含局限（本仓视角）

- 🔴 **(a) Sepsis 上输给专用不规则模型**（5.3 节）—— 论文的"robust to irregular sampling"主张在**唯一真·不规则的主战场**上只拿到"competitive"，不是"best"。而本仓的核心命题恰恰是"液态/连续时间在真·不等间隔下的结构性收益"。
- 🔴 **(b) "跨通道几何"的重要性依赖于任务是否真的需要它** —— 消融表显示 +2.2pp 来自 depth-2，但**这个增益在 EthanolConcentration 一类"吸收曲线"任务上会被放大到 +16.7pp**，在 Heartbeat/MotorImagery 上则接近 0 甚至为负（LogSig-SSM 76.9 vs PD-SSM 80.0；59.9 vs D-LinOSS 61.1）。⇒ **不能笼统说"跨通道几何普遍有用"**。
- 🔴 **(c) tokeniser 是离线的、不可微的（不在反向图里）** —— 论文明确说 "precomputed features are not in the training-time backpropagation graph"。⇒ **log-signature 的窗口/深度超参无法端到端学习**，且与随机数据增强的兼容性存疑。
- 🔴 **(d) 与本仓主线的直接可迁移性有限** —— LogSig-SSM 的效率优势来自"**离线 tokeniser + 对角扫描**"，而本仓主线是"**可学习 $\tau$ 的闭式/数值积分 ODE**"。二者**机制不同源**，不能直接把 LogSig-SSM 当成"LTC 的高效替代"。
- 🔴 **(e) 复现成本中等** —— 需要 `iisignature`/`Signatory`（JAX 系），与本仓 PyTorch 栈不同源；且本轮三个候选里**没有一篇给出端到端训练脚本之外的完整配置**（超参在 Appendix B/F）。

### 6.3 未来方向

- 学习式/自适应窗口与 stride 选择（作者明示）；
- 解决单通道退化（时间通道增广已给出方向）；
- tokeniser 与可学习 $\tau$ / 闭式 ODE 的结合（**本仓可以自己开这条**）。

## 7. 对本仓的意义

🔴 **(1) 本篇直接兑现了本仓 11-16 轮列为"最便宜、价值最高的新轴"的那张待办清单中的一项。** 11-16 的 Firebolt-VL 研读写下了："**SSM 消融（S4 > Mamba）直接提示把 S4/Mamba 加入本仓对照组 —— 本仓 `bench_mdl_cfc.py` / toy benchmark 至今只在液态家族内部互比（LTC vs CfC vs GRU），从未与'非液态的结构化线性时序模型'同台，这可能是本仓目前最便宜、价值最高的一个新对照轴**"。本篇给出了**这套对照需要的完整外部数字**：UEA 六数据集上 Mamba 58.6 / S5 61.8 / LRU 61.7 / LinOSS-IM 67.8 / D-LinOSS 68.0 / LogSig-SSM 71.7，**外加 EigenWorms 上的显存与时间表（362MB / 4s per 1000 steps vs Mamba 13486MB / 122s）**。⇒ 本仓可以直接把它作为**参照基准表**，不必每次重跑。

🔴 **(2) 一个决定性的反例，用来给本仓的"液态 = 连续时间机制有效"叙事再加一道约束。** 本篇证明：**一个完全不含 ODE、不含可学习 $\tau$、不含任何液态成分的模型（log-signature token + 对角 Mamba）**，在六个通用时序基准上**全面压过 NCDE / NRDE / Log-NCDE / Mamba / S5 / LRU**（71.7 vs 60.6 / 60.2 / 64.3 / 58.6 / 61.8 / 61.7），在物理回归上比 Mamba 好 58%，在长程天气预测上最好。🔴 **它证明"连续时间解 ODE"不是长序列建模的必要条件** —— 一个**离线、无参数、不学习**的路径几何摘要 + 一个**线性扩展**的对角扫描就够。⇒ **引用本篇时必须显式标注：这不是液态模型，但它是对"液态必须靠连续时间积分才有价值"的反向证据。**

🔴 **(3) 消融表给出了一条极其便宜、可立刻在本仓复现的消融轴：depth-1 log-signature vs depth-2 log-signature。** 本篇的证据链非常干净：**同样的 tokeniser、同样的骨干、同样的预算，depth 1 → depth 2 只增加"跨通道二阶项"这一步，就值 +2.2pp（在少通道数据集上领先最强替代描述子 6.2 / 2.5 / 5.3）**。本仓可以在自己的 toy benchmark 上做一个最小版：(a) 现有 LTC/CfC；(b) 加一阶 log-signature（跨通道一阶）作为额外输入；(c) 加二阶 log-signature（跨通道面积）作为额外输入。**若 (b) 相对 (a) 几乎无增益而 (c) 有增益，则本仓 11-18 的 "LogSigDrivenCfCCell" 变体从"直觉"升级为"有外部机制证据支撑的候选"**；反之则说明本仓场景不需要路径几何。这是本仓至今**最便宜的一条能区分"路径几何有用"与"路径几何没用"的实验**（不需要 Mamba，只需要 log-signature 特征）。

🔴 **(4) 一条跨篇的时间尺度观察（本仓应记录）：连续时间谱系正在从"解方程"走向"用闭式/代数摘要绕开方程"。** 本轮三篇的分工非常清楚：2402.18512 保留非线性向量场但把驱动换成 Lie 括号结构化的 log-sig（仍需解 ODE）；2607.05280 把向量场线性化从而**彻底消除 ODE 求解器**（闭式 flow + parallel scan，1500× 加速）；2610.05051 则更激进 —— **连连续时间积分都不需要，只保留 log-signature 作为离线 token**。⇒ **三条路都在把"连续时间"从"积分过程"退化为"代数结构"**。这与本仓 11-13 轮 CT-GRU 的负结果（"显式时间常数/多时间尺度归纳偏置做成完整连续时间模型后无优势"）和 11-18 的 Neural CDE 结论（"连续时间机制的收益是结构性的，但只在真·不等间隔下"）**共同构成一条本仓应该长期跟踪的主线**：**液态家族的可学习 $\tau$ 与闭式解，是否也被同样的"代数化"趋势逼近？**

**Verdict**：
- **"长序列 + 不规则采样的时序建模不必依赖 ODE 求解" → TARGET-POSITIVE（本轮最有价值的一条）**（LogSig-SSM 在四个基准上匹配或超越强 SSM / 连续时间基线，同时训练快 30×、显存少 37×；且连续时间解读本身是精确的而非近似）。**引用时须标注 Sepsis 上略输给专用不规则模型。**
- **"跨通道高阶几何（signature / Lévy area）值得显式建模" → TARGET-POSITIVE（机制级，证据链干净）**（消融逐级分解 + Prop 4.1 的"单对角 Mamba block 闭包不含 Lévy area"理论 + depth-2 的匹配预算对照）。
- **"log-signature 优于 signature" → TARGET-POSITIVE**（同深度同配置下 signature token 平均 −3.1pp 且 tokenisation 时间约翻倍；理论依据是 log 是无冗余坐标系且落在自由 Lie 代数内，支撑 Log-ODE 方法）。
- **"液态 / 连续时间网络在通用时序任务上更优" → TARGET-NEGATIVE-WITH-NUANCE**（本篇**不是**液态模型且**全面压过**连续时间基线；它对"液态必须靠连续时间积分才有价值"构成**反向证据**。⇒ 引用时**不得**用它支持液态命题，也**不得**把它当液态模型引用 —— 它的价值在于**为液态命题提供外部对照压力**）。
- **"不规则采样下连续时间机制有结构性收益" → TARGET-DEPENDENT-WITH-NUANCE**（本篇在 PhysioNet 10.3% 观测率上只拿到 competitive（90.4/87.4，低于 Neural LNSDE 91.1/88.1、Neural GSDE 90.9/88.4），其"鲁棒性"来自 token 的重参数化不变性这一**代数性质**，而非 ODE 积分本身）。
