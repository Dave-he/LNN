---
title: "Learning Adaptive Dynamical Features via Multi-τ Liquid-Mamba for All-in-one Image Restoration"
date: 2026-06
tags: [LNN, LTC, Liquid-Mamba, Multi-Tau, Mamba, State-Space-Model, Image-Restoration, All-in-One, Discrete-Time-Discretization, Low-Rank-Residual]
---

# 研读报告：Multi-τ Liquid-Mamba — 把"液态时间常数"塞进 Mamba 的离散化环节：多时间尺度 + 低秩残差调制 + 门控融合

> **本篇不在 digest 候选池里** —— 标题虽含 "Liquid"，但 arXiv 摘要层未触发本仓 selector 的强关键词打分（`2606.22801` 在 187 篇报告中零命中），是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。
> 🔴 **本篇对本仓的意义是"液态路线在 2026 年最主流的落地形态长什么样"** —— 它**没有**用 LTC 的 $\dot x = -\frac{x}{\tau(u)} + f(u)$ 连续时间形式，而是把"液态"降维成**离散化步长的输入条件化调制**（$\Delta_k\tau_m$），再叠一个**低秩残差矩阵** $A_m = A_0 + U_m\mathrm{diag}(\tau_m)V_m^\top$。**这是"液态 = 自适应时间常数"这条叙事迄今最工程化的一次实现，同时也最暴露它的解释力边界。**

## 1. 元数据

- **论文标题**：Learning Adaptive Dynamical Features via Multi-τ Liquid-Mamba for All-in-one Image Restoration
- **作者**：Hu Gao, Changshuo Wang, Yulong Chen, Lizhuang Ma†（†通讯）
- **机构**：上海交通大学计算机科学与技术系（Hu Gao / Changshuo Wang / Lizhuang Ma）；哈尔滨工业大学建筑与设计学院（Yulong Chen）
- **发表时间**：2026-06-22 (arXiv:2606.22801v1, cs.CV)
- **来源**：arXiv:2606.22801v1 [cs.CV]，**IEEE 会议模板稿**（全文使用 IEEEtran 排版，正文引用编号 [1]–[79]，含 "Index Terms" 段）⇒ 很可能投 IEEE 系会议/期刊，**当前为未评审预印本**
- **代码**：论文未给出公开仓库链接
- **本地 PDF**：[papers/2606.22801/2606.22801.pdf](../../papers/2606.22801/2606.22801.pdf) ／ 文本 [papers/2606.22801/2606.22801.txt](../../papers/2606.22801/2606.22801.txt)
- **页数**：18 页
- **关联概念**：Mamba / 选择性状态空间（selective SSM）/ ZOH 零阶保持离散化 / LTC (Hasani 2021) / Liquid-S4 / 多时间尺度（multi-τ）/ 低秩残差调制 / 门控融合 / 多输入多输出（MIMO）U-Net / NAFNet 骨干 / all-in-one 图像复原

## 2. 核心问题

**选择性状态空间模型（selective SSM）**的连续时间原型是

$$\frac{dx(t)}{dt} = A x(t) + B u(t), \qquad y(t) = C x(t). \tag{1}$$

Mamba 的贡献是让 $B$、$C$ 成为**输入的函数**（$B_k = B(u_k)$，$C_k = C(u_k)$），从而让状态传播变成内容感知的。但论文指出一个被忽略的结构性缺陷：

> "Although Mamba introduces input-dependent parameterization for $B_k$ and $C_k$, the hidden-state evolution is still governed by a **single transition operator** $\Phi_k = e^{A_0\Delta_k}$, where $A_0$ denotes the **shared** state matrix and $\Delta_k$ is generated from a **single discretization branch**."

即：**Mamba 的"选择性"只作用在输入/输出投影上，状态转移算子本身仍是单一时间尺度的、且 $A_0$ 在所有输入间共享**。论文用一句更直白的话概括这个隐含假设：

> "Such a formulation implicitly assumes that **all degradations can be modeled using a common dynamical system**."

**为什么这对图像复原是硬伤**：退化类型在空间上是**高度异质**的。论文的论证是——

> "image **denoising** mainly relies on short-range information aggregation, while motion **deblurring and haze removal** require substantially longer memory horizons to capture global degradation structures."
> "smooth regions generally require stable and gradual state evolution to preserve global structures, whereas edges and textured areas benefit from faster dynamics to recover fine details."

⇒ 单一时间尺度的状态转移无法同时服务"空间上局部的退化"与"空间上全局的退化"。

**问题被重述为**：*能不能在不破坏 Mamba 的线性复杂度与 selective scan 硬件实现的前提下，把隐状态演化的"有效时间尺度"变成输入条件化、且多尺度的？*

## 3. 方法论与核心思路

### 3.1 总体架构

Multi-τ Liquid-Mamba 是一个**即插即用模块**，替换 NAFNet 骨干中的 NAFBlock。整体流水线：

```
输入退化图像
   │
   ├─ 隐表示提取 r_k（浅层特征）
   │
   ├─ 从 r_k 估计 M 个时间常数 τ_m = Softplus(W_τm r_k + b_τm)   ← Eq.15
   │
   ├─ 对每个分支 m：构造调制后的状态矩阵
   │      A_m(u) = A_0 + U_m diag(τ_m(u)) V_m^T                    ← Eq.13/14
   │
   ├─ ZOH 离散化（注意：步长变成 Δ_k τ_m）
   │      Φ_m = exp(A_m Δ_k τ_m),  Ψ_m = A_m^{-1}(Φ_m - I)B(u_k)    ← Eq.17/18
   │
   ├─ 每条分支独立递推：x^(m)_{k+1} = Φ_m x^(m)_k + Ψ_m u_k       ← Eq.16
   │
   ├─ 门控融合：α_m = Softmax(W_α r_k)，x_{k+1} = Σ_m α_m x^(m)_{k+1} ← Eq.21/22
   │
   └─ 输出映射 C(u_k) x_{k+1}
```

**关键设计决策（论文自己反复强调）**：**不替换 Mamba 的状态矩阵 $A_0$，而是在其周围做低秩残差调制**。论文原文：

> "Unlike conventional Liquid-S4, which modifies the convolution kernel through **bilinear liquid approximations**, our formulation **directly modulates the state-transition operator** while preserving the selective scan mechanism."
> "directly replacing the transition matrix with a scalar liquid coefficient may **destroy the expressive transition structure** learned by Mamba."

### 3.2 液态状态转移动力学（LTC 的降维使用）

论文先回顾 LTC (Hasani 2021)：

$$\frac{dx(t)}{dt} = -\frac{x(t)}{\tau(u)} + f(u). \tag{10}$$

然后说明它**不直接沿用**，而是改造为输入条件化的连续时间系统：

$$\frac{dx(t)}{dt} = A(u)x(t) + B(u)u(t). \tag{11}$$

⇒ **注意这里的 "liquid" 已经不是 LTC 意义上的"可学习时间常数作用在状态衰减项上"，而是"输入条件化地改变状态转移算子"**。这是一个**语义漂移**，本报告第 6 节会专门标注。

### 3.3 多时间尺度 + 低秩残差调制（核心模块）

$$\frac{dx(t)}{dt} = A_m(u)x(t) + B(u)u(t). \tag{12}$$

$$A_m(u) = A_0 + \Gamma_m(u). \tag{13}$$

$$\Gamma_m(u) = U_m \,\mathrm{diag}\big(\tau_m(u)\big)\, V_m^\top. \tag{14}$$

时间常数由输入条件化生成：

$$\tau_m = \mathrm{Softplus}(W_{\tau_m} r_k + b_{\tau_m}). \tag{15}$$

**这里有一个必须指出的细节**：$U_m$、$V_m$ 是**可学习的低秩投影**（low-rank），而 $\tau_m(u)$ 是**输入条件化的**。所以 Eq.14 实际上是

$$\Gamma_m(u) = U_m \,\mathrm{diag}\big(\mathrm{Softplus}(W_{\tau_m}r_k + b_{\tau_m})\big)\, V_m^\top$$

⇒ **$A_m$ 的低秩方向是静态的（学出来的 $U_m, V_m$），只有对角增益是输入依赖的**。这与"整矩阵输入依赖"（如 NCP/NCDE 的 $A(u)$）有本质区别。

### 3.4 连续到离散的过渡（与本仓最相关的一段）

论文对 Eq.12 施加零阶保持（ZOH）离散化，**区间为 $\Delta_k\tau_m$**（注意：原来的 $\Delta_k$ 被 $\tau_m$ 缩放了）：

$$x^{(m)}_{k+1} = \Phi_m x^{(m)}_k + \Psi_m u_k. \tag{16}$$

$$\Phi_m = \exp(A_m \Delta_k \tau_m). \tag{17}$$

$$\Psi_m = A_m^{-1}(\Phi_m - I) B(u_k). \tag{18}$$

对比原始 Mamba 的 $\Phi_0 = e^{A_0\Delta_k}$ ⇒ **本文同时调制了"转移动力学"（$A_m \ne A_0$）与"演化速度"（$\Delta_k \to \Delta_k\tau_m$）**。论文原句：

> "Compared with the original Mamba transition operator $\Phi_0 = e^{A_0\Delta_k}$, the proposed formulation **simultaneously adapts both the transition dynamics and the evolution timescale**."

论文进一步用 Fréchet 展开说明这个构造的含义：

$$e^{(A_0+\Gamma_m)\Delta_k\tau_m} = e^{A_0\Delta_k\tau_m} + \mathcal{L}_{\exp}(A_0\Delta_k\tau_m,\ \Gamma_m\Delta_k\tau_m) + O(\|\Gamma_m\Delta_k\tau_m\|^2). \tag{19}$$

$$\mathcal{L}_{\exp} = \int_0^1 e^{(1-s)A_0\Delta_k\tau_m}\Gamma_m\Delta_k\tau_m e^{sA_0\Delta_k\tau_m}\,ds. \tag{20}$$

⇒ 作者的解读："Multi-τ Liquid-Mamba can be interpreted as **learning a continuous family of transition operators around the original Mamba dynamics**."

### 3.5 自适应门控融合

$$\alpha_m = \frac{\exp(W_{\alpha_m}r_k + b_{\alpha_m})}{\sum_{j=1}^{M}\exp(W_{\alpha_j}r_k + b_{\alpha_j})}. \tag{21}$$

$$x_{k+1} = \sum_{m=1}^{M}\alpha_m x^{(m)}_{k+1} = \sum_{m=1}^{M}\alpha_m \left[\Phi_m x^{(m)}_k + \Psi_m u_k\right]. \tag{22}$$

⇒ 这是**输入条件化的多分支混合动力学系统**：每个分支有各自的状态矩阵 $A_m$、各自的时间尺度 $\tau_m$，最终由输入决定的软权重融合。

### 3.6 与 LTC / CfC / NCP 的关系（**必须区分**）

| 维度 | 本文 (Multi-τ Liquid-Mamba) | LTC (Hasani 2021) | CfC (Hasani 2022) | NCP (Lechner 2020) |
|---|---|---|---|---|
| 时间常数语义 | $\tau_m$ **缩放离散化步长** $\Delta_k\tau_m$，并作为低秩增益的对角元 | $\tau$ **直接除在状态衰减项上** $\dot x = -x/\tau + f(u)$ | 闭式解中 $\tau$ 决定指数衰减率 | **固定**时间常数（离散接线） |
| 状态矩阵 | $A_0$（Mamba 共享）+ 低秩残差 $\Gamma_m(u)$ | 通常不含状态矩阵（纯状态衰减） | 闭式近似，矩阵形式 | 稀疏**固定**接线矩阵 |
| 输入依赖性 | $\tau_m(u)$、$\alpha_m(u)$ 输入依赖；$U_m,V_m$ 静态 | $\tau(u)$ 输入依赖 | CfC 的 $F$ 通常静态 | 稀疏模式静态 |
| 连续性 | **离散时间模型**，连续性只体现在"步长缩放"这一层 | 真连续时间 ODE | 真连续时间（闭式解） | 离散 |
| 上下文 | 图像 2D 扫描序列（4 方向扫描） | 时序 | 时序 | 时序 |

⇒ **最关键的口径**：本文的"liquid"是**"离散化步长的输入条件化缩放"**，不是"连续时间动力学"。它与本仓 `lnn/core/ltc.py` 的 $\dot x = -x/\tau + f$ **不是同一族**。引用时**不得**称其为"连续时间液态网络"，只能说"**把液态的自适应时间常数思想迁移到 Mamba 的离散化环节**"。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq.1–2 | $\dot x = Ax + Bu,\ y = Cx$ | 线性连续时间 SSM 原型 |
| Eq.3 | $x(t+\Delta) = e^{\Delta A}x(t) + \int_0^\Delta e^{(\Delta-s)A}Bu(t+s)ds$ | 连续系统的精确解 |
| Eq.4–6 | $x_{k+1} = \bar A x_k + \bar B u_k,\ \bar A = e^{\Delta A},\ \bar B = (\Delta A)^{-1}(e^{\Delta A}-I)(\Delta B)$ | ZOH 离散化（后文 Eq.17/18 是它的可学习推广） |
| Eq.7–9 | $x_{k+1} = \bar A x_k + B(u_k)u_k,\ y_k = C(u_k)x_k$ | Mamba 的选择性参数化 |
| Eq.10 | $\dot x = -x/\tau(u) + f(u)$ | LTC 原式（被引用为灵感来源） |
| Eq.11 | $\dot x = A(u)x + B(u)u$ | 本文改造后的输入条件化连续时间系统 |
| Eq.12 | $\dot x = A_m(u)x + B(u)u$ | 第 $m$ 条液态分支 |
| Eq.13 | $A_m(u) = A_0 + \Gamma_m(u)$ | 在 Mamba 基座上做加性残差 |
| Eq.14 | $\Gamma_m(u) = U_m\mathrm{diag}(\tau_m(u))V_m^\top$ | 低秩、时间常数承载在对角元 |
| Eq.15 | $\tau_m = \mathrm{Softplus}(W_{\tau_m}r_k + b_{\tau_m})$ | 输入条件化时间常数 |
| Eq.16 | $x^{(m)}_{k+1} = \Phi_m x^{(m)}_k + \Psi_m u_k$ | 分支递推 |
| Eq.17 | $\Phi_m = \exp(A_m\Delta_k\tau_m)$ | **步长被 $\tau_m$ 缩放**（本文最核心的一行） |
| Eq.18 | $\Psi_m = A_m^{-1}(\Phi_m - I)B(u_k)$ | 输入矩阵的 ZOH 形式 |
| Eq.19–20 | $e^{(A_0+\Gamma_m)\Delta_k\tau_m} = e^{A_0\Delta_k\tau_m} + \mathcal{L}_{\exp}(\cdot) + O(\|\Gamma_m\Delta_k\tau_m\|^2)$ | Fréchet 展开：说明这是"围绕 Mamba 动力学的连续族" |
| Eq.21 | $\alpha_m = \mathrm{Softmax}_m(W_{\alpha_m}r_k + b_{\alpha_m})$ | 退化感知的融合权重 |
| Eq.22 | $x_{k+1} = \sum_m \alpha_m[\Phi_m x^{(m)}_k + \Psi_m u_k]$ | 多分支软融合 = 自适应混合动力学 |
| Eq.23 | $\mathcal{L} = \sum_i[\mathcal{L}_c + \delta\mathcal{L}_e + \lambda\mathcal{L}_f]$，$\lambda=0.1,\ \delta=0.05$ | Charbonnier + 梯度域边缘保持 + 频域一致性 |

## 5. 关键成果与贡献

### 5.1 主结果（all-in-one，R+S+H+B+N，Table I）

| 方法 | 类型 | 平均 PSNR ↑ | 平均 SSIM ↑ |
|---|---|---|---|
| PGH2Net | task-specific（最优） | 31.84 | 0.909 |
| ACL | task-aligned（最优） | 32.78 | 0.922 |
| Perceive-IR | all-in-one | 33.51 | 0.940 |
| Defusion | all-in-one（最优基线） | 33.58 | 0.940 |
| **MLMIR (Ours)** | all-in-one | **34.23** | **0.957** |

⇒ 相对最优 all-in-one 基线 **+0.65 dB**；相对最优 task-aligned **+1.45 dB**；相对最优 task-specific **+2.39 dB**。逐任务：去雨 31.11 dB（最佳）、去雪 34.01 dB（最佳）、去雾 40.86 dB（最佳）、去模糊 31.41 dB（最佳）、去噪 33.76 dB（略低于 Perceive-IR 的 33.81）。

### 5.2 组件消融（Table XI）

| 配置 | Multi-τ Liquid-Mamba | MIMO | PSNR | ΔPSNR |
|---|---|---|---|---|
| (a) NAFNet 基线 | ✗ | ✗ | 32.34 | – |
| (b) | ✓ | ✗ | 33.96 | **+1.62 dB** |
| (c) | ✗ | ✓ | 32.55 | +0.21 dB |
| (d) | ✓ | ✓ | **34.23** | **+1.89 dB** |

⇒ **多时间尺度模块单独贡献了消融增益的 86%（1.62/1.89）**，MIMO 机制单独只有 +0.21 dB。**这是本篇最硬的一条证据：收益来自"多时间尺度离散化"，不是来自网络加深或跨尺度交互。**

### 5.3 多-τ 设计本身的消融（Table XIV / Table XVI）

| 变体 | PSNR | Δ |
|---|---|---|
| Multi-τ (M=4) | 34.23 | – |
| Single-τ (M=1) | 33.05 | **−1.18 dB** |
| w/o τ（退化为普通 Mamba） | 32.59 | **−1.64 dB** |

| M | PSNR | Δ | FLOPs(G) |
|---|---|---|---|
| 1 | 33.05 | – | – |
| 2 | 33.68 | +0.63 | +1.25 |
| 3 | 33.94 | +0.89 | +2.11 |
| **4** | **34.23** | **+1.18** | **+2.74** |
| 6 | 34.17 | +1.12 | +3.82 |

⇒ **M=4 达到最优，M=6 反而略降**（作者归因于"分支间相关性增加"）。这条曲线对本仓极有价值：**多时间尺度数量是一个可饱和的超参，不是越大越好**。

### 5.4 融合策略消融（Table XV）

| 策略 | PSNR | Δ vs Add |
|---|---|---|
| Add（直接求和） | 33.78 | – |
| Concat + Conv | 33.91 | +0.13 |
| **Adaptive (Softmax gating)** | **34.23** | **+0.45** |

⇒ **硬切换/求和明显劣于软路由**。这与本仓 11-13 记录的"$\ln\tau$ 空间 softmax 软路由 vs 硬分组多分支"待办轴**方向一致**（本篇是该轴的外部佐证）。

### 5.5 即插即用验证（Table XIII）

| 被替换的骨干 | 原 PSNR | 替换后 | Δ | ΔParams(M) | ΔFLOPs(G) |
|---|---|---|---|---|---|
| ACL | 32.78 | 33.99 | +1.21 | +0.013 | +2.74 |
| MambaIRv2 | 32.56 | 33.92 | +1.36 | +0.045 | +3.62 |
| TSPMamba | 31.79 | 33.65 | **+1.86** | +0.012 | +2.56 |

⇒ **极小的参数量代价（+0.012~0.045 M）换取 +1.21~1.86 dB**。这是本篇最有说服力的一条：**增益不来自容量**。

### 5.6 机制分析（Figure 12/13）

- **t-SNE**："w/o τ" 各退化类型特征**严重纠缠**；Single-τ 出现初步聚类但仍有重叠；Multi-τ 产生**分离且紧致的簇**。
- **融合权重 $\alpha_m$ 的退化依赖性**：雨与噪声 → 偏向**短时间尺度分支**（抑制高频、恢复局部细节）；雾与模糊 → 偏向**长时间尺度分支**（依赖缓慢演化与广域上下文聚合）；雪 → 分布均衡（局部遮挡 + 全局可见度畸变的混合特征）。
- **区域级 $\tau$ 统计**：平滑区与雾区 $\tau$ 更大（长记忆保全局一致性）；边缘与纹理区 $\tau$ 更小（快速状态转移恢复高频细节）。

### 5.7 泛化性实验

- **训练未见退化类型**（UIEB / C60 水下）：MLMIR 21.82 dB / 0.557 UCIQE / 2.557 UIQM，对 Perceive-IR 仅 +0.05 dB（**优势极小**）。
- **训练未见噪声强度**（σ=60/100）：CBSD68 σ=60 时 27.17 dB（最佳），σ=100 时 20.71 dB（略低于 Defusion 20.72）；Urban100 σ=100 时 21.58 dB（最佳）。
- **混合退化**（Table X）：H+R+N 30.45/0.901、H+R 32.52/0.965，平均 32.65/0.936（最佳）。

### 5.8 复杂度

$$\text{总复杂度} = O(M \cdot L \cdot D \cdot N + L \cdot D^2). \tag{§V-B}$$

其中 $L$ = 序列长度、$D$ = 特征维、$N$ = 状态维、$M$ = 时间尺度数。⇒ **对序列长度仍是 $O(L)$**，保留 Mamba 的线性复杂度优势；但**对分支数 $M$ 线性**。

## 6. 局限性与未来展望

### 6.1 作者自陈局限（§V-C 原文）

> "First, the **introduction of multiple timescale hyperparameters increases the model design space**. Second, the method **does not explicitly incorporate physical degradation priors**, which may limit interpretability under extremely complex or unseen degradation conditions. Third, the current formulation is **primarily validated on single-image restoration tasks**, and its behavior in temporally consistent scenarios such as **video restoration remains unexplored**."

作者的未来方向：① 引入显式物理退化模型与液态状态空间动力学结合；② 扩展到视频/多帧复原；③ **发展自适应机制自动确定最优时间尺度数**（即自动选 $M$）。

### 6.2 隐含局限（本仓视角）

1. 🔴 **"液态"一词的语义漂移（最需要标注的一条）**：本文的 $\tau_m$ **只缩放离散化步长**（Eq.17：$\Phi_m = \exp(A_m \Delta_k \tau_m)$），并作为低秩残差的对角增益（Eq.14）。**它没有把 $\tau$ 除在状态衰减项上**，因此**不是 LTC 意义上的连续时间液态网络**。引用本文时**不得**写"液态时间常数带来 +1.62 dB"——准确表述是"**输入条件化的离散化步长调制**带来 +1.62 dB"。这是本仓自 11-13 起持续强调的"关键词命中 ≠ 液态证据"的又一例。
2. 🔴 **"单一时间尺度的动态系统"这一前提只在 Mamba 语境下成立**：论文的对照是 Mamba 的 $\Phi_0 = e^{A_0\Delta_k}$。但本仓 `lnn/core/ltc.py` 的 $\tau$ 是**加性且作用在衰减项上**、CfC 的 $\tau$ 是**对率加性**，两者与 Mamba 的 $\Delta$ 缩放**不属同一族** ⇒ 本文的 $M=4$ 曲线**不能**直接迁移为本仓 $\tau$ 变体的最优数量。
3. **模型精度依赖"低秩 + 对角"这一人工结构选择**：$U_m$、$V_m$ 是可学习的，但**低秩秩数 $r$ 未在论文中给出**（文本提取中未出现具体秩值），也未做秩的消融 ⇒ "低秩"这一关键设计缺少实证支撑。
4. **$\Delta_k$ 与 $\tau_m$ 的乘法耦合未被消融**：论文没有做"只调 $\Delta_k$ 不调 $\tau_m$"或"只调 $\tau_m$ 不调 $A_m$"的对照，因此 **Eq.17 的步长缩放与 Eq.13 的矩阵调制各自的贡献无法分离**。这与本仓"变体间差异归因"问题同型。
5. **泛化优势微弱**：在训练未见退化类型上仅领先 Perceive-IR **+0.05 dB**（Table VIII），在 σ=100 上**不如** Defusion（20.71 vs 20.72 dB）⇒ "自适应多时间尺度 ⇒ 强泛化"这一因果链**证据不足**。
6. **门控融合与多分支选择的因果性未分离**：$\alpha_m$ 的可视化（Fig. 13）是**事后解读**，不是干预实验（没有"打乱 $\alpha_m$"或"固定 $\alpha_m$"的对照）⇒ 与本仓"打乱 $\Delta t$"诊断轴同型，本仓可补。
7. **与本仓已记录文献的对照位置**：
   - 相对 11-13 的 CT-GRU (1710.04110，"显式时间常数 + 多时间尺度归纳偏置"在 11 个数据集上**无优势**）：本文给出的是一个**正例**（+1.62 dB）。**两者不矛盾** —— CT-GRU 的时间常数在**连续时间 ODE 里**做多尺度归纳偏置且**未使用 $\Delta t$ 时间戳**；本文的 $\tau$ 在**离散化环节**做条件化。**这恰好说明"多时间尺度"本身不是关键，"多时间尺度 + 用在哪里"才是。**
   - 相对 11-22 的 DS3M (2611.25274，"连续时间的成本在调度不在表达力")：本文再次印证 —— **保留 selective scan 的线性复杂度是本文的核心卖点**，M=4 只让复杂度从 $O(LDN)$ 变成 $O(MLDN)$。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的变体**：
- 🔴 **`MultiTauLiquidCell`（最高优先级）** —— 实现 Eq.12–22 的离散版：在**离散递推**里加一个可学习的低秩残差矩阵族 $A_m = A_0 + U_m\mathrm{diag}(\tau_m)V_m^\top$ 与门控 $\alpha_m$。**与本仓现有 $\tau$ 变体的关键区别是：把 $\tau$ 从"作用于状态衰减"移到"作用于状态矩阵 + 融合权重"**。这是本仓至今没有的第三个作用位点（本仓现有：加性 $\tau$、对率 $\tau$、门控强度）。
- **`SoftmaxRoutedTauBank`** —— 把 Eq.21/22 的门控融合套到本仓**现有的** $\tau$ 变体库上（不改动底层 $\tau$ 语义），用来测本仓 11-13 待办轴"$\ln\tau$ 空间 softmax 软路由 vs 硬分组多分支"。**本篇是该轴的外部佐证（软路由 +0.45 dB 优于求和）。**

**可嫁接到哪个 bench 脚本**：
- 本仓 `curvature.py` / `variants.py` 的变体对比脚本：新增 **"分支数 $M$ 扫描"**（$M\in\{1,2,4,8\}$）—— 本篇给了明确的饱和曲线（$M=4$ 最优，$M=6$ 略降），本仓应验证**同一 $M$ 是否饱和**，以及饱和点是否与实现族相关。
- 本仓所有"连续时间 vs 离散"对比：**必须区分"$\tau$ 在连续时间衰减项里"与"$\tau$ 在离散化步长里"**，不能笼统写"液态变体"。

**提供哪个新 ablation axis**：
- ① 🔴 **"$\tau$ 的作用位点"轴（最高优先级）** —— 同一 $\tau$ 参数化，分别放在 (a) 连续衰减项 $\dot x = -x/\tau + f$、(b) 离散化步长 $\exp(A\Delta\tau)$、(c) 状态矩阵低秩对角元、(d) 融合门控 logits。**本仓至今只测过 (a)/(b 类)**，本篇直接说明"作用位点决定语义"。
- ② 🔴 **"低秩残差 vs 全矩阵"轴** —— 论文用低秩是因为要保留 Mamba 的 $A_0$；本仓没有 $A_0$ 基座，可直接测"低秩 $U_mV_m^\top$ vs 全矩阵 $A_m(u)$"，看输入条件化的状态矩阵到底需要多少参数。
- ③ **"门控干预"轴** —— 打乱 $\alpha_m$ / 固定 $\alpha_m = 1/M$ / 固定 $\alpha_m$ 为退化无关的常数，检验"退化感知融合"是否真被用到（补上论文 Fig. 13 缺失的干预实验）。
- ④ **"$\Delta_k$ 与 $\tau_m$ 解耦"轴** —— 固定 $\tau_m \equiv 1$ 只保留 $A_m$ 调制；固定 $A_m = A_0$ 只保留步长缩放。分离 Eq.17 与 Eq.13 的贡献。

**一条口径纪律**：
> 本篇**不能**被引用为"液态时间常数在连续时间建模中带来 1.62 dB 增益"。它证明的是：**在保留 Mamba 离散 selective scan 的前提下，把"有效时间尺度"做成输入条件化 + 多分支 + 软融合是有收益的（+1.62 dB，参数量几乎不变）**。而它把"液态"降维成了"步长缩放"——**这是 2026 年工程界使用"liquid"这个词的典型形态，值得本仓在索引里显式记录这一"语义漂移"现象**。

**Verdict**：
- 对 **"多时间尺度 + 自适应时间常数"这一归纳偏置**：**TARGET-POSITIVE**（+1.62 dB，消融干净，参数量代价可忽略）。
- 对 **"液态 = 连续时间动力学"**：**TARGET-NEGATIVE-WITH-NUANCE** —— 本文**不是**连续时间液态模型，$\tau$ 只作用于离散化步长；引用时必须加限定。
- 对 **"多时间尺度数量越多越好"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 本篇给出 $M=4$ 饱和、$M=6$ 略降的实证，本仓需在自己实现上复测饱和点。
