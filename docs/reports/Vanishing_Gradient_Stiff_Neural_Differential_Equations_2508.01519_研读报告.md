---
title: "The Vanishing Gradient Problem for Stiff Neural Differential Equations"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-ODE, Stiff-Systems, Vanishing-Gradient, Stability-Function, A-Stable, L-Stable, Numerical-Analysis, Preprint]
arxiv_id: 2508.01519
---

# 研读报告：刚性 Neural ODE 的梯度消失 — **不是网络太深，是积分器的稳定函数在数学上禁止你学**：一条所有 A-稳定/L-稳定方法都逃不掉的下界

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描，第七天）捞出来的。本仓**零覆盖**（`2508.01519` 在 180 篇研读报告中零命中）。
> 🔴 **本篇对本仓的意义有四层，是本轮三篇里对本仓实现影响最直接的一篇**：(a) 🔴 **它给"本仓为什么不能靠加层数/加 skip connection 解决液态训练问题"提供了一条**数学层面的答案**：**刚性区里的梯度抑制源自积分器的稳定函数 $R(z)$，与网络深度无关**（$R'(z)=O(|z|^{-2})$，且**任何** A-稳定方法的最好情况也只是 $O(|z|^{-1})$）；(b) 🔴 **它解释了本仓 CfC 相对 LTC 的一个潜在机制性优势（此前本仓只有"快"这一个理由）** —— **CfC 是闭式一步更新、根本没有稳定函数 $R(z)$**，所以它**在结构上不可能遭遇刚性梯度抑制** ⇒ **"液态要闭式"这条工程路线的收益里，可能藏着一条数学层面的收益**；(c) 🔴 **它给本仓的 `tau` 多尺度变体提供了一条硬性设计约束** —— 本仓 `lnn/core/cfc.py` 的多 $\tau$ 实现是 $\mathrm{decay}=\sigma(-f\tau\Delta t)$（**逐元素 sigmoid**），若要让本仓液态单元**也**具备"多时间尺度"，那么**时间尺度分离比 $|\tau_{\max}/\tau_{\min}|$ 越大，梯度越可能被抑制** ⇒ 这是一条**本仓至今零测量的设计边界**；(d) **一条必须加进本仓 bench 纪律的量** —— **报告每步的"刚度比" $|z|=|h\lambda|$**（本仓 bench 只报精度/时间/显存，从未报过刚度）。

## 1. 元数据

- **论文标题**：The Vanishing Gradient Problem for Stiff Neural Differential Equations
- **作者**：Colby Fronk（通讯）, Linda Petzold（通讯）
- **机构**：Department of Chemical Engineering / Department of Mechanical Engineering / Department of Computer Science, University of California, Santa Barbara
- **发表时间**：2025-08-02 (v1)（arXiv:2508.01519v1 [cs.LG]；论文自标 *Dated: 5 August 2025*）
- **发表渠道**：**Preprint**（自标 arXiv preprint，实验环境为 **XSEDE / NERSC** 集群 ⇒ 已实际跑过数值实验）
- **来源**：arXiv:2508.01519v1 [cs.LG]
- **代码**：论文未给出公开仓库地址
- **本地 PDF**：[papers/2508.01519/2508.01519.pdf](../../papers/2508.01519/2508.01519.pdf) ／ 文本 [2508.01519.txt](../../papers/2508.01519/2508.01519.txt)
- **页数**：10 页
- **关联概念**：Neural ODE / **Disc-Opt vs Opt-Disc** / **adjoint 伴随法** / **刚性（stiffness）** / **稳定函数 $R(z)$** / **A-稳定 / L-稳定** / Backward Euler / Trapezoid / Radau3 / Radau5 / BDF2 / **IF Euler（指数积分因子欧拉）** / 2/3 阶有理方法 / **参数不可辨识性（unidentifiability）** / 生物化学通路（caspase 凋亡、p53 通路）/ 混沌与刚性优化 / `torch.compile` / JAX
- **入选理由**：本仓**零覆盖**"积分器如何决定梯度可见性"这一整族；而本仓自 11-06 起连续多轮在索引里追问"**液态时间常数变体之间为什么差别这么小**"，本篇给出了一个**此前本仓完全没考虑过的解释候选**：**如果被冻结/不可辨识的恰是快模式参数，梯度抑制会让它们永远学不出来** —— 这是一条**能直接解释本仓"多 $\tau$ 变体可能是一回事"**（11-13 CT-GRU 悬案）的机制。

## 2. 核心问题

**背景事实**：Neural ODE 的训练必须**穿过数值求解器**求导（Opt-Disc 用伴随法反向积分；Disc-Opt 先离散再用框架 AD 精确求导）。**两种策略都要经过求解器**。

**论文指出的问题**：在**刚性（stiff）**系统里，**控制快衰减模式的参数**的梯度会**趋于零**，导致优化困难。作者把它形式化为一个关于**稳定函数导数**的问题：

$$y_{n+1}=R(h\lambda)\,y_n=R(z)\,y_n,\qquad z=h\lambda \tag{线性测试方程}$$

- **A-稳定**：$|R(z)|\le 1$ 对所有 $\Re(z)\le 0$；
- **L-稳定**：额外要求 $\lim_{|z|\to\infty}R(z)=0$（把极刚性的模式每步都压掉）；
- **参数敏感度**由 **$R'(z)=dR/dz$** 控制：每步链式法则都会乘上 $R'(z)$ 的因子 ⇒ **$R'(z)$ 的形状与量级直接决定参数敏感度如何沿时间传播**。

论文的**核心断言**（也是其标题的主张）：

> *"we show that this vanishing gradient phenomenon is **not an artifact of any particular method, but a universal feature of all A-stable and L-stable stiff numerical integration schemes**."*

并进一步给出**普适下界**：任何 A-稳定方法的 $R'(z)$ 衰减**不可能慢于** $O(|z|^{-1})$。

**问题被重述为**：**"刚性区里的梯度消失，其最坏情况的下界是多少？这个下界由什么决定、能否被绕过？"**

**一个必须记住的诚实补充**（论文 §DISCUSSION 末段，本篇最有价值的一句）：

> *"even the **exact sensitivity** for the linear test equation exhibits exponential decay with increasing stiffness... the vanishing gradient phenomenon is **intrinsic to stiff dynamical systems themselves, not merely a numerical artifact**."*

⇒ 即：**数值积分器会额外抑制梯度，但根子在刚性动力学本身** —— 这是"数值缺陷"与"问题本身困难"的分界，**论文把它讲清楚了**（这比大多数"梯度消失"论文诚实）。

## 3. 方法论与核心思路

### 3.1 分析框架：从 $R(z)$ 到 $R'(z)$

**精确解的敏感度（参照基准）**：对 $\dot y=\lambda y$，$y(h)=y_0e^{\lambda h}=y_0e^z$，

$$\frac{\partial y(h)}{\partial\lambda}=h\,y_0e^z,\qquad \Big|\frac{\partial y(h)}{\partial\lambda}\Big|=h|y_0|e^{\Re(z)}\to 0 \ (\Re z\to-\infty) \tag{参照基准}$$

⇒ **即使完全精确，敏感度也指数衰减** ⇒ 这是刚性问题本身的性质。

**数值侧的敏感度**：稳定函数写成有理形式 $R(z)=P(z)/Q(z)$（$\deg P=m$，$\deg Q=n$），则

$$R'(z)=\frac{P'(z)Q(z)-P(z)Q'(z)}{[Q(z)]^2} \tag{$R'$ 的表达式}$$

**三种情形**：

| 情形 | 结论 |
|---|---|
| $m>n$ | $R(z)\sim (a_m/b_n)z^{m-n}\to\infty$ ⇒ **违反 A-稳定**，这类积分器不用在刚性问题上 |
| $m<n$ | 分子最高次**不抵消** ⇒ $R'(z)=O(|z|^{m-n-1})$ |
| $m=n$ | 分子最高次**恒等抵消**（$P'Q$ 与 $PQ'$ 的 $\sim z^{2n-1}$ 项相等）⇒ $R'(z)=O(|z|^{-2})$ |

**Table I 汇总**：$m=n$ ⇒ $O(|z|^{-2})$；$m<n$ ⇒ $O(|z|^{m-n-1})$；$m>n$ ⇒ 不可能 A-稳定。

### 3.2 核心定理：普适下界（Cauchy 积分公式路线）

**定理（Universal Lower Bound for A-stable Schemes）**：设 $R(z)$ 是一致性（$R(0)=1$，$R'(0)=1$）、单步、A-稳定（解析于闭左半平面且 $|R(z)|\le1$）方法的稳定函数。则对任意 $\Re(z)<0$：

$$\Big|R'(z)\Big|\le \frac{1}{-\Re(z)}$$

且对任意固定 $\delta\in(0,\pi/2)$，若 $z$ 落在扇区 $S_\delta=\{z:\Re z<0,\ |\arg(-z)|\le\pi-\delta\}$，则

$$\Big|R'(z)\Big|\le \frac{1}{|z|\cos\delta}$$

**证明骨架（可复算，一段）**：取 $r=-\Re(z_0)>0$，考虑闭圆盘 $D=\{w:|w-z_0|\le r\}$。由 $|w-z_0|\le r$ 得 $\Re(w)\le \Re(z_0)+r=0$ ⇒ **$D$ 完全落在闭左半平面内** ⇒ 在 $D$ 上及内部应用 Cauchy 积分公式：

$$R'(z_0)=\frac{1}{2\pi i}\oint_{|w-z_0|=r}\frac{R(w)}{(w-z_0)^2}dw$$

在积分路径上 $|R(w)|\le1$（A-稳定）且 $|w-z_0|=r$ ⇒ $|R'(z_0)|\le \frac{1}{2\pi}\cdot\frac{1}{r^2}\cdot 2\pi r=\frac1r=\frac{1}{-\Re(z_0)}$。

扇区界：$z_0=|z_0|e^{i\theta}$，$-\Re(z_0)=|z_0|\cos\theta'$ 且 $|\theta'|\le\pi-\delta$ ⇒ $-\Re(z_0)\ge |z_0|\cos\delta$ ⇒ $|R'(z_0)|\le \frac{1}{|z_0|\cos\delta}$。

⇒ 🔴 **下界是锐的**：论文指出 Möbius 型有理稳定函数 $R_\beta(z)=1/(1-\beta z)$ 在合适扇区里**恰好**达到 $O(1/|z|)$ ⇒ **扇区界不可改进**。

### 3.3 常见刚性积分器的显式结果（Table II）

| 方法 | 放大因子 $R(z)$ | 敏感度衰减率 |
|---|---|---|
| Backward Euler | $\dfrac{1}{1-z}$ | $O(|z|^{-2})$ |
| Trapezoid | $\dfrac{2+z}{2-z}$ | $O(|z|^{-2})$ |
| Radau3 | $\dfrac{6+2z}{6-4z+z^2}$ | $O(|z|^{-2})$ |
| Radau5 | $\dfrac{60+24z+3z^2}{60-36z+9z^2-z^3}$ | $O(|z|^{-2})$ |
| BDF2 | $\dfrac{4+\sqrt{16-4(3-2z)}}{2(3-2z)}$ | $O(|z|^{-3/2})$（**较慢**） |
| **IF Euler** | $e^{z}$ | $O(e^{\Re z})$（**指数衰减，匹配精确解**） |
| 2nd Order Rational | $\dfrac{2+z}{2-z}$ | $O(|z|^{-2})$ |
| 3rd Order Rational | $\dfrac{12+6z+z^2}{12-6z+z^2}$ | $O(|z|^{-2})$ |

**两个关键观察**：
1. **绝大多数方法都是 $O(|z|^{-2})$**，只有 **BDF2 是 $O(|z|^{-3/2})$**（较慢 ⇒ 敏感度保留更多）；
2. **IF Euler 用精确指数处理线性部分**，敏感度衰减 $O(e^{\Re z})$，**与精确解一致** ⇒ 它是 Table II 中**唯一不被额外数值抑制**的方法（代价是一阶精度）。

### 3.4 讨论：为什么深度学习里的老办法全都没用

论文明确对比了经典梯度消失与刚性梯度消失：

| | 经典深度梯度消失 | 刚性 NDE 梯度消失 |
|---|---|---|
| 根因 | 深层非线性 Jacobian 连乘 | **积分器对刚性模式的数值阻尼** |
| 深度/结构是否相关 | **是** | **否** |
| residual / normalization / skip connection 能否缓解 | **能** | **不能**（*"they cannot address this issue, since it arises from the integrator's action on stiff modes, not from the depth or structure of the neural network itself"*） |
| 可选解法 | 初始化、ReLU 族、门控、归一化 | **需要新的计算策略**（论文明说 *"demand new strategies in how we approach both integration and sensitivity analysis"*） |

### 3.5 与 LTC / CfC / GRU 的关系 —— 本篇对本仓最重要的推论

- **与 LTC (Hasani 2021)**：本仓 `lnn/core/ltc.py` 的 `LTCODEFunc` 是 $\dot h=-h/\tau_{\text{eff}}+f(\cdot)A$，**通过 `torchdiffeq` 的 `odeint(rk4)` 积分** ⇒ **本仓的 LTC 路线完整暴露在本篇诊断的问题下**：若训练中出现的 $\lambda$ 谱很宽（$|\Re\lambda|$ 跨几个数量级），则 $\dot h$ 里最快的那个模式对应的参数会被**数值抑制**。
  ⚠️ **但必须精确**：RK4 是**显式**方法，**不是 A-稳定**（稳定域有限）⇒ 本篇的普适下界**不直接适用**于 RK4；适用的是"刚性区显式方法需要极小步长 ⇒ 成本爆炸"这条更朴素的路径。**本篇对 RK4 的适用部分只有"显式方法在刚性区不可行"这一层**。
- 🔴 **与 CfC**：本仓 `lnn/core/cfc.py` 的实现是
  ```python
  decay = torch.sigmoid(-f * self.time_scale * dt)
  return decay * g + (1.0 - decay) * h_out
  ```
  ⇒ **这是闭式一步更新，压根没有稳定函数 $R(z)$** ⇒ **在结构上不可能遭遇本篇描述的刚性梯度抑制**。
  ⇒ 🔴 **这是本篇给本仓的一条全新论证**：CfC 相对 LTC 的已知优势（速度、可并行、不规则采样）之外，**还有一条"梯度可见性"层面的机制性优势** —— 但**必须同时声明反面**：CfC 的闭式更新本身是**对精确解的一阶近似**（$\sigma(-f\tau\Delta t)$ 取代指数积分），因此它**用近似误差换掉了刚性阻尼**；**这不是免费午餐**，而是一次**偏差-方差式的取舍**（近似误差 vs 梯度抑制），本篇**没有**对 CfC 做任何分析（它不讨论闭式连续时间）⇒ **本仓若要主张这条优势，必须自己补一个对照实验**。
- **与 GRU**：离散更新，无积分器 ⇒ 无 $R(z)$，但也**无连续时间语义**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| 参照基准 | $\partial y(h)/\partial\lambda = h y_0 e^z$，$\;\lvert\cdot\rvert=h\lvert y_0\rvert e^{\Re z}\to 0$ | **精确解**的敏感度也指数衰减 ⇒ 问题固有于刚性动力学 |
| $R$ 表达式 | $R'(z)=\dfrac{P'(z)Q(z)-P(z)Q'(z)}{[Q(z)]^2}$ | 有理稳定函数的导数 |
| $m=n$ | $R'(z)=O(\lvert z\rvert^{-2})$ | 最高次抵消 ⇒ **最常见情形** |
| $m<n$ | $R'(z)=O(\lvert z\rvert^{m-n-1})$ | 分子不抵消 |
| $m>n$ | $R(z)\to\infty$ ⇒ **违反 A-稳定** | 此类方法不用 |
| **定理（全局界）** | $\lvert R'(z)\rvert\le \dfrac{1}{-\Re(z)}$ | Cauchy 积分公式 ⇒ A-稳定的**普适**上界 |
| **定理（扇区界）** | $\lvert R'(z)\rvert\le \dfrac{1}{\lvert z\rvert\cos\delta}$，$z\in S_\delta$ | 固定扇区上的 $O(1/\lvert z\rvert)$ ⇒ **下界锐**（Möbius $R_\beta=1/(1-\beta z)$ 可达） |
| Table II | Backward Euler / Trapezoid / Radau3 / Radau5 / 2nd-3rd Rational：$O(\lvert z\rvert^{-2})$；**BDF2：$O(\lvert z\rvert^{-3/2})$**；**IF Euler：$O(e^{\Re z})$** | 各方案的具体衰减率；**IF Euler 是唯一与精确解同阶者** |

## 5. 关键成果与贡献

### 5.1 主结果（理论为主）

1. 🔴 **普适下界定理**：任何一致、单步、A-稳定方法的 $R'(z)\le 1/(-\Re z)$，在固定扇区上 $O(1/|z|)$，**且该界是锐的**（存在达到它的 A-稳定有理函数）。
2. 🔴 **显式表**（Table II）：8 个常见刚性积分器的放大因子与敏感度衰减率，**6 个是 $O(|z|^{-2})$**，BDF2 $O(|z|^{-3/2})$，IF Euler $O(e^{\Re z})$。
3. 🔴 **归因澄清**：刚性梯度消失**不是**网络架构问题 ⇒ **深度学习的全套补救手段（residual / normalization / skip / gating）在此无效**。
4. 🔴 **"根子在问题本身"的诚实补充**：精确解的敏感度同样指数衰减 ⇒ **即使换成精确求解器，刚性系统的参数不可辨识性依然存在**。

### 5.2 机制分析

- **为什么 IF Euler 例外**：它把线性部分用**精确指数**积分（$\mathrm{e}^z$），因而其"数值阻尼"退化为精确解本身的阻尼 ⇒ 它的敏感度衰减 $O(e^{\Re z})$ 是**不可避免的下界**，而非额外的数值损失。
- **为什么 BDF2 稍慢**：其 $R(z)$ 的分子含 $\sqrt{16-4(3-2z)}$，$m<n$ 的差值更小 ⇒ 衰减率 $O(|z|^{-3/2})$ 介于 $O(|z|^{-1})$ 与 $O(|z|^{-2})$ 之间。
- **为什么"深度学习老办法"无效**：残差连接改变的是**网络内部的 Jacobian 连乘**，而刚性抑制发生在**积分器对模态的逐时间步放大**上 ⇒ 两者在数学上不重叠。

### 5.3 贡献清单

- 🔴 **一条不可绕过的下界**：*"所有 A-稳定方法在刚性区都必然抑制参数梯度，最好也就是 $O(|z|^{-1})$"* —— 这是一条**方法论级的硬约束**，任何"换个积分器就解决了"的说法必须先过这一关。
- 🔴 **一张可直接查表的衰减率表**（8 个方案 + 显式 $R(z)$ 与敏感度表达式）。
- 🔴 **一条归因纪律**：**"梯度消失"要分两类**（网络深度 vs 积分器刚性），**解法完全不同**。
- 🔴 **一条"问题固有性"声明**：精确解的敏感度也衰减 ⇒ **不要把所有失败都归咎于数值方法**。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **实际影响程度仍是开放问题**：*"the extent to which the vanishing gradient phenomenon limits practical learning remains an open question."*
- **不同模型/任务/数据区间上的影响尚未被充分理解**。
- **显式指数积分法的高阶版本不可用**：论文引述前人工作指出，**除一阶 IF Euler 外，没有找到同时具备可靠性、精度与稳定性的高阶显式指数积分法**（高阶显式有理 Taylor 序列方法虽有报道，但需每步一次线性求解）。
- **本文只分析线性测试方程的稳定函数**；真实网络的非线性耦合如何改变整体谱，本文未处理。

### 6.2 隐含局限（本仓视角）

- ⚠️ **本篇的定理不覆盖本仓 LTC 的 RK4 积分器**（显式、非 A-稳定）⇒ **引用时必须说明"本篇结论严格针对 A-稳定/L-稳定方法"**；对 RK4 只能引用"显式方法在刚性区步长受限"这层意思。**这是本篇最容易被误引的地方。**
- ⚠️ **本篇完全不讨论闭式连续时间（CdC/CfC）** ⇒ "CfC 天然避开刚性梯度抑制"是**本仓的推论，不是本篇的结论**，引用时必须标明是本仓的推论 + 需要自证。
- **本篇没有给出任何"怎么办"的方案** ⇒ 引用时不能写成"本文给出了解决方案"。
- **"刚性"在时序数据建模里的对应量未定义** —— 本仓的 $\tau$ 是**学习出来的**（不是物理特征值），"刚性"在本仓语境下更接近"**快慢时间尺度之比 $|\tau_{\max}/\tau_{\min}|$**"，而本篇的 $|z|=h\lambda$ 是"步长 × 特征值" ⇒ **两者不是同一个量**，不能直接套用 $O(|z|^{-2})$ 的数值。

### 6.3 未来方向（作者给出的）

- 寻找**根本上绕过**该下界的计算策略（而非在 A-稳定家族内挑更好的）；
- 显式指数积分法的高阶版本；
- 量化该现象对不同模型族/任务/数据区间的**实际**影响。

## 7. 对本仓的意义

**（a）可作为 `lnn/core/` 的什么变体？** —— 本篇**不直接产出变体**，但产出**一条设计边界**：本仓 `cfc.py` 的多 $\tau$ 实现（`tau_scales=(0.1, 1.0, 10.0)`，比例 100:1）若进一步拉大时间尺度分离比，**多 $\tau$ 分支之间的"快分支"参数可能学不动**（在 ODE 路线下）。⇒ 可考虑一个 **`StiffAwareCfCCell`** 变体：显式限制 $|\tau_{\max}/\tau_{\min}|$ 的上界（例如 $\le 50$），或用**梯度裁剪 / 参数归一化**把不同 $\tau$ 分支的更新速率拉齐，作为"多 $\tau$ 变体为何彼此等价"（11-13 CT-GRU 悬案）的一个候选解释。

**（b）可嫁接到哪个 bench 脚本？** —— 新增 **`scripts/bench_tau_separation_gradient.py`**：
- 对 `CfCCell(n_tau=K, tau_scales=...)` 扫时间尺度分离比 $r\in\{1,10,100,1000\}$；
- 报：(i) 每个 $\tau$ 分支参数的平均相对更新量 $|\Delta\theta|/|\theta|$（**梯度可见性的直接代理**）；(ii) 各分支隐状态的实际时间常数 $\tau_{\text{eff}}$ 的谱跨度；(iii) 与单 $\tau$ 基线的任务损失。
**这是本仓第一次能直接回答"多 $\tau$ 分支是不是有一支学不动"的实验。**

**（c）提供哪个新 ablation axis？** —— 🔴 **"梯度可见性"轴**（第三次补齐，本轮的第三条）：
| 轴 | 来源 | 状态 |
|---|---|---|
| 证书符号统计 $L_fV_\theta<0$ 的比例 | 11-22 `2611.21030` | 待做 |
| 梯度-观测序号敏感度 $R^2$ | 11-23 `2602.11738` | 待做 |
| **每分支梯度可见性 / 时间尺度分离比** | **本日 `2508.01519`** | **待做** |
| **$\lambda_{\max}(S_\Theta)$ 分布** | **本日 `2608.09404`** | **待做** |

⇒ 四条轴合起来 = **"本仓的液态模型到底学到了什么、没学到什么"** 的第一份完整审计清单。

**（d）一条 bench 纪律升级（承接 11-22 的 NFE 纪律）**：

> **凡本仓用连续时间模型跑的结果，除 NFE 与 wall time 外，必须报告"刚度代理量"** —— 对 ODE 路线：谱半径与谱间隙（$|\Re\lambda_{\max}|$ 与 $|\Re\lambda_{\min}|$）及其比值；对闭式路线：**各 $\tau$ 分支的有效时间常数跨度 $|\tau_{\max}/\tau_{\min}|$**。**理由**：$R'(z)=O(|z|^{-2})$ 意味着**梯度可见性本身是刚度的函数**，不报刚度 ⇒ 无法判断"这个变体没效果"是机制无效还是梯度被抑制了。

**（e）一条引用纪律（可并入 `analysis/negative_results/README.md` 纪律段，本次未擅自改台账）**：

> **凡引用本篇（或其结论）说明"梯度消失"时，必须标注三件事**：(i) 该结论**严格针对 A-稳定/L-稳定积分器**，本仓 `ltc.py` 用的 **RK4 是显式方法、不在定理覆盖范围内**；(ii) 刚性区**精确解**的敏感度同样指数衰减 ⇒ **梯度抑制有一部分是问题固有的，不能全归咎于数值方法**；(iii) 本篇**不讨论闭式连续时间**，任何" CfC 因此受益"的说法都是**本仓推论、需自证**。

**Verdict**：**"刚性区的参数梯度抑制是所有 A-稳定方法的数学必然后果，且下界 $O(|z|^{-1})$ 不可改进" → TARGET-POSITIVE（定理级证据，本篇最可靠的部分）**；**"CfC 因闭式而天然规避该问题" → TARGET-DEPENDENT-WITH-NUANCE（本篇未涉及 CfC，属本仓推论，需自测；且 CfC 是用近似误差换梯度可见性，是取舍不是免费）**。
