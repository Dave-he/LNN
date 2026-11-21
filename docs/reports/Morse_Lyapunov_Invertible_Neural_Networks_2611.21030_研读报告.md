---
title: "Learning Morse Lyapunov functions through invertible neural networks"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-ODE, Lyapunov, Morse-Theory, Invertible-Neural-Network, Sobolev-Approximation, Stability-Certificate, PLNet]
---

# 研读报告：Morse Lyapunov 函数的可学性 — **"$C^0$ 逼近不够"**：学一个稳定性证书要控制的是 Sobolev 范数，不是 sup 范数

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**（`2611.21030` 在 174 篇报告中零命中）。
> 🔴 **本篇对本仓的意义是"一条方法论级的正确性前提"**：它证明**"用神经网络拟合一个函数"这件事，在稳定性证书这一类任务上，sup 范数逼近是无效的**（$\|V-V_\theta\|_{C^0}<\epsilon$ **不蕴含** $L_f V_\theta<0$）⇒ 🔴 **这直接指向本仓一个至今未被检查的隐含假设**：本仓所有"连续时间模型学到了更好的动力学"的论断，其误差度量是 **MSE / 预测精度**，而**没有任何一条连续时间动力学的不稳定性证书**。本篇给出了让这件事可判定的完整工具链。

## 1. 元数据

- **论文标题**：Learning Morse Lyapunov functions through invertible neural networks
- **作者**：Wouter Jongeneel（单人作者）
- **机构**：KTH Royal Institute of Technology and Digital Futures
- **发表时间**：2026-11-16 (v1)
- **发表渠道**：**PMLR (Proceedings of Machine Learning Research) vol vvv:1–13, 2027**（模板占位符 `vvv` 未填，**正式卷期未定**）；作者自陈 "Currently under review"
- **来源**：arXiv:2611.21030v1 [math.OC]
- **代码**：论文未给出仓库链接
- **本地 PDF**：[papers/2611.21030/2611.21030.pdf](../../papers/2611.21030/2611.21030.pdf) ／ 文本 [papers/2611.21030/2611.21030.txt](../../papers/2611.21030/2611.21030.txt)
- **页数**：13 页
- **关联概念**：Lyapunov 函数 / **Morse 理论**（非退化临界点）/ **可逆神经网络 (INN)** / **Neural ODE** / **Sobolev 逼近（$W^{r,\infty}$）** / **PL Lyapunov 函数** / PLNet / BiLipNet / PolarNet / affine coupling flow / 全局渐近稳定 (GAS) / 吸引域 / 耗散性 / Kuramoto-Smale / 条件数与畸变参数 $\tau$

## 2. 核心问题

**数据驱动的稳定性证书**。经典理论（Massera 1956、Kurzweil 1963、Wilson 1969；Hahn 1967、Bacciotti & Rosier 2005）说：一个平衡点 $x^\*$ 全局渐近稳定 (GAS) **当且仅当**存在一个光滑 Lyapunov 函数 $V:\mathbb{R}^n\to\mathbb{R}_{\ge0}$ 满足

- **(i)** **强制性 (coercive)**：所有下水平集 $\{x: V(x)\le c\}$ 紧；
- **(ii)** $\{x^\*\}=V^{-1}(0)$；
- **(iii)** **沿轨严格下降**：$L_f V(x):=\langle\nabla V(x), f(x)\rangle < 0$，$\forall x\ne x^\*$。

**问题一（存在性有多"结构"？）**：论文聚焦"**结构**"而非仅正则性。**Morse Lyapunov 函数**（所有临界点非退化，$\nabla^2 V(x^\*)\succ0$）近年因一个拓扑事实而重要：**局部二次的 Lyapunov 函数可以被写成某个微分同胚的范数平方**（Morse 引理）⇒ 天然指向**可逆神经网络**的参数化。

**问题二（学得出来吗？）**：论文回答"**何时、如何**学出来"。

**论文指出的结构性缺陷（三条，逐级加码）**：

1. **简单的 Lyapunov 函数在一般情况下不够**。线性系统可以全局取二次型；非线性系统只在**局部**（双曲性假设下）成立。**Ahmadi et al. (2011) 已证伪"任何 Lyapunov 函数都可以取为多项式"**；**存在向量场使原点 GAS 但可证没有凸 Lyapunov 函数**（Jongeneel & Schwan 2024）；甚至**存在光滑 Lyapunov 函数但没有局部实解析的**。
2. 🔴 **存在性本身就可能被阻断**：论文给出 **Proposition 1（局部齐次性的阻碍）** —— 若 $Df(0)$ 在虚轴上有**非半单**特征值，则**任何**满足 $a_1\|x\|^p\le V(x)\le a_2\|x\|^p$ 的 $C^0$ 函数**都不可能是 Lyapunov 函数**。⇒ **不是"参数化选得不好"，是"这类系统的证书结构本身受限"**。
3. 🔴🔴 **通用逼近定理给出的范数不对口**：Zhong et al. (2026, PolarNet) 依据 Teshima et al. (2020) 的通用逼近结论，声称"渐近稳定 ⇒ 存在可被 PolarNet 精确建模的 Lyapunov 函数"。**但 Teshima 的结论是 $L_p$ / sup 范数下的逼近，而论文指出**：**$L_p$ 与 sup 范数都不控制 $C^k$ 范数（$k>0$）** ⇒ **这些范数在 Lyapunov 不等式的语境里毫无用处**。论文的定量反例：

$$\tilde h(x) = \epsilon\sin(x/\epsilon^2) \ \Rightarrow\ \|\tilde h\|_{C^0([0,1])}\le\epsilon \quad\text{但}\quad \|h+\tilde h\|_{C^1([0,1])}\ge h'(0)+1/\epsilon .$$

⇒ **sup 范数上差 $\epsilon$ 的两个函数，其一阶导可以差 $1/\epsilon$** ⇒ 🔴 **"$\|V-V_\theta\|_{C^0}<\epsilon$ 不蕴含 $L_f V_\theta<0$，即便 $L_f V<0$"** —— **这正是本仓最该记住的一句**。

**问题被重述为**：*在什么条件下，一个由 Neural ODE 构成的可逆神经网络（INN）**确实**能学出一个合法的（严格下降的）Lyapunov 函数？所需控制的到底是哪个范数？*

## 3. 方法论与核心思路

### 3.1 总体架构

论文是**理论 + 数值说明**的组合，结构为：设置与障碍（§1–§2）→ Morse Lyapunov 函数的结构刻画（§2）→ 需要控制哪些拓扑（§3）→ INN 的通用逼近性质（§4）→ 稳定性证书的逼近（§5）→ 数值实验（§6）→ 结论（§7）。

### 3.2 核心模块 1：Morse Lyapunov 函数的结构刻画（Theorem 2 + Morse 引理）

**Theorem 2**（引自作者的伴生论文 Jongeneel 2026b, Thm 3.1）：若 $f\in HS^r_0(\mathbb{R}^n)$（$0$ 是**双曲平衡点**，$\mathrm{Re}\{\lambda_i(Df(0))\}\ne0$），则**存在**光滑 Morse Lyapunov 函数 $V$ 证明 $0$ 是 GAS。

**关键在于 Morse 引理给出的表示**：$V$ 属于

$$\mathcal{V} = \Big\{ V\in C^\infty(\mathbb{R}^n;\mathbb{R}_{\ge0}) \ \Big|\ \exists\varphi\in\mathrm{Diff}^\infty(\mathbb{R}^n;\mathbb{R}^n):\ V(\varphi^{-1}(y)) = \inf_{y'}\! V(y') + \tfrac12\|y\|^2 \Big\}$$

⇒ **$V = \|\varphi\|^2$，$\varphi$ 是微分同胚** ⇒ **"学 Lyapunov 函数"被重述为"学一个微分同胚"** ⇒ **可逆神经网络的用武之地**。论文还指出（Remark 3）：吸引域 $B^\*$ 与 $\mathbb{R}^n$ 微分同胚 ⇒ 讨论可平移到一般流形。

### 3.3 核心模块 2：需要控制的拓扑 —— Sobolev 范数（§3）

对 $C^k$-光滑映射 $h:\mathbb{R}^d\to\mathbb{R}^n$，定义

$$\|h\|_{C^k(\Omega)} := \sup_{i\in\mathbb{N}[0,k]}\sup_{x\in\Omega}\|D^i h(x)\|.$$

🔴 **本节的论证要点**（论文原话大意）：**仅 $C^0$ 控制是无用的** —— $\|V-V_\theta\|_{C^0}(K)<\epsilon$ 不蕴含 $L_f V_\theta<0$；**$L_p$ 范数与 sup 范数都不控制 $C^k$ 范数（$k>0$）** ⇒ **在 Lyapunov 语境下几乎无用**。论文也点出 Harting et al. (2026) 考虑过 $C^1$ 控制，但**不是通过构造，而是诉诸通用逼近结果**。

### 3.4 核心模块 3：INN 的 $W^{r,\infty}$ 通用逼近（Theorem 4，引自 Ishikawa et al. 2023, Thm 44）

设 $G\subseteq\mathrm{Lip}_r(\mathbb{R}^n;\mathbb{R}^n)$（$C^r$-光滑全局 Lipschitz 映射），令 $\varphi_g$ 为 $\dot y=g(y)$ 的流，定义

$$\Phi(G) := \{x\mapsto \varphi^{1}_g(x): g\in G\}, \qquad \mathrm{INN}_{\Phi(G)} := \{W_1\circ\varphi_1\circ\cdots\circ W_k\circ\varphi_k : k\in\mathbb{N}_{\ge1},\ \varphi_i\in\Phi(G),\ W_i\in\mathrm{Aff}(\mathbb{R}^n)\}.$$

**Theorem 4**：若 $b_G\subseteq\mathrm{Lip}_r$ 是 $\mathrm{Lip}_r$ 的 $W^{r,\infty}$ 通用逼近器，则 $\mathrm{INN}_{\Phi(b_G)}$ 是 $\mathrm{Diff}^{\max\{r,1\}}$ 的 $W^{r,\infty}$ 通用逼近器。

**具体含义**（Eq. 5）：给定任意 $\varphi\in\mathrm{Diff}^{\max\{r,1\}}$、紧集 $K$、$\epsilon>0$，存在 $k$ 使得

$$\|\varphi - W_1\circ\varphi_1\circ\cdots\circ W_k\circ\varphi_k\|_{K,r,\infty} < \epsilon, \qquad \|\cdot\|_{K,r,\infty} := \sum_{|\alpha|\le r}\sup_{x\in K}\|D^\alpha\varphi(x)-D^\alpha\varphi_\theta(x)\| < \epsilon .$$

论文特别说明：**$b_G$ 的合适选择见 Ishikawa et al. (2023) p.19，例如"有限权重 + tanh 激活的 MLP"**（因为通常 $r>0$）。此外，通过与合适的 $W_0\in\mathrm{Aff}$ 复合，**可强制 $\varphi_\theta(0)=0$**（此时 $\|V_\theta\|$ 的平移误差 $<\epsilon+\|b_0\|<2\epsilon$）。

### 3.5 核心模块 4：稳定性证书的逼近（Theorem 5 + Corollary 7）

**Theorem 5**（超双曲平衡点的 Morse Lyapunov 函数通用逼近）：设 $f\in HS^r_0(\mathbb{R}^n)$，$n\ge2$。则

- **(i)** 存在 $V\in\mathcal{V}$ 使 $PA+A^\top P\prec0$（$A:=Df(0)$，$P:=\nabla^2V(0)$），从而证明 $0$ 是 GAS；
- **(ii)** 对任意紧邻域 $K$，存在 $\epsilon_K>0$，使得只要某个 $\varphi_\theta\in\mathrm{Diff}^1$ 满足 $\|\varphi_\theta-\varphi\|_{K,1,\infty}<\epsilon_K$，则 $V_\theta(x):=\|\varphi_\theta(x)-\varphi_\theta(0)\|^2$ **就是 Lyapunov 函数**，在 $K\setminus\{x^\*\}$ 上满足 $L_fV_\theta<0$，从而证明 $\{x\in K: V_\theta(x)\le \min_{x\in\partial K}V_\theta(x)\}$ 上的全局渐近稳定；
- **(iii)** 这样的 $\varphi_\theta$ 可由 INN 生成。

🔴 **证明的技术核心（值得记住）**：把 $K$ 分成 $B_\varrho(0)$ 与 $W:=K\setminus B_\varrho(0)$ 两块，**分别控制**：

- 在 $W$ 上：$L_fV_\theta = L_fV + (L_fV_\theta - L_fV)$，用 $D\varphi^\top\varphi - D\varphi_\theta^\top\varphi_\theta = (D\varphi-D\varphi_\theta)^\top\varphi + D\varphi_\theta^\top(\varphi-\varphi_\theta)$ 分解，得

$$|L_fV_\theta - L_fV| \le 2\|f\|_{W,\infty}\|D\varphi-D\varphi_\theta\|_{W,\infty}\|\varphi\|_{W,\infty} + 2\|f\|_{W,\infty}\|\varphi-\varphi_\theta\|_{W,\infty}\big(\|D\varphi\|_{W,\infty}+\|D\varphi_\theta-D\varphi\|_{W,\infty}\big),$$

只要 $\|\varphi-\varphi_\theta\|_{W,1,\infty}\le\epsilon$ 且 $\epsilon$ 满足一个显式的二次不等式，就有 $|L_fV_\theta-L_fV|\le\beta$，从而 $\max_W L_fV_\theta\le-\beta<0$。

- 在 $B_\varrho(0)$ 上：利用 $P=\nabla^2V(0)=2D\varphi(0)^\top D\varphi(0)$ 与 Lyapunov 不等式 $PDf(0)+Df(0)^\top P\prec0$，得存在 $\varrho>0$、$\kappa_\varrho>0$ 使 $L_fV(x)<-2\kappa_\varrho\|x\|^2$；再用中值定理得 $|L_f(V_\theta-V)(x)|\le 2\kappa_f\epsilon'(2\kappa_4+\epsilon')\|x\|^2$，最终 $\epsilon_K=\tfrac12\min\{\epsilon,\epsilon'\}$。

**Corollary 7**（**最实用的版本**）：若存在 $V\in\mathcal{V}$ 证明 GAS（**不要求双曲**），则对任意紧邻域 $K$ 与 $\varrho>0$（$B_\varrho(0)\subset\mathrm{int}(K)$），存在 $\epsilon_{K,\varrho}>0$ 使得 $\|\varphi_\theta-\varphi\|_{K,1,\infty}<\epsilon_{K,\varrho}$ 时，$V_\theta$ 是 Lyapunov 函数，**在 $K\setminus B_\varrho(0)$ 上满足 $L_fV_\theta<0$**。

🔴 **论文对 Theorem 5 与 Corollary 7 的区分（重要的诚实性）**："为了把 $B_\varrho(0)$ 也包括进来，需要对 $V$ 有更多**局部**控制（Lyapunov 方程 $PA+A^\top P\prec0$）。**这样的 $V$ 总是存在，但我们没有通过 $\mathrm{INN}_{\Phi(b_G)}$ 强制这个结构**。" ⇒ **Corollary 7 给出的是"实践上最相关的保证"，Theorem 5 的完整保证需要额外的结构约束。**

### 3.6 核心模块 5：PŁ 函数的等价性（Remark 8）

**在紧集上，光滑 PŁ Lyapunov 函数与 Morse Lyapunov 函数是等价的**（Rebjock & Boumal 2025 Cor 2.17 / Boumal et al. 2026 Thm 1.1 给出 C² 光滑 PŁ + 唯一临界点 ⇒ Morse；反方向在紧集上成立）。具体地，若 $V$ 是 Morse，则在 $x^\*=0$ 附近 $V(x)=\tfrac12\langle x,Px\rangle+o(\|x\|^2)$、$\nabla V(x)=Px+o(\|x\|)$，从而在 $B_\varrho(0)$ 上有 $\|V(x)\|\le 2\lambda_{\max}(P)\|x\|^2$ 且 $\|\nabla V(x)\|^2\ge\tfrac12\lambda_{\min}(P)\|x\|^2$ ⇒ $\|\nabla V(x)\|^2\ge\mu_\varrho V(x)$，$\mu_\varrho:=\tfrac18\lambda_{\min}(P)^2/\lambda_{\max}(P)>0$。

⇒ **PLNet 的 $\nu_1,\nu_2$（双 Lipschitz）约束里的畸变参数 $\tau:=\nu_2/\nu_1$**（对 $V_\theta$ 则是 $(\nu_2/\nu_1)^2$）**控制 $\nabla^2V_\theta(0)$ 的条件数上界 $\tau^2$** ⇒ **"找 $\tau$"就是"找可用的条件数"**。

### 3.7 数值实验（§6）

**实验 A**（系统 Eq. 4：$\dot x_1 = x_2 - x_1(x_1^2+x_2^2)$，$\dot x_2=-x_1-x_2(x_1^2+x_2^2)$，其 Morse Lyapunov 函数为 $V=x_1^2+x_2^2$，$\dot V=-2\|x\|^4$）：四种架构参数量大致相同 —— **NODE-INN**（3 层 + tanh 生成 Lipschitz 向量场，RK4 积分）、**Coupling-INN**（仿射耦合流）、**PLNet (BiLipNet)**、**MLP**（tanh，2 隐层 20 神经元）。训练 512 样本，margin $m=10^{-3}$，lr $10^{-2}$，Adam 500 次迭代，**展示 100 次独立 run 的结果**。损失是尺度无关的

$$\ell(\theta) := \frac1N\sum_{i=1}^{N}\max\big(0,\ L_fV_\theta(x_i)\,V_\theta(x_i)+m\big).$$

🔴 **论文的关键结构性说明**：**对 NODE-INN 生成的任何 $V_\theta$，正定性、强制性、无虚假临界点都"由构造保证"**（因为它是可逆映射的范数平方）；**只有 $L_fV_\theta<0$ 这条需要泛化到样本之外**。**而对 MLP，所有 Lyapunov 性质都需要泛化。**

**结果**：NODE-INN 与 PLNet 行为相似，**但 PLNet 更快**（每次迭代与收敛到总迭代数两方面都更快）。

**实验 B**（非线性系统与线性双曲系统**光滑共轭**：$\dot x=Ax$，$A=\begin{pmatrix}-1&k\\0&-1\end{pmatrix}$，$J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$，$y=h(x)$ 为 $\dot x=\beta\|x\|^2Jx$ 的 time-1 流 ⇒ $h$ 是 twist；$V(x)=\|x\|^2$ 满足 $\langle x,\beta\|x\|^2Jx\rangle=0$）：**NODE-INN 表现好，其他架构似乎抗拒收敛**。论文随后说明 BiLipNet 需要选 $\nu_1,\nu_2$，**畸变 $\tau$ 需调**（Figure 3 展示了 $\tau=2,16,256,4096$ 的影响）⇒ 论文称"这与 Remark 8 相关：PLNet 需要找合适的 $\tau$，而这个数可能很大（参见 $\mu_\varrho$）；同时这种搜索空间的限制可能有助于收敛"。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| GAS 条件 | (i) 强制性；(ii) $V^{-1}(0)=\{x^\*\}$；(iii) $L_fV(x)=\langle\nabla V(x),f(x)\rangle<0$ | 光滑 Lyapunov 函数的三个条件（论文采用 smooth/strict/proper 的用法） |
| Eq. 1 | $\dot x=f(x),\ x\in\mathbb{R}^n$ | $C^r$-光滑动力系统；$S^r_0$ = 使 $x^\*=0$ GAS 的向量场集合 |
| Eq. 2 | $\dot x_1=-\epsilon x_1+x_2,\ \dot x_2=-x_1-\epsilon x_2$ | **反例系统**：原点 GAS 但**不存在 Morse Lyapunov 函数**（线性化的虚部不可对角化） |
| Eq. 4 | $\dot x_1=x_2-x_1(x_1^2+x_2^2),\ \dot x_2=-x_1-x_2(x_1^2+x_2^2)$ | **存在 Morse Lyapunov 函数的典型非双曲例子**；$V=x_1^2+x_2^2$，$\dot V=-2\|x\|^4$ |
| Prop. 1 | 若 $Df(0)$ 有虚轴上非半单特征值，则任何 $a_1\|x\|^p\le V(x)\le a_2\|x\|^p$ 的 $C^0$ 函数都不是 Lyapunov 函数 | **存在性阻碍**：证书结构本身受限，不是参数化问题 |
| $\mathcal{V}$ | $\{V:\ \exists\varphi\in\mathrm{Diff}^\infty,\ V(\varphi^{-1}(y))=\inf V+\tfrac12\|y\|^2\}$ | Morse Lyapunov 函数 = **微分同胚的范数平方** ⇒ "学证书" ≡ "学同胚" |
| $C^k$ 范数 | $\|h\|_{C^k(\Omega)}:=\sup_{i\le k}\sup_{x\in\Omega}\|D^ih(x)\|$ | **真正需要控制的拓扑** |
| 反例 | $\|\tilde h\|_{C^0}\le\epsilon$ 但 $\|h+\tilde h\|_{C^1}\ge h'(0)+1/\epsilon$ | 🔴 **sup 逼近不蕴含 Lyapunov 条件** |
| $\mathrm{INN}_{\Phi(G)}$ | $\{W_1\circ\varphi_1\circ\cdots\circ W_k\circ\varphi_k:\ \varphi_i\in\Phi(G),\ W_i\in\mathrm{Aff}\}$ | 可逆神经网络族（NODE 流 + 仿射层） |
| Thm. 4 / Eq. 5 | $\|\varphi - W_1\circ\cdots\circ W_k\varphi_k\|_{K,r,\infty}<\epsilon$ | **INN 的 $W^{r,\infty}$ 通用逼近**（Ishikawa et al. 2023） |
| Thm. 5(i) | $PA+A^\top P\prec0,\ A=Df(0),\ P=\nabla^2V(0)$ | 局部 Lyapunov 方程 ⇒ 双曲平衡点处存在 Morse Lyapunov 函数 |
| Thm. 5(ii) | $\|\varphi_\theta-\varphi\|_{K,1,\infty}<\epsilon_K \Rightarrow L_fV_\theta<0$ on $K\setminus\{x^\*\}$ | **逼近 ⇒ 合法稳定性证书** |
| Eq. (7) | $\kappa_1=\|f\|_{W,\infty},\ \kappa_2=\|\varphi\|_{W,\infty},\ \kappa_3=\|D\varphi\|_{W,\infty}$；$\epsilon\le-(\kappa_2+\kappa_3)+\sqrt{(\kappa_2+\kappa_3)^2+2\beta/\kappa_1}\over2$ | $W$ 上 $|L_fV_\theta-L_fV|\le\beta$ 的显式阈值 |
| Remark 8 | $\|\nabla V(x)\|^2\ge\mu_\varrho V(x)$，$\mu_\varrho=\tfrac18\lambda_{\min}(P)^2/\lambda_{\max}(P)$ | 紧集上 PŁ 与 Morse 等价；**$\tau$ 控制 $\nabla^2V_\theta(0)$ 的条件数上界 $\tau^2$** |
| 损失 | $\ell(\theta):=\frac1N\sum_i\max(0,\ L_fV_\theta(x_i)V_\theta(x_i)+m)$ | 尺度无关的训练目标 |

## 5. 关键成果与贡献

### 5.1 主结果

1. **Proposition 1（存在性阻碍）** —— 局部齐次化的 Lyapunov 函数在"虚轴上非半单特征值"情形下**不存在**。这与 Ahmadi et al. (2011)（无多项式 Lyapunov 函数）、Jongeneel & Schwan 2024（无凸 Lyapunov 函数）、Ahmadi & El Khadir 2018（无局部实解析 Lyapunov 函数）**同属"存在性被阻断"的谱系**，但本文给的是**局部齐次性这一族**的阻碍。
2. **Theorem 2 + Morse 引理** —— **双曲平衡点 ⇒ 一定存在 Morse Lyapunov 函数，且可写成 $\varphi$ 的范数平方** ⇒ 论证了"用可逆神经网络学稳定性证书"这条路线**在结构上是有依据的**。
3. **Theorem 4（转引）** —— **INN 是 $\mathrm{Diff}^{\max\{r,1\}}$ 的 $W^{r,\infty}$ 通用逼近器** ⇒ 把"学同胚"这件事的误差控制权从 $L_p$/sup **升级到 Sobolev**。
4. **Theorem 5 / Corollary 7** —— **在明确的 $\epsilon_K$ 阈值下，"学到的 $\varphi_\theta$"真的产生一个合法的严格下降 Lyapunov 函数**（Theorem 5 覆盖 $B_\varrho(0)$ 内，Corollary 7 覆盖 $K\setminus B_\varrho(0)$）。
5. **Remark 8** —— **紧集上 PŁ 与 Morse Lyapunov 等价**，把 PLNet 的 $\tau$ 解释为"条件数上界"。

### 5.2 机制分析（实验 A/B 的关键结构性观察）

🔴 **"哪些性质由构造保证、哪些需要泛化"是本篇最实用的一句话**：

> 对 NODE-INN 生成的 $V_\theta$：**正定性、强制性、无虚假临界点由构造保证**（可逆映射的范数平方）；**只有 $L_fV_\theta<0$ 需要泛化到样本之外**。
> **对 MLP：所有 Lyapunov 性质都需要泛化。**

⇒ 这直接解释了实验结果：**PLNet 更快**（BiLipNet 的双 Lipschitz 结构天然给出"无虚假临界点 + 正定性"），而 **NODE-INN 在强共轭的非线性系统（实验 B）上抗收敛能力更强**（因为它能表示**非线性同胚**，而 $\mathrm{Aff}$ 复合 + Lipschitz 流的结构受 $\tau$ 的条件数约束）。

### 5.3 贡献清单

1. **一条口径纠偏**：**通用逼近必须指定范数**；PolarNet / Teshima et al. 的"$L_p$/sup 逼近"**不足以支撑 Lyapunov 结论**（论文点名 Zhong et al. 2026 与 Teshima et al. 2020）。
2. **一条"存在性 vs 参数化"的分离**：即便存在性被 Proposition 1 阻断，也不是"换个参数化就能救"的问题。
3. **一条可操作的工具链**：$W^{r,\infty}$ 误差界 + 显式阈值 $\epsilon_K$ ⇒ **"学出来的证书是否合法"变成可判定的量**。
4. **一条结构先验**：把 Lyapunov 函数写成**可逆映射的范数平方**，把"正定性/强制性/无虚假临界点"变成**构造性保证**而不是 loss 里的正则项。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **Theorem 5 只能处理 $n\ge2$**（依赖 Ishikawa et al. 2023）；$n=1$ 平凡（$V=x^2$ 总可用）。
- **Corollary 7 不能覆盖 $B_\varrho(0)$** —— "我们没有通过 $\mathrm{INN}_{\Phi(b_G)}$ 强制这个结构" ⇒ **实践上最相关的保证是"实践渐近稳定"（practical asymptotic stability）**，不是全局的。
- **Theorem 5(ii) 只保证 $K$ 内的吸引域**，不是全局 ⇒ 论文明确写 "certifying **global** asymptotic stability on $\{x\in K: V_\theta(x)\le\min_{x\in\partial K}V_\theta(x)\}$" —— **这是"区域吸引域"意义上的全局**。
- **Proposition 1 的推广留给未来**："可以继续把 (3) 推广到 $K^\infty$ 函数的子集，**我们把这以及完整刻画留给未来工作**"。
- **PŁ 函数的利用留给未来**："Exploiting this is future work"。
- **PLNet 的 $\tau$ 的精确刻画留给未来**："A precise characterization is left for future work"。
- **结论段的两个留白**："**补全逆理论（对其他架构）、以及给出严格的数值分析，都留给未来工作**"。
- **数值部分自陈是"illustration"**：只为可视化而用平面系统；且实验 B 只展示曲线与训练损失，**没有报告"最终 $V_\theta$ 是否真的满足 $L_fV_\theta<0$ 的全域验证数字"**。

### 6.2 隐含局限（本仓视角）

- 🔴 **本篇没有给出"用神经网络学稳定性证书"的实验结果** —— 只有**平面系统上的等高线图与训练损失曲线**，**没有数字化的证书验证**（没有 $\sup_x L_fV_\theta$ 的值、没有吸引域面积的数字、没有与其它方法的量化对比）⇒ **理论部分很强，实验部分只是示意**。
- 🔴 **训练损失的尺度无关形式 $\max(0, L_fV_\theta \cdot V_\theta + m)$ 有量纲问题** —— $L_fV$ 是"沿轨的 $V$ 变化率"，$V$ 是"能量"，两者相乘再与 margin $m$ 比较，**不同状态量级下 $m$ 的意义会变** ⇒ 论文未讨论这个量纲问题。
- 🔴 **$n\ge2$ 的限制 + 只在平面系统验证** ⇒ 对高维系统（本仓时序模型的状态维度常常是几十到几百）的可迁移性**完全没有证据**。
- ⚠️ **单作者、PMLR 模板占位符 `vvv`、作者自陈 "Currently under review"** ⇒ **尚未经过同行评审**。
- ⚠️ **论文的 AI 使用声明**：写明用了 **Claude Fable 5.2** 辅助理解文献、实现代码与实验。
- ⚠️ **多处结论依赖作者的伴生论文**（Jongeneel 2026b 的 Thm 3.1 / Prop 3.5 / Cor 3.9）⇒ **本文的可独立核查性受限**（伴生论文本仓未研读）。

### 6.3 未来方向

- 把 Proposition 1 推广到 $K^\infty$ 函数的完整刻画。
- 显式利用 PŁ 结构（Remark 8）。
- PLNet 畸变参数 $\tau$ 的精确刻画。
- 补全逆理论（对 NODE 之外的架构）与严格的数值分析。

## 7. 对本仓的意义

**🔴 本篇对本仓的价值几乎全在"**它说明本仓缺了什么**"，而不在"它给了本仓什么模型"。**

**(a) 🔴 一条本仓至今未被检查的隐含假设**

本仓所有关于"液态/连续时间模型学到了更好的动力学"的论述，其证据链都是 **MSE / 预测精度**（NR-01…NR-09 全部是这个口径）。**本仓从未为任何液态模型产出过一个"稳定性证书"** —— 没有 $L_fV<0$ 的验证，没有吸引域，没有 Lyapunov 函数。

本篇给出的判据是：**一个"逼近"结论在稳定性这类任务上必须带上它所控制的范数**。⇒ **对本仓的直接含义**：本仓若要说"LTC/CfC 学到了更接近真实动力学的状态演化"，**光有预测 MSE 不够** —— 至少还需要一个**动力学的正确性证据**（Lyapunov 证书、或者吸引域 / 可观性 / 轨迹收敛性）。本篇把这条要求从"未说"变成"可执行"。

**(b) 一个本仓完全缺失的实验类型：稳定性证书诊断**

本仓已研读的 `2603.27058`（MDN 模仿学习）、`2606.28827`（temporal dropout）、`2607.05280`（Neural CDE 线性化）、`2611.02340`（连续时间系统辨识与镇定）都在不同角度触碰"连续时间系统的稳定性/镇定"，但**本仓自己一次都没有为液态模型构造过稳定性诊断**。

⇒ **一条新的消融轴（"证书轴"，成本中等，证据价值高）**：对本仓的 LTC/CfC，构造一个标量 Lyapunov 候选 $V(x)=\tfrac12\|x\|^2$（或学一个小的 $V_\theta$），**在观测点上计算 $L_fV = \langle\nabla V, f_\theta(x,u)\rangle$ 的符号**，并统计"违反 $L_fV<0$ 的时间点比例"。这是一个**只用一次前向 + 一次反向**的诊断（$f_\theta$ 的导数可用 `torch.func.jvp`），但能回答本仓至今没回答过的问题：**"本仓的液态模型学到的向量场，在多大比例的输入上是收缩的？"** —— 这比单看 MSE 强得多，因为它直接检验"连续时间动力学"这个机制是否真的被学成"动力学"。

⚠️ **必须同时记录本篇的限定**：即便 $L_fV<0$ 在观测点成立，**按 Proposition 1 与本篇的 $C^1$ 分析，这也不构成任何稳定性保证**（需要 $\varphi_\theta$ 的 Sobolev 逼近、$Df(0)$ 的双曲性等）⇒ **本仓的证书轴只能作为"诊断指标"，不能作为"稳定性证明"**。这条限定必须写在报告里。

**(c) 与本仓既有工作的交叉**

本仓 11-13 研读的 `2006.12070`（Lipschitz RNN，**全局指数稳定性 + 对称-反对称分解**）与本篇是**同一条线的两端**：
- `2006.12070` 从**谱**角度给出**充分条件**（特征值实部 ⇒ 指数稳定），但**不保证学到的模型满足该条件**；
- 本篇从**证书构造**角度给出**必要与充分的存在性刻画 + 可学性工具**，但**只覆盖 $n\ge2$、只给区域吸引域、只做平面示意**。

⇒ **两篇合起来，本仓可以第一次说清楚"全局稳定性"这条线上的完整图景**：**谱条件（充分）与 Lyapunov 证书（充要）都已知，但"用神经网络学到满足条件的模型"这一环在两边都缺** ⇒ **这是一个明确的、本仓可以填的空白**，且是一个**方法论空白而非工程空白**。

**(d) 对本仓"变体库"的一条负面提醒**

本篇的 Theorem 5 与 Corollary 7 都建立在 **"学一个微分同胚"** 的参数化上 ⇒ **正定性、强制性、无虚假临界点由构造保证**。⇒ **一条对本仓 `lnn/core/variants.py` 的直接启示**：本仓的变体（多 $\tau$、加性 τ、门控强度、SnCP 稀疏接线……）**全部是"给一个已学好的向量场加约束"**，而不是"用构造保证某种性质" ⇒ **本仓至今没有一个变体具备"构造性保证"**（例如：把状态转移写成**可逆映射**的形式，从而让某个物理性质自动成立）。**这是一个全新的、与"多 $\tau$"正交的变体方向**，本轮不擅自实现，但应记入候选。

**Verdict**：
- **"用神经网络学 Lyapunov 函数需要控制 Sobolev 范数，sup/Lp 逼近不足" → TARGET-POSITIVE**（论证严谨、有显式反例（$\epsilon\sin(x/\epsilon^2)$）、直接点名并纠正了两个被广泛引用的结论（Zhong et al. 2026 的 PolarNet 主张、Teshima et al. 2020 的泛用性）⇒ **方法论级结论，本仓必须采纳**）。
- **"NODE-based INN 能学出合法的 Morse Lyapunov 函数" → TARGET-POSITIVE-WITH-NUANCE**（Theorem 4/5/Corollary 7 的推导完整且带显式阈值；**但**：(a) Theorem 5 需 $n\ge2$；(b) Corollary 7 **不覆盖 $B_\varrho(0)$**；(c) **数值实验只是平面系统的等高线示意，没有任何量化的证书验证数字**；(d) 依赖作者的伴生论文；(e) 未评审）。
- **"存在性障碍（Proposition 1）" → TARGET-POSITIVE（作为负向知识）** —— **对所有学稳定性证书的连续时间模型都是一条硬约束**：$Df(0)$ 虚轴非半单 ⇒ 连局部齐次证书都没有 ⇒ **引用任何"我们的模型学到了稳定的动力学"之前，必须先检查这一条件**。
- **对本仓液态路线（LTC / CfC）本身 → 不适用（族外，但有强方法论含义）** —— 本篇**不是**液态家族的证据（无 LTC/CfC 对比）；**但它给本仓一条必须补上的证明义务**：**连续时间机制的"正确性"不能只用预测精度来陈述**。
