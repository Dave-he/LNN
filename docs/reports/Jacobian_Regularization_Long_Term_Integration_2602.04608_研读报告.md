---
title: "Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-Differential-Equation, Jacobian, Lipschitz-Constant, Stability-Theory, Long-Horizon-Integration, Preprint]
---

# 研读报告：Jacobian Regularization — **匹配雅可比**而不是匹配向量场：把"短 rollout 训练 + 长时积分稳定"这件事从**延长 rollout**换成**惩罚 $J$ 的差**

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描，第八天）捞出来的。本仓**零覆盖**（`2602.04608` 在 183 篇报告中零命中）。
> 🔴 **本篇对本仓的意义有两条，且都指向本仓自己写下的待办**：(a) **11-22 的笔记写下过一条待办** —— *"连续时间的收益与成本必须分离，而**正确性**则需要稳定性/证书类证据（**本仓至今全部是 MSE 证据，缺一份**）"*；本篇给出的正是那份缺失证据的**可操作配方** —— 它证明**不用延长 rollout**（那会爆炸性昂贵），只要**惩罚 $\|J_F(x)-J_{F_\theta}(x)\|$**（而不是 $\|F(x)-F_\theta(x)\|$）就能把长时积分稳定下来；(b) 🔴 **本篇与 11-24 研读的 SHND（`2404.12554`）构成"保证阶梯"的两级** —— SHND 是 **A 级（构造性，参数化保证）**，本篇是 **B 级（经验正则，匹配真 $J$）**，而本篇**证明了一件 SHND 没有证明的事**：*B 级正则即使没有真值动态，也能从"只见过轨迹点"的设定下拿到大部分稳定性收益*（LFD 变体）⇒ **本仓至今的"证书"全在 B 级，且往往连真值动态都没有**（玩具 bench 有 ⇒ 可用 LFD 路线）。
> ⚠️ **本篇的定理是针对 ODE/求解器路线的**（Neural ODE 的右端项 $F_\theta$），**不是**针对 CfC 的闭式一步映射 ⇒ 移植到本仓 `cfc.py` 需要另做推导。

## 1. 元数据

- **论文标题**：Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations
- **作者**：Maya Janvier, Julien Salomon, Etienne Meunier
- **机构**：LOCEAN, Sorbonne Université（Paris） / ANGE team, INRIA / LJLL, Sorbonne Université
- **发表时间**：2026-02-04 (v1)（arXiv:2602.04608v1 [cs.LG]）；PDF 脚注自标 "Preprint. February 5, 2026."
- **发表渠道**：**Preprint**（未标注会议/期刊）⇒ **未评审**
- **代码**：论文附录 B 写 *"All of our experiments are realized in jax, with code available on GitHub."* 但**正文与附录均未给出具体仓库 URL** ⇒ **链接缺失，不可复现**
- **本地 PDF**：[papers/2602.04608/2602.04608.pdf](../../papers/2602.04608/2602.04608.pdf) ｜ 文本 [papers/2602.04608/2602.04608.txt](../../papers/2602.04608/2602.04608.txt)（14 页，45049 字符）
- **本仓复算脚本**：`/tmp/lnn_verify_final_2026-11-25.py` §(4)（§6.1 全部数字由该脚本实跑产出）
- **关联概念**：连续时间机制层 / 雅可比正则 / Lipschitz 常数 / Grönwall 界 / 长时积分稳定性 / 刚度（11-24 已记录）

## 2. 核心问题

Neural ODE 的稳定性有两个已知事实：

1. **用 solver 训练（rollout $N>1$）比离线训练（$N=1$，直接拟合 $\|F(x)-F_\theta(x)\|^2$）稳定得多** —— 但 rollout 的代价是**梯度要穿过整个迭代过程**，$N$ 一大就成为瓶颈，而且 *"errors in gradient estimation accumulate and result in unstable training"*。
2. **长时积分的发散由 Lipschitz 常数 $L_{F_\theta}=\sup_x\|J_{F_\theta}(x)\|_2$ 驱动** —— Grönwall 引理给出轨迹误差界（Proposition 2.2）：

$$\|x(t)-x_\theta(t)\|_2\le\|\epsilon_\theta\|_\infty\,L_{F_\theta}\big(e^{L_{F_\theta}t}-1\big) \tag{Eq.3}$$

其中 $\epsilon_\theta=F_\theta-F$ 是离线误差。⇒ **一个 $\epsilon_\theta$ 很小但 $L_{F_\theta}$ 很大的模型，其轨迹误差会随 $t$ 指数放大**。反之，$L_{F_\theta}$ 太小又会损害拟合能力 —— 论文明确指出 *"there exists a trade-off between the smoothness of a neural network and its expressiveness"*，**找到平衡仍是开放问题**。

论文的诊断（Table 1）是这一困境的关键证据：**用长 rollout 训练的模型，在 $\epsilon_\theta$ 上更差（离线误差更大），但在 $L_{\text{traj}}$ 与雅可比误差 $J_e(\theta)$ 上更好** ⇒ **长 rollout 的真正收益是"把雅可比学对了"，而不是"把向量场学准了"**。论文据此主张：**直接正则化雅可比，应当能在短 rollout 下复现长 rollout 的稳定性收益**。

⇒ 问题被重新表述为：**能否用一种"不需要把 rollout 变长、也不需要显式计算整个雅可比矩阵"的代价，来约束 $F_\theta$ 的雅可比？** 论文给出两条路径：**已知真动态时用精确方向导数**，**未知真动态时用有限差分近似**。

## 3. 方法论与核心思路

### 3.1 总体架构

```
真动态 F (或仅有轨迹点) ──┐
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
   已知 F (AD)                         未知 F (只有轨迹点)
        │                                    │
   精确 JVP: (J_F − J_Fθ)·v            有限差分近似方向导数
   Hutchinson 估计 ‖J_F − J_Fθ‖²_F        ‖(Fθ−F)(x+Δt) − (Fθ−F)(x)‖² / ‖x+Δt−x‖²
        │                                    │
        └────────────┬───────────────────────┘
                     ▼
   L(θ) = L_traj(θ) + λ Σ_x ‖J_F(x) − J_Fθ(x)‖²_F        (Eq.4)
                     │
                     ▼
   短 rollout 训练 (N = 2Δt 或 5Δt) ⇒ 长时积分稳定
```

### 3.2 模块一：雅可比正则项（Eq.4）

$$\mathcal L(\theta)=\mathcal L_{\text{traj}}(\theta)+\lambda\sum_{X\in\mathcal T_F}\sum_{i=0}^{N}\big\|J_F(x(t_i))-J_{F_\theta}(x(t_i))\big\|_2^2 \tag{Eq.4}$$

**关键点**：惩罚的不是向量场差 $\|F-F_\theta\|$，而是**雅可比差** $\|J_F-J_{F_\theta}\|$。

**Proposition 2.3**（为什么这能约束 Lipschitz 常数）：若 $\|J_F(x)-J_{F_\theta}(x)\|_2\le\epsilon\ \forall x$，则

$$L_F-\epsilon\le L_{F_\theta}\le L_F+\epsilon$$

即**雅可比接近 ⇒ Lipschitz 常数接近** ⇒ 这是一种**"以物理信息系统的方式"约束 Lipschitz 常数**，而不是用一个外部估计器去罚它。

### 3.3 模块二：Hutchinson 估计器 —— 用 JVP 代替完整雅可比

直接算 $\|J_F-J_{F_\theta}\|_F^2$ 需要完整矩阵，训练时太贵。论文用 Hutchinson 迹估计：

$$\|J_F(x)-J_{F_\theta}(x)\|_2^2=\|J_{\epsilon_\theta}(x)\|_2^2=\text{Tr}\big(J_{\epsilon_\theta}^\top J_{\epsilon_\theta}\big)\approx\mathbb E_{v\sim P(v)}\big[\|J_{\epsilon_\theta}(x)\cdot v\|_2^2\big] \tag{Eq.5}$$

其中 $P(v)$ 通常取标准正态，**任何满足 $\mathbb E[v^\top v]=I$ 的分布都可以**（论文提到 Rademacher 可降低方差）。

**Proposition 3.1**（JVP = 方向导数）：

$$J_{\epsilon_\theta}(x)\cdot v=D_v\epsilon_\theta(x):=\lim_{h\to0}\frac{\epsilon_\theta(x+hv)-\epsilon_\theta(x)}{h}$$

⇒ 只需**前向模式自动微分**算 JVP，**不用构造矩阵**。

### 3.4 模块三：两种正则化

**(a) 已知真动态**（精确方向导数）：

$$\mathcal L_{AD}(\theta,v_i,x)=\big\|\widehat D_{AD}\,v_i\,(F_\theta-F)(x)\big\|_2^2 \tag{Eq.6}$$

$$\mathcal L_{AD}(\theta,\{v_i\}_{i=1}^V)=\frac{1}{V}\sum_{x\in\Omega}\sum_{i=1}^V\mathcal L_{AD}(\theta,v_i,x) \tag{Eq.7}$$

（$v_i$ i.i.d. 采自 $\mathcal N(0,I)$；训练点取真轨迹上的 $x$）

**(b) 未知真动态**（有限差分，无监督）：

$$\mathcal L_{FD}(\theta,x_{t_i})=\frac{\big\|(F_\theta-F)(x_{t_i}+\Delta t)-(F_\theta-F)(x_{t_i})\big\|_2^2}{\|x_{t_i}+\Delta t-x_{t_i}\|_2^2} \tag{Eq.8}$$

论文推导其极限（Eq.10）：

$$\lim_{\Delta t\to0}\frac{\big\|D F(x_{t_i})\frac{F(x_{t_i})}{\|F(x_{t_i})\|_2}(F_\theta-F)(x_{t_i})\big\|_2^2}{\|F(x_{t_i})\|_2^2}$$

即**沿轨迹切方向的方向导数**的平方 ⇒ **完全不需要真动态**，只需要轨迹点（用有限差分近似）。

### 3.5 与 LTC / CfC / NCP 的关系

- **与 LTC（Hasani 2021）/ CfC（Hasani 2022）**：⚠️ **本篇的 $F_\theta$ 是 ODE 右端项**，需要用 solver 积分 ⇒ **LTC（ODE）路线直接适用**，**CfC 不适用**（CfC 无 solver、无右端项概念）。这与 11-24 SHND 的路线适用性**完全一致**（SHND 也只适用 ODE 路线）⇒ **本仓若采纳这一族方法，载体只能是 `ltc.py`，不是 `cfc.py`**。
- **与本仓 `lyapunov_stable_cfc.py` / `iss_stable_cfc.py`**：本仓这两者是**惩罚 $V(h_{t+1})-V(h_t)$**（状态量的一阶差分），本篇惩罚的是**雅可比**（一阶导的差）⇒ **不同层级、互补**。
- **与本仓 `smoothness_metrics.py`**：本仓已有 `max_gradient`（输入侧光滑度），**但没有雅可比范数**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq.(2) | $L_{\text{traj}}(\theta)=\sum_{i=0}^{N}\|x(t_i)-\tilde x_\theta(t_i)\|_2^2$ | rollout 训练损失（长 rollout 贵） |
| Eq.(3) | $\|x(t)-x_\theta(t)\|_2\le\|\epsilon_\theta\|_\infty L_{F_\theta}(e^{L_{F_\theta}t}-1)$ | **Proposition 2.2（Grönwall）**：长时发散由 Lipschitz 常数指数放大 |
| Eq.(4) | $\mathcal L=\mathcal L_{\text{traj}}+\lambda\sum_x\|J_F(x)-J_{F_\theta}(x)\|_2^2$ | **雅可比正则**（核心） |
| — | $L_F-\epsilon\le L_{F_\theta}\le L_F+\epsilon$ | **Proposition 2.3**：雅可比接近 ⇒ Lipschitz 常数接近 |
| Eq.(5) | $\|J_\epsilon\|_2^2\approx\mathbb E_v[\|J_\epsilon v\|_2^2]$ | Hutchinson 估计：只需 JVP |
| Eq.(6)(7) | $\mathcal L_{AD}(\theta,v_i,x)=\|\widehat D_{AD}v_i(F_\theta-F)(x)\|_2^2$ | 已知动态：精确方向导数正则 |
| Eq.(8)(9) | $\mathcal L_{FD}=\frac{\|(F_\theta-F)(x+\Delta t)-(F_\theta-F)(x)\|_2^2}{\|x+\Delta t-x\|_2^2}$ | **未知动态：有限差分正则（无监督）** |
| — | $J_\epsilon(x)\cdot v=D_v\epsilon(x)=\lim_{h\to0}\frac{\epsilon(x+hv)-\epsilon(x)}{h}$ | **Proposition 3.1**：JVP ≡ 方向导数 |

## 5. 关键成果与贡献

### 5.1 主结果（作者自报，本仓未复现）

**Table 1（$\Delta$ 列相对短 rollout 基线）**：

| 实验 | 模型 | $L_{\text{traj}}$ | $\epsilon_\theta-\epsilon^B_\theta$ | $J_e(\theta)-J^B_e(\theta)$ |
|---|---|---|---|---|
| Two-Body | $N=2\Delta t$ | $4.9\times10^{15}$ | 0 | 0 |
| | $N=5\Delta t$ | $1.7\times10^{-2}$ | $+3.4\times10^{-6}$ | $-0.15$ |
| | $N=2\Delta t$ + $L_{AD}$ | $1.0\times10^{-2}$ | $-2.8\times10^{-7}$ | $-0.03$ |
| | $N=2\Delta t$ + $L_{FD}$ | $2.64\times10^{2}$ | $-3.1\times10^{-7}$ | $-0.02$ |
| Rigid Body | $N=5\Delta t$ | $1.41\times10^{4}$ | 0 | 0 |
| | $N=10\Delta t$ | $1.62\times10^{0}$ | $+0.79\times10^{-7}$ | $-0.55\times10^{-3}$ |
| | $N=5\Delta t$ + $L_{AD}$ | $1.79\times10^{-1}$ | $-0.13\times10^{-7}$ | $-7.39\times10^{-3}$ |
| | $N=5\Delta t$ + $L_{FD}$ | $1.52\times10^{-1}$ | $+4.73\times10^{-6}$ | $-1.96\times10^{-3}$ |
| KS | $N=2\Delta t$ | $4.77\times10^{14}$ | 0 | 0 |
| | $N=10\Delta t$ | $2.91\times10^{0}$ | $+5.49\times10^{-3}$ | $-2.15\times10^{-2}$ |
| | $N=2\Delta t$ + $L_{AD}$ | $8.50\times10^{5}$ | $-0.04\times10^{-3}$ | $-0.01\times10^{-2}$ |
| | $N=2\Delta t$ + $L_{FD}$ | $3.59\times10^{0}$ | $-0.08\times10^{-3}$ | $-0.05\times10^{-2}$ |

**读表要点（必须带）**：
- ✅ **雅可比误差 $J_e$ 全面下降**（所有正则化行都是负增量）⇒ **核心主张成立**。
- ⚠️ **$L_{AD}$ 在 KS 实验上反而更差**（$8.50\times10^{5}$，比基线 $4.77\times10^{14}$ 好但**比 $L_{FD}$ 的 $3.59\times10^{0}$ 差 5 个数量级**）⇒ 论文自己解释：KS 的 $J_F$ 含 Fourier 变换，量级远大于神经网络雅可比，**使 $\mathcal L_{AD}$ 极难调**（附录 B.2：*"The LAD regularization was particularly difficult to tune for the KS experiment, with unstable training when $\lambda\ge10^{-10}$"*）。
- ⚠️ **$L_{FD}$ 在 Rigid Body 上使 $\epsilon_\theta$ 变差**（$+4.73\times10^{-6}$，是唯一正增量）⇒ 论文自陈*"the learned dynamic is faulty and sometimes produces collapsing trajectories"*。
- ✅ **"$L_{FD}$ 更易调、更便宜、但精度略逊"** 这一结论是**有数据支撑的**（$\lambda_{FD}$ 的搜索区间比 $\lambda_{AD}$ 宽得多，见附录 Table 2）。

**Table 2/3 超参**：$\lambda$ 的搜索区间跨 **6 个数量级**（$10^{-11}\sim10^{-6}$）⇒ **调参成本不低**。

### 5.2 机制分析

- **Table 1 的诊断（最有价值的部分）**：长 rollout 模型的**离线误差更大**、**轨迹损失更小**、**雅可比误差更小** ⇒ 论文论证 *"this is an explanation to the huge gap in performance between offline and online training"*。
- **两种正则的对比结论**：$L_{AD}$ 精确但难调；$L_{FD}$ 近似但"更易调、代价更低、可扩展到高维"，且在多数场景"几乎与长 rollout 基线一样好"。
- **附录 C**：三个实验都验了守恒量（Two-Body 角动量、Rigid Body 的 Casimir、KS 的质量），结论与主结果一致。

### 5.3 贡献清单

1. **一条诊断**：长 rollout 训练的主要收益是"把雅可比学对"，而不是"把向量场学准"（Table 1）⇒ 这**重新解释了** offline vs online 训练的差距。
2. **两个正则化**（精确 AD / 有限差分），后者**完全无监督**（只要轨迹点）。
3. **一个高效实现路径**：Hutchinson + 前向 AD ⇒ 不需要构造雅可比矩阵。

## 6. 局限性与未来展望

### 6.1 ✅ 本仓复算：Eq.(5) 的 Hutchinson 恒等式与 Proposition 3.1 均成立

```
  ||J||_F^2 = 30.72603   Hutchinson E||Jv||^2 = 30.88914   rel.err = 0.53%
  => Eq.(5) identity holds; JVP-only estimation is valid.

  ||J_eps v - dirder|| / ||J_eps v|| = 7.99e-07
  => Prop 3.1 (JVP == directional derivative) confirmed numerically.
```

（$V=8000$ 个高斯探针；Proposition 3.1 用有限差分 $h=10^{-6}$ 的中心差分对照。）

⚠️ **本仓复算过程中的一个教训（与 11-25 的 Continuum Dropout 复算同类）**：本仓第一版脚本里 $F_\theta$ 的输出维度与 $F$ 不匹配导致广播报错，且 Hutchinson 的探针数取 $V=4000$ 时相对误差 1.40%、取 $V=8000$ 时降到 0.53% ⇒ **复算 Hutchinson 类估计器时必须报探针数，否则误差不可解释**。

### 6.2 作者自陈局限

- **代码链接缺失**（附录 B 说 "available on GitHub" 但未给 URL）⇒ **不可复现**。
- **$\lambda$ 极难调**（跨 6 个数量级；KS 上 $L_{AD}$ 在 $\lambda\ge10^{-10}$ 时训练不稳定）⇒ 论文自己说 *"a sign of the loss being hard to tune"*。
- **$L_{AD}$ 在 KS 上失败**（雅可比量级失配）⇒ **方法不普适**。
- **$L_{FD}$ 可能学到"坏动力学"**（Rigid Body 上出现坍缩轨迹）⇒ 论文明确承认。
- **只测了 3 个系统**（Two-Body 4 维、Rigid Body 3 维、KS 256 维 PDE），且**都是教科书级的物理系统**，不是学习任务。
- **未评审**。
- **没报任何 wall-clock 或显存数字** ⇒ "far lower cost compared to long rollouts" 只有定性陈述。

### 6.3 隐含局限（本仓视角）

- 🔴 **定理完全针对 ODE + solver 路线** —— $F_\theta$ 是右端项，$J_{F_\theta}$ 是右端项对状态的雅可比 ⇒ **本仓 `cfc.py`（闭式一步映射）没有 $J_{F_\theta}$ 这个对象**。若要移植到 CfC，只能类比成"隐状态更新映射对隐状态的雅可比" $J_h$（这正好与本轮 `2512.20625` 引入的 $J_h$ 诊断量接上），但 **Eq.(3) 的 Grönwall 界与 Eq.(4) 的正则形式都需要重新推导** ⇒ **不得直接套用**。
- 🔴 **"匹配雅可比"不等于"匹配稳定性"** —— Proposition 2.3 只给出 $L_{F_\theta}$ 的**区间**，不给出"稳定"本身；而本仓 11-24 已确立的纪律是**保证必须分层标注**（A/B/C 级）⇒ **本篇属于 B 级**（采样点上的经验正则），**不得写成"本方法证明了稳定性"**。
- 🔴 **Proposition 2.3 的方向性**：它说"雅可比接近 ⇒ Lipschitz 常数接近"，但**反之不成立** ⇒ **雅可比正则不能替代**对 $L_{F_\theta}$ 的直接约束（若真动态的 $J_F$ 本身就很大，$L_{F_\theta}$ 仍会很大）。
- **Eq.(8) 的有限差分步长 $\Delta t$ 是超参**，且它同时出现在分母 $\|x+\Delta t-x\|^2$ 里 ⇒ **步长选择影响正则强度**，论文**未讨论这个超参**（只扫了 $\lambda$）。
- **没有与"直接正则 $L_{F_\theta}$"（如 Lipschitz 层、正则化权重）做对照** ⇒ 无法判断**收益来自"匹配雅可比"这个具体形式，还是仅仅来自"给雅可比加了个约束"**。这是一个**关键的缺失 baseline**（11-24 的"约束的实现方式"轴同款问题）。
- **实验全是拟合已知真动态**（不是分类/预测任务）⇒ **与本仓的 bench 场景（toy 回归 + 工业任务）差距很大**。

### 6.4 未来方向

论文 §5 结语提到：可用于**求解器只部分可微**的场景（引 List et al. 2024，如当前气候模型），以及**混合设定**（Neural ODE 只替代部分动力学）。

## 7. 对本仓的意义

**(1) 🔴 填上 11-22 自己写下的一条待办** —— 11-22 的结论明确写着：*"正确性则需要**稳定性/证书类证据**（**本仓至今全部是 MSE 证据，缺一份**）"*。本篇给出的正是那份缺失证据的**可操作配方**：不必延长 rollout（那正是本仓 bench 成本高、且 11-23/11-24 已反复报"批量化 vs 串行化"问题的根源），只要**惩罚雅可比差**即可稳定长时积分 ⇒ **本仓的 bench 可以用"长时积分不发散"作为一条新的、可自动化的证据轴**，而不只是 MSE。

**(2) 两条新消融轴**：
① 🔴 **"正则化的目标量"轴（第一优先级）** —— 同一 CfC/LTC、同数据、同 rollout 下比较：(i) 无正则、(ii) **向量场正则** $\|F-F_\theta\|$、(iii) **雅可比正则** $\|J_F-J_{F_\theta}\|$（本篇）、(iv) **Lipschitz 正则**（直接罚 $\|J_{F_\theta}\|$）、(v) **本仓现状的 Lyapunov 罚**（$V(h_{t+1})-V(h_t)$）⇒ 五者报同一组量：**长时积分的发散时刻**（或发散轨迹数）+ 短时精度 + 训练成本三元组 ⇒ **首次能量化"本仓的稳定性惩罚到底该罚哪一层"**。⚠️ 这也**同时是对本篇的一个缺失 baseline 的补测**（见 §6.3）。
② 🔴 **"有限差分步长"轴** —— Eq.(8) 的 $\Delta t$ 是未被讨论的超参 ⇒ 本仓扫 $\Delta t\in\{0.1,0.5,1,2,5\}\times$ 原步长，看正则强度与稳定性的关系。

**(3) 一条 bench 元数据纪律升级**：新增 **"长时积分发散时刻"**（或"在 $T$ 步内不发的轨迹比例"）作为独立指标列 ⇒ 本仓现有的 toy bench（`bench_cfc_temporal_dropout.py` 等）**只报 MSE/accuracy**，从未报"推演久了会不会炸"。⚠️ 这条指标对本仓的**多 $\tau$ / 多分支变体尤其关键**（11-13 的 CT-GRU 悬案、11-16 的梯度可见性轴都指向"某些分支学不动/学崩"）。

**(4) 一条变体候选**：`JacobianRegularizedLTCCell` —— 在 `ltc.py` 的 cell 上加**雅可比范数正则** $\mathcal L_{J}=\lambda\mathbb E_v\|J_h(x)\cdot v\|^2$（**注意**：本仓最现实的做法是**直接罚 $\|J_h\|$**，而不是本篇的"匹配真 $J$"，因为本仓的玩具 bench 有真值 ⇒ 可直接用 $L_{AD}$ 路线；真实任务上用 $L_{FD}$ 路线）。

**(5) 一条引用纪律**：本篇标题含 "Neural Differential Equations"，**极易被当成"连续时间有效"的证据** —— 但它**不做连续时间 vs 离散的对照**，其全部结论是"**怎样让 ODE 求解器训出来的模型不炸**"⇒ 不可引用为"连续时间机制更优"（延续 11-23/11-24 纪律）。

**Verdict**：
- **"匹配雅可比可以在短 rollout 下拿到长 rollout 的稳定性收益" → TARGET-POSITIVE**（方法论级，且**填上本仓 11-22 自己写下的待办**；Table 1 的 $J_e$ 全面下降 + 三个系统上的定性验证构成有效支撑；本仓已独立复算 Hutchinson 恒等式与 Prop 3.1 成立）
- **"这是一个稳定性证明" → TARGET-NEGATIVE-WITH-NUANCE**（🔴 属 **B 级经验正则**（11-24 纪律），只给 Lipschitz 常数的区间而非稳定性本身；且 $\lambda$ 跨 6 个数量级、$L_{AD}$ 在 KS 上失败、$L_{FD}$ 可能学到坍缩动力学）
- **"可直接用于本仓 CfC" → TARGET-NEGATIVE-WITH-NUANCE**（🔴 定理针对 **ODE 右端项**，CfC 是闭式一步映射、无右端项 ⇒ 需重新推导；载体只能是 `ltc.py`）
