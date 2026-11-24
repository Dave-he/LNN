---
title: "Projected Neural Differential Equations for Learning Constrained Dynamics"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-ODE, Hard-Constraints, Projection, Conservation-Laws, Riemannian-Geometry, Energy-Conserving, Preprint]
arxiv_id: 2410.23667
---

# 研读报告：PNDE — 把守恒律从"软惩罚"升级为**构造性硬约束**：以及一篇把"显式约束"与"隐式约束"彻底分开的坐标论证

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描，第七天）捞出来的。本仓**零覆盖**（`2410.23667` 在 180 篇研读报告中零命中）。
> 🔴 **本篇对本仓的意义有三层**：(a) **一个本仓完全没有的结构化先验工具箱** —— **把约束写成代数等式 $g(u)=0$，然后把向量场投影到切空间** ⇒ 约束**由构造保证**，且**不需要**在广义坐标里写方程；(b) **一条与本仓 10-28 研读的 `2012.02334`（能量守恒网络基准）**直接对撞的**分歧点** —— 那篇论文的结论倾向"隐式约束（在守恒坐标里学）更简单可靠"，而本篇的结论恰恰是**"显式约束（在简单坐标里学 + 投影回去）更好，且 N 越大差距越大"** ⇒ **本仓索引里必须把这两篇标成"同一问题的对立结论"**，不能只引一端；(c) **一条本仓必须执行的方法论纪律** —— **"硬约束"与"软约束"的差别不在精度，而在"推理时是否保证满足"**：PNDE 的 $\frac{d}{dt}g(u(t))=Dg(u)[\mathrm{Proj}_u(\bar f_\theta(u))]=0$ 是**恒等式**，而 PINN/物理损失只是"希望"权重落在满足约束处 ⇒ **本仓所有"结构化/物理先验"变体都应被重新归类为软约束**。

## 1. 元数据

- **论文标题**：Projected Neural Differential Equations for Learning Constrained Dynamics
- **作者**：Alistair White†, Anna Büttner, Maximilian Gelbrecht, Valentin Duruisseaux, Niki Kilbertus, Frank Hellmann, Niklas Boers
- **机构**：Technical University of Munich（Munich Climate Center / Earth System Modelling Group）· Potsdam Institute for Climate Impact Research · Caltech · Helmholtz Munich · MCML · University of Exeter
- **发表时间**：2024-10-31 (v1) / **2026-09-09 (v2)**（arXiv:2410.23667v2 [cs.LG]）
- **发表渠道**：**Preprint**（自标 arXiv preprint；v2 日期 2026-09-09 晚于 v1 一年多，说明长期修订中）
- **来源**：arXiv:2410.23667v2 [cs.LG]
- **代码**：论文未给出公开仓库地址（正文与附录均未见 URL）
- **本地 PDF**：[papers/2410.23667/2410.23667.pdf](../../papers/2410.23667/2410.23667.pdf) ／ 文本 [2410.23667.txt](../../papers/2410.23667/2410.23667.txt)
- **页数**：20 页（PDF 16.95 MB，含大量实验图）
- **关联概念**：Neural ODE / Neural Differential Equation (NDE) / **Universal Differential Equation** / **约束流形 $M=\{u\in E; g(u)=0\}$** / **切空间 $T_uM$** / **Moore–Penrose 伪逆** / **Cholesky / QR 分解** / **Zygote.jl（Julia AD）** / MAGMA / cuSOLVER / **Stabilized NDE (SNDE)** / 守恒律（能量、动量）/ **holonomic 约束** / N-pendulum（Cartesian vs 广义坐标）/ **Lotka–Volterra** / **IEEE 14-bus 电网** / 二维 quartic 势 / 捕食者-被捕食者
- **入选理由**：本仓**零覆盖**"把守恒律做成构造性硬约束"这一整族；而本仓自 10-28 研读 `2012.02334`（十个能量守恒网络的统一基准）以来，一直只有"软约束"这一族可选（PINN 式残差惩罚、Hamiltonian/端口哈密顿结构），**没有"硬约束"工具**。本篇同时给出一个**本仓至今零做的对照维度：约束是隐式的还是显式的**。

## 2. 核心问题

**任务**：从数据学习一个动力系统 $\dot u = f_\theta(u)$，但要求学到的解**始终**落在已知约束集上

$$\mathcal{M}=\big\{u\in E;\ g(u)=0\big\},\qquad g:E\to\mathbb{R}^m,\quad m<n \tag{Eq. 2}$$

（$g$ 是 $m$ 个**独立显式代数约束**，$g_i(u)=0$；假设 $Dg(u)$ 在 $\mathcal{M}$ 上**满行秩**，由 Regular Level Set Theorem 可知 $\mathcal{M}$ 是 $\mathbb{R}^n$ 的光滑 $(n-m)$ 维嵌入子流形）

论文指出 NDE 的核心缺陷是**软约束的四宗罪**（§I）：

1. **复合损失昂贵**：物理损失要算导数；
2. **优化困难**：条件数变差、损失项互相冲突、导数近似不准；
3. **存在伪解（spurious solutions）**：物理损失在配点上被满足，但解不是真解；
4. 🔴 **最致命的一条**：*"they provide **no guarantees** that the physical constraints will be satisfied **at inference time**; at best, they can only hope to guide the model weights toward physically plausible solutions during training."*

并且点名了长程仿真的具体后果：*"in climate modeling, for example, small per-timestep energy conservation errors can accumulate over time and introduce significant biases in the simulated long-term climate."*

于是问题被重述为：**如何在保留"在简单坐标里学"这一工程红利的同时，让约束在推理时被构造性保证？** 论文的答案是**不要去学"守恒的能量函数"，而是把向量场投影到约束流形的切空间** —— 这样**约束与坐标选择解耦**，且不需要把系统写进广义/正则坐标。

## 3. 方法论与核心思路

### 3.1 总体架构（Fig. 1）

```
自由欧氏空间 E = R^n
   │
   │  f̄_θ(u) = NN(u)          ← 无约束向量场（红箭头）
   ▼
   Proj_u: E → T_u M           ← 投影到约束流形切空间（绿箭头）
   │
   ▼
   u̇ = Proj_u(f̄_θ(u)) ∈ T_u M  ← 动力学永远在 M 上（橙线）
```

**关键点**：约束流形由**代数等式**定义 ⇒ 可以是**各种**约束（守恒律、完整约束），且**不要求**系统以特定（广义/正则）坐标表达。

### 3.2 核心模块 1：投影算子的构造（沿 Boumal 的正交投影路线）

微分与伴随：$Dg(u)[v]=(\langle\nabla g_1(u),v\rangle,\dots,\langle\nabla g_m(u),v\rangle)^\top$（Eq. 4a），$Dg(u)^\star[\alpha]=\sum_i\alpha_i\nabla g_i(u)$（Eq. 4b）。

任意 $v\in E$ 的**唯一分解**：

$$v=\mathrm{Proj}_u(v)+Dg(u)^\star[\alpha] \tag{Eq. 5}$$

$\alpha$ 由最小二乘给出：$\alpha=(Dg(u)^\star)^\dagger[v]$（Eq. 6，$\dagger$ 为 Moore–Penrose 伪逆），代入得

$$\mathrm{Proj}_u(v)=v-Dg(u)^\star\Big[(Dg(u)^\star)^\dagger[v]\Big] \tag{Eq. 7}$$

矩阵形式（$G(u)\in\mathbb{R}^{m\times n}$，$G_{ij}=\partial g_i/\partial u_j$，满行秩 ⇒ 伪逆简化为 $(G^\top)^\dagger=(GG^\top)^{-1}G$）：

$$\mathrm{Proj}_u(v)=\Big[I-G(u)^\top\big(G(u)G(u)^\top\big)^{-1}G(u)\Big]v \tag{Eq. 10}$$

### 3.3 核心模块 2：投影后的微分方程与约束保证

$$\dot u=\mathrm{Proj}_u\big(\bar f_\theta(u)\big)\in T_u\mathcal{M} \tag{Eq. 8}$$

**命题 1（约束保持）**：若 $u_0\in\mathcal{M}$，则 PNDE (8) 的解对所有 $t\ge 0$ 满足 $g(u(t))=0$。

**证明**（可复算，一行）：

$$\frac{d}{dt}g(u(t))=Dg(u)\big[\mathrm{Proj}_u(\bar f_\theta(u))\big]=0 \tag{Eq. 9}$$

因为 $\mathrm{Proj}_u(\bar f_\theta(u))\in T_u\mathcal{M}=\ker Dg(u)$ ⇒ $g(u(t))$ 恒等于 $g(u(0))=0$。

⇒ 🔴 **这就是"硬约束"的全部含义**：不是"训练时惩罚"，而是**沿流形的不变量（invariant）**。

### 3.4 核心模块 3：可微实现（这是本篇真正的工程贡献）

**前向**：不解显式逆，而是解线性方程 $GG^\top\lambda=Gv$（Eq. 11），再算 $\mathrm{Proj}_u(v)=v-G^\top\lambda$（Eq. 12）。用 **Cholesky**（$GG^\top=LL^\top$，$m\times m$）或 **QR**（当 $G$ 病态时，因为 $\kappa(GG^\top)=\kappa(G)^2$，平方条件数是隐患）。

**反向**（Zygote.jl 无法对外部求解器库求导 ⇒ 作者**手推解析梯度**）：

$$\bar v=\bar p-G^\top\mu,\qquad GG^\top\mu=G\bar p \tag{Eq. 13, 14}$$

$$\bar G=-\mu p^\top-\lambda\bar v^\top \tag{Eq. 15}$$

**Eq. 15 是本篇最漂亮的一行**：$G$ 既显式出现在 $p=v-G^\top\lambda$ 里、又隐式通过 $\lambda$ 出现，链式法则化简后得到一个**异常紧凑**的形式，且右侧全部量在前向已算过 ⇒ **整个反向退化为矩阵-向量乘 + 一次 $m\times m$ 三角求解**，且**复用前向的 Cholesky 分解**。

**复杂度**：前向 $O(m^2n)$，反向 $O(mn)$（复用分解）⇒ **对状态维 $n$ 线性**，瓶颈在约束维 $m$；$m\ll n$ 时最划算。

### 3.5 核心模块 4：广义坐标 vs 笛卡尔坐标的对照（本篇的"第二贡献"）

**N-pendulum**：在广义坐标 $\theta$ 里，运动方程随 $N$ 增大**急剧复杂化**；换到笛卡尔坐标后方程简单，但**臂长不再自动守恒**，必须显式施加 holonomic 约束：

$$g_i^{(\mathrm{pos})}(q)=\|q_i-q_{i-1}\|^2-1=0,\quad i=1,\dots,N \tag{Eq. 27}$$

$$g_i^{(\mathrm{vel})}(q,\dot q)=(q_i-q_{i-1})\cdot(\dot q_i-\dot q_{i-1})=0 \tag{Eq. 28}$$

共 $2N$ 个约束，$g:\mathbb{R}^{4N}\to\mathbb{R}^{2N}$（Eq. 29）。

**结论**：*"the benefits of learning in Cartesian coordinates with constraints become pronounced as $N$ is increased."*

⇒ 🔴 **这与 10-28 的 `2012.02334`（Zhong et al.）形成直接对立**：那篇把"隐式约束（守恒坐标）vs 显式约束（广义坐标 + Lagrange 乘子）"列为消融维度之一，倾向于隐式/结构化路线更省事；本篇主张**显式约束 + 简单坐标**才是正解。⇒ **两篇必须并置引用**。

### 3.6 与 LTC / CfC / Hamiltonian-NN / SNDE 的关系

- **与 LTC/CfC**：PNDE 是**通用 ODE 层的约束机制**，与液态族正交 —— 可与 `lnn/core/ltc.py`（ODE 路线）叠加；但**与 CfC 不兼容**（CfC 无积分器，约束需在闭式更新的流形上重新推导）。
- **与 Hamiltonian-NN / Lagrangian-NN（CHNN/CLNN）**：那些是**换参数化**（把 $H(q,p)$ 或 $\mathcal{L}(q,\dot q)$ 作为网络输出，靠结构自动守恒）；PNDE 是**投影**（网络仍输出任意向量场，只是方向被约束）⇒ **PNDE 的一般性更强**（可施加非能量类约束），但**结构性更弱**（不保证任何超出 $g$ 的守恒量）。
- **与 SNDE（Stabilized NDE）**：SNDE 用增益 $\gamma$ 把 $g(u(t))$ 拉回 0（软约束 + 投影项），**需要调 $\gamma$**，且 $\gamma$ 大时**变刚性、求解变贵**；PNDE **无超参、无刚性**。
- **与本仓 `LRCCell`**：⚠️ **命名撞车警示**（承接 10-28 记录）—— 本仓 `lnn/core/variants.py:346` 的 `LRCCell` 名字来自 arXiv:2403.08791，但**实现的是该文要修正的"耦合 LC"版本**；本篇的 LNN 与 Hamiltonian-NN 一族同样是"名字与机制可能脱节"的样本。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 2 | $\mathcal{M}=\{u\in E;\ g(u)=0\}$ | 约束流形（$m<n$，独立代数约束） |
| Eq. 3 | $f_\theta(u)=\mathrm{Proj}_u[\bar f_\theta(u)]\in T_u\mathcal{M}$ | **核心定义**：先学自由向量场，再投影 |
| Eq. 7 | $\mathrm{Proj}_u(v)=v-Dg(u)^\star[(Dg(u)^\star)^\dagger[v]]$ | 正交投影的通用形式（伪逆） |
| Eq. 8 | $\dot u=\mathrm{Proj}_u(\bar f_\theta(u))$ | **PNDE 方程** |
| Eq. 9 | $\frac{d}{dt}g(u(t))=Dg(u)[\mathrm{Proj}_u(\bar f_\theta(u))]=0$ | **约束保持的证明**（一步）⇒ 硬约束的本质 |
| Eq. 10 | $\mathrm{Proj}_u(v)=[I-G^\top(GG^\top)^{-1}G]v$ | 矩阵形式（$G$ 满行秩 ⇒ 伪逆化简） |
| Eq. 11 | $GG^\top\lambda=Gv$ | 前向：**解线性方程**而非求逆（Cholesky/QR） |
| Eq. 12 | $\mathrm{Proj}_u(v)=v-G^\top\lambda$ | 前向：投影向量 |
| Eq. 13–14 | $\bar v=\bar p-G^\top\mu,\ GG^\top\mu=G\bar p$ | 反向：$\bar v$ 的伴随系统（**复用前向 Cholesky**） |
| Eq. 15 | $\bar G=-\mu p^\top-\lambda\bar v^\top$ | 反向：$G$ 的梯度（**异常紧凑**，全部量前向已算） |
| Eq. 27–29 | $g_i^{(\mathrm{pos})}=\|q_i-q_{i-1}\|^2-1$，$g_i^{(\mathrm{vel})}=(q_i-q_{i-1})\cdot(\dot q_i-\dot q_{i-1})$ | N-pendulum 的 $2N$ 个显式约束 |

## 5. 关键成果与贡献

### 5.1 主结果

**对比对象**：NDE（无约束）、SNDE（$\gamma\in\{0.1,1,20,200\}$）、PNDE。**所有测试统计在 100 个未见初值上平均**。误差度量：相对状态误差 $\|u(t)-\hat u(t)\|_2/\|u(t)\|_2$；约束误差 $\|g(\hat u(t))\|_2$（在流形上为零）。

**（a）约束精度 —— PNDE 数量级碾压**

- 二维 quartic 势、Lotka–Volterra、N-pendulum（$N=4,8$）、IEEE 14-bus 电网：**PNDE 的约束误差比最好的 SNDE 低若干个数量级**，且**接近机器精度**。
- 电网实验（Fig. 6）：所有模型的**状态预测几乎相同**（SNDE 与 NDE 不可区分），但 **NDE 与 SNDE 在 PQ 母线有功功率约束上误差大**；增大 $\gamma$ 能改善约束但**引入刚性、求解代价激增**；**PNDE 无刚性且约束精度高若干个数量级**。
- **长程 rollout（500 s）**：无约束 NDE 的 PQ 母线注入有功**漂移甚至变正**（*"they spontaneously generate power"*）⇒ 论文称之为*"clearly unphysical"*；PNDE 不发生。

**（b）状态精度 —— 显式约束在简单坐标里反而更好**

- N-pendulum：**笛卡尔坐标 + 显式约束（PNDE）优于广义坐标下的 NDE**，且**$N$ 越大差距越明显**（Fig. 4，$N=4$ 与 $N=8$ 对照）。
- 电网：**状态预测持平**，收益全在约束侧。

**（c）计算开销（Table I，batch 1024，单张 NVIDIA H100）**

| 系统 | 模型 | Train (min) | Fwd. Pass (ms) | Solve (ms) |
|---|---|---|---|---|
| 2-D Quartic | NDE | 3.9 | 0.17 | 2163 |
| | SNDE ($\gamma=1$) | 4.5 | 0.50 | 2743 |
| | SNDE ($\gamma=10$) | 4.5 | 0.50 | 3908 |
| | **PNDE** | 4.5 | **0.42** | **2531** |
| Lotka–Volterra | NDE | 5.5 | 0.20 | 3201 |
| | **PNDE** | 6.6 | 0.38 | 3199 |
| 4-Pendulum | NDE (Angular) | **238** | **1.22** | **2235** |
| | SNDE ($\gamma=20$) | 703 | 1.68 | 3696 |
| | SNDE ($\gamma=200$) | 755 | 1.70 | 6089 |
| | **PNDE** | 607 | 1.50 | 3506 |
| 8-Pendulum | NDE (Angular) | **320** | **1.23** | **3629** |
| | SNDE ($\gamma=20$) | 1005 | 1.95 | 5980 |
| | SNDE ($\gamma=200$) | 890 | 1.90 | 8778 |
| | **PNDE** | 825 | 1.75 | 7725 |
| IEEE 14-Bus | NDE | **9.8** | **0.36** | 36 |
| | SNDE ($\gamma=0.1$) | 25.5 | 1.22 | 75 |

论文自陈的总体开销：**投影的训练/推理开销约 1.0–2.8×**（相对无约束 NDE），**随约束数 $m$ 二次增长**。

### 5.2 机制分析

- **为什么 PNDE 不需要 $\gamma$**：SNDE 的 $\gamma$ 是**软约束的强度旋钮**，要手工调、且大 $\gamma$ 换刚性；PNDE 的约束**由 Eq. 9 的恒等式保证**，$\gamma$ 不存在 ⇒ **超参更少**（论文摘要与 §V 均明确列为贡献）。
- **为什么投影不引入刚性**：投影是把向量场**分量替换**为切向分量，**不改变其模长上界**，因此不制造 SNDE 那种"约束项随误差增大而增大 ⇒ 局部刚化"的机制。
- **为什么在简单坐标里学更好**：约束显式化后，**网络只需学"动力学"不必学"守恒"**，等价于把守恒约束从学习负担里**移出去**；$N$ 越大广义坐标方程越复杂，差距越大。

### 5.3 贡献清单

- 🔴 **一条范式**：**"软约束"→"硬约束"** 的升级路径 —— 不靠惩罚项，靠**投影 + 不变量证明**。
- 🔴 **一个可移植的算法**：投影的**解析反向梯度**（Eq. 13–15）+ **Cholesky 复用** ⇒ **反向 $O(mn)$**，且不依赖外部求解器库的可微性（这正是 Zygote.jl 场景下的关键）。
- 🔴 **一个坐标论证**：**显式约束 + 简单坐标 ≻ 隐式约束 + 复杂坐标**（当系统维数上升时）—— 与 `2012.02334` 的倾向**相反**，必须并置引用。
- 🔴 **一个能量学观察**：无约束 NDE 在长程 rollout 里**自发产生功率**（约束漂移 → 非物理）⇒ 这是"结构化归纳偏置不是可选项"的直接物证。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **开销随约束数 $m$ 二次增长**（训练/推理 1.0–2.8×）；约束很多时成瓶颈（实践中最紧的 $m$ 尚可接受）。
- **假设约束 Jacobian 处处满行秩**（否则 $\mathcal{M}$ 不是光滑流形、投影算子定义失效）。
- **电网模型是微分-代数方程**（DAE），与纯 ODE 的处理不同（论文附录专门说明）。
- **用投影代替结构**会**丢失除 $g$ 之外的任何结构**（例如不会自动守恒能量，除非 $g$ 里写了）。
- 附录 H 明确说明：**除电网实验外，其余实验都有特定的数据切分与设置**，跨系统的数字不可直接横比。

### 6.2 隐含局限（本仓视角）

- **与 CfC 不兼容**：本仓主力是 **CfC（闭式一步）**，不是 ODE 积分 ⇒ Eq. 8 的"投影流"在 CfC 上没有直接对应（除非把 CfC 的更新写成"流形上的离散映射"再投影状态，**这本身是一个可做的变体**）。
- **约束由人工给定**：$g$ 需已知 ⇒ 对"不知道守恒量是什么"的场景无能为力（与"学守恒量"那一族正交）。
- **投影会改变最优解**：把 $\bar f_\theta$ 投影到切空间后，**网络在"正交方向"上的自由度变成无用参数** ⇒ 存在**参数冗余**，可能需要更正则化/更小模型；论文未讨论这一点。
- **本仓的"结构化变体"实际全在软约束一侧**：本仓此前的"物理先验"型变体（Hamiltonian 型、端口哈密顿型、对称-反对称型）都是**靠结构让约束自动成立**（比 PNDE 的"投影"更强），但**都没有 PNDE 那种"任意代数约束"的通用性** ⇒ 两族是**互补**而非替代，索引必须分层。
- ⚠️ **与 `2012.02334` 的结论冲突必须显式记录**，否则本仓索引会变成"挑着引用"。

### 6.3 未来方向

- 形式化区域证书（见本日 `2608.09404` 的同类诉求）；
- 非欧氏/黎曼度量下的投影（论文 Appendix A 已给出推导，本文只做欧氏）；
- 更多 DAE / 混合系统；
- 把"发现约束"（learned constraints）与"施加约束"（hard projection）结合。

## 7. 对本仓的意义

**（a）可作为 `lnn/core/` 的什么变体？** —— **`ProjectedCfCCell` / `ProjectedLTCCell`**：把本仓任一 cell 的输出向量场（`bar_f`）投影到约束流形切空间。具体到 CfC：`h' = d·g + (1-d)·h_out` 中的 `g` 与 `h_out` 合成向量 `v` ⇒ 先算 `v` 再投影：`h' = Proj_h(v)`。约束 $g$ 的选型建议用**守恒量**（如 $\sum$ 质量、能量、谱半径）或**已知单调量**。
⚠️ **诚实提示**：CfC 的更新不是 ODE 流，投影保证的是"**状态落在流形上**"而非"**沿流形积分**" ⇒ 必须在报告里写清这个区别。

**（b）可嫁接到哪个 bench 脚本？** —— 新增 **`scripts/bench_projected_conservation_cfc.py`**：在 toy_sin / structured / random 三个 toy 数据集上比较
- (i) 无约束 CfC；
- (ii) 软约束 CfC（loss 里加 $\|g(h)\|^2$）—— **本仓至今的默认做法**；
- (iii) 硬约束 CfC（投影）。
**必报三列**：预测 MSE、**约束违反量 $\|g(h)\|$ 的 max 与均值**（区分硬/软的**唯一**判据）、**墙钟 + NFE**（11-22 `2611.25274` 纪律）。

**（c）提供哪个新 ablation axis？** —— 🔴 **"约束的实现方式"轴（软 / 硬 / 结构）**：本仓至今所有"结构化归纳偏置"变体其实只在"结构"这一格；本篇补上"硬投影"这一格 ⇒ 形成 **2×2（结构 vs 投影）×（软 vs 硬）** 的可比较空间。

**（d）一条对 10-28 `2012.02334` 的强制口径修正** —— 本仓索引必须新增一条**并置记录**：

> `2012.02334`（能量守恒网络基准）倾向"**隐式约束**（在守恒坐标里学）"更简单可靠；`2410.23667`（PNDE）主张"**显式约束 + 简单坐标 + 投影**"更好，且在高维（$N$-pendulum）下差距更大。**两篇结论方向相反**（一个说显式约束更麻烦、一个说显式约束更好），差异可能来自：(i) 基准论文比较的是**十个守恒网络**（全是结构化路线），PNDE 比较的是**投影路线**；(ii) PNDE 的显式约束是**逐分量可解析求梯度**的 holonomic 形式；(iii) 两者的任务与系统族不同。⇒ **引用任何一方都必须并置另一方**。

**（e）一条引用纪律（可并入 `analysis/negative_results/README.md` 纪律段，本次未擅自改台账）**：

> **凡本仓报告"物理/守恒先验有效"，必须标明该先验是"结构化（构造保证）"、"投影（构造保证）"还是"惩罚（软约束）"** —— 三者的**保证强度**依次递减，而本仓 bench 的默认实现**普遍是软约束**。

**Verdict**：**"硬约束（投影）是本仓结构化先验工具箱里缺失的一格，且其保证强度远高于软约束" → TARGET-POSITIVE（方法论层面，证据充分：多系统 × 100 初值 + 精度/开销双表）**；**"PNDE 可直接用于 CfC" → TARGET-DEPENDENT-WITH-NUANCE**（需重新推导 CfC 上的投影语义，只能保证"状态在流形上"而非"沿流形积分"）。
