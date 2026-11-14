---
title: Liquid-Graph Time-Constant Network for Multi-Agent Systems Control
date: 2026-11
tags: [LNN, LTC, CfC, graph-ODE, contraction-analysis, closed-form, distributed-control, multi-agent]
arxiv_id: 2404.13982
---

# 研读报告：LGTC / CfGC — 把 LTC 抬到**图结构**上，并用**收缩分析（contraction analysis）**给闭式解背书

## 1. 元数据
- **论文标题**：Liquid-Graph Time-Constant Network for Multi-Agent Systems Control
- **作者**：Antonio Marino, Claudio Pacchierotti, Paolo Robuffo Giordano —— **Univ Rennes / CNRS / Inria / IRISA – Rennes**（法国）
- **发表时间**：arXiv 2024-04-22（v3 更新 2025-03-01, cs.MA）
- **代码**：论文**未给出开源仓库 URL**（按摘要与正文表述，属**不可直接复现**）
- **本地 PDF**：[papers/2404.13982/2404.13982.pdf](../../papers/2404.13982/2404.13982.pdf) ／ 文本 [2404.13982.txt](../../papers/2404.13982/2404.13982.txt)
- **arXiv 分类**：cs.MA
- **关联概念**：LTC（Hasani 2021）/ CfC 闭式解 / 图神经网络 / 连续时间图 ODE / 收缩理论（contraction analysis）/ δISS（增量输入到状态稳定）/ 分布式控制 / 群集（flocking）/ 通信负载优化
- **入选理由**：本仓**零覆盖**"液态网络 + 图结构 + 稳定性理论"这一整条线。`docs/LNN_训练方向_图时空与通信系统_可行报告.md` 里只把 LGTC 列成一行**候选条目**（"把 LTC 与图结构结合，处理多智能体动态"），**从未做过正式研读**。本篇提供了本仓目前**唯一一份"连续时间机制 + 可证明稳定性"的第三方理论背书**——这正是本仓 §0 想回答而始终没有外部支撑的问题。

## 2. 核心问题

**GNN 在分布式控制里的核心矛盾是：通信代价 vs 表达力。** 每个智能体（agent）只能跟邻居交换信息，而标准 GNN 需要交换**完整的高维隐状态**。N 个 agent、N 维状态 → 通信量随规模爆炸，"分布式"名存实亡。

同时，**图上的循环网络还有第二个隐患**：图过滤的谱半径会随图规模增长，长程信息在图上衰减/震荡，状态容易失衡。论文作者此前的工作（[22]，GGNN 的 δISS 条件）已经处理过这个离散问题。

液态网络（LTC）恰好提供了一条路：**输入依赖的可变时间常数**让每个状态元素自己认领不同的动力学，理论上可以用**更少的隐状态**表达同样丰富的时间行为，从而**少通信**。而 CfC（Lechner et al. 2022）提供了 LTC 的**闭式解**，免去 ODE 求解器。

但把 LTC 直接搬到图上有三个未解的硬问题：

1. **稳定性怎么证？** 连续时间下，LTC 的隐状态会不会炸？多智能体系统里这个担忧是致命的（一个 agent 发散 = 整机失控）。
2. **闭式解怎么落？** LTC 的闭式解是逐元素的；**图结构的 Kronecker 积项让闭式解无法在 agent 间分布式计算**——这是本文最核心的技术障碍。
3. **"少通信"和"必须多次迭代"会不会互相抵消？** 用 ODE 求解器就要迭代 n 步，每步都要重新通信一次，**迭代直接把省下来的通信量又吃回去**。

于是问题被重述为：**能否构造一个连续时间图网络（LGTC），既在图上证明 δISS 稳定性与收缩率，又给出保持同一收缩率的闭式近似（CfGC），从而在不牺牲稳定性的前提下真正降低通信量？**

## 3. 方法论与核心思路

### 3.1 总体架构

```
                 ┌─── 每个 agent 相同的网络 (GNN 层) ───┐
  邻居信号 x_j ──┤  f = ρ(ÂS(x)+b_x) + ρ(B̂S(u)+b_u)     │  f = 输入/状态依赖的
                 │  ẋ = −(b+f)∘x − Σ_k S_k x A_k + f∘tanh(BS(u))  │  "液态"时间常数
                 └──┬────────────────────────────────────┘
                    │  稳定性: 收缩率 c > 0 (Theorem 2)
                    │  闭式解: x(T)| = exp(−(diag(b)+diag(f_x)+f_i)T)·diag(σ(2f_σ))·x(0)|
                    │           + diag(1−σ(2f_σ))·σ_u          (Eq. 16)
                    ▼
              CfGC (Eq. 20): 一次迭代 = 一次通信
```

### 3.2 图基元：图移位算子与图滤波

论文用**图移位算子** $S \in \mathbb{R}^{N\times N}$（实验中取**归一化拉普拉斯**）统一描述所有图算子。图滤波为：

$$H_S(x) = \sum_{k=0}^{K} S^k x H_k \tag{Eq. 2}$$

其中 $H_k \in \mathbb{R}^{G\times F}$ 是滤波器组。**关键性质：$S^k$ 可以只用 1-hop 通信的反复应用来计算**，所以整个滤波是**分布式**的——这是后面"通信负载"论证的基础。

### 3.3 LGTC：连续时间图 ODE（Eq. 8）

$$\begin{aligned} f &= \rho(\hat{A}S(x) + b_x) + \rho(\hat{B}S(u) + b_u) \\ \dot{x} &= -(b + f)\circ x - \sum_{k=1}^{K} S_k x A_k + f\circ \tanh(BS(u)) \end{aligned} \tag{Eq. 8}$$

$\rho = \text{ReLU}$。与 LTC 的差别只在：LTC 的输入项是 $f(I(t))$，这里换成了**图过滤后的** $\hat{A}S(x)$、$\hat{B}S(u)$，而 $-\sum_k S_k x A_k$ 就是 **GGNN 的离散状态更新搬到连续时间**。

**这里的"液态"落在 $-(b+f)\circ x$ 上**：$f$ 由输入和**经图通信回来的邻居状态**共同决定 ⇒ **每个 agent 的时间常数同时依赖自己的输入和邻居状态**。这是论文自陈与 LGNN/GGNN 的核心差异。

### 3.4 稳定性：δISS 与收缩率（Theorem 2）

用 **log-norm** $\mu_\infty(X) = \max_i(x_{ii} + \sum_{j\neq i}|x_{ij}|)$ 定义收缩（Definition III.1：$\mu_\infty(D_xF) < -c$，$c>0$ 为收缩率）。Theorem 2 给出闭式可验的充分条件：

$$c \geq 0;\quad b \geq 0;\quad \mu_\infty\Big(\sum_{k=1}^K A_k^T \otimes S_k\Big) \geq 0 \tag{Eq. 11}$$

$$\boxed{\,c = \|b\|_\infty + \|A_{1,K}\|_\infty \|\tilde{S}_{1,K}\|_\infty - \|b_x\|_\infty + \|\hat{A}^T_{0,K}\|_\infty \|\bar{S}_{0,K}\|_\infty\,} \tag{Eq. 12}$$

注意 $c$ 是**可由权重直接算出的数**（全是 ∞-范数），**不需要跑 forward 也不需要梯度**——这比本仓 `lnn/core/iss_stable_cfc.py` 的做法（把 ISS 条件做成**训练损失**）便宜一个数量级。

稳定性在训练中通过 **Softplus 正则** $\Pi = \sum_i \mathrm{Softplus}(p_i)$ 强制（Eq. 21），$\beta=10$。

### 3.5 闭式近似 CfGC（Eq. 16 → Eq. 20）

**问题**：Eq. 8 的 ODE 积分没有闭式解，且迭代 n 步就要 n 次通信。论文先给出**理想闭式解**：

$$x(T)| = e^{-(\mathrm{diag}(b|) + \mathrm{diag}(f_x|) + f_i)T}\,\mathrm{diag}(\sigma(2f_\sigma)|)\,x(0)| + \mathrm{diag}(1-\sigma(2f_\sigma)|)\,\sigma_u| \tag{Eq. 16}$$

**三个工程修正把它变成可分布式训练的 CfGC（Eq. 20）**：

$$\begin{aligned} f_\sigma &= \rho(\hat{B}S(u) + b_u) + \rho(\hat{A}S(x) + b_x) \\ f_x &= \rho(\hat{A}S(x) + b_x) \\ f_i &= -D_{xf}\,\hat{A}S(x) + \sum_{k=1}^K \frac{S_k x A_k}{x+\epsilon} \\ x^+ &= \big(x \circ \sigma(-(b + f_x + f_i)t + \pi) - \sigma_u\big)\circ\sigma(2f_\sigma) + \sigma_u \end{aligned} \tag{Eq. 20}$$

1. **Kronecker 积项除以 $x+\epsilon$**（element-wise）→ 让 $f_i$ **可以在每个 agent 上局部计算**，不必传递整个 Kronecker 结构 ⇒ **恢复分布式性**。
2. **指数项换成 sigmoid** $\sigma(\cdot)$ → 缓解指数项导致的**梯度消失**（与 CfC 论文里的完全同款技巧，见 §"与 CfC 的关系"）。
3. **加 $\pi$ 偏置** → 使 $t=0$ 时 $\sigma(-Ft+\pi)\approx 1$。

Theorem 3 声称：**CfGC 保持与 LGTC 完全相同的收缩率 $c$**（证明路径：$\sup_x \mu_\infty(D_xF) = \partial_t \sup_x \|e^{D_xFt}\|_\infty|_{t=0^+}$，Eq. 17-19）。

### 3.6 与 LTC / CfC / GGNN 的关系

| 维度 | GGNN（离散） | LGTC（本文，ODE） | CfGC（本文，闭式） | CfC（Lechner 2022） |
|---|---|---|---|---|
| 时间语义 | 离散步 | 连续 ODE | **离散一步 = 一整个时间窗** | 闭式、显式 t |
| 稳定性保证 | δISS（作者前作） | δISS + 收缩率 $c$ | **同上（Theorem 3）** | 无 |
| 通信/迭代 | 1 次/步 | **n 次（ODE 步）** | **1 次** | 1 次 |
| 稀疏性 | 图稀疏 | 图稀疏 | 图稀疏 | 密集（除非 NCP） |

**与 CfC 的关系要特别说清楚**：CfGC 的 Eq. 20 与 CfC 的 Eq. 10 **结构上同族**——都是"**用 sigmoid 时间门控插值两个分支**"（$\sigma(\cdot)\odot g + (1-\sigma(\cdot))\odot h$）。差别在于 CfGC 的门控**被图过滤修正**了、且额外带 Kronecker 修正项 $f_i$。**换句话说：本文把 CfC 的"时间门控 + 闭式"范式搬进了图结构，并额外给它配了一套收缩率证明。**

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 2 | $H_S(x) = \sum_{k=0}^{K} S^k x H_k$ | 图滤波；$S^k$ 可用 1-hop 通信迭代实现 ⇒ 分布式 |
| Eq. 8 | $f = \rho(\hat{A}S(x)+b_x)+\rho(\hat{B}S(u)+b_u)$；$\dot{x} = -(b+f)\circ x - \sum_k S_k x A_k + f\circ\tanh(BS(u))$ | **LGTC 主方程**：连续时间图 ODE，时间常数依赖输入+邻居状态 |
| Eq. 9 | $\tilde{q} = \sigma(\tilde{A}S(x)+\tilde{B}S(u)+\tilde{b})$；$\hat{q} = \sigma(\hat{A}S(x)+\hat{B}S(u)+\hat{b})$；$x^+ = \sigma_c(\hat{q}\circ AS(x) + \tilde{q}\circ BS(u)+b)$ | **GGNN 状态空间表示**（对照基线） |
| Eq. 11 | $c \geq 0;\ b \geq 0;\ \mu_\infty(\sum_{k=1}^K A_k^T\otimes S_k)\geq 0$ | **Theorem 2 稳定性充分条件**（三行可验） |
| Eq. 12 | $c = \|b\|_\infty + \|A_{1,K}\|_\infty\|\tilde{S}_{1,K}\|_\infty - \|b_x\|_\infty + \|\hat{A}^T_{0,K}\|_\infty\|\bar{S}_{0,K}\|_\infty$ | **收缩率闭式表达式**（纯权重范数，无需 forward） |
| Eq. 14 | $x(t+\Delta) = x(t) + \Delta\big(-\sum_k S_kx(t)A_k + f\circ\tanh(BS(u))\big)/(1+\Delta(b+f))$ | **混合 ODE 求解器**（LTC 的 hybrid solver 图版） |
| Eq. 16 | $x(T)| = e^{-(\mathrm{diag}(b|)+\mathrm{diag}(f_x|)+f_i)T}\mathrm{diag}(\sigma(2f_\sigma)|)x(0)| + \mathrm{diag}(1-\sigma(2f_\sigma)|)\sigma_u|$ | **理论闭式解**（含 Kronecker 项，不可分布式） |
| Eq. 17 | $\sup_x \mu_\infty(D_xF) = \left.\partial_t \sup_x \|e^{D_xFt}\|_\infty\right\|_{t=0^+}$ | log-norm 收缩与矩阵指数的关系（Theorem 3 证明枢纽） |
| Eq. 20 | $f_i = -D_{xf}\hat{A}S(x)+\sum_k \frac{S_kxA_k}{x+\epsilon}$；$x^+ = (x\circ\sigma(-(b+f_x+f_i)t+\pi)-\sigma_u)\circ\sigma(2f_\sigma)+\sigma_u$ | **CfGC 实际部署方程**（Kronecker 局部化 + sigmoid 替代指数） |
| Eq. 21 | $\Pi = \sum_i \mathrm{Softplus}(p_i)$，$p_i \in \{\delta A_i - 1;\ [-c,-b,-\mu_\infty(\sum A_k^T\otimes S_k)]\}$ | **稳定性正则项**（把 Theorem 2 的约束做成训练损失） |
| Eq. 22 | $u_f(t) = \frac{1}{N}\sum_{i=1}^N v_i(t) - \nabla_{r}CA(r(t),r_j(t))$；$u_l(t) = -W_p(r_l(t)-d(t)) - \nabla_{r_l}CA(r_l(t),r_j(t))$ | **群集专家控制器**（模仿学习 teacher） |

## 5. 关键成果与贡献

### 5.1 主结果（flocking 控制，Fig. 2）

任务：$N$ 个 agent 的分布式群集控制（双积分器动力学），模仿学习学一个**分布式**控制器；每个 agent 局部可观测，通信半径 $R$，采样/通信周期 $T=0.05\text{s}$。数据集 60 条轨迹（70/10/20 划分），DAGGER 每 20 epoch 一次，ADAM lr=1e-3，120 epoch。

**模型配置**（这是可比性关键）：隐状态 $F=50$，滤波器长度 $K=2$，**只通信 $F'=4$ 个状态特征 + $G'=4$ 个输入特征**（即通信量降到 1/12.5），归一化拉普拉斯作 $S$。

| 模型 | leader 误差 | 群集误差 | 通信/迭代 |
|---|---|---|---|
| 专家控制器（中心化 teacher） | 基准 | 基准 | 需全局状态 |
| **GGNN**（δISS） | 最差 | 最差 | 1 次 |
| **GraphODE** | **最差** | **最差** | n 次 |
| **LGTC** | 好 | 好 | n 次 |
| **CfGC** | 好（与 LGTC 接近） | 好（部分 N=10 场景**比 LGTC 好 1–2%**） | **1 次** |

论文口径：**LGTC/CfGC 相对 GGNN 在群集误差上"最多好 40%"、leader 误差"最多好 10%"**（原文："up to 40% for the flocking error and 10% for the leader error closer to the expert controller"）。

**通信范围鲁棒性实验**（训练用 $R=4\text{m}$，测试改 $R=2\text{m}$）：GGNN/LGTC/CfGC 群集误差均掉 **0.18 m/s**，GraphODE 掉 **0.3 m/s**。

### 5.2 机制分析

- **CfGC 偶尔反超 LGTC 1–2%**：论文归因于 CfGC 是**离散模型**，训练更容易、梯度消失更少。**这条对本仓很有价值——它说明"闭式化"不只是省算力，在训练动力学上可能反而更好。**
- **GraphODE 最差的原因**：论文明确指出 GraphODE **不直接使用输入特征**（输入只用来初始化内部状态），实为自回归模型。**这条是"用不用输入"与"怎么建模时间"两个因素的分离证据。**
- **压缩通信的正确性论证**：论文借用 LTC 的 PCA 轨迹深度分析，说明 LTC 的表达力在短轨迹上"最差不过离散 RNN"，长轨迹上不衰减 ⇒ 可减少需通信的状态维度；并论证**用这些缩减后的图滤波器，Theorem 2 依然成立**。

### 5.3 贡献清单

1. **LGTC**：首个把 LTC 时间常数机制与图结构结合的连续时间图网络，且**给出了 δISS + 收缩率的闭式可验条件**。
2. **CfGC**：一个**保持同一收缩率**的闭式近似，通过"Kronecker 项局部化 + 指数换 sigmoid"两项工程修正**恢复了分布式可计算性**，把通信次数从 $n$（ODE 步）降到 **1**。
3. **收缩率的纯范数表达**（Eq. 12）——**不需要跑前向、不需要梯度**即可评估稳定性，这是一个可以直接移植的**零成本诊断量**。
4. **通信量–表达力的联合论证**：在通信量降到 1/12.5 的前提下仍优于全通信的 GGNN。

## 6. 局限性与未来展望

### 6.1 作者自陈局限
- 通信负载**仍高于基于模型的（model-based）方法**——作者在结论里明确说"still high compared to model-based approaches"，未来工作包含进一步降低通信量。
- **只在 flocking 一个案例上验证**，且是**仿真**；未来工作是四旋翼 UAV 的实验验证。
- 论文**未给出开源代码**。

### 6.2 隐含局限（本仓视角）
- **收缩率 $c$ 是一个充分条件，不是必要条件**。$c>0$ 靠 Eq. 11/12 的**保守范数界**得到，实践中大概率**远松于真实收缩**。**论文没有报告实际 $c$ 的数值**，也没有做"理论 $c$ 与实测收缩的对照"——这是稳定性论证里最该有但没有的一张表。
- **$c$ 的正负号判据是"$\mu_\infty(\sum A_k^T\otimes S_k)\geq 0$"** —— 论文 Eq. 11 写的是 $\geq 0$，而 Eq. 12 的 $c$ 里是 $-\|A_{1,K}\|_\infty\|\tilde{S}_{1,K}\|_\infty$ 这种**减项**。**这两处的符号一致性没有在正文里被解释**，读者需要自己核对（Eq. 12 是从 Eq. 11 推出 $c$ 的最终表达式，Eq. 19 复现了同一式子，因此**三处一致**，但 Eq. 11 的"$\geq 0$"是**约束**、Eq. 12 的 $c$ 是**收缩率**，两者容易被误读为同一个量）。
- **通信量的口径不严格**：论文比较的是"通信的**特征维度**"（$F'=4$ vs $F=50$），**没有报告实际通信的比特数、频率或总带宽**，也没有给出"同等通信预算下"的对照曲线。**"少通信"是维度上的，不是系统级带宽上的。**
- **只有一个任务、一个仿真环境**，没有跨任务验证收缩条件的普适性。
- **$K=2$ 的滤波器长度**——论文自己说 $K$ 会"约束更多参数"，所以 $K=2$ 是个偏保守的选择；**$K$ 与稳定裕度的 trade-off 曲线缺失**。

### 6.3 未来方向
作者给出：进一步降低通信负载；四旋翼 UAV 实验验证。
**本仓可补的**：把收缩率作为**独立消融轴**（见 §7）；把 Eq. 20 的两个工程修正**分别消融**（Kronecker 局部化 vs sigmoid 替代），本文没有做这两个修正的单独 ablation。

## 7. 对本仓的意义

### 7.1 可作 `lnn/core/variants.py` 的什么变体
本文最直接可落地的东西**不是网络结构，而是那个收缩率**。本仓已有 `lnn/core/iss_stable_cfc.py`（ISS Lyapunov 损失，**训练时**约束）和 `lnn/core/lyapunov_stable_cfc.py`，但**都是在 loss 里做约束**。本文给的是一条**纯范数、零前向**的诊断：

$$c = \|b\|_\infty + \|A_{1,K}\|_\infty\|\tilde{S}_{1,K}\|_\infty - \|b_x\|_\infty + \|\hat{A}^T_{0,K}\|_\infty\|\bar{S}_{0,K}\|_\infty$$

虽然它是针对**图**结构的（引入了 $S$），但**标量版（$S=I$，即普通图/序列）直接退化成本仓可用的量**。建议新写一个 `lnn/core/contraction_rate.py`，提供 `contraction_rate_cfc(cell) -> float`，作为**每个 bench 结束后自动打印的伴随指标**——它比 Lyapunov 损失更便宜，且能回答"这个变体在**权重层面**离稳定还有多远"。

### 7.2 可嫁接到哪个 bench 脚本
- **`scripts/bench_cfc_vs_ssm.py`**：本仓已有 CfC vs SSM 对照。SSM（状态空间模型）的一个核心卖点就是**可证明稳定性**，而本仓的 CfC 侧**目前没有任何稳定性证书**。把 Eq. 12 的标量版作为 CfC 侧的稳定性读数，与 SSM 的理论保证并列，是**补上本仓 CfC/SSM 对照里最缺的那一栏**。
- **`scripts/bench_curvature_routing.py`**：本仓 `lnn/core/curvature.py` 只有 Ollivier-Ricci 曲率。**收缩率是另一种（更硬的）稳定性几何量**，两者可互为交叉验证。
- **通信/计算量的对照**：本仓所有 bench 都是单体的，没有"分布式通信量"这一维。若要引 LGTC 范式，需要先在**单体 CfC 上做"状态维度 vs 性能"的削减曲线**——这是 LGTC 论点的单体版本，可以直接检验"LTC 的表达力是否真的允许少用状态"。

### 7.3 提供哪个新 ablation axis
1. **收缩率 $c$ 作为连续谱的纵轴**（新轴）：把本仓每个 CfC 变体的 $c$ 算出来，与它的训练损失/测试误差画在同一张图上。**本仓至今没有"稳定性裕度 vs 精度"的二维图**，而这恰恰是"液态连续时间"这条线最该回答的问题。
2. **指数 vs sigmoid 时间门控**（新轴）：CfGC Eq. 20 与 CfC Eq. 10 **都**把指数换成 sigmoid，但**本仓 `lnn/core/cfc.py` 的 `CfCCell` 用的就是 sigmoid 版本**（`f_gate` = Sigmoid，`time_scale` 参数 + `select_step_delta`）。**也就是说本仓实现的是"论文 Eq. 10 的 sigmoid 版"，不是 Eq. 9 的指数版。** 本仓可以补一个"指数门控 vs sigmoid 门控"的对照，验证这**不是**一个无关紧要的工程近似，而是有实际性能差异的建模选择——**论文对这一点只有定性说法（"better condition on the loss surface"），没有数字**。
3. **$K$（滤波器/跳数）与稳定裕度的 trade-off**（LGTC 提出、本仓无对应）：本仓 `bench_cfc_temporal_smoothness.py` 涉及时间平滑度，可与收缩裕度联合看。
4. **"少状态维度"的削减曲线**（LGTC 论点的单体版）：**本仓的 §0 至今无法回答"LNN 的表达力是否真的允许用更少的状态"** —— 这正是 LGTC 的核心论证。

**Verdict**：
- **LGTC（连续时间图 ODE + 收缩率定理）→ TARGET-POSITIVE**：这是本仓至今**第一份由第三方给出、可用权重范数直接计算的连续时间稳定性证书**，且其"表达力支撑通信压缩"的论证方式值得本仓在单体尺度上复刻。**引用时必须带两条限定**：(a) 收缩率是**充分条件**，论文未报告实际 $c$ 数值，也未做理论–实测对照；(b) 通信压缩的口径是**特征维度**（50→4），**不是系统级带宽**，无等预算对照曲线。
- **CfGC（闭式近似）→ TARGET-DEPENDENT-WITH-NUANCE**：工程上是**正确且优雅**的（Kronecker 局部化 + sigmoid 替代指数两项修正都指向同一个真实障碍），**"闭式化反而更好训"这一点（+1–2%）值得本仓直接验证**。但**数值证据强度弱**：单一仿真任务、单一环境、无方差报告、无超参敏感性分析。
- **"LGTC/CfGC 优于 GGNN 40%" 这一条 → 不能写成"LNN 优于 GNN"**：对照组的 GGNN 本身就是**更老一代的离散图 RNN**，比较里**没有 Transformer、没有现代图 transformer（GAT/GraphGPS）、也没有等通信预算的强基线**。按本仓 NR-06 的纪律，**不得据此声称"LNN 在图任务上 SOTA"**。

## 8. 引用纪律（本篇专用）

1. **本篇是"LTC 谱系"证据，不是"LNN 缩写撞车"样本** —— 全文的 "LTC" 均指 Liquid Time-Constant，与 Lagrangian/Localized 无关。（对照：本仓 11-14 研读的 `2608.13215` 与 `2012.02334` 属 Lagrangian 撞车，**两者不可混为一谈**。）
2. **引用 "40% / 10% 改善" 时必须带上"vs GGNN（离散图 RNN）"的口径**，且必须说明通信量口径是**状态特征维度 50→4**。
3. **引用 Theorem 2/3 时必须说明这是充分条件**，且论文**未实测 $c$ 值**。
4. **不可引用为"液态网络在分布式控制上已达到 SOTA"** —— 见 §7 Verdict 第三条。

---
*研读依据：`paper-analyzer` 技能 SOP + AGENTS.md §2 摘要 Agent 标准模块（6 段）+ 研读骨架模板。PDF 全文经 `scripts/extract_pdf_text.py` 提取（42,033 字节 / 949 行），所有公式编号与 PDF 原文对齐。*
