---
title: Benchmarking Energy-Conserving Neural Networks for Learning Dynamics from Data
date: 2026-11
tags: [LNN, Lagrangian-NN, Hamiltonian-NN, energy-conserving, benchmark, negative-result, acronym-collision]
arxiv_id: 2012.02334
---

# 研读报告：Benchmarking Energy-Conserving Neural Networks — 十种"能量守恒网络"横向基准，以及一次**缩写撞车**的教科书级样本

> ⚠️ **本报告首条即为引用纪律，不是本论文的结论**：
> **本论文里的 "LNN" = Lagrangian Neural Network（拉格朗日神经网络），与本仓关注的 Liquid Neural Network（液态神经网络）除了首字母缩写相同之外没有任何关系。**
> 本仓的 digest 流水线之所以把它收进候选池，靠的是 `all:LNN AND all:forecasting` 这条布尔词组（见下文 §7.3），
> 这是一次**纯缩写撞车**，不是液态网络证据。本报告把这个撞车当作**方法论样本**来读。

## 1. 元数据
- **论文标题**：Benchmarking Energy-Conserving Neural Networks for Learning Dynamics from Data
- **作者**：Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty（Siemens Corporation, Princeton NJ）
- **发表时间**：arXiv 2012-12-03（v1）/ v6 2023-04-28；正式发表于 **PMLR vol. 144, pp. 1–12, L4DC 2021**（3rd Annual Conference on Learning for Dynamics and Control）
- **代码**：论文自陈"acknowledgement 里提到的 Finzi et al. (2020) codebase"（即 CLNN/CHNN 的实现），**未给出独立仓库 URL**
- **本地 PDF**：[papers/2012.02334/2012.02334.pdf](../../papers/2012.02334/2012.02334.pdf) ／ 文本 [2012.02334.txt](../../papers/2012.02334/2012.02334.txt)
- **arXiv 分类**：cs.LG, cs.AI, eess.SY, math.DS
- **关联概念**：Lagrangian Neural Network / Hamiltonian Neural Network / Neural ODE / 能量守恒归纳偏置 / 显式约束 vs 隐式约束 / 角度感知（angle-aware）/ 质量矩阵参数化 / 数据效率
- **入选理由**：本仓**零覆盖**"能量守恒神经网络"这一整族；且该论文对本仓**最刺眼的一点**是：它把同一批底层机制（连续时间 ODE + 结构化质量矩阵 + 物理归纳偏置）在**四种截然不同的归纳偏置维度**上做了统一的消融，结论对本仓的 CfC/LTC 结构选择有直接的方法论参考价值。

## 2. 核心问题

作者观察到，近几年"把物理规律写进深度学习计算图"的模型出现了一条清晰的分支线：**用拉格朗日/哈密顿形式重写 Newton 第二定律**，从而让网络**自动满足能量守恒**。但这条线上同时冒出了**十个名字不同、机制相近**的模型（HNN、LNN、DeLaN、SymODEN、CHNN、CLNN 及其变体），而**它们之间到底差在哪、差多少，缺少统一口径的对照**。

论文指出的结构性缺陷有两层：

**第一层是"命名掩盖机制"**。作者明确写道 *"As the original names of these models might not reflect the difference between them, we renamed a few models to emphasize the difference"* —— 同一篇论文里，`HNN`（参数化哈密顿量本身）与 `HNN-structure`（参数化质量矩阵 + 势能以复用式 (9) 的结构）是两个模型；`LNN`（参数化拉格朗日量 $L(q,\dot q)$ 本身）与 `LNN-structure`（即 DeLaN，参数化 $M(q)$ 与 $V(q)$）也是两个模型。**在未读源码的情况下按名字引用，会把耦合版当成解耦版**（本仓在 `LRCCell` 上已经踩过同类的坑，见 §7.4）。

**第二层是"约束的实现方式"从未被当作变量系统比较**。同一套拉格朗日/哈密顿动力学，可以用两种方式落地：
- **隐式约束**：直接用广义坐标 $q \in \mathbb{R}^M$，坐标数 = 自由度；
- **显式约束**：用笛卡尔坐标 $x \in \mathbb{R}^D$，用 $K = D - M$ 个 holonomic 约束 $\Phi_k(x) = 0$ 加上 Lagrange 乘子显式施加。

作者主张二者**不是实现细节之别，而是物理模型之别**（质量矩阵在笛卡尔坐标下是常数，在角坐标下是坐标的复杂函数），但此前没有人把两者放在同一张表上比过。

于是问题被重述为：**在这十个模型、四个物理系统、三个评估维度（精度 / 长时间外推 / 数据效率）上，哪些结构化归纳偏置（角度感知、质量矩阵结构、约束显式性）真正带来收益，哪些只是看着更"物理"？**

## 3. 方法论与核心思路

### 3.1 四种约束 × 两种能量形式的统一推导

论文的贡献形式是**先把全部十个模型统一到一套推导里**，再在同一协议下比较。

**隐式约束下的拉格朗日动力学**（由 D'Alembert 原理导出）：
$$\delta q^{\top}\left[\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q}\right) - \frac{\partial L}{\partial q}\right] = 0 \quad\Longrightarrow\quad \ddot q = \left(\nabla_{\dot q}\nabla_{\dot q}^{\top}L\right)^{-1}\left[\nabla_q L - \left(\nabla_q\nabla_{\dot q}^{\top}L\right)\dot q\right] \tag{Eq. 3}$$

其中结构化拉格朗日量 $L(q,\dot q) = \tfrac12 \dot q^{\top} M(q)\dot q - V(q)$（Eq. 4），代入后得
$$\ddot q = M(q)^{-1}\left[-\dot M(q)\dot q + \tfrac12\nabla_q\left(\dot q^{\top}M(q)\dot q\right) - \nabla_q V(q)\right] \tag{Eq. 5}$$

**隐式约束下的哈密顿动力学**（Legendre 变换）：$\dot q = \nabla_p H$，$\dot p = -\nabla_q H$（Eq. 6），且
$$\dot H = (\nabla_q^{\top}H)\dot q + (\nabla_p^{\top}H)\dot p = 0 \tag{Eq. 7}$$
这就是"能量守恒"的**结构性来源** —— 不是损失项罚出来的，是形式本身给的。结构化哈密顿量 $H(q,p) = \tfrac12 p^{\top}M^{-1}(q)p + V(q)$（Eq. 8）。

**显式约束下的哈密顿动力学**引入 $2K$ 个 Lagrange 乘子，最终得到
$$\dot z = J\nabla_z H - J(D_z\Psi)^{\top}\left[(D_z\Psi)J(D_z\Psi)^{\top}\right]^{-1}(D_z\Psi)J\nabla_z H \tag{Eq. 20}$$
其中 $z = (x^{\top}, p_x^{\top})^{\top}$，$J$ 为辛矩阵，$\Psi(z) = (\Phi^{\top}, \dot\Phi^{\top})^{\top}$。

**关键结构差异**（作者在 §2.3 明说）：显式约束下**质量矩阵 $M$ 是常数矩阵且分块对角**，其元素可直接设为可学习参数；隐式约束下质量矩阵是坐标的复杂函数，**其初始化的参数化往往病态**（§5 讨论）。

### 3.2 十个模型的统一命名表

| 论文内部名 | 对应原名 | 参数化了什么 | 约束方式 |
|---|---|---|---|
| HNN | Greydanus 2019 | 哈密顿量 $H$ 本身 | 隐式 |
| HNN-structure | SymODEN (Zhong 2020a) / Dissipative SymODEN / SRNN | $M(q)$ + $V(q)$ | 隐式 |
| CHNN | Finzi 2020 | 常数分块对角 $M$ + $V$ | **显式** |
| LNN | Cranmer 2020 | 拉格朗日量 $L(q,\dot q)$ 本身 | 隐式 |
| LNN-structure | DeLaN (Lutter 2019) | $M(q)$ + $V(q)$ | 隐式 |
| CLNN | Finzi 2020 | 常数分块对角 $M$ + $V$ | **显式** |
| \*-angle 变体 | — | 上述四个隐式模型 + 角度嵌入 $(\cos q, \sin q)$ | 隐式 |

**参数化技术细节**：凡参数化质量矩阵的模型，都利用正定性 $M(q) = L(q)L(q)^{\top}$（Cholesky 分解）改为参数化下三角 $L(q)$，而非直接参数化 $M$ 或 $M^{-1}$（§3 末段）。

### 3.3 评估协议

- **物理系统**（4 个）：1-pendulum、2-pendulum、4-pendulum（2D，角坐标；$N\ge2$ 是**混沌系统**，作者明确说"准确长期预测不可能"）、gyroscope（3D，Euler 角；含进动 + 章动）。
- **数据**：每系统 800 条训练初始条件 × 100 步；另 100 条测试；每坐标独立加 $\sigma=0.01$ 高斯噪声。积分步长 N-pendulum 0.03s、gyroscope 0.02s，**用 RK4 而非辛积分器**（作者理由：辛积分器不便引入控制与耗散）。
- **网络**：全连接前馈，3 隐层 × 256 神经元，Softplus 激活；AdamW，1000 epoch，lr 1e-3。
- **指标**：训练用 $L_1$（对离群点鲁棒）；评估用**轨迹相对误差** $\sum_t \|\hat s_t - s_t\|_2 / \|s_t\|_2$ 与**真实能量绝对误差** $\|\hat E_t - E_t\|_2$（作者明确说**不用相对能量误差**，因为势能零点是任意选的）。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $\delta q^{\top}\left[\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q}\right)-\frac{\partial L}{\partial q}\right]=0$ | D'Alembert 原理（虚功原理）的拉格朗日形式 |
| Eq. 3 | $\ddot q = (\nabla_{\dot q}\nabla_{\dot q}^{\top}L)^{-1}[\nabla_q L - (\nabla_q\nabla_{\dot q}^{\top}L)\dot q]$ | Euler-Lagrange 方程的显式展开（**LNN 直接参数化 $L$ 后走这条**） |
| Eq. 5 | $\ddot q = M(q)^{-1}[-\dot M(q)\dot q + \tfrac12\nabla_q(\dot q^{\top}M(q)\dot q) - \nabla_q V(q)]$ | 结构化拉格朗日动力学（**LNN-structure / DeLaN 走这条**） |
| Eq. 6-7 | $\dot q=\nabla_p H$，$\dot p=-\nabla_q H$；$\dot H=(\nabla_q^{\top}H)\dot q+(\nabla_p^{\top}H)\dot p=0$ | 哈密顿动力学与**能量守恒的结构性来源**（不是惩罚项） |
| Eq. 14 | $\lambda_L = -[(D_x\Phi)M^{-1}(D_x\Phi)^{\top}]^{-1}[(D_x\Phi)M^{-1}\nabla_x V - (D_x\dot\Phi)\dot x]$ | 显式约束下的 Lagrange 乘子解析解 |
| Eq. 20 | $\dot z = J\nabla_z H - J(D_z\Psi)^{\top}[(D_z\Psi)J(D_z\Psi)^{\top}]^{-1}(D_z\Psi)J\nabla_z H$ | 显式约束下的哈密顿动力学（**CHNN 走这条**） |

## 5. 关键成果与贡献

### 5.1 主结果（Figure 2 / 3 / 4，论文只给图不给数表 —— 见 §6.1）

作者在三个评估维度上的**定性结论**（论文原文措辞，不做外推）：

1. **显式约束模型（CLNN / CHNN）在全部 4 个系统上一致优于隐式约束模型**。"The gap is the largest in the 3D system (Gyroscope) since with explicit constraints, the dynamics under Cartesian coordinates do not suffer from **gimbal lock**"。作者归因：笛卡尔坐标下的质量矩阵**坐标无关**，而角坐标下质量矩阵是坐标的复杂函数。
2. **角度感知设计普遍有益**（LNN-Angle vs LNN、HNN-Angle vs HNN）。作者理由："respecting the geometry of coordinates lets models identify $q$ and $q \pm 2\pi$"。
3. **结构化拉格朗日/哈密顿量对 2-pendulum 与 4-pendulum 有益，但对 1-pendulum 和 gyroscope 无益**（作者明确说 1-pendulum 动力学简单、不用结构也能学好；对 3D gyroscope "**it is unclear why the structure does not help. We leave it as an open question for future work**"）。
4. **长时间外推（T=20，训练只到 T=4）**：CHNN/CLNN 最好；2-pendulum 与 4-pendulum 是混沌系统，"accurate long-term prediction is not possible"。
5. **数据效率（2-pendulum，10 → 10,000 条训练轨迹）**：CLNN/CHNN 全面最优；"prior knowledge (structure/angle-aware) benefits data efficiency"。
6. **能量漂移（Figure 4，本篇最重要的一条）**：**所有十个模型都漂离真实恒定的能量**。"although each model conserves its own learned energy, they all drift away from the constant true energy as time goes by"。CLNN/CHNN 漂移相对最低。

### 5.2 机制分析：参数化病态与计算代价（§5 结论段）

- **所有 Hamiltonian 模型（CHNN 除外）与 LNN-structure / LNN-structure-angle**都需要学习坐标相关的质量矩阵，**其初始化的参数化"tend to be ill-conditioned, which makes training unstable"**。论文的处理是"加一个小的正对角常数"来缓解，并明确指出"**this caveat complicates the training of these implicit models**"。
- **LNN / LNN-angle**（直接参数化 $L$）不学质量矩阵 —— 它由拉格朗日量的 **Hessian** 隐式导出；由于 Hessian 需要两次梯度运算，这两个模型**额外付出计算代价、训练更慢**。

### 5.3 贡献清单
1. **统一推导 + 统一协议**下横向比较十个能量守恒网络模型，覆盖 4 个物理系统 × 3 个评估维度。
2. **把"约束的显式性"确立为一个独立的、可比较的结构化归纳偏置维度**，并给出机制解释（质量矩阵坐标无关性 + 无万向节死锁）。
3. **把"角度感知"与"结构化能量函数"拆成两个可独立消融的轴**，并给出**非单调**的结论（结构化对 2D 有益、对 3D gyroscope 无益且原因未明）。
4. **指出"能量守恒"与"能量正确"是两件事**：结构守恒的是**学到的**能量，真能量仍会漂移。

## 6. 局限性与未来展望

### 6.1 作者自陈局限
- **主结果全部以图给出，论文正文没有数值表**（Figure 2/3/4/5 均为 log-scale 曲线图），因此**任何"CLNN 比 LNN 好 X%"式的数字都无从引用**。
- **gyroscope 上结构化无益的原因未明**，作者明确留作 open question。
- **用的是 RK4 而非辛积分器**（理由是辛积分器不便引入控制与耗散）—— 这意味着**本基准测的是"用哪种物理结构"，不是"用哪种积分器"**，且对守恒性最严格的场景（纯 Hamiltonian）放弃了最合适的工具。
- **10 epoch/1000 epoch 单次运行，无方差报告**，无统计检验。
- 只在 **synthetic** 系统上评测（1/2/4-pendulum + gyroscope），**无真实实验数据**。

### 6.2 隐含局限（本仓视角）
- **(a) 🔴 最关键：本论文与本仓主线（Liquid NN）无关。** 论文里的 LNN = Lagrangian NN。**不得**把本论文的任何结论（LNN-structure 更优、CLNN/CHNN 最优）当作"液态网络"的证据引用。**本仓 AGENTS.md §0 的九条禁止 claim 中没有一条能被本论文支持或反驳** —— 它连 CfC/LTC 都没提。
- **(b) 与 LTC/CfC 的关系是"祖孙"而非"同类"**：本论文的 Eq. 2（LTC 的形式）与 Eq. 3（CfC 的闭式解）**恰好在 §2.1.1 被 LFLDNets 那篇引用过**，但**本论文本身完全没有讨论 LTC/CfC**（它比 LTC 早 1 年、比 CfC 早 3 年）。论文里"连续时间 ODE"的写法与本仓 `ltc.py` 的实现**不共享任何机制**。
- **(c) 隐式约束下的"病态质量矩阵"结论对本仓有间接意义，但外推需谨慎** —— 本仓的 CfC/LTC **没有质量矩阵概念**，质量矩阵是力学系统的属性；把这条结论搬到连续时间 RNN 上是**范畴错误**。
- **(d) Figure 3 的长时间外推结论在混沌系统上被作者自己标为"不可能准确"** —— 因此"CLNN/CHNN 长时间外推最好"实际只在**非混沌/近极限环**区间成立。
- **(e) 缺一个关键对照**：论文**没有与"黑盒 Neural ODE（同参数量）"在同等条件下的对比** —— Neural ODE 是它引言里的基线，但正文把 Neural ODE 当作**实现载体**（用它来学 (5)/(9)），而不是**竞争模型**。因此"结构化归纳偏置 vs 无归纳偏置"这一最核心的问题**没有在本论文内被回答**。

### 6.3 未来方向
- gyroscope 上结构化为何无益（作者留白）。
- 从结构化动力学**设计基于能量的控制器**（作者展望：CLNN/CHNN 的结构可用来做 energy-based controllers）。
- 引入控制项与耗散项的更完整比较（Dissipative SymODEN / Port-Hamiltonian NN 被引用但未进基准）。

## 7. 对本仓的意义

### 7.1 引用纪律（本报告最重要的一条）
**"LNN" 这个缩写在本仓有两条互不相干的谱系**：
- **Liquid Neural Network**（液态神经网络，Hasani/Lechner 2020-2021，LTC / CfC / NCP 谱系）—— 本仓主线。
- **Lagrangian Neural Network**（拉格朗日神经网络，Cranmer 2020，HNN / DeLaN / CHNN / CLNN 谱系）—— 力学系统归纳偏置。

**两者共享首字母缩写，且都在 arXiv 上用 "LNN" 自称**。本仓 digest 流水线的 `应用场景` 词组 `all:LNN AND all:forecasting` 之类就是被这个撞车污染的（详见 §7.3）。**今后凡见 "LNN" 必须先看标题/摘要里有没有 "Lagrangian" 或 "Liquid" 字样**；本仓 `docs/LNN_深度研读报告.md` 的研读条目里，**每一条都应显式标注属于哪一条谱系**。

### 7.2 方法论可落地：三条消融轴
1. **角度感知轴（angle-aware）** —— 本仓的 CfC/LTC 状态若用于表示角度变量（如机械臂关节角、相位、周期信号），**是否存在 $q$ 与 $q\pm2\pi$ 的等变性破损**？本仓**尚无任何 angle-aware 实现**（`lnn/core/` 下所有 cell 都是普通实值状态）。可落地为 `scripts/bench_cfc_angle_aware.py`：对带角度的合成序列，对比 `(cos q, sin q)` 嵌入 vs 原始实值。
2. **约束显式性轴** —— 本仓库**没有任何模型显式施加物理约束**（本仓是序列建模/形变场/医疗影像方向，不是力学系统）。因此这条轴**对本仓主线优先级低**，但对 `analysis/` 下的任何力学/机器人实验（如 `bench_*` 里的机械臂）是可借鉴的外部协议。
3. **积分器阶数轴（与 11-13 Lipschitz RNN 报告的发现合并）** —— 11-13 报告新增的"积分器阶数（Euler vs RK2 vs odeint RK4）"消融轴，在本篇得到**一个外部佐证**：本篇为了引入控制/耗散而**放弃辛积分器改用 RK4**，说明"积分器选择"与"物理结构选择"在本领域确实是**两个被混在一起的变量**。本仓应把这两条**分开测**。

### 7.3 🔴 本篇暴露的流水线缺陷：布尔词组造成系统性污染

本篇进入候选池的**唯一路径**是 `all:LNN AND all:forecasting`（`应用场景` 词组）。按本仓流水线的 `ARXIV_TERM_GROUPS` 实测（2026-11-14）：

- **今日候选池 95 篇里，有 13 篇的唯一来源是 `all:LNN AND ...` 布尔词组**；
- 其中**至少 5 篇是纯 Lagrangian-NN 撞车**：`2608.13215` (HiLNN)、`2603.14333` (STRIDE/quadruped LNN)、`2603.08478` (STRIDE)、`2506.16079`、`2403.00381`；
- 其余的还有 `2606.28827` (LNN-Fly)、`2506.16079`、`2410.01855` (Localized NN，**LNN = Localized Neural Network** —— 第三种撞车)、`2306.05593`、`2405.12803` 等。

**结论**：`all:LNN AND all:<场景词>` 这条词组的**精确率极低**，因为 arXiv 的 `all:` 域匹配的是**自由文本**，`LNN` 这个 token 在 2026 年的 arXiv 上至少有 **4 种互不相干的展开**（Liquid / Lagrangian / Localized / Large？）。**这条词组不是"覆盖应用场景"，它是"收集缩写撞车"。**

### 7.4 一条可直接迁移的教训：本仓 `LRCCell` 命名陷阱的同款
本仓在 10-28 索引中已记录：`lnn/core/variants.py:346` 的 `LRCCell` **命名来自 arXiv:2403.08791，实现却是该文要修正的"耦合 LC"**。本论文展示了**同一类错误在学术界的普遍形态**：十个模型名字各不相同（DeLaN / SymODEN / SRNN / CHNN / CLNN），**但"名字"与"实际参数化了什么"是正交的**。作者为此**重命名了模型**（HNN vs HNN-structure、LNN vs LNN-structure）—— 这说明**按论文名引用结构化模型是不可靠的，必须核对源码/公式**。本仓引用任何"某某-Net"时，应同样**先核对公式再核对名字**。

### 7.5 Verdict
- **"十个能量守恒网络模型的统一横向基准" → TARGET-POSITIVE（外部方法论）**：为"哪些物理归纳偏置真正有用"提供了本仓目前缺失的**多轴消融范式**（角度感知 / 结构化 / 约束显式性 / 数据效率 / 能量漂移）。
- **"显式约束（CLNN/CHNN）全面最优" → TARGET-DEPENDENT-WITH-NUANCE**：论文自己指出 gyroscope 上结构化无益且原因未明；主结果只有图无数值；chaotic 系统的长期外推被作者标为"不可能准确"。**引用须带这三条限定**。
- **"能量守恒 ⇒ 能量正确" → TARGET-NEGATIVE**：论文自己证伪 —— 十个模型**全部**漂离真实恒定能量。"结构守恒的是学到的能量，不是真实能量"这条应写进本仓的物理先验引用纪律。
- **"本论文可作为液态网络（CfC/LTC/NCP）的证据" → TARGET-NEGATIVE（范畴错误）**：**本论文与液态网络谱系无关**，LNN 在此 = Lagrangian Neural Network。**引用即构成 §0 违规。**
