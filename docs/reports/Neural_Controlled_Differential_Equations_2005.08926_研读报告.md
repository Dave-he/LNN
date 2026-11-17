---
title: Neural Controlled Differential Equations for Irregular Time Series
date: 2026-11
tags: [LNN, Continuous-Time, Neural-CDE, Irregular-Sampling, Rough-Path, Universal-Approximation, Adjoint, Partially-Observed]
---

# 研读报告：Neural CDE — 用受控微分方程把"数据本身"变成 ODE 的驱动源

## 1. 元数据

- **论文标题**：Neural Controlled Differential Equations for Irregular Time Series
- **作者**：Patrick Kidger, James Morrill, James Foster, Terry Lyons (Mathematical Institute, University of Oxford; The Alan Turing Institute, British Library)
- **发表时间**：2020-05-05 (v1) / 2020-11-05 (v2)，NeurIPS 2020
- **来源**：arXiv:2005.08926v2 [cs.LG]
- **代码**：https://github.com/patrick-kidger/NeuralCDE ，库 https://github.com/patrick-kidger/torchcde
- **本地 PDF**：[papers/2005.08926/2005.08926.pdf](../../papers/2005.08926/2005.08926.pdf) ／ 文本 [papers/2005.08926/2005.08926.txt](../../papers/2005.08926/2005.08926.txt)
- **数据集**：CharacterTrajectories（不规则、部分观测）、PhysioNet 2019 sepsis（不规则 + 部分观测，40335 条序列，观测率仅 10.3%）、Speech Commands（等间隔完全观测）
- **关联概念**：Neural ODE / CDE / 粗糙路径 / Riemann–Stieltjes 积分 / 不规则采样 / 通用逼近 / adjoint 反向传播 / LTC / CfC

## 2. 核心问题

连续时间模型（Neural ODE 系）在序列建模上的根本缺陷不是数值，而是**因果结构上的**：一个 ODE 的解由初值唯一确定，一旦 $\theta$ 学完，轨迹就被锁死，**后到的观测没有任何机制去修正已经发生的轨迹**。论文原文：

> "the solution to an ordinary differential equation is determined by its initial condition, and there is no mechanism for adjusting the trajectory based on subsequent observations."

这个缺陷在**不规则采样 + 部分观测**的数据上被放大到不可回避：真实 ICU 监测里 10.3% 的格子有值，剩下全是"没测"而非"测到 0"；若沿用 RNN 路线，常规做法是**分箱（binning）或插补（imputing）**——这是把连续时间问题伪装成离散时间问题，代价是丢掉了"观测本身携带信息"这一维度（观测频率本身可能是信号：医生对高风险病人测得更勤）。

已有的连续时间补救方案各有硬伤：(a) **GRU-ODE** 一类"把 $X_s$ 直接喂进向量场"的做法，其向量场 $\dot z = h_\theta(z, X_s)$ 依赖 $X_s$ 的**当前值**而非**增量**；(b) **ODE-RNN / GRU-$\Delta t$** 一类在每个观测点**打断** ODE 做隐状态更新的做法，破坏了 adjoint 反向传播的连续性，内存从 $O(H)$ 退化到 $O(LH)$；(c) **GRU-D** 靠衰减掩码修补缺失，属于事后启发式。

论文把问题重新表述为一个数学问题：**能否让隐藏状态的连续演化被"观测过程本身"所驱动，且驱动的方式是路径增量（differential）而非路径取值（value）**？答案是用粗糙路径理论里的**受控微分方程（CDE）**：$X$ 不是被"读"进模型的，而是**模型的驱动信号**。

## 3. 方法论与核心思路

### 3.1 总体架构

Neural CDE 的骨架只有三层，但每一层都换掉了 RNN 的一个根本部件：

| 部件 | RNN 的做法 | Neural CDE 的做法 |
|---|---|---|
| 状态演化 | 离散跳变 $h_t = f(h_{t-1}, x_t)$ | 连续受控流 $z_t = z_{t_0} + \int_{t_0}^t f_\theta(z_s)\,dX_s$ |
| 观测进入方式 | 作为**取值** $x_t$ | 作为**路径增量** $dX_s$（路径导数） |
| 时间戳 | 忽略（或塞进 $\Delta t$ 通道） | 天然编码进路径 $X_{t_i} = (x_i, t_i)$ 的时间分量 |
| 缺失值 | 分箱 / 插补 / 衰减掩码 | 每通道独立插值成路径，缺失即"没有增量" |
| 反向传播 | BPTT，$O(LH)$ 内存 | **adjoint**，$O(L+H)$ 内存 |
| 观测强度 (intensity) | 需专门学一个强度函数 | 追加一个索引通道，**免学习** |

论文的类比很贴切：**Neural ODE 是 ResNet 的连续类比，Neural CDE 是 RNN 的连续类比**。

### 3.2 核心模型：受控微分方程 (Eq. 2 / Eq. 3)

设 $X:[\tau,T]\to\mathbb{R}^v$ 是有界变差路径（例如 Lipschitz 即可保证），$f:\mathbb{R}^w\to\mathbb{R}^{w\times v}$ 连续。定义 CDE：

$$
z_t = z_\tau + \int_\tau^t f(z_s)\,dX_s, \qquad t\in(\tau,T]
\tag{Eq. 2}
$$

这是 **Riemann–Stieltjes 积分**（非 Lebesgue/Riemann），$f(z_s)\in\mathbb{R}^{w\times v}$ 与 $X_s\in\mathbb{R}^v$ 做矩阵–向量乘积。在 $f$ 满足全局 Lipschitz 时解**全局存在且唯一**（引 [20, Theorem 1.3]）。

具体到神经模型：给定不规则采样序列 $x=((t_0,x_0),\dots,(t_n,x_n))$，取 $X:[t_0,t_n]\to\mathbb{R}^{v+1}$ 为以 $\{t_i\}$ 为节点、满足 $X_{t_i}=(x_i,t_i)$ 的**自然三次样条**（选自然样条是因为它具有处理边界情形所需的"最低正则性"）。于是：

$$
z_t = z_{t_0} + \int_{t_0}^t f_\theta(z_s)\,dX_s,\qquad z_{t_0}=\zeta_\theta(x_0,t_0)
\tag{Eq. 3}
$$

**与 Neural ODE 的唯一结构性差异**：Eq. 3 被 $X$ 驱动，而 Neural ODE 只被恒等映射 $\iota:\mathbb{R}\to\mathbb{R}$ 驱动。形式上极小，语义上极大——**数据的变化会即时改变系统的局部动力学**。

### 3.3 求解：把 Stieltjes 积分化回普通 ODE (Eq. 4 / Eq. 5)

因为 $X$ 实际可微，可以定义

$$
g_{\theta,X}(z,s) = f_\theta(z)\,\frac{dX}{ds}(s)
\tag{Eq. 4}
$$

于是

$$
z_t = z_{t_0} + \int_{t_0}^t f_\theta(z_s)\,dX_s = z_{t_0}+\int_{t_0}^t g_{\theta,X}(z_s,s)\,ds
\tag{Eq. 5}
$$

**这是本文最实用的一句话**：Neural CDE 可以直接用现成的 `torchdiffeq` 求解，**零修改**。也就是说，模型的理论形态是 CDE，工程形态仍是标准 ODE 求解器。

### 3.4 与替代 ODE 模型的表达力关系（Theorem C.1）

论文给出一个关键的**单向包含**定理（非正式表述）：

> 任何形如 $z_t = z_0+\int_{t_0}^t h_\theta(z_s,X_s)\,ds$ 的方程，都能被 Neural CDE **精确**表示；**反过来不行**。

关键直觉：Neural CDE 能轻松表示路径之间的**恒等映射**，而 $h_\theta(z,X_s)$ 形式做不到。论文明确把这一理论差异与实验差异挂钩：*"In our experiments, we find that the Neural CDE substantially outperforms the GRU-ODE, which we speculate is a consequence of this result."*（注意措辞是 speculate，因果未严格证明。）

### 3.5 adjoint 反向传播与内存

Neural ODE 的 adjoint 法只需 $O(H)$ 内存（$H$ 为向量场内存足迹），但**跨越 ODE 的跳跃点无法做 adjoint**。前人做时间序列时正是"在每个观测处打断 ODE 并更新隐状态"，于是内存退回 $O(LH)$。

Neural CDE 因为通过 $dX_s$ **连续地**吸收数据、**不打断**微分方程，adjoint 可以全程使用 ⇒ **总内存 $O(L+H)$**。论文同时提醒：adjoint 需谨慎使用（有些系统正向/反向不稳定），但**有限时间爆炸不是问题**（全局 Lipschitz 保证，ReLU/tanh 即可满足）。

### 3.6 观测强度作为通道（免学习）

医生对高风险病人测得更频繁——**观测频率本身携带信息**。以往做法要专门学一个强度函数。本文给了一个**完全免学习**的做法：把观测的**索引** $i$ 与 $x_i$ 拼接，构造路径 $X$；由于模型用的是 $\frac{dX}{ds}$，该通道的导数**就是**累积观测强度。部分观测时，给每个原始通道 $o_j$ 配一个独立强度通道 $c_j$（每观测一次就自增）。

### 3.7 与 LTC / CfC / NCP 的关系（本仓视角）

这是本仓最需要澄清的一层，因为多篇历史研读把"连续时间"当成一个统一卖点：

- **与 LTC / CfC 的关系 = 弱耦合，不是同族**。LTC 的动力学是 $dx/dt = -\frac{1}{\tau + f(x,I)}x + f(x,I)A$，CfC 是它的闭式解。**两者的 ODE 都是"自驱动"的**——向量场只依赖状态与输入，$\Delta t$ 是积分步长。而 Neural CDE 的向量场**显式依赖数据路径的导数** $\frac{dX}{ds}$，这是一个**受控**（controlled）而非**自治**（autonomous）系统。
- **与 NCP 的关系 = 家族不同**。NCP（Lechner 2020）是固定时间常数 + 离散稀疏接线；Neural CDE 是连续受控流。两者都属于"稀疏/低秩接线 + 连续时间"的话语空间，但机制不同。
- **一个关键的实践差异**：LTC/CfC 的 $\Delta t$ 通常被当作"积分了多少时间"使用（因此**时间戳的绝对值和不等间隔信息被部分抹掉**），而 Neural CDE 把时间戳**放进路径 $X$**（$X_{t_i}=(x_i,t_i)$），因此时间轴本身参与驱动。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $y\approx \ell_1^\theta(z_T),\quad z_t=z_0+\int_0^t f_\theta(z_s)ds,\ z_0=\ell_2^\theta(x)$ | Neural ODE 基线：无数据驱动，轨迹由初值锁定 |
| Eq. 2 | $z_t=z_\tau+\int_\tau^t f(z_s)\,dX_s$ | **CDE 定义**（Riemann–Stieltjes 积分），驱动源是路径 $X$ |
| Eq. 3 | $z_t=z_{t_0}+\int_{t_0}^t f_\theta(z_s)\,dX_s,\ z_{t_0}=\zeta_\theta(x_0,t_0)$ | **Neural CDE 模型**，$X$ 为以 $\{t_i\}$ 为节点的自然三次样条，$X_{t_i}=(x_i,t_i)$ |
| Eq. 4 | $g_{\theta,X}(z,s)=f_\theta(z)\frac{dX}{ds}(s)$ | 把 Stieltjes 积分折成向量场 |
| Eq. 5 | $z_t=z_{t_0}+\int_{t_0}^t g_{\theta,X}(z_s,s)\,ds$ | **可用标准 ODE 求解器求解**（`torchdiffeq` 零修改） |
| Thm C.1 | $h_\theta(z_s,X_s)$ 型 $\Rightarrow$ 可被 Neural CDE 精确表示；**反向不成立** | Neural CDE 严格强于 GRU-ODE 型模型 |
| Thm B.14 | 线性映射作用于 Neural CDE 终态 ⇒ **通用逼近**（流形上函数基的逼近） | 形式化通用逼近定理 |
| — | 内存 $O(L+H)$ vs BPTT 的 $O(LH)$ | adjoint 跨观测点可用，因路径连续吸收数据 |

## 5. 关键成果与贡献

### 5.1 主结果：CharacterTrajectories（不规则 + 部分观测）

Test accuracy（5 次运行 mean ± std）与训练内存：

| Model | 30% dropped | 50% dropped | 70% dropped | Memory (MB) |
|---|---|---|---|---|
| GRU-ODE | 92.6% ± 1.6% | 86.7% ± 3.9% | 89.9% ± 3.7% | 1.5 |
| GRU-$\Delta t$ | 93.6% ± 2.0% | 91.3% ± 2.1% | 90.4% ± 0.8% | 15.8 |
| GRU-D | 94.2% ± 2.1% | 90.2% ± 4.8% | 91.9% ± 1.7% | 17.0 |
| ODE-RNN | 95.4% ± 0.6% | 96.0% ± 0.3% | 95.3% ± 0.6% | 14.8 |
| **Neural CDE (ours)** | **98.7% ± 0.8%** | **98.8% ± 0.2%** | **98.6% ± 0.4%** | **1.3** |

**关键读法**：Neural CDE 在丢 30%→70% 观测时**准确率基本不变**（98.7 → 98.6），而其余全部模型随丢弃率上升而下降。内存 1.3 MB，是 ODE-RNN（14.8 MB）的 **1/11**，是 GRU-$\Delta t$（15.8 MB）的 **1/12**。

### 5.2 主结果：PhysioNet 2019 sepsis（不规则 + 部分观测）

Test AUC：

| Model | OI（有观测强度） | No OI | Memory (MB) OI / No OI |
|---|---|---|---|
| GRU-ODE | 0.852 ± 0.010 | 0.771 ± 0.024 | 454 / 273 |
| GRU-$\Delta t$ | 0.878 ± 0.006 | 0.840 ± 0.007 | 837 / 826 |
| GRU-D | 0.871 ± 0.022 | 0.850 ± 0.013 | 889 / 878 |
| ODE-RNN | 0.874 ± 0.016 | 0.833 ± 0.020 | 696 / 686 |
| **Neural CDE (ours)** | **0.880 ± 0.006** | 0.776 ± 0.009 | **244 / 122** |

**关键读法（含论文自陈的不公平）**：有观测强度时 Neural CDE 最好（0.880），但 ODE-RNN（0.874）与 GRU-$\Delta t$（0.878）**非常接近**——差距在 std 量级内，**不能写成"大幅领先"**。去掉观测强度后所有模型都显著变差，Neural CDE 尤其（0.880 → 0.776）。**论文自己脚注承认这一对比不完全公平**：ODE-RNN / GRU-D / GRU-$\Delta t$ 始终接收 $\Delta t$ 作为输入，即使在 "No OI" 情形下它们仍知道数据是不规则采样的。数据集极度不平衡（5% 正例率），故报 AUC 而非 accuracy。

### 5.3 主结果：Speech Commands（等间隔 + 完全观测）

| Model | Test Accuracy | Memory (GB) |
|---|---|---|
| GRU-ODE | 47.9% ± 2.9% | 0.164 |
| GRU-$\Delta t$ | 43.3% ± 33.9% | 1.54 |
| GRU-D | 32.4% ± 34.8% | 1.64 |
| ODE-RNN | 65.9% ± 35.6% | 1.40 |
| **Neural CDE (ours)** | **89.8% ± 2.5%** | **0.167** |

**关键读法**：**基线的巨大方差（±33.9%、±34.8%、±35.6%）说明它们经常训练失败**，作者"无法通过调优化器解决"。作者明确写道：*"The best GRU-$\Delta t$, GRU-D and ODE-RNN models did match the performance of the Neural CDE, suggesting that on a regularly-spaced problem all approaches can be made to work equally well."* ⇒ **在等间隔场景下，Neural CDE 的优势主要体现为训练稳定性而非精度上限**。论文坦承"未对此现象做正式调查"。

### 5.4 贡献清单

1. **把"数据驱动连续时间模型"的表达力缺口定位到结构层面**，并用粗糙路径理论给出精确的解法（$dX_s$ 而非 $X_s$）。
2. **给出严格单向包含定理**（Thm C.1）：Neural CDE ⊃ GRU-ODE 型模型，反向不成立——这是本仓"连续时间机制优越性"叙事**最需要的一个反例来源**（见 §7）。
3. **在 adjoint 反向传播下实现 $O(L+H)$ 内存**，且**跨观测点不打断 ODE**（前人做法因打断而退化到 $O(LH)$）。
4. **部分观测与观测强度的统一免费处理**：每通道独立插值 + 追加索引通道作为免学习强度。
5. **端到端实证**：3 个数据集、6 个基线，覆盖不规则/部分观测/等间隔三种采样形态。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

1. **adjoint 方法需谨慎**：某些系统正向与反向评估不稳定（引 [29,30]）；有限时间爆炸不构成问题（全局 Lipschitz 保证）。
2. **训练稳定性的说法是轶事式的**：*"we did not perform a formal investigation of this phenomenon"*——"Neural CDE 训练快、对超参鲁棒"这一观察**未经系统验证**。
3. **Thm C.1 与实验差异之间只是推测**：作者用 "we speculate" 连接理论与 GRU-ODE 的表现差距。
4. **通用逼近定理的技术性**：Thm B.14 被放在附录，作者称其"somewhat technical"。

### 6.2 隐含局限（本仓视角）

1. **对 $X$ 的正则性要求**：自然三次样条要求节点处二阶连续，构造 $X$ 本身有开销；论文没给"路径构造成本 vs 收益"的分析（只给了显存，未给 wall-clock）。
2. **CDE 的 $f_\theta$ 是自治的、但依赖数据路径——这引入了一个耦合难题**：$f_\theta$ 的输出被 $dX_s$ 调制，意味着**数据噪声直接进入向量场**，论文没有系统讨论对噪声/异常值的鲁棒性（而这正是本仓 NR-02 的核心议题）。
3. **未与 LTC/CfC 做对照**：论文比较的全部是 ODE-RNN / GRU-D / GRU-$\Delta t$ / GRU-ODE，**没有任何 LTC 或 CfC 基线**。因此"Neural CDE 优于连续时间家族"这一命题在本文中**未被检验**。
4. **无方差报告的 benchmark 对比**：Speech Commands 上其他方法 ±35% 的方差意味着"GRU-ODE 47.9%"与"ODE-RNN 65.9%"这两个数字本身极不稳定，跨论文引用单点数值风险高。
5. **数据集规模偏小/任务偏经典**：CharacterTrajectories 与 Speech Commands 都是小规模分类任务，与本仓关心的长序列连续控制/流形建模有距离。

### 6.3 未来方向

- 论文给出的开放问题：**sup-universality 是否能推广到比 $\mathcal{D}^2$ 更大的映射类**（此问题在同期的 Teshima et al. 2012.02414 中被部分回答——见同日研读）。
- 量化"逼近一个给定双 Lipschitz 微分同胚需要多少层 NODE"。
- 本仓可推进的方向：把 Neural CDE 的**受控**机制（$dX_s$ 驱动）移植到 CfC 上，形成 **Controlled CfC**，检验"受控 vs 自治"是否真的是性能来源。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的什么变体**：新增一个 **ControlledCfCCell**。本仓现有 `cfc.py` 的闭式解是

$$x(t)=\sigma(-f(x,I;\theta_f)\,t)\,g(x,I;\theta_g) + \big[1-\sigma(-f(x,I;\theta_f)\,t)\big]\,h(x,I;\theta_h)$$

其中 $t$ 就是 $\Delta t$，**输入只通过 $f,g,h$ 的值进入**，$\Delta t$ 只决定积分长度——即**自治**的。而 Neural CDE 的做法是把 $x$ 与 $\Delta t$ 一起构成路径 $X$，用 $\frac{dX}{ds}$ 调制向量场。落地方式：把闭式解中的 $t$ 换成一个**由路径导数驱动的调制项**（例如让 $f$ 的输入包含 $dx/ds$ 的低维投影）。**这条改造同时测试本仓两个未解问题**：① 连续时间收益到底来自"多时间尺度"还是来自"数据驱动"；② 不等间隔采样是否真的能带来增益（见下）。

**可嫁接到哪个 bench 脚本**：本仓 toy benchmark 目前**只有等间隔采样**。本篇提供的现成诊断是**直接沿用论文的 Table 1 协议**（丢 30%/50%/70% 观测 + 训练内存），以及 §6 的核心对照——**论文的 GRU-$\Delta t$ 基线**。注意 §6 的诚实负结果（CT-GRU 1710.04110）已经证明"**打乱 $\Delta t$ 但不使用它**"会掉 ≤5%；Neural CDE 提示了对照的另一面：**把 $\Delta t$ 真正用于驱动状态**（而非丢弃）是否带来增益。这两条可以组合成一条**此前本仓完全缺失的消融轴**：

$$\text{轴 A：}\ \Delta t \text{ 的三种处置}\quad
\{\ \text{忽略},\ \ \text{仅作为标量输入通道},\ \ \text{作为路径导数驱动向量场}\ \}$$

本仓现有全部 bench 脚本**只实现了第一种**。

**提供哪个新 ablation axis**：
1. 🔴 **"受控 vs 自治"轴**（本文 Theorem C.1 给出的严格单向包含关系）——这是**本仓至今缺失的最强理论基准**：它说明"把数据放进向量场"不是实现细节，而是**表达力的严格提升**。任何声称"LNN 因连续时间而强"的论证，都应先回答"你的模型受控还是自治？"
2. 🔴 **观测强度通道轴**（§3.5 的免学习做法）：本仓所有动态系统都隐含"固定采样率"，把观测索引作为通道是**零成本**的鲁棒性增强。
3. **内存轴**：adjoint vs BPTT 的 $O(L+H)$ vs $O(LH)$，对本仓长序列 bench 的可扩展性有直接意义（尤其多 $\tau$ 变体库）。

**Verdict**：
- **TARGET-NEGATIVE-WITH-NUANCE（对"连续时间 = 更强"这一命题）**：本文**不是** LNN 家族的论文（无 LTC/CfC），但它提供了一个**严格的反向标尺**——在同样"连续时间"的大类里，**"受控"比"自治"表达力严格更强，而 GRU-ODE（自治 + $\Delta t$ 通道）与受控版差距巨大**。这与本仓 11-13 轮 CT-GRU 的负结果方向一致（"显式时间常数建模未必带来优势"）——**共同指向：连续时间机制的收益更可能来自"数据如何进入模型"而非"状态是否连续演化"**。
- **TARGET-DEPENDENT-WITH-NUANCE（对"不规则采样场景"）**：在**真正不规则 + 部分观测**的任务上本文证据很强（丢弃率 70% 仍 98.6%，且内存 1/11）；但**在等间隔场景上优势主要来自训练稳定性而非精度**（作者自陈 best baseline 可持平）。因此引用时必须带"仅在不规则/部分观测任务上成立"这一限定。

**引用纪律**：① 本篇**没有** LTC/CfC 对照，**不得**用作"LNN 家族优于其他连续时间模型"的证据；② PhysioNet 上 0.880 vs 0.878 的差距**在 std 内**，不得写成"大幅领先"；③ Speech Commands 上其他基线的 ±35% 方差意味着该表的绝对数值不可跨论文引用。
