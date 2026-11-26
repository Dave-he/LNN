---
title: "Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time"
date: 2026-04
tags: [LNN, Continuous-Time, Neural-ODE, Causal-Inference, Observability, FrontDoor, Dynamic-Treatment-Regime, Irregular-Sampling, Clinical, Preprint]
---

# 研读报告：ObsNODE — 把"可观测性"从**检验假设**变成**按构造保证**，再用它换来连续时间因果识别

> **本篇与本仓主线的位置**：论文标题里的 "Neural ODE" 会让人误以为是"液态/连续时间有效"的证据。**它不是。** 全文（含附录、参考文献）**`Liquid` / `LTC` / `CfC` / `NCP` / `Closed-Form` 零出现**，基线全部是 Neural ODE / Neural CDE 系（IGC-Net、SCIP-Net、OptAB、DoseAI），且其中 OptAB 与 DoseAI 就是作者团队自己的前作。⇒ 本篇的**新意在因果识别的数学结构**（可观测性 ⇒ 条件前门 ⇒ 状态空间调整公式），**不在时序骨干**。本仓引用它时必须显式标注"族外 / 与液态机制优劣无关"。
>
> 🔴 **但本篇对本仓有一条真价值**：它是本仓见过的**第一篇把"结构性保证"落到神经 ODE 潜变量参数化上的论文** —— 不是在损失里加正则，而是**直接把潜状态参数化成三角可观测标准形**，使"状态可从观测历史唯一重建"成为构造性事实。本仓 11-22 结论里列的"构造性保证型变体"候选清单（`PortHamiltonianCfCCell` 等）正属于这一类，而本篇提供了一个**已被同行评审社区接受、且给出识别定理**的具体实例。
>
> ⚠️ **本篇正文存在一处可复算的口径错标**（把"平均绝对 CB"写成"mean absolute ATE error"），本仓已复算并定位，见 §7。

## 1. 元数据

- **论文标题**：Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time
- **作者**：Jennifer Wendland（wendland@uni-koblenz.de）、Nicolas Freitag（nfreitag@uni-koblenz.de）、Maik Kschischo（kschischo@uni-koblenz.de）—— 单一机构，无标注共同一作 †
- **机构**：Department of Computer Science, University of Koblenz, 53424 Koblenz, Germany（单一机构）
- **发表时间**：2026-04-28（arXiv 首次提交，arXiv 元数据 `published: 2026-04-28T19:18:42Z`；PDF 页眉标注 `arXiv:2604.26070v3 [cs.LG] 6 Oct 2026`，即当前读到的是 **v3 版（2026-10-06）**）。**论文正文本身未写首次提交日期。**
- **arXiv 分类**：PDF 页眉仅标 `[cs.LG]`；arXiv API 元数据为 `cs.LG, math.OC, math.ST, q-bio.QM`（其余三个来自 arXiv API，非论文正文）
- **出版状态**：论文第 1 页页脚标 `Preprint.`。**论文未提及**投稿/接收的会议或期刊；**论文未提及**利益冲突声明（「Acknowledgments and Disclosure of Funding」一节在正文抽取文本中**内容为空**，未披露具体资助来源）
- **代码**：`https://github.com/JenniferJaschob/ObsNODE.git`（正文 A.3.1 "Code availability" 明确给出）
- **本地 PDF**：[papers/2604.26070/2604.26070.pdf](../../papers/2604.26070/2604.26070.pdf) ／ 文本 [papers/2604.26070/2604.26070.txt](../../papers/2604.26070/2604.26070.txt)
- **页数**：25 页（PDF 实测 `/Count 25`；正文文本页码标记 1–25）
- **全文结构**：正文 1–7 节 + 附录 A（A.1 式(4)连续时间推导 / A.2 序列相关隐混杂下的路径空间调整 / A.3 训练细节与代码可得性 / A.4 数据仿真细节 / A.5 超参调优 / A.6 计算效率 / A.7 LLM 使用声明）
- **关联概念**：Neural ODE / Latent ODE / ODE-RNN / Neural CDE / 控制论可观测性（observability）/ 条件前门准则 CFD / 动态处理策略 DTR / 时依混杂与处理—混杂反馈 / 隐混杂 / 连续时间因果识别 / 不规则采样临床数据 / MIMIC-IV / AmsterdamUMCdb
- **与本仓主线的关系**：🔴 **族外**。论文零次提及 LNN/LTC/CfC/NCP，其 Δt 处理方式与液态家族不同（见 §3.8）。可迁移的是"**构造性可观测性保证**"这一架构原则，以及"**有反事实真值时的 CB/ATE 评测协议**"。

## 2. 核心问题

**现状：连续时间序贯决策里的因果预测，缺的不是"更强的时序骨干"，而是"在存在隐混杂时如何被识别"。** 论文第 1 节把场景讲得很清楚：医学中的动态处理策略（DTR）要求"治疗随患者纵向历史反复调整"（Murphy, 2003; Robins, 2004），而在观察性数据中，协变量既影响治疗分配又影响结局、同时自身还被既往治疗影响（**时依混杂 + 处理—混杂反馈**，Hernán & Robins, 2023）。当疾病状态的关键分量不可观测、且隐藏变量同时影响处理与结局时，识别变得更难。

**现有方法为何不够用（论文第 6 节的逐条论据）：** 经典纵向因果推断只估总体平均潜在结果，不条件于个体的治疗与结局历史，因此不能做个体化预测；CAPO 一类离散时间方法依赖固定时间离散，而临床测量是**不规则时间点**上的；平衡法类 TE-CDE 在序贯设定下未必完全解决时依混杂；SCIP-Net 基于稳定化逆概率加权，被限制在离散处理水平；结构动态因果模型（Bongers et al., 2022）采用了与本文不同的干预表述。最接近的前作 **CF-ODE** 假设"处理时刻的观测唯一决定潜状态"（即 Definition 3.1 的可观测性条件），**但这一假设既不被结构性强制、也难以在实践中验证**，且 CF-ODE 只条件于既往观测、没有显式纳入处理历史。

**结构性缺陷（本论文诊断出的那一条）：** 潜在状态空间模型里，"潜状态能否从观测历史唯一重建"是整个识别链条的枢纽 —— 而标准的 latent Neural ODE **不提供任何关于该性质的保证**。论文的写法是把它当成一个**需要在实践中验证、却往往无法验证**的运行假设；一旦它不成立，整条识别链（潜中介 $Z_{t+s}$ 不可识别 ⇒ CFD 不可用）就断掉，模型的"因果预测"就退化为"相关预测"。

⇒ **问题被重述为**：*在连续时间、存在时依混杂与隐混杂的动态处理策略中，如何把"潜状态可从处理—结局历史唯一重建"从一条**待验证的假设**变成一条**按构造成立的架构约束**，从而使条件潜在结局密度在隐混杂下仍可识别，并落成一个可训练、可预测的模型？*

论文自己的答案是一条三段链条：**可观测性 ⇒ 潜中介 $Z_{t+s}$ 满足条件前门准则（CFD）⇒ 连续时间状态空间调整公式（式 4）**，并用"**连续三角可观测标准形**"（式 7）在架构层把第一段变成构造性事实。

## 3. 方法论与核心思路

### 3.1 总体架构

ObsNODE 是"**一个数学定理 + 一个把它实现出来的参数化 + 一套自监督训练协议**"的三段式：

```
观测: 不规则采样 Y_t (多变量结局) + A_t (处理)
   │
   ├─(1) 识别层（数学）  可观测性 ⇒ CFD ⇒ 调整公式 (4)
   │      └─ 需要 p(z_t|y_t,o_{[0,t)}) · p(z_{t+s}|a,z_t) · p(y_{t+s}|z_{t+s})
   │
   ├─(2) 参数化层（架构） 把潜动力学写成连续三角可观测标准形 (7)
   │      └─ Z_t = (Z_t^{(1)},...,Z_t^{(m)}), d_z = m·d_y;  Y_t = Z_t^{(1)}
   │         ⇒ 可观测性按构造成立, 不再是"待检验的假设"
   │
   ├─(3) 推断层（工程）  LSTM recognition network 近似滤波分布 p(z_t|y_t,o_{[0,t)})
   │      └─ 显式缺失插补层 (指示函数 + 可学习 b_j) 处理不规则采样
   │
   └─(4) 预测层  给定假设处理路径 a_{[t,t+s)}, 前向积分式 (7) 并投影回观测空间
          └─ 自监督训练: 掩码平方误差 (按观测次数与经验方差归一化), 随机决策时刻 t_c, Adam
```

三者的耦合点值得单独点出：**式 (4) 的识别论证是在"精确滤波分布"上成立的，而实现里滤波分布是 LSTM 的近似**。论文对此只有一句 "We **approximate** the filtering distribution ... with an RNN encoder"，并在附录 A.3 自陈 "State reconstruction from finite, noisy observations remains **approximate**"。这一"识别定理 vs 近似实现"的落差，论文未量化（见 §6.2）。

**上下文关系**：骨干是成熟的 latent Neural ODE（式 3，引 Chen 2018 / Rubanova 2019 / Kidger 2020）+ ODE-RNN 风格的 LSTM 识别网络（Rubanova 2019）。**本文的结构性创新全部在"状态参数化"这一层**，不在骨干。

### 3.2 潜在 Neural ODE 模型（式 3）

论文用最一般的潜变量形式建模观测过程（式 3）：

$$
Z_0 \sim p_\theta(Z_0), \qquad \dot Z_t = f_\theta(Z_t, A_t), \qquad Y_t \sim q_\theta(\cdot\mid Z_t). \tag{3}
$$

其中 $p_\theta$ 参数化初始状态分布、$f_\theta$ 是向量场、$q_\theta$ 描述边缘观测分布。论文把**处理解释为对潜在动力学的干预**（Fig. 1(a)），典型情形是动态处理策略：

$$
A_t = \pi\big(y_t,\, y_{[0,t)},\, a_{[0,t)}\big) \qquad (\text{Fig. 1(b), 绿色虚线箭头}).
$$

Fig. 1(a) 的时序 DAG 明确：隐扰动 $\epsilon_t$ **同时**影响当代处理分配与当代观测，且 $\epsilon_t$ **在时间上互相独立**（temporally independent）—— 这正是后文"tractable"化的关键结构点。

### 3.3 可观测标准形（式 7）—— 本文的架构核心

论文的continuation任务是"把可观测性从假设变成构造"。做法是把潜动力学**参数化为连续三角可观测标准形**（引 Bernard 2019; Adamy 2024）：

$$
\dot Z_t=
\begin{pmatrix}
\dot Z_t^{(1)}\\ \vdots\\ \dot Z_t^{(i)}\\ \vdots\\ \dot Z_t^{(m)}
\end{pmatrix}
=
\begin{pmatrix}
Z_t^{(2)}+\phi^{(1)}_{\theta_1}\!\left(Z_t^{(1)},a_t\right) & \cdots\\
Z_t^{(i+1)}+\phi^{(i)}_{\theta_i}\!\left(Z_t^{(1)},\ldots,Z_t^{(i)},a_t\right) & \cdots\\
\phi^{(m)}_{\theta_m}\!\left(Z_t,a_t\right) &
\end{pmatrix},
\qquad Y_t = Z_t^{(1)}. \tag{7}
$$

要点：
- 状态被分块为 $m$ 块，每块 $Z_t^{(i)}\in\mathbb{R}^{d_y}$，故 $d_z = m\,d_y$（**潜维度被强制为观测维度的整数倍**）；
- 每个 $\phi^{(i)}_{\theta_i}$ 由一个神经网络实现（附录 A.5.1：实验中**所有分块共用同一套超参**，未单独调优）；
- 测量模型允许加性噪声：$Y_t = Z_t^{(1)}+\eta_t$，$\eta_t$ 为中心化噪声过程 —— 论文明确 "This noise does not affect identification."，而隐藏扰动 $\epsilon_t$ **不要求中心化**，由对应的调整公式处理；
- 论文承认该标准形与一般可观测系统的等价性有前提："**Under suitable conditions**, observable systems can be transformed into this normal form without loss of expressiveness (Bernard, 2019)."（论文未给出"这些条件"的具体表述，也未在真实数据上检验）

**关键对比**：论文对 CF-ODE 的批评正落在这里 —— CF-ODE 把可观测性当作假设，"not structurally enforced and is difficult to verify in practice"；ObsNODE 用式 (7) "guarantees observability **by construction**"。这是本文与最接近前作的唯一结构性差异，也是唯一消融实验所检验的对象。

### 3.4 连续时间条件前门调整（CFD，式 2）

连接因果识别与潜状态空间模型的桥是**条件前门准则**（Xu et al., 2024），front-door 准则在"存在未观测共因"下的扩展。

**Theorem 3.2**（原文引 Xu et al., 2024）：$M$ 满足相对有序变量对 $(A,Y)$ 的 CFD 准则，若 (i) $M$ 截断从 $A$ 到 $Y$ 的所有有向路径；(ii) 存在条件集 $W$ 阻断从 $A$ 到 $M$ 的所有后门路径；(iii) $M$ 与 $Y$ 之间的所有后门路径被 $\{A\}\cup W$ 阻断。则因果效应由

$$
P\big(Y(a)\big)=\sum_{m,w,a'}P(y\mid a',m,w)\,P(a'\mid w)\,P(m\mid a,w)\,P(w) \tag{2}
$$

给出（"where the sum over $a'$ runs over alternative treatments."）。

### 3.5 主定理 4.3：可识别性公式（式 4）

**Theorem 4.3**：在 consistency、SUTVA、overlap(positivity)、Assumption 4.1 与 4.2、以及**潜状态可观测**这组条件下，$a_{[t,t+s)}$ 下的条件潜在结局分布由下式识别：

$$
P_{Y_{t+s}}\!\left(a_{[t,t+s)}\right)\,\Big|\,Y_t,O_{[0,t)}\big(y_{t+s}\mid y_t,o_{[0,t)}\big)
=\int_{z_{t+s}}\!\int_{z_t}
p\big(z_t\mid y_t,o_{[0,t)}\big)\,
p\big(z_{t+s}\mid a_{[t,t+s)},z_t\big)\,
p\big(y_{t+s}\mid z_{t+s}\big). \tag{4}
$$

**这是全篇最重要的一行**：它把"因果预测"拆成三个可分别建模/估计的因子 —— ① **滤波分布**（推断当前潜状态）、② **潜在转移密度**（处理如何推动潜状态）、③ **测量模型**（潜状态如何变成观测）。论文强调该式在"$\epsilon_t$ 时间独立 ⇒ 处理与观测之间存在未被观测的同期依赖"时**依然成立**，且可扩展到 DTR（需要 sequential positivity）。

**证明的中间量**（式 5、6 及其展开）体现了 CFD 求和被替换成积分、并对替代处理 $a'_t$ 积分掉的过程：

$$
p\big(y_{t+s}\mid a'_t,z_{t+s},z_t,y_t,o_0\big)=p\big(y_{t+s}\mid z_{t+s}\big),\qquad
p\big(z_{t+s}\mid a_t,z_t,y_t,o_0\big)=p\big(z_{t+s}\mid a_t,z_t\big).
$$

**离散→连续的桥梁**：附录 A.1 先在有限分区 $\{t_k\}$ 上推导（式 8–11），令 $K\to\infty,\ L\to\infty$ 即得式 (4)。同时，模型预测在式 (3) 下是**确定性**的，故

$$
p(z_{t+s}\mid a_{[t,t+s)},z_t)=\delta\!\left(z_{t+s}-z_t-\int_t^{t+s}f(z_\tau,a_\tau)\,d\tau\right),
$$

即转移密度是 Dirac 点质量 —— 论文据此指出："If we replace the ODE by a stochastic differential equation we need a notion of **stochastic observability** (Van Handel, 2009)."（把 ODE 换成 SDE 需另立概念，论文未展开）

### 3.6 "Tractable" 从哪来：时间不相关的同期扰动

摘要与第 1 节的措辞是：推导一个**一般**调整公式，并证明它在"**未观测同期扰动时间上不相关**"时**化简为 tractable 的状态空间公式**。机理在 Remark 4.4 与附录 A.2：

- 当 $\epsilon_t$ 是**外生、时间独立**的扰动时，Assumption 4.2 成立 ⇒ $p(y_{t+s}\mid a'_{[t,t+s)},z_{t+s},\ldots)=p(y_{t+s}\mid z_{t+s})$ ⇒ **观测处理路径分布 $p(a'|\cdot)$ 可以被积分掉**，式 (2) 的求和塌缩为式 (4) 的三项乘积；
- 反之，若 $\epsilon_t$ **序列相关**（Fig. 5），上述条件独立失效，但识别仍由未化简的路径空间公式给出（附录 A.2，式 13）：

$$
P_{Y_{t+s}}(a_{[t,t+s)})\mid Y_t,O_{[0,t)}\big(y_{t+s}\mid y_t,o_{[0,t)}\big)
=\int_{z_t}\!\int_{z_{t+s}}\!\int_{A_{[t,t+s)}}\Big\{
p\big(y_{t+s}\mid a'_{[t,t+s)},z_{t+s},z_t,y_t,o_{[0,t)}\big)
\times P\big(da'_{[t,t+s)}\mid z_t,y_t,o_{[0,t)}\big)
\,p\big(z_{t+s}\mid a_{[t,t+s)},z_t\big)
\,p\big(z_t\mid y_t,o_{[0,t)}\big)
\Big\}. \tag{13}
$$

这里的 $P(da'|\cdot)$ 是**条件路径测度**（Kallenberg, 2021），因为连续时间轨迹空间是无限维的。

🔴 **论文对式 (13) 的可实现性给了明确警告（原文）**：

> "Equation 13 could in principle be approximated by learning the conditional observational treatment process... However, its non-parametric estimation additionally requires path-space and mediator positivity, **which may fail for deterministic latent dynamics**. Practical implementations may therefore require stochastic latent dynamics or an augmented observable state that captures the temporally persistent confounding process. The ObsNODE implementation considered in this work instead uses Assumption 4.2, under which the outcome density is independent of $a'_{[t,t+s)}$ given $Z_{t+s}$. The path integral then equals one, and (13) reduces to the tractable adjustment formula (4)."

⇒ **一句话**：**论文实现的 ObsNODE，恰好不覆盖论文自己强调为最难的"persistent confounding"情形**，而是选了那条可 tractable 的支路。

### 3.7 滤波分布、测量模型与训练目标

**滤波分布**（§4.3）："We approximate the filtering distribution $p(z_t\mid y_t,o_{[0,t)})$ with an RNN encoder, following (Chen et al., 2018). Given the inferred state $z_t$, forecasts under a hypothetical treatment path $a_{[t,t+s)}$ are obtained by **integrating (7) forward in time** and projecting to the observed space."（附录 A.1 补充：这是"连续时间 ODE 模型配离散时间观测"的滤波分布，Jazwinski, 2007）

**实现细节**（附录 A.3）：LSTM recognition network 推断初始状态 $z_0$；缺失插补层用指示函数与可学习参数 $b_j$：

$$
I_{t_{ij}}=\begin{cases}1 : y^t_{ij}\ \text{observed}\\ 0 : y^t_{ij}\ \text{missing}\end{cases},
\qquad
y^t_{ij}=y^t_{ij}\cdot I_{t_{ij}}+b_j\,(1-I_{t_{ij}}).
$$

模型支持**直接长时程预测**与**递归预测**（ODE-RNN 风格，"without using any ground-truth observations beyond $t_c$"），并自陈"augmenting the sequence with intermediate predictions can improve stability for longer forecasting horizons in some settings"。

**训练目标**：掩码平方误差，按每个成分的观测次数与经验方差归一化：

$$
\mathcal{L}_{(t_c,t_f]}=\sum_{i=1}^{n}\sum_{j=1}^{d_y}
\frac{1}{\big(n\,\sigma_j^2\,|T_{ij}(t_c,t_f)|\big)}
\sum_{t\in T_{ij}(t_c,t_f)}\big(y^t_{ij}-\hat y^{\,dat}_{t_{ij}}\big)^2
$$

（$T_{ij}(t_c,t_f)=T_{ij}\cap(t_c,t_f]$ 为该窗口内第 $i$ 个样本第 $j$ 个成分的观测时刻集合；$\sigma_j^2$ 为该成分在训练集全时间点上的经验方差，标准化数据可置 1）。训练时对批内所有样本选同一 $t_c\in[0,t_f)$，"both a fixed set of possible values for $t_c$ evenly distributed over $[0,t_f)$ or a uniform random choice led to very similar results"。

### 3.8 与 LTC / CfC / NCP / GRU 的关系

🔴 **本节结论先行：本篇与液态家族零交集，既不能作"LNN 更好"的证据，也未与该家族做任何对照。**

| 维度 | ObsNODE（本篇） | LTC (Hasani 2021) / CfC (Hasani 2022) / NCP (Lechner 2020) | GRU |
|---|---|---|---|
| 状态演化 | **数值积分**连续时间 ODE（代码用 `odeint_adjoint`，`method='rk4'`, `step_size=1.0`） | CfC/NCP：**闭式/近似闭式**更新（指数衰减形式）；LTC：连续时间微分方程数值积分 | 离散步、隐状态按固定步更新 |
| 时间进入方式 | 时间只作为 **ODE 自变量** $t$（出现在 $f_\theta(Z_t,A_t)$ 与积分限中） | LTC/CfC/NCP：**显式 $\Delta t$** 通道驱动状态松弛 | 隐式步数 |
| $\Delta t$ | 🔴 **全文未使用**（无 $\Delta t$、无时间间隔嵌入；时间间隔不作为模型输入） | $\Delta t$ 是核心输入 | 无 |
| 不规则采样 | ✅ 明确面向不规则采样（§2、§4.3、§6）；用**掩码损失 + 缺失插补层**吸收；连续时间方程"天然"支持任意查询时刻 | ✅ 同为设计目标；CfC 走 $\Delta t$ 条件化路线 | ❌ 需显式插值/对齐 |
| 与 GRU 的差别 | 状态由 ODE 积分推进，$t_c$ 之后可在**任意时刻**求值（论文给出直接式与递归式两套） | 同（但更新是闭式近似） | 逐 token 递推 |
| 骨干家族 | latent Neural ODE + LSTM encoder | 液态族 | RNN |

三条具体差异点：
1. **求解方式**：ObsNODE 是**真·连续时间数值积分**（vendored torchdiffeq 的 adjoint RK4），不是闭式解 ⇒ 与 CfC/NCP 的"闭式"路线相反；与 LTC 的"数值积分"路线同侧，但 LTC 用 $\Delta t$ 松弛式动力学，ObsNODE 用显式向量场。
2. **不规则性的吸收机制不同**：ObsNODE 靠"连续时间方程 + 掩码损失 + 插补层"；LTC/CfC 靠"$\Delta t$ 显式通道"。**两种机制在本仓 bench 中从未直接对照过**（本仓 `scripts/bench_irregular_dt.py`、`scripts/bench_quite_irregular_ts.py` 已具备跑这种对照的基础，但本篇不提供这条轴的任何证据）。
3. **本文的新意不落在时序骨干上**，落在**潜变量参数化与因果识别**上 ⇒ 与液态家族是"正交的两条轴"，不是"更强的液态模型"。

## 4. 核心公式提取

> 论文编号公式到 **(18)** 为止（式 14/15 各含 a/b 两个子式）；**不存在 Eq. 19 及以后的编号公式**（论文未提及）。未编号但被正文使用的重要式子（训练损失、插补层、Dirac 转移密度、加性观测噪声）单列在末尾。

| 编号 | 公式 (LaTeX) | 含义 |
|---|---|---|
| Eq. 1 (1a)(1b)(1c) | $\dot z_t=f(z_t,a_t)$；$y_t=h(z_t)$；$z_0=\zeta$ | 连续时间确定性状态空间模型三元组：动力学、测量、初始条件（$h$ 未必可逆，$d_z>d_y$ 时尤其） |
| Def. 3.1 | $y_t(a,\zeta)=y_t(a,\eta)\ \forall t\in[0,T]\Rightarrow\zeta=\eta$ | **一致可观测性定义**：输出轨迹相同 ⇒ 初始状态相同（$[0,T]$ 为两解共同存在的最大区间） |
| Eq. 2 | $P(Y(a))=\sum_{m,w,a'}P(y\mid a',m,w)\,P(a'\mid w)\,P(m\mid a,w)\,P(w)$ | 条件前门准则（CFD）的因果效应估计公式（求和版） |
| Eq. 3 | $Z_0\sim p_\theta(Z_0),\ \dot Z_t=f_\theta(Z_t,A_t),\ Y_t\sim q_\theta(\cdot\mid Z_t)$ | 潜变量 Neural ODE 模型：初始分布、向量场、观测模型 |
| Eq. 4 | $P_{Y_{t+s}}(a_{[t,t+s)})\mid Y_t,O_{[0,t)}(y_{t+s}\mid y_t,o_{[0,t)})=\int_{z_{t+s}}\int_{z_t}p(z_t\mid y_t,o_{[0,t)})\,p(z_{t+s}\mid a_{[t,t+s)},z_t)\,p(y_{t+s}\mid z_{t+s})$ | 🔴 **主结果**：可识别性公式 = 滤波分布 × 潜在转移 × 测量模型（三项乘积） |
| Eq. 5 | $p(y_{t+s}\mid a'_t,z_{t+s},z_t,y_t,o_0)$ | CFD 展开式中的被积因子（潜在结局密度项） |
| Eq. 6 | $p(a'_t\mid z_t,y_t,o_0)\,p(z_{t+s}\mid a_t,z_t,y_t,o_0)\,p(z_t,y_t,o_0)$ | CFD 展开式中剩余三个因子（处理分布、中介分布、联合先验） |
| Eq. 7 | $\dot Z_t=\begin{pmatrix}\dot Z_t^{(1)}\\ \vdots\\ \dot Z_t^{(i)}\\ \vdots\\ \dot Z_t^{(m)}\end{pmatrix}=\begin{pmatrix}Z_t^{(2)}+\phi^{(1)}_{\theta_1}(Z_t^{(1)},a_t)&\cdots\\ Z_t^{(i+1)}+\phi^{(i)}_{\theta_i}(Z_t^{(1)},\ldots,Z_t^{(i)},a_t)&\cdots\\ \phi^{(m)}_{\theta_m}(Z_t,a_t)&\end{pmatrix},\quad Y_t=Z_t^{(1)}$ | 🔴 **可观测标准形**：三角结构 + 直接观测首块 ⇒ 可观测性按构造成立；$d_z=m\,d_y$ |
| Eq. 8 | $p(y_{t+s}\mid a'_{K+1:L},z_{t+s},z_t,y_t,o_{0:K})$ | 附录 A.1 有限分区下未化简 CFD 的被积因子 |
| Eq. 9 | $p(a'_{K+1:L}\mid z_t,y_t,o_{0:K})\,p(z_{t+s}\mid a_{K+1:L},z_t,y_t,o_{0:K})\,p(z_t,y_t,o_{0:K})$ | 附录 A.1 有限分区下的完整 CFD 调整积分 |
| Eq. 10 | $P_{Y_{t+s}(a_{K+1:L})}\mid Y_t,O_{0:K}(y_{t+s}\mid y_t,o_{0:K})$ | 有限分区下条件潜在结局密度的左端 |
| Eq. 11 | $\int_{z_{t+s}}\int_{z_t}p(y_{t+s}\mid z_{t+s})\,p(z_{t+s}\mid a_{K+1:L},z_t)\,p(z_t\mid y_t,o_{0:K})$ | 有限分区下的 tractable 调整公式（式 4 的离散对应） |
| Eq. 12 | $p(y_{t+s}\mid a'_{K+1:L},z_{t+s},z_t,y_t,o_{0:K})\times P(a'_{K+1:L}\mid z_t,y_t,o_{0:K})\,p(z_{t+s}\mid a_{K+1:L},z_t)\,p(z_t\mid y_t,o_{0:K})$ | 序列相关隐混杂下（有限分区）的未化简调整公式 |
| Eq. 13 | $P_{Y_{t+s}}(a_{[t,t+s)})\mid Y_t,O_{[0,t)}(y_{t+s}\mid y_t,o_{[0,t)})=\int_{z_t}\int_{z_{t+s}}\int_{A_{[t,t+s)}}\{p(y_{t+s}\mid a'_{[t,t+s)},z_{t+s},z_t,y_t,o_{[0,t)})\times P(da'_{[t,t+s)}\mid z_t,y_t,o_{[0,t)})\,p(z_{t+s}\mid a_{[t,t+s)},z_t)\,p(z_t\mid y_t,o_{[0,t)})\}$ | 序列相关隐混杂下的**路径空间**一般调整公式（条件路径测度，Kallenberg 2021） |
| Eq. 14a | $\frac{dV(t)}{dt}=\frac{\big(\rho\log\frac{K}{V(t)}-\beta_cC(t)-(\alpha_r d(t)+\beta_r d(t)^2)+\epsilon(t)\big)}{V(t)}$ | 附录 A.4.1 合成数据生成器：Gompertz 肿瘤生长 + 化疗/放疗抑制 + 高斯白噪声 |
| Eq. 14b | $\frac{dW_{net}(t)}{dt}=k_gW_{net}(t)\big(1-\frac{W_{net}(t)}{G}\big)-\beta_{wc}C(t)-\alpha_{wr}d(t)-\lambda V(t)$ | 附录 A.4.1 净体重 logistic 生长 + 治疗/疾病影响 |
| Eq. 15a | $C(t)=C_{max}\,\sigma\big(\gamma_c\frac{\alpha^{dose}_c}{D_{max}}(D(t)-\delta_c)\big)$ | 化疗剂量的时依随机生成（$\gamma$ 为混杂强度，$C_{max}=14$ mg/m³） |
| Eq. 15b | $d(t)=d_{max}\,\sigma\big(\gamma_r\frac{\alpha^{dose}_r}{D_{max}}(D(t)-\delta_r)\big)$ | 放疗剂量的时依随机生成（$d_{max}=3$ Gy，$D_{max}=13$ cm） |
| Eq. 16 | $\tilde Y^{j,(i)}_t=\alpha^s_j B\text{-}spline(t)+\alpha^g_j g_{j,(i)}(t)+\alpha^\phi_j\phi^Y_j(\epsilon^{(i)}_t)+\eta_{j,(i)}_t$ | 附录 A.4.2 半合成数据：未治疗结局 = B 样条 + GP（Matérn 核）+ RFF 隐混杂 + 噪声 |
| Eq. 17 | $E^j_t=\sum_{k=t-w_l}^{\,t}\min_{l=1,\dots,d_a}\mathbb{1}[A^l_k=1]P_{A^l_k}\beta_{lj}(w_l-k)^2$ | 附录 A.4.2 时依处理效应（上界由 $\beta_{lj}\in\{0,\beta\}$ 界定） |
| Eq. 18 | $Y^{j,(i)}_t=\tilde Y^{j,(i)}_t+E^j_t$ | 附录 A.4.2 处理后结局 = 未治疗结局 + 处理效应 |
| （无编号） | $\mathcal{L}_{(t_c,t_f]}=\sum_{i=1}^n\sum_{j=1}^{d_y}\frac{1}{n\sigma_j^2|T_{ij}(t_c,t_f)|}\sum_{t\in T_{ij}(t_c,t_f)}(y^t_{ij}-\hat y^{dat}_{t_{ij}})^2$ | 训练损失：按观测次数与经验方差归一化的掩码平方误差 |
| （无编号） | $I_{t_{ij}}=\{1:\ y^t_{ij}\ \text{observed}; 0:\ y^t_{ij}\ \text{missing}\}$，$y^t_{ij}=y^t_{ij}\cdot I_{t_{ij}}+b_j(1-I_{t_{ij}})$ | 附录 A.3 缺失值插补层（$b_j$ 可学习） |
| （无编号） | $p(z_{t+s}\mid a_{[t,t+s)},z_t)=\delta\big(z_{t+s}-z_t-\int_t^{t+s}f(z_\tau,a_\tau)\,d\tau\big)$ | 确定性 ODE 的转移密度为 Dirac 点质量（可观测性重建的基础） |
| （无编号） | $Y_t=Z_t^{(1)}+\eta_t$ | 加性观测噪声（中心化，不影响识别） |

## 5. 关键成果与贡献

### 5.1 主结果

**实验设置与统计口径**：基线为 IGC-Net（Hess et al., 2026）、SCIP-Net（Hess & Feuerriegel, 2025）、两个 TE-NODE/TE-CDE 模型 OptAB（类别型处理，Wendland et al., 2024）与 DoseAI（类别型 + 连续型处理，Wendland & Kschischo, 2025）；消融基线 NODE（**无**观测标准形的普通 latent Neural ODE）。结果为 "**five independent runs** with disjoint training, validation, and test sets" 的 mean ± std；RMSE 按各动态变量在测试集上的标准差缩放。合成/半合成数据另扫混杂强度 $\gamma$。

**(a) 合成癌症数据（Table 1：肿瘤体积 RMSE，$t_c=5$，$s=t_f-t_c$ 以月计）**

| $\gamma$ | 模型 | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 |
|---|---|---|---|---|---|---|---|
| 4 | OBSNODE | 0.06±0.00 | 0.11±0.00 | 0.14±0.00 | 0.17±0.00 | 0.19±0.00 | 0.21±0.01 |
| 4 | DOSEAI | 0.16±0.02 | 0.15±0.02 | 0.12±0.01 | 0.12±0.01 | 0.11±0.00 | 0.10±0.01 |
| 4 | IGC-NET | 0.13±0.01 | 0.18±0.04 | 0.22±0.05 | 0.22±0.03 | 0.17±0.01 | 0.26±0.14 |
| 4 | SCIP-NET | 0.16±0.00 | 0.27±0.02 | 0.32±0.04 | 0.30±0.05 | 0.28±0.06 | 0.27±0.07 |
| 4 | NODE | 0.61±0.00 | 0.47±0.00 | 0.37±0.00 | 0.30±0.00 | 0.25±0.00 | 0.22±0.00 |
| 6 | OBSNODE | 0.06±0.00 | 0.10±0.00 | 0.13±0.00 | 0.15±0.00 | 0.17±0.00 | 0.18±0.01 |
| 6 | DOSEAI | 0.12±0.02 | 0.11±0.01 | 0.10±0.01 | 0.10±0.01 | 0.10±0.00 | 0.10±0.01 |
| 6 | IGC-NET | 0.12±0.01 | 0.18±0.03 | 0.20±0.04 | 0.20±0.03 | 0.16±0.01 | 0.25±0.15 |
| 6 | SCIP-NET | 0.14±0.00 | 0.24±0.02 | 0.31±0.05 | 0.29±0.05 | 0.27±0.06 | 0.26±0.06 |
| 6 | NODE | 0.56±0.00 | 0.43±0.00 | 0.34±0.00 | 0.28±0.00 | 0.24±0.00 | 0.20±0.00 |
| 8 | OBSNODE | 0.05±0.00 | 0.09±0.00 | 0.12±0.00 | 0.14±0.00 | 0.16±0.00 | 0.18±0.01 |
| 8 | DOSEAI | 0.12±0.01 | 0.11±0.00 | 0.11±0.00 | 0.11±0.01 | 0.12±0.01 | 0.13±0.02 |
| 8 | IGC-NET | 0.11±0.01 | 0.18±0.03 | 0.20±0.04 | 0.19±0.03 | 0.16±0.01 | 0.25±0.17 |
| 8 | SCIP-NET | 0.14±0.00 | 0.24±0.02 | 0.31±0.05 | 0.29±0.05 | 0.27±0.06 | 0.26±0.06 |
| 8 | NODE | 0.55±0.00 | 0.43±0.00 | 0.35±0.00 | 0.29±0.00 | 0.25±0.00 | 0.22±0.00 |

⚠️ **论文正文自述长时程最优是 doseAI**："For long horizons, **doseAI achieves the lowest error, followed by ObsNODE and IGC-Net**, whereas **ObsNODE performs best for short horizons ($\tau\le 2$ cycles)**."（Table 1 的 $s=5,6$ 两列可直接核对这一说法：ObsNODE 0.19/0.21 vs doseAI 0.11/0.10）

**(b) 半合成 MIMIC-IV（Table 3：RMSE，固定 $t_c=1$ 小时，$s$ 以小时计）**

| 模型 | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 |
|---|---|---|---|---|---|---|
| OBSNODE | 0.13±0.00 | 0.21±0.00 | 0.25±0.00 | 0.27±0.00 | 0.30±0.00 | 0.33±0.00 |
| OPTAB | 0.34±0.04 | 0.35±0.04 | 0.41±0.02 | 0.42±0.02 | 0.44±0.02 | 0.45±0.02 |
| IGC-NET | 0.10±0.03 | 0.21±0.01 | 0.25±0.01 | 0.29±0.02 | 0.34±0.02 | 0.38±0.01 |
| SCIP-NET | 0.34±0.10 | 0.73±0.10 | 1.32±0.16 | 1.32±0.16 | 1.33±0.16 | 1.32±0.16 |
| NODE | 0.88±0.06 | 0.63±0.11 | 0.48±0.12 | 0.43±0.06 | 0.41±0.04 | 0.42±0.05 |

⚠️ $s=1$ 时 IGC-Net（0.10）优于 ObsNODE（0.13）。

**(c) 真实临床数据（Table 5a：MIMIC-IV 脓毒症，$t_c=1$ 小时，$s$ 以小时计）**

| 变量 | 模型 | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 |
|---|---|---|---|---|---|---|---|
| SOFA-SCORE | OBSNODE | 0.24±0.00 | 0.32±0.01 | 0.37±0.01 | 0.42±0.02 | 0.47±0.03 | 0.50±0.04 |
| SOFA-SCORE | OPTAB | 0.41±0.01 | 0.46±0.01 | 0.50±0.02 | 0.53±0.02 | 0.57±0.02 | 0.59±0.02 |
| SOFA-SCORE | IGC-NET | **0.09**±0.00 | 0.37±0.00 | 0.40±0.00 | 0.43±0.00 | 0.45±0.00 | 0.48±0.01 |
| SOFA-SCORE | SCIP-NET | 0.57±0.06 | 1.03±0.44 | 1.25±0.43 | 1.23±0.39 | 1.21±0.35 | 1.19±0.33 |
| CREATININE | OBSNODE | 0.34±0.00 | 0.44±0.00 | 0.53±0.00 | 0.53±0.02 | 0.58±0.04 | 0.65±0.07 |
| CREATININE | OPTAB | 1.14±0.02 | 0.93±0.02 | 0.89±0.03 | 1.23±0.03 | 0.96±0.01 | 0.62±0.02 |
| CREATININE | IGC-NET | **0.09**±0.02 | 1.13±0.01 | 0.89±0.00 | 0.92±0.01 | 0.78±0.15 | 0.95±0.01 |
| CREATININE | SCIP-NET | 1.01±0.04 | 1.06±0.26 | 1.36±0.28 | 1.39±0.30 | 1.17±0.39 | 0.99±0.42 |
| BILIRUBIN-TOTAL | OBSNODE | 0.34±0.00 | 0.38±0.00 | 0.46±0.02 | 0.54±0.04 | 0.62±0.07 | 0.70±0.11 |
| BILIRUBIN-TOTAL | OPTAB | 1.36±0.05 | 0.66±0.01 | 0.73±0.07 | 1.05±0.04 | 1.56±0.14 | 0.97±0.05 |
| BILIRUBIN-TOTAL | IGC-NET | **0.07**±0.01 | 1.40±0.02 | 0.69±0.01 | 0.91±0.00 | 1.07±0.01 | 1.59±0.03 |
| BILIRUBIN-TOTAL | SCIP-NET | 1.23±0.17 | 0.72±0.12 | 1.13±0.11 | 1.24±0.12 | 1.84±0.14 | 1.17±0.14 |
| ALT | OBSNODE | 0.09±0.00 | 0.22±0.01 | 0.27±0.03 | 0.33±0.06 | 0.39±0.10 | 0.47±0.13 |
| ALT | OPTAB | 0.72±0.20 | 1.25±0.09 | 0.65±0.16 | 0.76±0.25 | 0.67±0.21 | 0.54±0.11 |
| ALT | IGC-NET | **0.06**±0.02 | 0.64±0.17 | 1.34±0.21 | 0.41±0.02 | 0.36±0.03 | 0.90±0.23 |
| ALT | SCIP-NET | 1.75±0.20 | 1.51±0.30 | 0.81±0.28 | 0.77±0.37 | 2.24±0.45 | 1.45±0.26 |

⚠️ **$s=1$ 时 IGC-Net 在全部四个变量上都优于 ObsNODE**（0.09/0.09/0.07/0.06 vs 0.24/0.34/0.34/0.09）；论文正文亦承认 "IGC-Net performs better for very short-term predictions of selected laboratory variables"。

**(d) 外部验证（Table 5b：MIMIC-IV 训练 → AmsterdamUMCdb 测试，$t_c=1$ 小时）**

| 变量 | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 |
|---|---|---|---|---|---|---|
| SOFA-SCORE | 0.26±0.00 | 0.37±0.01 | 0.43±0.02 | 0.44±0.02 | 0.51±0.02 | 0.59±0.03 |
| CREATININE | 0.14±0.00 | 0.18±0.02 | 0.22±0.05 | 0.27±0.09 | 0.34±0.13 | 0.39±0.17 |
| BILIRUBIN-TOTAL | 0.01±0.00 | 0.11±0.02 | 0.18±0.05 | 0.25±0.09 | 0.33±0.13 | 0.40±0.17 |
| ALT | 0.01±0.00 | 0.28±0.01 | 0.34±0.03 | 0.41±0.05 | 0.50±0.07 | 0.86±0.07 |

🔴 **表 5b 只有 ObsNODE 一列 —— 外部队列上没有任何基线。** 外部测试集仅 **399 例**（364 头孢曲松 + 35 万古霉素，**0 例**哌拉西林/他唑巴坦），且两队列分布差异巨大（平均 SOFA 6.8 vs 4.43；无总胆红素记录 23.3% vs 56.8%；无 ALT 记录 16.8% vs 56.1%），论文自陈 "may contribute to the observed distribution shift and affect model robustness and generalizability across cohorts"。

**(e) 计算效率（附录 A.6）**：ObsNODE 与 DoseAI/OptAB/SCIP-Net 用两张 40 GB A100，IGC-Net 用一张 96 GB H100。单张 A100 上 ObsNODE 训练耗时：合成癌症 **140 min**、半合成 **49 min**、真实 sepsis **65 min**；"similar to the training times required for the baseline models IGC-Net and SCIP-Net"，而 "Training OptAB and DoseAI required between **40 and 74 hours**"。**论文未提及**参数量、推理延迟、显存占用对比。

### 5.2 机制分析 / 消融

论文的消融**只有一个，且是组件级的最粗粒度版本**："we compare ObsNODE with an otherwise identical latent Neural ODE **without the observable normal form**"（附录 A.5.5：NODE 基线使用与 ObsNODE 相同的 encoder-LSTM 架构，潜维度设为与 ObsNODE 相同 "to ensure a fair comparison"）。

**消融协议**：对每个测试样本，两个模型都吸收到 $t_c=5$ 的事实历史，并在五个预设处理策略（CF1–CF5）下预测结局；**反事实偏差 CB** = 预测潜在结局与真值潜在结局之**带符号**差，对全部测试样本与**六个预测时间点**取平均；**ATE** 为各策略相对基线处理的平均处理效应。

**Table 2（合成，$t_c=5$，$\gamma=4$）**

| 指标 | 变量 | 模型 | CF1 | CF2 | CF3 | CF4 | CF5 |
|---|---|---|---|---|---|---|---|
| CB | TUMOR VOLUME | OBSNODE | 0.1738 | 0.1739 | 0.1724 | 0.1708 | 0.1708 |
| CB | TUMOR VOLUME | NODE | -0.5152 | -0.5288 | -0.5364 | -0.5321 | -0.5321 |
| CB | BODY WEIGHT | OBSNODE | 0.0952 | 0.0941 | 0.0924 | 0.0924 | 0.0926 |
| CB | BODY WEIGHT | NODE | -2.0229 | -2.0272 | -2.0313 | -2.0315 | -2.0315 |
| ATE | TUMOR VOLUME | TRUE | -0.0105 | 0.0031 | 0.0107 | 0.0063 | 0.0063 |
| ATE | TUMOR VOLUME | OBSNODE | 0.1633 | 0.1770 | 0.1831 | 0.1771 | 0.1771 |
| ATE | TUMOR VOLUME | NODE | -0.5257 | -0.5257 | -0.5257 | -0.5257 | -0.5257 |
| ATE | BODY WEIGHT | TRUE | -0.0016 | 0.0227 | 0.0068 | 0.0070 | 0.0070 |
| ATE | BODY WEIGHT | OBSNODE | 0.0937 | 0.0968 | 0.0992 | 0.0995 | 0.0995 |
| ATE | BODY WEIGHT | NODE | -2.0245 | -2.0245 | -2.0245 | -2.0245 | -2.0245 |

**Table 4（半合成，$t_c=5$）**

| 指标 | 变量 | 模型 | CF1 | CF2 | CF3 | CF4 | CF5 |
|---|---|---|---|---|---|---|---|
| CB | SEMI-SYNTHETIC OUTCOME | OBSNODE | 1.9165 | 1.2978 | 1.3962 | 2.6466 | -0.0149 |
| CB | SEMI-SYNTHETIC OUTCOME | NODE | 5.1987 | 2.4121 | 2.4121 | 7.6795 | -0.5663 |
| ATE | SEMI-SYNTHETIC OUTCOME | TRUE | -5.7596 | -5.1849 | -5.1849 | -5.1849 | -3.9458 |
| ATE | SEMI-SYNTHETIC OUTCOME | OBSNODE | -3.8431 | -3.8860 | -3.7887 | -3.9403 | -3.7794 |
| ATE | SEMI-SYNTHETIC OUTCOME | NODE | -0.5608 | -2.7728 | -2.7728 | 1.2556 | -4.5121 |

**消融结论（可读出的三条机制信号）**：
1. **可观测标准形把"系统性反事实偏差"压下去了**：NODE 基线的 CB 在合成数据上几乎全部为**大额负偏差**（体重 −2.02 ~ −2.03），ObsNODE 降到 +0.09 量级；半合成上 NODE 的 CB 波动剧烈（5.20 / 2.41 / 2.41 / 7.68 / −0.57），ObsNODE 为 1.92 / 1.30 / 1.40 / 2.65 / −0.01 ⇒ **偏差幅度与跨策略一致性都改善**。
2. **ATE 方向性仍不完美**：ObsNODE 的 ATE 在合成数据上与 TRUE 值（≈0）差一个量级（0.16~0.18 vs 0.006），说明"**可观测标准形显著缩小偏差，但不等于无偏**"；半合成上 ObsNODE 的 ATE（−3.8~−3.9）明显小于 TRUE（−5.2~−5.8），仍有系统性低估。
3. **方差极小**：ObsNODE 的 RMSE 标准差在多数格子里是 ±0.00，NODE/IGC-Net/SCIP-Net 多为 ±0.01~±0.17 ⇒ 论文自述 "ObsNODE consistently shows the smallest variance across different runs"。

🔴 **但这个消融没有拆解标准形的内部组件** —— 没有对块数 $m$、$Z^{(2)}$ 链式耦合的具体形式、滤波近似（LSTM vs 精确滤波）、损失归一化等分别消融。**"可观测性"是不是唯一起作用的因素，论文没有分离。**

### 5.3 贡献清单

- **建立了"控制论可观测性 ⇒ 连续时间因果可识别性"的明确链条**：证明在 consistency / SUTVA / positivity、Assumption 4.1（受控连续时间 Markov 潜状态）、Assumption 4.2（给定潜状态则观测与历史独立）、以及潜状态可观测这组条件下，动态处理效应由一个**连续时间条件前门调整公式**识别（Theorem 4.3、式 4）。
- **把可观测性从"待验证的运行假设"改写为"按构造成立的架构约束"**：提出连续三角可观测标准形（式 7，$d_z=m\,d_y$，$Y_t=Z_t^{(1)}$），从而使最接近前作 CF-ODE 所依赖的假设不再依赖实践检验。
- **给出了 tractable 化的确切条件并划出了它的边界**：证明一般调整公式在"未观测同期扰动时间上不相关"时化简为三项乘积的状态空间公式（式 4），并在附录 A.2 用路径空间公式（式 13）明确指出序列相关混杂下该化简失效、且其非参数估计"**may fail for deterministic latent dynamics**"。
- **实现了面向不规则采样临床数据的完整可训练管线**：LSTM recognition network + 显式缺失插补层 + 按观测次数与经验方差归一化的掩码平方误差 + 随机决策时刻 $t_c$ 的自监督训练，并在 MIMIC-IV 脓毒症（3,556 例）与 AmsterdamUMCdb（外部 399 例）上完成训练与跨队列验证。
- **在有反事实真值的三种设定（合成癌症 / 半合成 MIMIC-IV）上量化了反事实偏差与 ATE 误差**，并给出了一个可复现的评测协议（CF1–CF5 预设策略、$t_c=5$、六个预测时间点平均、5 次运行）。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

**第 7 节「Conclusion and Limitations」原文（VERBATIM）**：

> "The approach relies on consistency, positivity, and the stated state-space assumptions. The tractable adjustment formula permits contemporaneous hidden confounding but **assumes temporally independent disturbances; serially correlated confounding requires the general path-space adjustment derived in the Appendix**. **State reconstruction from finite, noisy observations remains approximate**, and **causal accuracy cannot be assessed on clinical data without counterfactual ground truth**. **Future work should address persistent confounding and uncertainty quantification.**"

**附录 A.2 的自陈局限（原文）**："...its non-parametric estimation additionally requires path-space and mediator positivity, **which may fail for deterministic latent dynamics**. Practical implementations may therefore require stochastic latent dynamics or an augmented observable state that captures the temporally persistent confounding process."

**其他自陈的范围条件**：
- 干预形式限于"对连续时间潜在动力学的干预"，不同于结构动态因果模型（Bongers et al., 2022）的干预表述；
- 若把 ODE 换成 SDE，需要"stochastic observability"这一新概念（Van Handel, 2009）；
- $f$ 关于 $z$ Lipschitz（保证解存在唯一）；
- 标准形与一般可观测系统的等价性只在 "**under suitable conditions**" 下成立；
- IGC-Net 只能处理二元处理指示变量（合成数据中以 $C(t)/C_{max}$、$d(t)/d_{max}$ 解释为 Bernoulli 概率）；DoseAI 只在合成癌症数据上处理连续剂量；OptAB 限于类别型处理；
- 真实数据部分**只有 RMSE**，无任何 CB/ATE（因无反事实真值）。

### 6.2 隐含局限（本仓视角）

- 🔴 **识别公式与实现之间存在未量化的距离。** 式 (4) 需要**精确**的滤波分布 $p(z_t\mid y_t,o_{[0,t)})$；实现是 LSTM 近似（"We **approximate** the filtering distribution ... with an RNN encoder"）。**识别论证建立在真实滤波分布上，模型输出是近似滤波分布的函数**。论文既未报告该近似的误差，也未做"精确滤波 vs 近似滤波"的对齐分析 —— 而这正是式 (4) 能否在数值上兑现的唯一薄弱环节。⇒ **本仓若引用"ObsNODE 是可识别的"，必须带上"这是定理层面的可识别，不是数值层面的已验证"**。
- 🔴 **可观测性是"按构造保证"，但没对真实数据是否满足做任何检验。** 式 (7) 强制 $d_z=m\,d_y$（真实 MIMIC-IV 用 $11\times3$），论文既未检验临床系统的真实可观测性，也未报告 $\phi^{(i)}_{\theta_i}$ 的可观测性条件（如 Jacobian 秩条件）是否数值满足。
- 🔴 **唯一消融不足以支撑"可观测性是功劳所在"。** 消融对象是"有/无可观测标准形的 latent NODE"这一个粗粒度开关，未拆解块数 $m$、链式耦合形式、滤波近似、损失归一化。且 NODE 的超参表（Table 14）**只覆盖合成癌症与半合成两个数据集，无真实 MIMIC-IV 配置** ⇒ "NODE 在真实数据上的表现"未定义。
- ⚠️ **"five independent runs with disjoint training, validation, and test sets" 表述有歧义**：是指每个 run 内三集合互不相交，还是不同 run 之间划分互不相交 —— 论文未澄清。若是前者，则 5 次运行共享同一份测试集，"robustness analysis" 的强度被削弱。
- ⚠️ **主图（Fig. 2–4）的逐格数值不在论文内**，正文写明 "The RMSE values of these heatmaps are given in https://github.com/JenniferJaschob/ObsNODE.git." ⇒ **论文中关于"短/长时程谁更好"的结论只能靠正文叙述，无法从论文表格核验**；且 Fig. 2/3 明确 "RMSE values larger than one were clipped at one"（截断影响读数，Fig. 4 未说明是否截断）。
- ⚠️ **半合成的隐混杂恰好用三个生命体征构造**（Creatinine、Bilirubin、ALT，附录 A.4.2 / Table 7），即 $\epsilon_t$ **本身是三个**有持续性的生理量 ⇒ **论文未报告该半合成数据是否真的满足 Assumption 4.2（时间独立）**；若不满足，则本文最强的一组数字（Table 4）建立在一个未验证的假设上。
- ⚠️ **基线含作者自己的前作（OptAB、DoseAI），且论文无利益冲突披露**（「Acknowledgments and Disclosure of Funding」节内容为空）。
- ⚠️ **CF-ODE 的对比是纯定性论断，无任何数字** —— CF-ODE **没有**被纳入任何基线表，"CF-ODE 的假设不可验证"这一批评因此**未被实验检验**。
- ⚠️ **未做显著性检验**：全部结果只有 5 次运行的 mean ± std，无 p 值、无置信区间、无配对检验；合成数据每个 $\gamma$ 只跑 5 次。

### 6.3 未来方向

- **作者明示（仅一句）**："Future work should address **persistent confounding** and **uncertainty quantification**."
- **作者暗示（附录 A.2）**：需要"**stochastic latent dynamics**"或"**an augmented observable state that captures the temporally persistent confounding process**"；以及随机可观测性概念（Van Handel, 2009）。
- **本仓推断（不外推论文结论，仅作选题提示）**：把"可观测性"从"潜状态可被重建"推广到"**观测误差有界时的近似可观测**"，并给出**可判定的残差读数**（这与本仓 11-22 引入的"证书轴"思路互补：证书问"稳定吗"，近似可观测性问"重建误差多大"）。

## 7. 诚实性核查（本仓复算与原文核对）

> 本节是**报告的一等组成部分**，不是附录。以下每条都基于论文原文 + 本仓独立复算，可复核。
> 复算脚本：`/tmp/lnn_verify_ate_v2_1127.py`（本仓自建，输入全部取自论文 Table 2 / Table 4 原数）。

### 7.1 🔴 主要发现：正文把"平均绝对 CB"误称为 "mean absolute ATE error"

**论文正文原文**：
- §5.1：*"reducing the **mean absolute ATE error** from **0.5289 to 0.1723** for tumor volume and from **2.0289 to 0.0933** for body weight."*
- §5.2：*"reduces the **mean absolute ATE error** from **3.6537 to 1.4523**".*

**本仓从 Table 2 / Table 4 原始数字复算**（两种口径并排）：

| 数据集 / 变量 | 口径 | ObsNODE | NODE | 论文正文所报 | 命中口径 |
|---|---|---|---|---|---|
| 合成 · 肿瘤体积 | mean abs(ATE − TRUE) | **0.1723** | **0.5289** | 0.5289 → 0.1723 | ✅ ATE 误差（且与 CB 同值） |
| 合成 · 肿瘤体积 | mean abs(CB) | 0.1723 | 0.5289 | 同上 | ✅ 两口径重合（NODE 的 CB≈ATE 误差） |
| 合成 · 体重 | mean abs(ATE − TRUE) | **0.0894** | **2.0329** | — | ❌ |
| 合成 · 体重 | **mean abs(CB)** | **0.0933** | **2.0289** | 2.0289 → 0.0933 | ✅ **命中 CB，不命中 ATE 误差** |
| 半合成 · 结局 | mean abs(ATE − TRUE) | **1.2045** | **3.4060** | — | ❌ |
| 半合成 · 结局 | **mean abs(CB)** | **1.4544** | **3.6537** | 3.6537 → **1.4523** | ⚠️ **NODE 侧命中 CB（3.6537），ObsNODE 侧差 0.0021（1.4544 vs 1.4523）** |

**结论（这是本仓发现的一处真实的口径/标注不一致，非推测）**：
1. **合成 · 体重**：正文所报 "2.0289 → 0.0933" 是 **mean|CB|**，而真正的 mean-absolute-ATE-error 是 **2.0329 → 0.0894**。⇒ **标签错位**（把 CB 读数标成 ATE 误差）。
2. **半合成**：正文所报 "3.6537 → 1.4523" **节点侧**等于 mean|CB|（3.6537 完全命中），但 **ObsNODE 侧 1.4523 与 mean|CB|=1.4544 差 0.0021**，且与 mean-absolute-ATE-error=1.2045 **不符**。⇒ **既错标口径，数值本身也与任一自洽口径不完全一致**（0.0021 的偏差来源论文未解释，可能是四舍五入或未公开的中间量）。
3. **合成 · 肿瘤体积**恰好两个口径重合（0.5289 / 0.1723），因此该处"碰巧正确"，不能作为正文用词正确的旁证。
4. ⇒ **正确表述应为**：ObsNODE 相对普通 NODE，**平均绝对反事实偏差**从 2.0289 降到 0.0933（体重）、从 3.6537 降到 ≈1.45（半合成）；而**平均绝对 ATE 误差**则是 2.0329 → 0.0894（体重）、3.4060 → 1.2045（半合成）。**两套口径的改善幅度不同（ATE 误差口径下的改善其实更大：3.4060 → 1.2045），因此本条错误并不改变"ObsNODE 显著优于 NODE"的定性结论，但改变了数字口径。** ⇒ 引用本篇数字时**必须写明是 CB 口径还是 ATE 误差口径**。

### 7.2 摘要的"strong performance over recent sequence models, including external validation"强于论文自身的表格证据

- 摘要：*"Experiments on synthetic, semi-synthetic, and real-world clinical data demonstrate **strong performance over recent sequence models, including external validation**."*；结论：*"perform **strongly** on synthetic, semi-synthetic, and real-world clinical data, including external validation."*
- 但论文自述：合成数据**长时程最优是 doseAI**（"For long horizons, **doseAI achieves the lowest error**, followed by ObsNODE and IGC-Net"；Table 1 可直接核对 $s=5,6$ 列）；真实 MIMIC-IV 上 **IGC-Net 在 $s=1$ 时四个变量全部优于 ObsNODE**（SOFA 0.09 vs 0.24；肌酐 0.09 vs 0.34；总胆红素 0.07 vs 0.34；ALT 0.06 vs 0.09）；半合成 $s=1$ 时 IGC-Net 0.10 优于 ObsNODE 0.13。
- ⇒ **ObsNODE 的真实优势集中在"方差极小 + 中长时程稳定 + 合成数据短时程"**，而"over recent sequence models"这一措辞未披露长时程被 doseAI 超越、以及短时程被 IGC-Net 全面超越这两件事。**摘要不能单独引用。**

### 7.3 🔴 "外部验证"没有基线，因此不支持任何跨队列优劣结论

- **Table 5b 只有 ObsNODE 一列**，外部队列上**没有 IGC-Net / SCIP-Net / OptAB 的任何数字**。
- ⇒ 外部验证只支持"**ObsNODE 能跨队列迁移且不崩**"，**不支持**"ObsNODE 在外部队列上优于基线"。摘要把这两件不可比的事合并成一句（"strong performance over recent sequence models, **including external validation**"）。
- 外部队列本身也小且极不平衡：**399 例**（364 头孢曲松 + 35 万古霉素，**0 例**哌拉西林/他唑巴坦），两队列分布差异巨大（平均 SOFA 6.8 vs 4.43；无总胆红素记录 23.3% vs 56.8%；无 ALT 记录 16.8% vs 56.1%），论文自陈这会造成 "substantial heterogeneity ... affect model robustness and generalizability across cohorts"。

### 7.4 无显著性检验

全部定量结论只有 **5 次运行的 mean ± std**（合成/半合成另扫 $\gamma$），**无 p 值、无置信区间、无配对检验、无多重比较校正**。论文也未讨论"5 次运行"中不同 run 的训练/验证/测试划分是否互相独立（见 §6.2）。⇒ **本仓引用时必须写"5-run mean±std，无显著性检验"。**

### 7.5 【代码事实，非论文内容】官方仓库用 vendored torchdiffeq + `odeint_adjoint(method='rk4', step_size=1.0)`

本仓克隆论文正文给出的代码地址 `https://github.com/JenniferJaschob/ObsNODE.git` 后核查：

- `Code/Model_observableNODE/utils_observableNODE.py` 与 `Code/Model_observableNODE/torchdiffeq/adjoint_method/adjoint.py` 表明仓库**内置了一份 torchdiffeq**，调用形式为
  `odeint_adjoint(func=self.Node, y0=z_now, t=time, method=method, adjoint_params=self.parameters(), options=options)`；
- 三个训练脚本（`train_observableNODE_seed_cancer.py` / `..._semisynMimicIV.py` / `..._MimicIV.py`）统一设置 **`method = 'rk4'`**、**`step_size = 1.0`**、`options=None`、`adjoint_method='original'`。
- ⇒ **它用可微分 ODE 求解器（RK4 定步长 + adjoint 灵敏度）求解，不是闭式解，也不是 CfC/NCP 的近似闭式更新。**

### 7.6 论文正文从不指名 ODE 求解器；LTC/CfC/NCP 零出现；Δt 从不使用

- **求解器**：全文检索 `torchdiffeq` / `diffeq` / `RK4` / `solver` / `dopri` / `euler` **零命中**；正文只写 "forecasts ... are obtained by **integrating (7) forward in time** and projecting to the observed space"。**论文未提及**具体求解器、步长、容差。（上节的 RK4 是**代码事实**，不能写成论文内容。）
- **家族**：全文（含附录、参考文献）**`Liquid` / `LTC` / `CfC` / `NCP` / `Closed-Form` 零出现**；连续时间相关工作为 Chen 2018（Neural ODEs）、Rubanova 2019（Latent ODEs）、Kidger 2020（Neural CDEs）、Seedat 2022（Neural CDEs for counterfactuals）、Brouwer 2022（CF-ODE），**无一篇来自液态家族**。
- **Δt**：**全文没有 $\Delta t$、delta t、时间间隔作为模型输入的表述**；时间只作为 ODE 自变量出现（$t$ 出现在 $f_\theta(Z_t,A_t)$ 与积分限中），观测时刻 $T_{ij}$ 只用于损失掩码与插补。**论文未提及** LTC/CfC 式的 $\Delta t$ 通道或时间间隔嵌入。⚠️ **这一点值得强调：论文把"不规则采样"当作核心卖点（§2、§4.3、§6 三处），但其机制是"连续时间方程 + 掩码损失 + 插补层"，与液态家族"$\Delta t$ 条件化"是两条不同路线 —— 不可混谈。**（代码侧同样没有 $\Delta t$ 通道：时间以 `time_obs` 拼接到观测输入 `def_x_obs`，另有 `data_time_step(x, step_size)` 的规则网格下采样/插值工具用于在网格上评估损失。）

### 7.7 其余可核查的口径提醒

- **热力图逐格数值不在论文内**（正文指向 GitHub 仓库）⇒ "短/长时程谁更好"只能靠正文叙述，**无法从论文表格核验**；Fig. 2/3 明确 "RMSE values larger than one were clipped at one"，Fig. 4 未说明是否截断。
- **真实临床数据上没有任何因果有效性证据**（论文自陈 "causal accuracy cannot be assessed on clinical data without counterfactual ground truth"）⇒ "因果"性质只在合成与半合成（有真值）上验证。
- **识别结论的条件清单很长**（consistency、SUTVA、positivity/sequential positivity、Assumption 4.1、Assumption 4.2、observability、Lipschitz、$d_z=m\,d_y$、标准形等价性的 "suitable conditions"），摘要把它们压缩成 "under explicit structural assumptions" 属合理压缩，但**单独引用摘要会让读者以为识别结论普遍成立**。
- **论文未给出**：参数量、推理延迟、显存占用对比；真实数据上的 CB/ATE；任何显著性检验；样本量敏感性分析（合成数据固定 3,000 例）。
- **致谢/资助节内容为空**，无利益冲突声明；基线 OptAB 与 DoseAI 为作者团队前作。
- **LLM 使用声明**（附录 A.7）仅一句："The core method development in this research does not involve LLMs as any important original, or non-standard components."

## 8. 对本仓的意义

**(a) 🔴 一条可迁移的架构原则："把可观测性写成参数化约束，而不是写成假设再祈祷它成立"。** 本仓 11-22 的结论已指出：本仓所有多 $\tau$ / 加性 $\tau$ / 门控 / SnCP 变体都是"给已学好的向量场加约束"，**无一具备构造性保证**；而本篇提供了一个**已被接受且带识别定理**的具体形态 —— **直接把潜状态参数化成三角结构，使"状态可从观测唯一重建"成为构造性事实**（式 7）。这条原则与本仓"构造性保证型变体"候选（`PortHamiltonianCfCCell` / `PortHamiltonianLTCCell` 等）**同源但方向不同**：本篇保证的是"可辨识/可重建"，本仓候选保证的是"耗散/稳定"。⇒ **两者可合并成一条统一的设计范式：把"我们希望这个模型具备的性质"翻译成参数化层面的构造性约束，而不是损失里的软惩罚。** 这不要求液态路线改变，只要求变体的**性质主张**必须能被结构兑现。

**(b) 🔴 一条本仓完全没有的评测轴："有反事实真值时，报 CB 与 ATE，而不是只报 RMSE"。** 本篇的消融协议非常干净可抄：**CF1–CF5 预设处理策略、$t_c=5$、六个预测时间点平均、5 次运行、同时报"带符号反事实偏差 CB"与"相对基线处理的 ATE"**。本仓目前的 bench 全部只报预测类指标（MSE / R² / AUPRC），**从没有任何一个 bench 报告过反事实偏差**。⇒ **本仓可立即加的一条新 bench 纪律**：任何声称"能做反事实/干预预测"的变体（尤其涉及门控、路由、观测-状态耦合的变体），**必须在一个有解析反事实真值的合成系统上同时报 CB 与 ATE**；若变体只能报 RMSE，则它**没有资格**声称具备反事实能力。⚠️ 这条纪律对本仓尤其重要，因为本仓的若干变体（`conservation_buys_stability`、因果门控类）本来就在谈"反事实"，而**反事实能力至今零独立证据**。

**(c) ⚠️ 一条对本仓"液态收益"叙事的边界提醒 —— 且这一条独立于本篇自身的证据强度。** 本篇是一篇**族外**论文：零次提及 LTC/CfC/NCP，$\Delta t$ 从不使用，求解器是 RK4 定步长 adjoint。**它不能被用作"液态/连续时间更好"的任何证据**，也不应进入本仓关于连续时间收益的证据链。**本篇在本仓的合法用途只有两个**：① 架构原则（构造性可观测性）；② 评测协议（CB/ATE 口径 + 明确的报告口径纪律）。⇒ 引用时必须标注"**族外 / 与液态机制优劣无关，仅提供可观测性参数化范式与因果评测协议**"，以免污染本仓索引的证据链。

**(d) 一条本仓必须记住的"读数纪律"（来自 §7.1）：论文级数字必须回到表格自证口径。** 本篇把 mean|CB| 标成 "mean absolute ATE error"，且其中一个数字（1.4523）与两个口径都对不齐。⇒ **本仓引用任何论文的"改善幅度"时，都应独立复算一遍该指标的两种可能口径并注明用的是哪一种。** 本仓已有先例（NR-08 把 +0.09 dB 判为"持平"而非"改进"、NR-09 的逐行复核）—— 本条是同一纪律在**指标口径**维度上的延伸。

**Verdict**：

- **"可观测性可以通过潜状态参数化（三角可观测标准形）被构造性地保证，并由此换来连续时间的因果识别" → TARGET-POSITIVE（族外，但可迁移）** —— 有定义（Def. 3.1）、有定理（Thm 4.3 / 式 4）、有构造（式 7）、有消融证据（Table 2/4 中 NODE 与 ObsNODE 的 CB 差距量级），且有可复现代码（RK4 + vendored torchdiffeq）。⚠️ 引用时必须标注"族外 + 定理层面的可识别 ≠ 数值层面已验证（滤波分布是 LSTM 近似）"。
- **"ObsNODE 在真实临床数据上强于近期序列模型（含外部验证）" → TARGET-NEGATIVE-WITH-NUANCE（该强表述不成立）** —— 论文自身表格即证：合成数据长时程 doseAI 更优（$s=6$：0.21 vs 0.10）；真实数据 $s=1$ 时 IGC-Net 在四个变量上全面更优（0.09/0.09/0.07/0.06 vs 0.24/0.34/0.34/0.09）；外部验证表只有 ObsNODE 一列、队列仅 399 例且分布偏移大。⇒ **正确表述**：ObsNODE 的优势集中在**极小方差 + 中长时程稳定 + 合成数据短时程**，并能跨队列迁移而不崩。
- **"ObsNODE 的因果改进幅度是 2.0289→0.0933（体重）与 3.6537→1.4523（半合成）的 ATE 误差" → TARGET-NEGATIVE-WITH-NUANCE（本仓复算证伪该口径）** —— 这两个数字是 **mean|CB|**；真正的 mean-absolute-ATE-error 是 **2.0329→0.0894** 与 **3.4060→1.2045**（本仓复算，脚本 `/tmp/lnn_verify_ate_v2_1127.py`），且半合成 ObsNODE 侧的 1.4523 与任一口径都不完全一致（与 CB 差 0.0021）。⇒ **引用时必须写明口径**。
- **"本篇支持'LNN / CfC / LTC / NCP 更优'或'连续时间优于离散'" → 不适用（族外）** —— 全文零次提及该家族，$\Delta t$ 从不使用，未做任何液态对照。**不得**用它支持本仓任何液态叙事。
- **对本仓可执行项**：① 新增"可观测性/构造性保证"作为变体设计原则的第三个维度（辨识性 × 稳定性 × 可重建性）；② 新增 **CB/ATE 评测轴**（本仓至今零反事实证据），首批可挂在 `scripts/bench_causality_gated_orth.py` 与 `lnn/core/bench_suite.py` 的合成真值系统上；③ 采纳"读数必须回到表格自证口径"的引用纪律；④ ⚠️ 属用户决策范畴的变体新增（如 `ObservableNormalFormCfcCell`），本轮**不擅自实现**。
