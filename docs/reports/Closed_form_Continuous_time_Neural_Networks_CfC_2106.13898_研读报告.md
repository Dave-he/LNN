---
title: Closed-form Continuous-time Neural Networks (Closed-form Continuous-depth Models, CfC)
date: 2026-11
tags: [LNN, CfC, LTC, NCP, closed-form, continuous-depth, irregular-sampling, foundation-paper]
arxiv_id: 2106.13898
---

# 研读报告：CfC — 本仓技术谱系的**奠基论文**，以及它自己给出的"哪些任务**不该**用 CfC"的清单

## 1. 元数据
- **论文标题**：Closed-form Continuous-time Neural Networks（arXiv 元数据标题为 "Closed-form Continuous-time Neural Models"；期刊版 *Closed-form continuous-time neural networks*, **Nature Machine Intelligence** 4(11):992–1003, 2022）
- **作者**：Ramin Hasani†, Mathias Lechner†, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus（† 共同一作）—— **MIT CSAIL** / **IST Austria** / **Aalborg University** / **University of Vienna**
- **发表时间**：arXiv 2021-06-25（v2 2022-03-02, cs.LG）
- **代码**：论文正文引用 NCP 的公开仓库；**CfC 自身代码在期刊版补充材料中开放**，本仓已归档 PDF/文本
- **本地 PDF**：[papers/2106.13898/2106.13898.pdf](../../papers/2106.13898/2106.13898.pdf) ／ 文本 [2106.13898.txt](../../papers/2106.13898/2106.13898.txt)
- **arXiv 分类**：cs.LG
- **关联概念**：LTC（Hasani 2021）/ NCP（Lechner 2020）/ CfC（闭式连续深度）/ Neural ODE / 不规则采样 / 连续时间门控 / 混合记忆架构（mmRNN）
- **入选理由**：**本仓把 `2106.13898` 当作"L3 CfC"基准引用了 4 次（索引第 77、548、558、670 行）、把它的 PDF 归档在 `papers/foundational/lechner_2022_cfc.pdf`，但本仓至今**没有任何一篇正式研读报告**，也没有任何一次**独立复算**它的核心定理。**本仓 `lnn/core/cfc.py` 的实现就是这篇论文的 Eq. 10** —— 也就是说，**本仓最核心的代码基座，其理论依据从未被本仓自己审过一遍**。这是本轮最值得做的补课。

## 2. 核心问题

**连续时间模型（Neural ODE / LTC / CT-RNN）的表达力被数值求解器锁死了。** 论文原文：

> "Continuous-time neural processes are performant sequential decision-makers that are built by differential equations (DE). However, their expressive power when they are deployed on computers is bottlenecked by numerical DE solvers."

具体地说，Neural ODE 要积分

$$\frac{dy(t)}{dt} = \mathrm{NN}_{dyn}(t, y(t), I(t); w_{dyn})$$

必须用求解器。Table 1 给出的复杂度代价是：

| 方法 | 计算复杂度 | 局部误差 |
|---|---|---|
| $p$ 阶求解器 | $O(K \cdot p)$ | $O(\epsilon^{p+1})$ |
| 自适应步长求解器 | – | $O(\tilde{\epsilon}^{p+1})$ |
| Euler 超求解器 | $O(K)$ | $O(\delta\epsilon^2)$ |
| $p$ 阶超求解器 | $O(K\cdot p)$ | $O(\delta\epsilon^{p+1})$ |
| **CfC（本文）** | **$O(\tilde{K})$** | **不适用** |

**误差与复杂度都被消掉了** —— 因为不再有"步长"这个概念。

但问题在于：**LTC 的 ODE 有一个此前没有已知闭式解的积分**：

$$\int_0^t f(I(s))\, ds$$

其中 $f$ 是正、连续、单调递增、有界的非线性，$I(s)$ 是任意定义的输入（如真实世界传感读数）。这个积分**不是初等函数**，无法解析求解。

于是问题被重述为：**能否为这个积分构造一个近似闭式解，并且证明这个近似是"紧"的（tight），即误差有可计算的指数衰减上界？**

## 3. 方法论与核心思路

### 3.1 总体架构（论文 Fig. 4）

```
输入 batch (I(t), t 向量)
      │
      ▼
  backbone 共享前几层  ──┬──► f(x,I;θ_f)  → 时间门控 (time-continuous gating)
                        ├──► g(x,I;θ_g)  → 分支 1
                        └──► h(x,I;θ_h)  → 分支 2
      │
      ▼
  x(t) = σ(−f(x,I;θ_f)·t) ⊙ g(x,I;θ_g) + [1−σ(−f(x,I;θ_f)·t)] ⊙ h(x,I;θ_h)
```

**backbone + 三头（f/g/h）的设计**：共享 backbone 让"时间常数"与"状态非线性"之间**建立耦合**（这是液态网络因果表征学习的前提），分离的头让"时间依赖"与"结构依赖"**独立探索**。

### 3.2 LTC 的 ODE 语义（Eq. 1）

$$\frac{dx}{dt} = -\big(w_\tau + f(x, I, \theta)\big)x(t) + A\,f(x, I, \theta) \tag{Eq. 1}$$

$w_\tau$ 是时间常数参数向量，$A$ 是偏置（反转电位），$f$ 是参数化神经网络。

### 3.3 核心定理：标量闭式解（Theorem 1）

$$\boxed{\;x(t) = (x_0 - A)\,e^{-\left[w_\tau + f(I(t),\theta)\right]t}\,f(-I(t),\theta) + A\;} \tag{Eq. 2}$$

**推导路径**（论文自己给了完整链条，这是本文最有价值的部分）：

1. **标量情形线性化**（Eq. 3）：单神经元 + 单维输入时 Eq. 1 对 $x$ 是线性的。
2. **线性 ODE 理论给出双积分**（Eq. 4）：
$$x(t) = e^{-\int_0^t [w_\tau+f(I(s))]ds}x(0) + \int_0^t e^{-\int_s^t[w_\tau+f(I(v))]dv}A(w_\tau+f(I(s)))ds$$
3. **代换消掉内层积分**：令 $u(s):=\int_s^t[w_\tau+f(I(v))]dv$，则 $du/ds = -(w_\tau+f(I(s)))$，积分换元后**双积分塌缩为单积分**，再整理得（Eq. 5）：
$$x(t) = (x(0)-A)e^{-w_\tau t}e^{-\int_0^t f(I(s))ds} + A$$
4. **分段常值输入有精确闭式**（Eq. 6）：若 $I(t)=\gamma_i$（$t\in[\tau_i,\tau_{i+1})$），则
$$\int_0^t f(I(s))ds = f(\gamma_k)(t-\tau_k) + \sum_{i=0}^{k-1} f(\gamma_i)(\tau_{i+1}-\tau_i)$$
5. **一般输入的近似**（Lemma 1）：
$$\tilde{x}(t) = (x(0)-A)e^{-\left[w_\tau t + f(I(t))t\right]}f(-I(t)) + A \tag{Eq. 8}$$
**误差界（这是"紧"的正式含义）**：
$$\left|x(t) - \tilde{x}(t)\right| \leq |x(0)-A| e^{-w_\tau t} \quad \forall t \geq 0$$
且**上界是锐的（sharp）**：$\sup\left\{\frac{1}{c}|x(t)-\tilde{x}(t)| \mid I:[0,t]\to\mathbb{R}\right\} = e^{-w_\tau t}$（$c=x(0)-A$）。下界则为 $e^{-w_\tau t}(e^{-t}-1)$。

**关键含义**：误差随 $t$ **指数衰减**（因为 $w_\tau>0$），所以**近似只在"早期时间"较差，长时间反而更准**——这与直觉相反，值得注意。

### 3.4 实例化与编译（Algorithm 1）

论文给出一个**把训练好的 LTC 网络"编译"成闭式版本**的算法。核心是：遍历每个神经元、每个突触，用该突触的参数 $(x_0, A_{ij}, \tau_i, \sigma_{ij}, \mu_{ij})$ 直接代入闭式解求和：

$$\hat{x}_i \mathrel{+}= (x_0 - A_{ij})\exp\Big[-t_x(t)\odot\Big(\tfrac{1}{\tau_i} + \tfrac{1}{1+e^{-\sigma_{ij}(x_{pre,ij}-\mu_{ij})}}\Big)\Big]\odot\frac{1}{1+e^{\sigma_{ij}(x_{pre,ij}-\mu_{ij})}} + A_{ij}$$

**这一步很关键：它意味着"LTC → CfC"是零重训练代价的**，只换推理路径。

### 3.5 实证紧致性（Fig. 3）

取一个**真实训练好的 NCP**（19 神经元 / 253 突触，自动驾驶车道保持任务），把每个神经元的参数代进闭式解，**离线仿真**该 ODE，与闭式解对比：**MSE = 0.006**。这是理论界（Lemma 1）在真实网络上的落地验证。

### 3.6 工程化：为什么最终公式不是 Eq. 9

**Eq. 9**（直接从标量解扩展到向量）：
$$x(t) = B \odot e^{-[w_\tau + f(x,I;\theta)]t} \odot f(-x,-I;\theta) + A$$

**它有训练问题**：指数项会把状态快速推向 0、把隐状态推向 $A$；**存在循环连接时这会造成梯度消失**。论文列了四项修正，**每一条都对应一个具体问题**：

| 修正 | 动机 |
|---|---|
| 指数 → **反向 sigmoid** $\sigma(\cdot)$ | 缓解梯度消失；$t=0$ 时 ≈1，$t\to\infty$ 时 →0，但过渡更平滑，"better condition on the loss surface" |
| 偏置 $B$ 并入 $f(-x,-I;\theta)$ | 让偏置可训练 |
| $A$ 换成网络 $h(\cdot)$ | 增加表达力 |
| **门控平衡**：$h$ 项乘 $(1-\sigma(\cdot))$ | 让时间衰减 sigmoid 承担**门控**角色，在 ODE 轨迹的 $t\to-\infty$ / $t\to\infty$ 两个极限间插值 |

**最终 CfC（Eq. 10）**：
$$x(t) = \underbrace{\sigma(-f(x,I;\theta_f)t)}_{\text{time-continuous gating}}\odot g(x,I;\theta_g) + \underbrace{\left[1-\sigma(-[f(x,I;\theta_f)]t)\right]}_{\text{time-continuous gating}}\odot h(x,I;\theta_h) \tag{Eq. 10}$$

**时间 $t$ 怎么取**（论文明确给出，本仓实现必须对齐）：**时变数据集**（不规则采样、事件驱动、稀疏）→ $t$ 取每个样本的**时间戳**；**顺序任务**（时间无关）→ $t$ 按 batch 长度等距采样，在两个超参 $a,b$ 之间。

### 3.7 与 LTC / NCP / Neural ODE 的关系

| | LTC（Hasani 2021） | **CfC（本文）** | Neural ODE |
|---|---|---|---|
| 时间语义 | ODE | **显式 $t$，闭式** | ODE |
| 求解器 | 需要（hybrid solver） | **不需要** | 需要 |
| 训练开销 | 高 | **低一个数量级** | 高 |
| 不可逆性 | 可逆（ODE 保证） | **不可逆**（作者明说 CfC 不保证双射） | 可逆 |
| 稳定性理论 | 无 | 无 | 无 |

**本仓实现的定位**：`lnn/core/cfc.py` 的 `CfCCell` docstring 直接写着 Eq. 10 的形式（`f_gate`/`g_branch`/`h_branch` + `time_scale` + `select_step_delta`），**与论文 Eq. 10 同构**。本仓的 `n_tau>1` 多时间尺度扩展则是**论文没有的**（论文是单一 $w_\tau$）。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $\frac{dx}{dt} = -(w_\tau + f(x,I,\theta))x(t) + A f(x,I,\theta)$ | **LTC 的 ODE 语义**（被闭式化的对象） |
| Eq. 2 | $x(t) = (x_0-A)e^{-[w_\tau+f(I(t),\theta)]t}f(-I(t),\theta)+A$ | **Theorem 1：标量闭式近似** |
| Eq. 3 | $\frac{d}{dt}x(t) = -[w_\tau+f(I(t))]\cdot x(t)+A f(I(t))$ | 标量线性化 |
| Eq. 5 | $x(t) = (x(0)-A)e^{-w_\tau t}e^{-\int_0^t f(I(s))ds}+A$ | 双积分塌缩后的**精确**线性 ODE 解 |
| Eq. 6 | $\int_0^t f(I(s))ds = f(\gamma_k)(t-\tau_k)+\sum_{i=0}^{k-1}f(\gamma_i)(\tau_{i+1}-\tau_i)$ | **分段常值输入的精确闭式** |
| Eq. 8 | $\tilde{x}(t) = (x(0)-A)e^{-[w_\tau t+f(I(t))t]}f(-I(t))+A$ | **Lemma 1：一般输入的近似** |
| Lemma 1 界 | $\lvert x(t)-\tilde{x}(t)\rvert \le \lvert x(0)-A\rvert e^{-w_\tau t}$；$\sup = e^{-w_\tau t}$（**锐**） | **近似紧致的正式保证** |
| Eq. 9 | $x(t) = B\odot e^{-[w_\tau+f(x,I;\theta)]t}\odot f(-x,-I;\theta)+A$ | 向量化闭式（**梯度有问题**，故未直接使用） |
| Eq. 10 | $x(t) = \sigma(-f(x,I;\theta_f)t)\odot g(x,I;\theta_g) + [1-\sigma(-[f(x,I;\theta_f)]t)]\odot h(x,I;\theta_h)$ | **CfC 最终公式（本仓 `cfc.py` 的实现对象）** |
| Alg. 1 | $\hat{x}_i \mathrel{+}= (x_0-A_{ij})e^{-t_x(t)\odot(1/\tau_i+1/(1+e^{-\sigma_{ij}(x_{pre}-\mu_{ij})}))}\odot\frac{1}{1+e^{\sigma_{ij}(x_{pre}-\mu_{ij})}}+A_{ij}$ | **LTC → CfC 零重训练编译** |

## 5. 关键成果与贡献

### 5.1 主结果

**（a）自动驾驶车道保持**（Lexus RX450H 真车，1km 私有测试路；30Hz RGB，3 小时数据 + 增强）——所有网络均能完成任务。参数效率（Table 3）：

| 模型 | 总参数 | **RNN 部分参数** |
|---|---|---|
| CNN（无 RNN） | 2,124,901 | – |
| LSTM | 259,733 | 33,089 |
| NCP | 233,139 | 6,495 |
| **Cf-S** | 227,728 | **1,084** |
| **CfC** | 230,828 | **4,184** |
| CfC-NoGate | 230,828 | 4,184 |
| **CfC-mmRNN** | 235,052 | 8,408 |

**CfC 的 RNN 部分只有 4,184 个可训练参数**，比 NCP 还少 36%。且注意力剖面（Fig. 5）在**冬季/夏季/高噪声（$\sigma^2=0.35$）**三种条件下都保持稳定。

**（b）Bit-stream XOR（Table 4）** —— 密集采样全部 100%，但**事件采样（不规则）**下分野明显：

| 模型 | 等间隔 | **事件采样（不规则）** | min/epoch |
|---|---|---|---|
| ODE-RNN | 50.47% | 51.21% | 4.11 |
| LTC | 100% | **49.11%** | 2.67 |
| CT-GRU | 100% | 61.36% | 0.80 |
| GRU-D | 100% | 97.90% | 0.58 |
| ODE-LSTM | 100% | 98.89% | 1.18 |
| **Cf-S** | 100% | 85.42% | **0.36** |
| **CfC-NoGate** | 100% | 96.29% | 0.78 |
| **CfC** | 100% | **99.42%** | 0.75 |
| **CfC-mmRNN** | 100% | **99.72%** | 1.26 |

**🔴 这张表里最刺眼的一行是 LTC：等间隔 100%，事件采样 49.11%（≈随机）**。**这是本仓最该记住的一条外部证据：LTC（ODE 路线）在不规则采样上会直接崩掉，而 CfC（闭式路线）不会。** 论文的机制解释是：ODE 路线受梯度消失/爆炸困扰（无论用哪种求解器），闭式路线把 $t$ 显式放进指数/门控，因而对 $\Delta t$ 变化鲁棒。

**（c）PhysioNet 2012（Table 5）**：CfC AUC 竞争力强（与 ODE-RNN 的 0.833 同档），但**训练快 160×（vs ODE-RNN）/ 220×（vs 连续隐变量模型）**，平均比先进离散门控模型快 3×。

**（d）IMDB 情感（Table 6）**：CfC-mmRNN **88.3%**（最佳），CfC-noGate 87.5%，CfC 85.9%；而 **LTC 只有 61.8% ± 6.1**。注意 HiPPO-LagT/LegS 为 88.0%。

**（e）物理动力学建模（Walker2D，Table 7）** —— **不规则采样**的物理状态回归：

| 模型 | 平方误差 | min/epoch |
|---|---|---|
| ODE-RNN | 1.904 | 0.79 |
| CT-GRU | 1.172 | 0.18 |
| CT-LSTM | 1.014 | 0.31 |
| ODE-LSTM | 0.883 | 0.29 |
| **LTC** | 0.662 | 0.78 |
| **CfC** | **0.643** | **0.08** |
| **CfC-mmRNN** | **0.617** | 0.34 |
| Transformer | 0.761 | 0.8 |

论文称 CfC 在此任务上**比 Transformer 好 18%**（0.643 vs 0.761）。**⚠️ 口径提醒：这是"每时间步回归"任务，不是标准主基准，不足以支撑"LNN 超越 transformer"的通用主张（见本仓 NR-06）。**

### 5.2 机制分析
- **紧致性的实证**：真实 NCP（19 神经元/253 突触）上 ODE vs 闭式 **MSE = 0.006**。
- **CfC-mmRNN > CfC 的模式贯穿三个任务**（IMDB 88.3 vs 85.9；Walker2D 0.617 vs 0.643）→ **混合记忆架构（把 CfC 作为 LSTM 的记忆状态）是更稳的用法**，论文在结论里明确建议长依赖任务用 CfC-mmRNN。

### 5.3 贡献清单
1. **Theorem 1 + Lemma 1**：为 LTC 的 ODE 给出**带锐界的近似闭式解**（误差 $\le |x_0-A|e^{-w_\tau t}$，上界可达到）。
2. **Eq. 10 的 CfC 架构**：时间门控 + backbone/三头，**训练/推理比 ODE 快 1–5 个数量级**。
3. **Algorithm 1**：**零重训练**地把训练好的 LTC 编译成闭式版本。
4. **系统性的经验证据**（5 个任务 + 真车）：闭式路线在**不规则采样**上相对 ODE 路线有**结构性优势**。

## 6. 局限性与未来展望

### 6.1 作者自陈局限（论文"Scope, Discussions and Conclusions"节，逐条照录）
- **CfC 可能表达梯度消失问题**；长依赖任务应配**混合记忆网络**（CfC-mmRNN）。
- **从 ODE 网络推断因果性可能比闭式更直接**（作者只是"推测"，未验证）。
- **验证一个连续神经流，用 ODE 表示还是闭式表示更可行**，也待评估。
- **不可逆性**：ODE 保证可逆（唯一性条件下可反向运行），**CfC 只是近似，不再必然是双射** ⇒ **生成建模（连续归一化流）不适用 CfC**。
- **CfC 是 ODE 的近似 ⇒ 逼近误差随结构/时间尺度变化**。

### 6.2 隐含局限（本仓视角）
- **本论文的基线里没有 Transformer 的"同口径"比较**，除了 Walker2D 的 0.761。**"比 Transformer 好 18%"是单任务结论**（本仓 NR-06 纪律：不得推广）。
- **表 4 的基线数字大量转引自其他论文**（† 标记），**不是同代码同超参下的复现** —— 这削弱了"LTC 在事件采样下崩到 49%"这条结论的强度。**但这条结论对本仓极其重要，值得本仓自己复现验证。**
- **误差界 $|x_0-A|e^{-w_\tau t}$ 依赖 $w_\tau > 0$ 且 $f$ 单调有界** —— 论文假设的是"正、连续、单调递增、有界"的 $f$。**本仓 `cfc.py` 的 `f_gate` 是 Sigmoid（单调有界 ✓），但 `g_branch`/`h_branch` 是 Tanh（非单调）** —— **Tanh 不满足"单调递增"假设**，所以本仓的 CfC 实现**严格来说不在 Lemma 1 的适用范围内**。论文没有讨论这一点。
- **"闭式"仍不是免费的**：Eq. 10 里的 $t$ 是**显式超参**，$t$ 怎么设（不规则采样用时间戳 vs 顺序任务等距）**完全依赖使用者的约定**，论文没有给出自动选择机制 —— **这是 CfC 在实践中最大的自由度，也是最容易出错的地方。**
- **Table 7 的 CfC-noGate 0.650 vs CfC 0.643** —— **门控机制的增益在 Walker2D 上只有 1%**，但 IMDB 上是 85.9 vs 87.5（1.6%）**。论文说 CfC-mmRNN 更好，但门控本身的独立贡献其实很小。**

### 6.3 未来方向
论文未列专门章节，从"Scope"节可归纳：混合记忆架构、因果推断、连续流验证、生成建模需 ODE。
**本仓可补的**：把"$t$ 的设定方式"做成消融轴（本仓 `select_step_delta` / `CfCDT` 变体已经有基础设施，见 `lnn/core/variants.py:453` 的 `CfCDTCell`）。

## 7. 对本仓的意义

### 7.1 可作 `lnn/core/variants.py` 的什么变体
**本仓最大的缺口是"指数门控 vs sigmoid 门控"的对照**。论文明确说指数项带来梯度消失、sigmoid 是替代，且**论文自身从未把 Eq. 9 与 Eq. 10 做消融对比**（Eq. 9 只作为"理论式"出现，未进任何实验表）。本仓 `lnn/core/cfc.py` 实现的是 sigmoid 版；**补一个 `ExponentialGateCfCCell`（Eq. 9 形式）是零理论成本的、且论文明确留白的对照**。

其次，本仓的 `n_tau>1` 多时间尺度扩展**在原论文里没有对应物** —— 本仓可以反向做一件事：**把多时间尺度的 $\tau$ 代入论文的紧致性分析**，看误差界 $|x_0-A|e^{-w_\tau t}$ 在**不同 $\tau_i$ 分组**下如何分别衰减（每组一个 $e^{-w_{\tau_i}t}$）—— 这是一条**论文没写、本仓能写**的分析。

### 7.2 可嫁接到哪个 bench 脚本
- **`scripts/bench_cfc_temporal_dropout.py` / `bench_cfc_input_dropout.py`**：直接服务于**"不规则采样"**这条本仓最关心的轴。论文 Table 4 的"LTC 在事件采样下 49%" vs "CfC 99.42%"是本仓该做的对照的**外部锚点**。
- **`scripts/experiment_physics_lnn.py`**：物理系统仿真 = Walker2D 的近亲，可直接对照论文 Table 7 的口径（每时间步回归 + 不规则采样）。
- **`scripts/bench_cfc_backward_coherence.py`**：论文明说 CfC **不保证可逆**（ODE 保证）—— 这给本仓现有的"反向一致性"bench 提供了一个**理论预期值**：CfC 的反向一致性**本就应该比 ODE 模型差**，若本仓测出 CfC 反向一致性很好，那说明实现里其实混了 ODE 语义，需要核对。

### 7.3 提供哪个新 ablation 轴
1. **🔴 "等间隔 vs 不规则采样"必须成为本仓的**一等公民**轴**（论文 Table 4 的 LTC 49.11% 是最有力的外部证据）。本仓现有 bench 大多等间隔采样 —— **若不规则采样普遍让液态模型崩掉，本仓目前大部分结论的可迁移性都要打折。**
2. **指数门控 vs sigmoid 门控**（论文留白，本仓可做）。
3. **"$t$ 的设定方式"**：时间戳 vs 顺序索引 vs batch 内等距（本仓 `CfCDTCell` 已有基础设施）。
4. **"闭式 vs ODE 求解器"的同参数量对照**：本仓 `lnn/core/ltc.py`（ODE 路线，`ode_method="euler"`）与 `lnn/core/cfc.py`（闭式路线）**本来就并存** —— 但**本仓至今没有把它们放在同一任务上做过 $\Delta t$ 扰动对照**。论文 Table 4 说明这正是两者分野最大的地方。

**Verdict**：
- **CfC 作为本仓的代码基座 → TARGET-POSITIVE，且优先级最高**：本仓 `lnn/core/cfc.py` 直接实现这篇论文的 Eq. 10，但**从未审过它的理论**。本轮研读后可以确认：**实现方向正确**，且论文的 Lemma 1 给出了"闭式近似紧致"的正式保证 —— **但该保证的假设（$f$ 单调递增有界）被本仓的 Tanh 分支悄悄违反了**，这一点论文未讨论、也未在实现中体现。**建议在 `cfc.py` 的 docstring 里补一行"本实现的 $g/h$ 分支为 Tanh，不满足 Lemma 1 的单调性假设"。**
- **"CfC 在不规则采样上远优于 ODE 路线" → TARGET-POSITIVE（本仓最该立刻验证的一条）**：Table 4 的 LTC 49.11% vs CfC 99.42% 是本仓最关心的轴上最有力的外部证据。**但该数字转引自其他论文**，本仓应自建复现后再引用为定论。
- **"CfC 比 Transformer 好 18%" → 不能这样引用**：单任务（Walker2D 每时间步回归）、单次对照、无方差以外的统计。**按本仓 NR-06，不得推广为"LNN 在主基准上超过 transformer"。**
- **"CfC 保留了 ODE 的全部性质" → 明确不成立**：作者自己说 CfC **不再保证双射/可逆**（生成建模不适用），且**因果推断可能不如 ODE 直接**。

## 8. 引用纪律（本篇专用）

1. **引用 Eq. 10 时必须注明"这是 sigmoid 门控版本（Eq. 10），不是指数版本（Eq. 9）"** —— 本仓 `cfc.py` 实现的是前者；两者论文没有做消融对比。
2. **引用 Lemma 1 的误差界时必须附上"$f$ 正、连续、单调递增、有界"这一假设**，并注意本仓 `g_branch`/`h_branch` 用 Tanh，**不满足单调性**。
3. **引用"LTC 在不规则采样下崩到 49%"时必须标注"数字转引自 (9)（Rubanova et al.），非同代码复现"**。
4. **不得引用为"LNN 在标准主基准上 SOTA"**（本仓 NR-06）；Walker2D 是"每时间步回归"，不是主基准。
5. **本篇与本仓 `2412.02215`（LTC 隐式物理模型反推）是同一任务方向的两篇不同路线**：2412.02215 走"黑盒求解器进 loss"，本篇走"闭式编译"。**引用"隐式物理/参数辨识"时不应把两者混为一谈。**

---
*研读依据：`paper-analyzer` 技能 SOP + AGENTS.md §2 摘要 Agent 标准模块（6 段）+ 研读骨架模板。PDF 全文经 `scripts/extract_pdf_text.py` 提取（63,269 字节 / 1,111 行），所有公式编号与 PDF 原文对齐；Theorem/Lemma/Algorithm 编号沿用原文。*
