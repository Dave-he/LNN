---
title: "Adaptive higher order reversible integrators for memory efficient deep learning"
date: 2024-10
tags: [LNN, Continuous-Time, Neural-ODE, Reversible-Integrator, Adaptive-Stepping, Yoshida-Composition, ALF, Memory-Efficiency, System-Identification, PDE-Learning]
---

# 研读报告：Adaptive Higher Order Reversible Integrators — 用 Yoshida 复合把"可逆 + 自适应步长"从 2 阶推到任意偶数阶

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描：*neural ODE / discretization / ODE solver / numerical error*）捞出来的。本仓**零覆盖**（`2410.09537` 在 187 篇报告中零命中）。
> 🔴 **本篇对本仓的意义是"给液态/连续时间路线补上一条被长期忽略的成本维度"**：**连续时间模型的训练成本不只由 NFE（串行求值次数）决定，还由"积分器阶数 × 步长"共同决定**；反过来，**阶数越高，步长可以越大、总步数越少、单步越贵**。本篇给出了这三者之间的**定量交换关系**（Table 1 的 $N = T\cdot O(\varepsilon^{-1/(p+1)})$），本仓做变体对比时**从未报过这一项**。

## 1. 元数据

- **论文标题**：Adaptive higher order reversible integrators for memory efficient deep learning
- **作者**：Sofya Maslovskaya, Sina Ober-Blöbaum, Christian Offen, Pranav Singh, Boris Wembe
- **机构**：
  - Paderborn University, Department of Mathematics, Warburger Str. 100, 33098 Paderborn, Germany（Maslovskaya / Ober-Blöbaum / Offen / Wembe）
  - Department of Mathematical Sciences, University of Bath, Bath BA2 7AY, United Kingdom（Pranav Singh）
- **发表时间**：2024-10-12 (v1)；**v2 更新于 2025-02-19**（arXiv:2410.09537v2 [math.NA]）
- **来源**：arXiv:2410.09537 [math.NA]，**纯预印本，未见期刊/会议发表记录**
- **代码**：论文未给出公开仓库链接；**明确说明其实现基于 MALI 网络 [64] 的代码**（附录 C："our implementation of the Yoshida composition method uses the code of the MALI network [64]"）
- **本地 PDF**：[papers/2410.09537/2410.09537.pdf](../../papers/2410.09537/2410.09537.pdf) ／ 文本 [papers/2410.09537/2410.09537.txt](../../papers/2410.09537/2410.09537.txt)
- **页数**：26 页
- **关联概念**：Neural ODE / 连续深度网络 / adjoint 伴随法 / 可逆神经网络 / 对称积分器 / Yoshida 复合 / 异步蛙跳（ALF）/ 状态空间增广 / 自适应步长 / checkpointing / 几何数值积分 / 系统辨识 / PDE 学习

## 2. 核心问题

论文在摘要里把三个问题并列：

> "The **memory requirement for backpropagation scales linearly with the number of layers**, which leads to memory bottlenecks during training. Moreover, **deep networks are often unable to handle time-series data appearing at irregular intervals**. These issues can be resolved by considering continuous-depth networks based on the neural ODE framework in combination with reversible integration methods that allow for variable time-steps. **Reversibility of the method ensures that the memory requirement for training is independent of network depth, while variable time-steps are required for assimilating time-series data on irregular intervals. However, at present, there are no known higher-order reversible methods with this property.**"

最后一句是全文的**靶心**：

> "**there are no known higher-order reversible methods with this property**"

展开成三条子问题：

**(a) 内存维度。** 反向传播需要保存全部中间状态 $z_j$，内存随深度 $N$ 线性增长。伴随法（adjoint）虽然不保存中间状态，但**梯度不正确**：

> "it was quickly realized [21, 63] that using this approach with arbitrary discretization methods leads to **incorrect gradients**."
> "the values of $z_j$ obtained by the numerical integration backward do not always coincide with the values obtained in the forward pass. This is why this approach usually leads to **inexact gradients**."

checkpointing 是另一条路，但对长轨迹的辨识任务，"the number of checkpoints becomes large and can lead to **memory leaks**"。

**(b) 不规则时间间隔维度。** 辛可逆积分器（symplectic reversible integrators）**不允许自适应步长**，且要求特殊的神经 ODE 结构 ⇒ "This makes them **unsuitable for use in time-series applications where data appears at irregular intervals**"。

**(c) 阶数维度（本文的原创贡献）。** 目前仅有两种支持变步长的可逆方法：**异步蛙跳 ALF** [40, 64] 与**可逆 Heun** [32]，且两者都是 **$(2,1)$ 阶**（在 $(z,v)$ 上），即速度分量只有 1 阶：

> "Both methods are known to be of order of accuracy (2, 1) in (z, v) [40], which makes them **computationally costly in learning tasks that require high accuracy** in the integration of the neural ODE... To be able to reach high accuracy, the **lower order methods are forced to use small step sizes** and, as a result, have higher computational costs."

论文进一步指出**阶数与参数辨识精度存在直接关系**（附录 C.1）：

> "we show in C.1 that **in parameter identification tasks there is a direct relation between the order of a numerical integrator and the order of accuracy of identified parameters**. Therefore, **there is a need for higher order reversible methods**."

**问题被重述为**：*能否构造一族"既可逆（内存与深度无关）、又支持自适应步长（处理不规则时间间隔）、又具任意偶数阶（高阶精度 + 少步数）"的数值积分器，并把它做成可训练的可逆神经网络？*

## 3. 方法论与核心思路

### 3.1 背景：可逆性的定义

论文**明确区分**了两种"可逆"（这是一个容易被误读的地方）：

> "This notion of reversibility **for neural networks** means that there exists an **explicit, efficient formula to invert the forward pass**. In numerical integration theory, a **time-reversible or symmetric** numerical integrator is a formula to advance the solution ... such that its inverse is obtained by substituting $h$ by $-h$."

论文证明了二者的**蕴含关系**：若离散化 $\sigma_h$ 是对称的，则

$$z_j = \sigma_{-h}(z_{j+1}, \theta_{j+1}), \qquad\text{即}\qquad z_j = \tilde\sigma(z_{j+1}) \quad\Longrightarrow\quad \text{可逆}. \tag{§2.3}$$

⇒ **对称性 ⇒ 可逆性**，因此可以使用"构造高阶对称方法"的经典数值分析工具。

### 3.2 ALF 方法（基座）

因为优化参数 $\theta(t)$ 依赖于时间，需把时间视为状态的一部分并增广速度 $v \approx f_\theta(z(t), t)$。ALF 的前向步：

$$\begin{pmatrix} z_{j+1} \\ v_{j+1} \end{pmatrix} = \Psi^{\mathrm{ALF}}_h(z_j, v_j, t_j) = \begin{pmatrix} z_j + h f_\theta\!\left(z_j + \tfrac{h}{2}v_j,\ t_j + \tfrac{h}{2}\right) \\[2pt] 2 f_\theta\!\left(z_j + \tfrac{h}{2}v_j,\ t_j + \tfrac{h}{2}\right) - v_j \end{pmatrix},\qquad t_{j+1} = t_j + h. \tag{2.5}$$

反向步：

$$\begin{pmatrix} z_j \\ v_j \end{pmatrix} = \Psi^{\mathrm{ALF}}_{-h}(z_{j+1}, v_{j+1}, t_{j+1}),\qquad t_j = t_{j+1} - h. \tag{2.6}$$

⇒ **前向与反向是同一个 $\Psi$ 取 $-h$** ⇒ 对称 ⇒ 可逆。

### 3.3 关键引理：ALF 的复合提升阶数（Theorem 3.1）

ALF 单独看是 $(2,1)$ 阶（$v$ 分量只有 1 阶一致性），但论文证明：

> **Theorem 3.1.** "Composition of two steps of ALF methods, i.e. $\Psi^{\mathrm{ALF}}_{h/2}\circ\Psi^{\mathrm{ALF}}_{h/2}$, applied to $\dot z = f(z,t)$ provides **second order accurate approximations of position $z$ and velocity $v = \dot z$**."

⇒ 这与已有文献 [40] 的分析**相反**（论文明确指出："This is surprising as the order of consistency of ALF in $v$ is only 1 [64]"）。作者用数值实验独立验证（附录 A）：ALF 在 $z$ 和 $v$ 上**都是 2 阶误差**。

**这一条是整篇论文的技术支点**：只有先证到 ALF2 是 $(2,2)$ 阶，才能对它套用"复合方法理论"来构造任意偶数阶。

### 3.4 Yoshida 复合（构造任意偶数阶）

记 $\Psi^{\mathrm{ALF2}}_h = \Psi^{\mathrm{ALF}}_{h/2}\circ\Psi^{\mathrm{ALF}}_{h/2}$。然后用 Yoshida 复合 [61]：

$$\Psi^{Y}_h = \Psi^{2k}_{ah}\circ\Psi^{2k}_{bh}\circ\Psi^{2k}_{ah}, \qquad a = \frac{1}{2 - 2^{1/(2k+1)}},\quad b = 1 - 2a. \tag{§3}$$

> **Theorem 3.2 ([61]).** "Yoshida composition of a reversible method $\Psi^{2k}$ of order $2k$ has **order $2k+2$ and is reversible**."

递归定义：$\Psi^{Y,2}_h \equiv \Psi^{\mathrm{ALF2}}_h$（起点），$\Psi^{Y,2k}_h = \Psi^{Y,2k-2}_{ah}\circ\Psi^{Y,2k-2}_{bh}\circ\Psi^{Y,2k-2}_{ah}$。

论文实际使用的 **Y4**（4 阶）展开为：

$$\Psi^{Y,4}_h = \Psi^{\mathrm{ALF2}}_{ah}\circ\Psi^{\mathrm{ALF2}}_{bh}\circ\Psi^{\mathrm{ALF2}}_{ah}. \tag{Algorithm 1}$$

⚠️ **Remark 3.3（作者自己标注的坑，必须知道）**：

> "The approach based on Yoshida composition **might require checkpoints** in case of certain neural ODEs, e.g., when the learning task is to learn a **dispersive partial differential equation such as heat equation**. This is because the **Yoshida composition forces the use of negative time-steps**, which can be a problem in dissipative cases, where it can lead to **instability**."

⇒ **负步长**是 Yoshida 复合的固有代价，在耗散系统上会失稳。

### 3.5 自适应步长

自适应步长的实现与 ALF 完全一致：**删掉计算图与所有步长计算所需的中间变量，只保存被接受的新步长 $h_j$**。前向保存 $h_1,\dots,h_N$，反向时取用。误差估计用 $(2k+1)$ 阶积分器的输出比较（Algorithm 1 第 6a–6b 步）。

⇒ **内存代价 = $N$ 个标量步长**，与中间状态维度无关。

### 3.6 梯度计算（两条路径）

增广变量 $\phi = (z,v)$，学习问题：

$$\min_{\{\theta_j\}} J = L(P_z(\phi_N), y),\qquad \phi_{j+1} = \Psi^{Y,2k}_h(\phi_j,\theta_j),\quad \phi_0 = (z, f(z,\theta_0)). \tag{3.1}$$

**路径 A（Algorithm 3，需存计算图）**：离散伴随方程

$$(\lambda_N)^\top = \nabla L(\phi_N),\qquad \lambda_j = \left[\frac{\partial\phi_{j+1}}{\partial\phi_j}\right]^\top\lambda_{j+1}, \tag{3.2}$$

$$\frac{\partial J(\theta)}{\partial\theta_j} = \lambda_{j+1}^\top \frac{\partial\phi_{j+1}}{\partial\theta_j}. \tag{3.3}$$

**路径 B（附录 B，无需存中间状态）**：利用复合结构，**直接递归写出伴随方程的数值格式**，从而**不必计算 $\partial\phi_{j+1}/\partial\phi_j$**。论文的 Theorem B.2 给出递归：

$$(\phi_k,\lambda_k) = \tilde\Psi^{Y,2k-2}_{ah}\circ\tilde\Psi^{Y,2k-2}_{bh}\circ\tilde\Psi^{Y,2k-2}_{ah}(\phi_{k+1},\lambda_{k+1}). \tag{B.4}$$

其中 $\tilde\Psi$ 是 state-adjoint 的反向步。论文明确：

> "In this case, there is **no need to compute $\partial\phi_{j+1}/\partial\phi_j$**, which makes the approach **computationally more efficient and memory efficient**." 且 "we only need to store **one intermediate state** obtained in the composition at a time. This makes the method of **the same memory cost as MALI**."

### 3.7 成本比较（Table 1 —— 本篇最该被本仓引用的一张表）

记 $d$ = $z$ 维数、$T$ = 时间区间长度、$N$ = 层数、$M$ = $f$ 内部的层数、$s$ = 步长选择所需步数、$p$ = 方法阶数、$r$ = $f$ 的求值次数（复合次数）：

| Method | 计算成本 | 内存成本 | 达到精度 $\varepsilon$ 所需的 $N$ |
|---|---|---|---|
| Backprop | $r\times d\times M\times N\times s\times 2$ | $r\times d\times M\times N\times s$ | $T\times O(\varepsilon^{-1/(p+1)})$ |
| NODE | $r\times d\times M\times N\times s\times 2$ | $d\times M$ | $T\times O(\varepsilon^{-1/(p+1)})$ |
| ACA | $r\times d\times M\times N\times(s+1)$ | $d\times(M+N)$ | $T\times O(\varepsilon^{-1/(p+1)})$ |
| MALI | $d\times M\times N\times(s+2)$ | $d\times(M+1)$ | $T\times O(\varepsilon^{-1/3})$ |
| **Proposed** | $r\times d\times M\times N\times(s+2)$ | $r^{**}\times d\times(M+1)$ | $T\times O(\varepsilon^{-1/(p+1)})$ |

其中 $r^{**}$ 注明："Memory costs of the proposed method can be **reduced to $r=1$**, if the gradients are computed as in Appendix B"。

⇒ **两个关键读法**：
1. **内存与深度 $N$ 无关**（所有非 Backprop 行都是 $d\times(\cdot)$ 或 $d\times(M+1)$），这是可逆方法的全部意义。
2. **层数 $N$ 随阶数 $p$ 变化**：$N = T\cdot O(\varepsilon^{-1/(p+1)})$。MALI 固定为 $\varepsilon^{-1/3}$（对应 2 阶），本文的 $p$ 阶方法给出 $\varepsilon^{-1/(p+1)}$ ⇒ **阶数越高，同精度下步数越少**。论文原句："N depends on the order p of the discretization method and **becomes smaller when the order is higher** for fixed $\varepsilon$ and $T$."

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| (2.1) | $\dot z(t) = f(z(t),\theta(t))$ | 连续深度网络的 ODE 形式 |
| (2.2) | $z_{j+1} = \sigma_h(z_j,\theta_j)$ | 离散化（层 = 步） |
| (2.3) | $\min J = L(z_N,y)$ | 学习问题 |
| (2.4) | $\dot p = -\left(\frac{\partial f}{\partial z}\right)^\top p,\quad p(T)=\nabla L$ | 连续伴随方程 |
| (2.5) | $\Psi^{\mathrm{ALF}}_h(z_j,v_j,t_j)$（显式展开） | 异步蛙跳前向步；对称 ⇒ 可逆 |
| (2.6) | $\Psi^{\mathrm{ALF}}_{-h}(z_{j+1},v_{j+1},t_{j+1})$ | ALF 反向步 = 前向取 $-h$ |
| Thm 3.1 | $\Psi^{\mathrm{ALF}}_{h/2}\circ\Psi^{\mathrm{ALF}}_{h/2}$ 在 $(z,v)$ 上 2 阶 | **ALF 复合升阶（本文技术支点，与 [40] 结论相反）** |
| §3 (Yoshida) | $\Psi^Y_h = \Psi^{2k}_{ah}\circ\Psi^{2k}_{bh}\circ\Psi^{2k}_{ah}$，$a=\frac{1}{2-2^{1/(2k+1)}}$，$b=1-2a$ | 复合方法：$2k$ 阶 ⇒ $2k+2$ 阶且可逆 |
| Thm 3.2 | Yoshida 复合保序且保可逆 | 高阶方法的合法性依据 |
| (3.1) | $\phi_{j+1} = \Psi^{Y,2k}_h(\phi_j,\theta_j)$，$\phi=(z,v)$ | 增广状态的可逆网络 |
| (3.2) | $\lambda_j = \left[\frac{\partial\phi_{j+1}}{\partial\phi_j}\right]^\top\lambda_{j+1}$ | 离散伴随递推（路径 A） |
| (3.3) | $\frac{\partial J(\theta)}{\partial\theta_j} = \lambda_{j+1}^\top\frac{\partial\phi_{j+1}}{\partial\theta_j}$ | 参数梯度 |
| Thm B.2 | $(\phi_k,\lambda_k) = \tilde\Psi^{Y,2k-2}_{ah}\circ\tilde\Psi^{Y,2k-2}_{bh}\circ\tilde\Psi^{Y,2k-2}_{ah}(\phi_{k+1},\lambda_{k+1})$ | **路径 B：直接递归伴随格式 ⇒ 内存回到 MALI 级别** |
| Table 1 | 各方法成本 $\times$ $N = T\cdot O(\varepsilon^{-1/(p+1)})$ | **本篇最该被引用的定量结论** |
| (4.1) | $\dot q = v,\quad \dot v = -\alpha\frac{q}{\|q\|^3}$ | 开普勒问题（辨识 $\alpha$） |
| (4.2) | $\dot q_i = v_i,\quad \dot v_i = -a_iq_i - b_iq_i^3 - \sum_j e_{i,j}(q_i-q_j)$ | 耦合 Duffing 振子（65 个未知参数，20 维） |
| (4.3) | $\dot q = v,\quad \dot v = -\nabla V(q)$ | 势函数学习（$-\nabla V$ 由 NN 逼近，51,500 参数） |
| §C.1 | $O(h^p)$ 扰动 ⇒ $O(h^{2p})$ 均方损失扰动 ⇒ 参数极小值在 $O(h^p)$ 邻域内 | **阶数 ↔ 参数辨识精度的直接关系** |

## 5. 关键成果与贡献

### 5.1 开普勒问题：达到 $10^{-8}$ 精度所需墙钟时间（Table 2，自适应步长）

| 初始 $\alpha$ | adaptive ALF | adaptive Y4 | 加速比 |
|---|---|---|---|
| 0.1 | 7.68 s | **2.42 s** | 3.2× |
| 0.7 | 4.07 s | **1.02 s** | 4.0× |
| 0.75 | 3.26 s | **0.803 s** | 4.1× |
| 0.8 | 2.5 s | **0.44 s** | 5.7× |
| 1.3 | 8.39 s | **3.85 s** | 2.2× |

⇒ 论文结论："in all tests, **Y4 is at least four times faster than ALF**"（表中最慢一档是 2.2×，严格说并非全部 ≥4×，**这是论文表述与自身数据的一处小出入**）。**机制**："The reason of the faster training for Y4 is in **using larger step sizes** for the forward integration. The lower order method requires smaller step sizes to reach the same accuracy."

### 5.2 耦合 Duffing 振子参数辨识：达到 $10^{-4}$ 精度（Table 3）

| 初始平均参数误差 | adaptive ALF | adaptive Y4 | 加速比 |
|---|---|---|---|
| 0.28897009 | 81608 s | **51720 s** | 1.58× |
| 0.29821727 | 68645 s | **40646 s** | 1.69× |
| 0.30549358 | 56764 s | **29990 s** | 1.89× |
| 0.30289593 | 96301 s | **46524 s** | 2.07× |
| 0.29106813 | 22161 s | **13790 s** | 1.61× |

⇒ "The time required by Y4 to reach accuracy $10^{-4}$ is **almost two times smaller**"。

### 5.3 势函数学习（Duffing，NN 参数化，Table 4）

| 随机初始化 | adaptive ALF | adaptive Y4 |
|---|---|---|
| 1 | 2504 s | 1974 s |
| 2 | 2961 s | 1857 s |
| 3 | 4185 s | 2627 s |
| 4 | 3542 s | 2125 s |
| 5 | 3396 s | 2616 s |

（目标：训练损失降到 $10^{-2}$ 以下）

### 5.4 离散化波动方程（PDE 学习，Table 5）

一维波动方程 $u_{tt} = u_{xx} - \nabla V(u)$，$V(u)=\frac12 u^2$，空间-时间域 $[0,1]\times[0,0.3]$，周期边界，网格 $\Delta x = 1/40$，未知函数 $f$ 用 1 隐藏层（宽 100）的全连接 ReLU 逼近，状态维 $(u_d, v_d)$ 维 40。

| 随机初始化 | adaptive ALF | adaptive Y4 |
|---|---|---|
| 1 | 336.68 s | **138.59 s** |
| 2 | 169.50 s | **133.68 s** |
| 3 | 180.82 s | **140.32 s** |
| 4 | 153.74 s | **128.28 s** |
| 5 | 142.77 s | 142.92 s |

（目标：训练损失低于 $10^{-3}$）⇒ **前 4 个初始化 Y4 明显更快，第 5 个两者持平**（论文正文说 "Yoshida is faster in finishing each epoch"，**Table 5 第 5 行是唯一的持平案例**）。

### 5.5 固定步长下的阶数差异（Figure 7，最有信息量的一条）

> "we test ALF and Y4 with the step size fixed to $h = 0.1$ and the training until either the training accuracy reaches $10^{-4}$ or the number of epochs reaches 500. Figure 7 shows that **ALF is stuck at the training accuracy $10^{-2}$ and the training stops because of reaching 500 epochs, while Y4 converges to accuracy $10^{-4}$ with 181 epochs**. The same behaviour is observed for different parameter initialization. **Decreasing the step size to $h = 0.01$ permits ALF to reach accuracy $10^{-4}$**."

⇒ **这才是"高阶"真正的价值所在**：不是"同样步长下更准"，而是"**同样的步长预算下能收敛到更优的极小值**"。低阶方法必须靠缩小步长（更多步数、更多成本）才能追上。论文的机理解释（附录 C.1）：

> "The application of a numerical integrator as a **perturbation of size $O(h^p)$** to the exact $(q^N_j(\alpha))_i$... yields a perturbation $\tilde L$ of $L$ of size $O(h^{2p})$ in case of the mean-square loss. Thus, assuming that the local minima of $L$ at the true parameter value is non-degenerate, $\tilde L$ has a local minimum **within a ball around the true parameter of size $O(h^p)$**."

⇒ **积分器阶数 $p$ 直接决定辨识参数的可达精度上限 $O(h^p)$。** 这是本篇对连续时间路线最有价值的定量洞见。

### 5.6 与不可逆 RK45 的对照

论文在 Kepler 与振子实验中一并比较了 **RK45（Runge–Kutta 4(5)）**：

> "While **RK45 is the fastest to get to accurate parameters**, it also requires the **storing of 80 additional states** during integration at each epoch, which means a considerable contribution to the memory costs."

⇒ **速度-内存的经典权衡被明确量化**：非可逆高阶方法（RK45）最快但内存不可控；可逆方法内存与深度无关但慢一些。**这正是本仓做变体对比时必须报的量。**

### 5.7 贡献清单

1. **证明 ALF 的两步复合在 $(z,v)$ 上是 2 阶**（Thm 3.1）—— 与既有文献 [40] 的 $(2,1)$ 结论相反，是把复合方法理论用到 ALF 上的前提。
2. **用 Yoshida 复合构造任意偶数阶的可逆 + 自适应步长积分器**（Thm 3.2），并给出前向/反向/梯度三个算法。
3. **给出 state-adjoint 的显式递归格式**（Thm B.2），使高阶可逆方法的内存成本**从 $r$ 倍降回 MALI 同级**（$r=1$）。
4. **给出阶数 ↔ 参数辨识精度的定量关系**（附录 C.1，$O(h^p)$）与**成本对照表**（Table 1）。

## 6. 局限性与未来展望

### 6.1 作者自陈（结论 §5）

> "While the examples in the article focus on **system identification tasks for systems governed by differential equations**, extensions to **neural stochastic differential equations** [32] are of interest and applications to **normalizing flows or image processing** [1] can be an exciting avenue to explore in future works."

另在 §4.1 提到与基于梯度的 simulation-based inference 结合是"an interesting avenue for future research"。

### 6.2 隐含局限（本仓视角）

1. 🔴 **Yoshida 复合强制负步长**（Remark 3.3）⇒ **耗散型 PDE（如热方程）上可能失稳**，作者自己建议改用 checkpointing ⇒ **"内存与深度无关"这一核心卖点在耗散系统上有折扣**。本仓若要在长时序/耗散任务上用，须实测。
2. 🔴 **只做了 4 阶（Y4）的实验**。方法宣称"任意偶数阶"，但**所有数值实验都是 Y4**，Y6/Y8 的实证缺失（$a,b$ 系数随 $k$ 变化，负步长幅度加大，风险上升）。
3. 🔴 **阶数 ↔ 参数辨识精度的关系只在"无噪声、参数为退化方向"的情形下论证**。附录 C.1 明确说这是"**asymptotic upper bound**"，并指出几何数值积分里误差可能以高度对称方式进入（辛积分的能量误差是无偏的），"If the sought parameter is related to the geometric structure that is preserved by the geometric numerical integrator, parameters can potentially be estimated to higher accuracy than expected"——**但作者随即说"These techniques, however, are tailored to the geometric problem at hand, while the approach of this article considers a more general case."** ⇒ **本篇的 $O(h^p)$ 是保守上界，不是可达下界。**
4. **学习问题的"已知结构"前提**：参数辨识实验里"**The structure of the differential equations is assumed to be known, but some parameters in the equations are unknown**"，且"the same $\theta_1,\dots,\theta_s$ appear at each layer" ⇒ **不是端到端学向量场**。NN 参数化的两个例子（势函数、波动方程）才是"向量场由 NN 给出"，但那时 $\theta$ 仍是 NN 权重。
5. **成本数字的解读陷阱**：Table 3/4/5 的墙钟时间是**单机总训练时间**（可达 96301 s ≈ 26.8 小时），**没有报 GPU/硬件、batch size、并行度、N（实际层数）、实际 $r$**（Y4 的 $r$ = ALF2 三次复合 = 6 次 ALF 步）⇒ **"加速 1.6–2×"是端到端墙钟比，不是单步成本比**；单步成本 Y4 必然更高。
6. **内存优势未被直接测量**：全文**没有给出实测峰值显存**数字；内存优势是从 Table 1 的**渐近式**推出的（"$d\times(M+1)$ 与 $N$ 无关"）⇒ 属于**理论论证而非实验验证**。
7. **Table 2 的 "at least four times faster" 与自身数据不完全一致**（$1.3$ 那一档只有 2.2×）；Table 5 第 5 个初始化两者持平。**引用时应给具体比值，不引原文结论句。**
8. **自适应步长的误差估计**依赖 $(2k+1)$ 阶积分器做对照 ⇒ **自适应本身要多算一个高阶方法**，这部分成本未计入 Table 1 的 $r$。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的变体**：
- **`ReversibleALF2Cell` / `YoshidaY4Cell`** —— 把本仓的连续时间 cell 的离散化从"定步长显式 Euler"换成"ALF2 复合"或"Y4 Yoshida 复合"。**这不是新动力学，而是新离散化** ⇒ 它测的是"本仓结论对离散化方案的敏感性"，对本仓全部既有结论都是一次鲁棒性体检。
- **`AdaptiveStepCell`** —— 步长由局部误差估计自适应给出（本仓现有多 $\tau$ 变体是"学出来的 $\tau$"，本篇是"算出来的步长"，**两者是不同机制**）。

**可嫁接到哪个 bench 脚本**：
- 本仓所有变体对比脚本：🔴 **必须补报"达到同一精度所需的总步数 $N$ 与总墙钟"两个量**（不能只比"每个 epoch 的时间"）。本篇 Table 1 给了这个量的形式：$N = T\cdot O(\varepsilon^{-1/(p+1)})$。
- 本仓 **`curvature.py` / 稳定性相关分析**：本篇附录 C.1 的"数值扰动 → 极小值位置偏移 $O(h^p)$"是一条**本仓完全没有的误差传播链** —— 本仓目前假设"训练收敛即参数正确"，本篇说明**离散化阶数本身就是一个参数精度上限**。

**提供哪个新 ablation axis**：
- ① 🔴 **"离散化阶数轴"（最高优先级）** —— 同一模型、同一数据，用 1 阶 / 2 阶 / 4 阶（ALF2 / Y4）积分器各跑一遍。**若本仓"液态优于离散"的结论在高阶离散器下消失，说明它测的是离散化误差而非模型机制。** 这是本仓至今**完全没有**的一维。
- ② 🔴 **"自适应步长 vs 固定步长"轴** —— 本仓所有 $\tau$ 变体都是隐式的"自适应时间常数"，但**从未与"数值积分器的自适应步长"区分开**。两者在数学形式上都是"每步的 $\Delta$ 不同"，**但前者改变的是动力学，后者改变的是求解器** ⇒ 本仓必须显式声明自己属于哪一类。
- ③ **"步长-精度换手曲线"轴** —— 沿 $\varepsilon$ 扫描，画"墙钟 vs 精度"曲线（而非单点比较）。本篇的数据显示**低阶方法在固定步长下会卡在精度上限**，单点比较会漏掉这个现象。
- ④ **"可逆性/内存轴"** —— 若本仓要训大深度变体，应报"峰值显存 vs 深度"，验证可逆离散化的收益。

**一条口径纪律**：
> 本篇**不能**被引用为"液态网络更省内存"或"连续时间模型更快"。它证明的是：**在神经 ODE 框架内，"可逆 + 自适应步长 + 高阶"这三者可以兼得，而此前文献认为不可兼得**；并且**积分器阶数直接设定参数辨识精度的渐近上限 $O(h^p)$**。本仓引用时应落在"**连续时间路线的误差来源被低估了：求解器阶数是与模型架构同级的变量**"这一点上。

**Verdict**：
- 对 **"高阶 + 可逆 + 自适应可兼得"**：**TARGET-POSITIVE**（有定理、有算法、有 5 组实验的端到端墙钟证据）。
- 对 **"本仓既有变体结论的鲁棒性"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 本篇的 $O(h^p)$ 参数精度上限意味着**本仓必须在高阶离散器下重测**，否则无法排除"测的是求解器误差"。
- 对 **"内存优势"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 仅有渐近理论论证，**无实测显存数字**，且 Yoshida 的负步长在耗散系统上会打破这一优势（作者自陈）。
