---
title: The Information Limits of Neural Controlled Differential Equations Under Sparse Observations
date: 2026-11
tags: [LNN, Continuous-Time, Neural-CDE, Partial-Identification, Zonotope, Lie-Bracket, Minimax-Rate, Sparse-Observation, Sampling-Design]
---

# 研读报告：Neural CDE 的信息论上限 — 稀疏观测下，连续时间模型的预测有一半是"观测没告诉你"的

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**。
> 🔴 **本篇对本仓的意义是"划边界"而非"提方案"** —— 它第一次给出了"**在稀疏观测下，一个 CDE 的预测里有多少是不可辨识的**"的定量答案。

## 1. 元数据

- **论文标题**：The Information Limits of Neural Controlled Differential Equations Under Sparse Observations
- **作者**：Rostislav Gusev, Evgenia Romanenkova, Alexey Zaytsev（Applied AI Institute）
- **发表时间**：2026-11-02 (v1)；**preprint**（comment 明示 "Preliminary version. Further theoretical extensions are under development"）；52 页
- **来源**：arXiv:2611.03887v1 [cs.LG]
- **代码**：N/A（preprint，未见代码链接）
- **本地 PDF**：[papers/2611.03887/2611.03887v1.pdf](../../papers/2611.03887/2611.03887v1.pdf) ／ 文本 [papers/2611.03887/2611.03887v1.txt](../../papers/2611.03887/2611.03887v1.txt)
- **实验对象**：训练在稠密观测平面轨迹上的 Neural CDE；合成的有符号面积目标 + 非线性受控系统输出 + 端点依赖负对照
- **关联概念**：Neural CDE / 受控微分方程 / 部分识别（partial identification）/ zonotope / **传输 Lie 括号** / Dido 定理 / 极小极大信息半径 / 观测设计 / NRDE / Log-NCDE / LTC / CfC

## 2. 核心问题

Neural CDE 靠**把离散观测插值成一条连续路径**来驱动隐藏状态。但插值是**建模选择**，不是数据给的：

> "Sparse measurements leave the input between observation times undetermined, and admissible continuous paths consistent with the same data can produce different terminal states."

具体地：观测只确定每段的**端点增量** $x_{i+1}-x_i$，而路径在两端之间**扫过的有符号面积** $a_i(X)$ **完全未确定**。在**向量场不可交换**的 CDE 里，这个面积通过 Lie 括号 $[V_1,V_2]$ 影响隐藏状态，并沿后续动力学传播。

这引出两个此前无人回答的问题：

1. **幅度问题**：对同一份观测记录 $S$，所有与 $S$ 相容的连续路径 $X\in X_L(S)$ 会产生一个**终态集合** $Z_T(S)$ 与一个**标量输出集合** $Y_T(S)$。这个集合有多大？⇒ **模型给出的那个点预测，有多少是"观测其实没告诉你"的自由发挥？**
2. **速率问题**：$N$ 个观测区间能把这个不确定性压到多小？在**确定性自适应采样策略**下，最优最坏情形的半径是多少？

已有相关工作都不直接回答这两个问题：
- Morrill et al. (2022) / Bleistein & Guilloux (2024)：分析**插值方案**与**采样依赖的泛化/逼近界**（即"选哪个插值好"），不刻画"同一个插值下还有多少自由度"。
- Walker et al. (2024)（Log-NCDE，本轮同日研读）、Malham & Wiese (2014)、Bailleul & Diehl (2015)：分析路径几何（signature / Lévy area）如何**进入**动力学，不问"进入的量有多少是未定的"。
- 随机设定下有 Lévy area 的条件分布（Malham & Wiese 2014），但那是**给定端点的 Brownian 路径**，与本文的**确定性正则性约束类**不同。
- 最优恢复 / 信息论复杂度（Micchelli 1976; Traub 1988）给的是**函数逼近**的框架，本文要的是**一个固定 CDE 的终态响应**在观测约束下的波动范围。

问题因此被重述为一个**部分识别（partial identification）**问题：不是"模型准不准"，而是"**这份观测记录最多能确定多少**"。

## 3. 方法论与核心思路

### 3.1 问题设定

受控动力学：

$$dZ_t=\sum_{j=1}^{d}V_j(Z_t)\,dX^j_t,\qquad Z_0=z_0,\qquad F(X)=g(Z_T(X))$$

观测记录 $S=\{(t_i,x_i)\}_{i=0}^{n}$，$0=t_0<\cdots<t_n=T$。**相容路径类**（$k=1,2$）：

$$X_L(S)=\Big\{X\in W^{1,\infty}([0,T];\mathbb{R}^d): X_{t_i}=x_i,\ \|\dot X\|_{L^\infty}\le L\Big\}$$
$$X_{L,M}(S)=\Big\{X\in X_L(S)\cap W^{2,\infty}: \|\ddot X\|_{L^\infty}\le M\Big\}$$

🔴 **一个关键且常被忽略的事实**（论文自己点出）：观测**确定最小可行速度界**，但**对未观测的补全路径不设任何有限上界**：

> "The observations therefore determine the minimum feasible speed bound, but place no finite upper bound on the speed of an unobserved completion."

⇒ **插值器选什么，等于替观测做了它没做的假设。**

**识别集**（identified set）：$Z_T(S;\mathcal C)=\{Z_T(X):X\in\mathcal C(S)\}$，$Y_T(S;\mathcal C)=\{F(X):X\in\mathcal C(S)\}$。

**极小极大信息半径**（确定性自适应策略）：

$$R_N(F;\mathcal C):=\inf_{\pi\in\Pi_N}\ \sup_{X\in\mathcal C}\ \mathrm{rad}\Big(F\big(F_\pi(X;\mathcal C)\big)\Big),\qquad \mathrm{rad}(A)=\frac{\sup A-\inf A}{2}$$

其中 $\Pi_N$ 允许策略自由选观测时刻（可非时间序），$N$ 是最终观测区间数。

### 3.2 几何：未观测面积 → 终态（Theorem 4.1）

**参考路径**：逐段线性插值 $\bar X$，其对应状态 $\bar z_t=Z_t(\bar X)$（分段线性插值在速度界下总可行）。

**有符号面积**（相对弦）：

$$a_i(X)=\frac12\int_{t_i}^{t_{i+1}}\Big[(X^1_t-x^1_i)\,dX^2_t-(X^2_t-x^2_i)\,dX^1_t\Big]$$

**Dido 定理**给出每段的可行面积**精确区间** $[-A_i^\star,A_i^\star]$，其中 $A^\star(r,\ell)$ 是长度 $\le\ell$、端距 $r$ 的平面路径相对其弦的最大绝对面积；**各段的面积可以独立选取**。

**一阶展开**（区间 $[t_i,t_{i+1}]$ 上，两路径同端点）：

$$\Phi^{X}_{t_i,t_{i+1}}(\bar z_i)-\Phi^{\bar X}_{t_i,t_{i+1}}(\bar z_i)=a_i(X)\,[V_1,V_2](\bar z_i)+O\Big(L^3(t_{i+1}-t_i)^3\Big)$$

其中 $[V_1,V_2]=DV_2V_1-DV_1V_2$。**这一项就是"路径几何进入动力学"的通道** —— 与 2402.18512 的 toy 实验机制完全一致。

**传输**：$J_i:=D_z\Phi^{\bar X}_{t_i,t_{i+1}}(\bar z_i)$，$P_i=J_{n-1}\cdots J_{i+1}$，于是第 $i$ 段一个单位面积对应的终态方向为

$$b_i=P_i[V_1,V_2](\bar z_i)\in\mathbb{R}^m$$

**zonotope**：

$$K(S)=\Big\{\sum_{i=0}^{n-1}a_i b_i\ :\ |a_i|\le A_i^\star\ \forall i\Big\}\subseteq\mathbb{R}^m$$

**Theorem 4.1**：存在常数 $C_Z=C_Z(V_1,V_2,L,T)>0$，对**任意**观测记录 $S$（只要 $X_L(S)\ne\emptyset$），

$$d_H\big(Z_T(S;X_L),\ \bar z_T+K(S)\big)\le C_Z\big(S_3+S_2^2\big),\qquad S_2:=\sum_i L^2(t_{i+1}-t_i)^2,\ S_3:=\sum_i L^3(t_{i+1}-t_i)^3$$

**关键性质**：$C_Z$ **与观测数量和位置无关** ⇒ 误差只由最大间隔 $\delta$ 决定（$S_2\le L^2T\delta$，$S_3\le L^3T\delta^2$ ⇒ 误差 $O(\delta^2)$）。

**直径公式**：

$$\mathrm{diam}\,Z_T(S;X_L)=2\max_{\|u\|=1}\sum_{i=0}^{n-1}A_i^\star\big|u^\top b_i\big|+O(S_3+S_2^2)$$

### 3.3 目标特定的不确定性证书（Theorem 4.2）

标量预测 $F(X)=g(Z_T(X))$ 的敏感度 $c_i=\nabla g(\bar z_T)^\top b_i$。**目标特定半径**：

$$R_2(S)=\sum_{i=0}^{n-1}A_i^\star\,|c_i|$$

**Theorem 4.2**：存在 $C_Y=C_Y(V_1,V_2,g,L,T)>0$，

$$d_H\Big(Y_T(S;X_L),\,[F(\bar X)-R_2(S),\,F(\bar X)+R_2(S)]\Big)\le C_Y(S_3+S_2^2)$$

⇒ $\big|\mathrm{diam}\,Y_T(S;X_L)-2R_2(S)\big|\le 2C_Y(S_3+S_2^2)$。

**计算上友好**：不需要显式形成传输积 $P_i$，只需**反向传播敏感度** $\lambda_n=\nabla g(\bar z_T)$，$\lambda_i=J_i^\top\lambda_{i+1}$，然后 $c_i=\lambda_{i+1}^\top[V_1,V_2](\bar z_i)$ ⇒ **一次 adjoint 反向传播即可得到全部系数**。

🔴 **这条性质对本仓极重要**：**$c_i=0$ 的区间即使有非零可行面积，也对预测零贡献** ⇒ **"观测了没有"和"观测的位置重要"是两件事** —— 观测应放在动力学对该区间几何**敏感**的位置。

### 3.4 极小极大观测极限（Theorem 5.1 / 5.2）

**非退化条件**：存在单位向量 $u$、速度 $0<v<L$、区间 $I\subset(0,T)$，使得连续时间版本的面积敏感度

$$\beta_{v,u}(t)=\big|\lambda_t^\top[V_1,V_2](\bar z_t)\big|\ge\beta_0>0\quad \forall t\in I$$

**Theorem 5.1（有界速度）**：

$$\liminf_{N\to\infty} N\,R_N(F;X_L)\ge \frac{A^\star(v,L)}{\Big(\int_I\sqrt{\beta_{v,u}(t)}\,dt\Big)^2}\ \Longrightarrow\ R_N(F;X_L)=\Theta(N^{-1})$$

**Theorem 5.2（同时有界加速度）**：

$$\liminf_{N\to\infty} N^2R_N(F;X_{L,M})\ge \frac{vM}{3}\Big(\int_I\beta_{v,u}(t)^{1/3}dt\Big)^3\ \Longrightarrow\ R_N(F;X_{L,M})=\Theta(N^{-2})$$

**加速界的机制**：加速度约束限制了兼容路径能偏离分段线性参考多远 ⇒ 面积自由度更小 ⇒ 收敛更快。

**精确的典范情形（Proposition 5.3）**：三维系统 $Z_0=0$，$V_1(z)=(1,0,-z_2/2)^\top$，$V_2(z)=(0,1,z_1/2)^\top$，读出 $g(z)=z_3$ —— **输出恰是平面路径相对起点的有符号面积**。则

$$R_N(F;X_L)=\frac{L^2T^2}{2\pi N}$$

且**均匀非自适应观测网格在所有确定性自适应策略中达到这个半径** ⇒ 🔴 **在这个典范情形下，自适应选点不能改进最坏情形信息半径**。

### 3.5 与 LTC / CfC 的关系

🔴 **本篇与本仓主线几乎无交集，这是它的价值所在 —— 它不是"另一个更好的模型"，而是"一个衡量模型的尺子"**：

- LTC / CfC 是**自主式**（$\dot h = -h/\tau + \mathrm{MLP}(h,\mathrm{input})/\tau$），**没有控制路径 $X$**，因此**没有"插值自由度"这个概念**，也**不存在本篇的识别集**。
- 反过来，本篇讨论的一切（未观测面积、传输 Lie 括号、Dido 定理）都建立在**受控路径**这个前提上 ⇒ **对 LTC/CfC 完全不适用**。
- ⇒ **引用纪律**：本篇**不能**用来支持或否定液态命题；它只能用来**约束"连续时间机制在稀疏观测下能宣称什么"**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $$dZ_t=\sum_{j=1}^{d}V_j(Z_t)\,dX^j_t,\quad Z_0=z_0,\quad F(X)=g(Z_T(X))$$ | 受控动力学与标量读出 |
| Eq. 2 | $$Z_T(S;\mathcal C)=\{Z_T(X):X\in\mathcal C(S)\},\quad Y_T(S;\mathcal C)=\{F(X):X\in\mathcal C(S)\}$$ | **识别集**（终态 / 标量输出） |
| Eq. 3 | $$R_N(F;\mathcal C):=\inf_{\pi\in\Pi_N}\sup_{X\in\mathcal C}\mathrm{rad}\Big(F(F_\pi(X;\mathcal C))\Big)$$ | **极小极大信息半径** |
| — | $$X_L(S)=\{X\in W^{1,\infty}:X_{t_i}=x_i,\ \|\dot X\|_\infty\le L\};\quad X_{L,M}(S)=X_L(S)\cap W^{2,\infty},\ \|\ddot X\|_\infty\le M$$ | **相容路径类**（速度 / 速度+加速度界） |
| — | $$a_i(X)=\frac12\int_{t_i}^{t_{i+1}}\Big[(X^1_t-x^1_i)dX^2_t-(X^2_t-x^2_i)dX^1_t\Big]$$ | **未观测有符号面积** |
| — | $$A_i^\star=A^\star(\|x_{i+1}-x_i\|,\ L(t_{i+1}-t_i))$$ | **Dido 定理**：可行面积的精确上界 |
| — | $$\Phi^X_{t_i,t_{i+1}}(\bar z_i)-\Phi^{\bar X}_{t_i,t_{i+1}}(\bar z_i)=a_i(X)[V_1,V_2](\bar z_i)+O\big(L^3(t_{i+1}-t_i)^3\big)$$ | **面积 → 状态的一阶效应（Lie 括号通道）** |
| — | $$J_i=D_z\Phi^{\bar X}_{t_i,t_{i+1}}(\bar z_i),\quad P_i=J_{n-1}\cdots J_{i+1},\quad b_i=P_i[V_1,V_2](\bar z_i)$$ | **传输 Lie 括号方向** |
| Eq. 4 | $$K(S)=\Big\{\sum_{i=0}^{n-1}a_i b_i:\ |a_i|\le A_i^\star\ \forall i\Big\}\subseteq\mathbb{R}^m$$ | **zonotope（终态识别集的一阶近似）** |
| Eq. 5 (Thm 4.1) | $$d_H\big(Z_T(S;X_L),\ \bar z_T+K(S)\big)\le C_Z(S_3+S_2^2)$$ | **终态识别集的 zonotopic 近似（一致界）** |
| — | $$\mathrm{diam}\,Z_T(S;X_L)=2\max_{\|u\|=1}\sum_i A_i^\star|u^\top b_i|+O(S_3+S_2^2)$$ | 终态直径的一阶量 |
| Eq. 6 | $$R_2(S)=\sum_{i=0}^{n-1}A_i^\star\,|c_i|,\qquad c_i=\nabla g(\bar z_T)^\top b_i$$ | **目标特定不确定性半径** |
| Eq. 7 (Thm 4.2) | $$d_H\Big(Y_T(S;X_L),[F(\bar X)-R_2(S),F(\bar X)+R_2(S)]\Big)\le C_Y(S_3+S_2^2)$$ | **标量输出的识别集近似** |
| — | $$\lambda_n=\nabla g(\bar z_T),\quad \lambda_i=J_i^\top\lambda_{i+1},\quad c_i=\lambda_{i+1}^\top[V_1,V_2](\bar z_i)$$ | **一次 adjoint 反传得到全部系数** |
| — | $$\beta_{v,u}(t)=\big|\lambda_t^\top[V_1,V_2](\bar z_t)\big|$$ | **非退化条件：面积敏感度** |
| Eq. 8 (Thm 5.1) | $$\liminf_{N\to\infty}N R_N(F;X_L)\ge\frac{A^\star(v,L)}{\Big(\int_I\sqrt{\beta_{v,u}(t)}dt\Big)^2}\ \Rightarrow R_N=\Theta(N^{-1})$$ | **有界速度下的极小极大速率** |
| Eq. 9 (Thm 5.2) | $$\liminf_{N\to\infty}N^2R_N(F;X_{L,M})\ge\frac{vM}{3}\Big(\int_I\beta_{v,u}(t)^{1/3}dt\Big)^3\ \Rightarrow R_N=\Theta(N^{-2})$$ | **加加速度界下的极小极大速率** |
| Eq. 10 (Prop 5.3) | $$R_N(F;X_L)=\frac{L^2T^2}{2\pi N}$$ | **面积 CDE 的精确最优半径（均匀网格达最优）** |

## 5. 关键成果与贡献

### 5.1 主结果汇总

| 结果 | 内容 |
|---|---|
| **Theorem 4.1** | 平面驱动、有界速度下，终态识别集有 **zonotopic 一阶近似**，Hausdorff 误差 $O(S_3+S_2^2)=O(\delta^2)$，**常数与观测数量/位置无关** |
| **Theorem 4.2** | 标量预测有**目标特定**的识别集区间 $[F(\bar X)\pm R_2(S)]$，$R_2=\sum A_i^\star|c_i|$，**adjoint 一次反传可算** |
| **Theorem 5.1 / 5.2** | 非退化平面 CDE 下，**确定性自适应采样**的最优最坏情形半径：$\Theta(N^{-1})$（仅速度界）/ $\Theta(N^{-2})$（速度+加速度界） |
| **Proposition 5.3** | 面积 CDE 的**精确**最优半径 $L^2T^2/(2\pi N)$，且**均匀网格在所有确定性自适应策略中最优** |

### 5.2 实验结果（§6）

**证书精度（§6.1）**：在 96 次路径依赖目标的评估上，预测直径 $2R_2(S)$ 与"可行输出扩散"的 **Spearman 相关 0.9987**；**可行/预测直径比值中位数 0.981**。观测数从 5 增到 33 时，中位相对偏差从 **5.49% 降到 1.69%**。拟合斜率：可行扩散对最大间隔 **1.01**，绝对偏差 **1.55**。

**负对照**：端点依赖目标的中位证书直径只有路径依赖目标的 **0.07%** ⇒ **证书确实在度量"路径依赖性"而非"一般噪声"**（这是一个设计得体的负对照）。

**路径正则性的作用（§6.2）**：72 次评估中，加速度感知的外层近似中位宽度只有速度版证书的 **7.11%**；拟合采样指数分别为 **1.04（速度）vs 2.26（速度+加速度）**。🔴 但论文自己承认：加速感知近似的**中位幅度只有其外层近似的 9.11%**（相关 0.975），且"**这个差异无法归因于保守性来自哪一源**"（全局 $C^1$ 路径还要满足跨区间的速度相容性，而数值搜索只探索了受限的相容补全族）。

**观测设计（§6.3）**：24 份初始观测记录，每步策略在中点选一处加测，共 6 次。用"归一化证书半径曲线下面积"衡量（越低越快）：

| 策略 | 归一化面积（中位） |
|---|---|
| 最大局部歧义 $A_i^\star|c_i|$ | **3.633** |
| 中点缩减分数 | **3.633** |
| 一步前瞻 oracle | 3.626 |
| 最长间隔选择 | 3.823 |
| 随机选择 | 4.544 |

⇒ **最大局部歧义规则几乎追平一步前瞻 oracle**（3.633 vs 3.626），且**优于中点缩减分数**（两者在 24 例中 4 例选了不同区间，中点缩减在其中 1 例给出更小的重算半径）。

### 5.3 贡献清单

1. **部分识别几何**：给出平面 CDE 终态识别集的 **zonotopic 一阶刻画**（可行有符号面积 × 传输 Lie 括号），并给出**与观测布局无关**的一致 Hausdorff 界。
2. **目标特定证书**：把"状态不确定性"转成"**某个具体预测**的不确定性"，且**一次 adjoint 反传即可计算** ⇒ 可作为**观测设计的打分函数**。
3. **极小极大观测极限**：给出 $\Theta(N^{-1})$ / $\Theta(N^{-2})$ 速率，**对确定性自适应策略成立**，并证明在典范情形下**自适应选点无收益**。
4. **经验验证**：证书与可行扩散 Spearman 0.9987；证书引导的选点几乎追平一步前瞻 oracle。

## 6. 局限性与未来展望

### 6.1 作者自陈局限（§7 Discussion and Limitations）

1. **几何刻画只对平面驱动成立**。更高维输入下，"未观测的二阶几何包含耦合的面积分量，其联合可行性必须先被刻画" ⇒ 超出本文范围。
2. **极小极大速率额外要求非退化的传输 Lie 括号敏感度**，且只针对**确定性自适应**观测策略。
3. **证书引导的采集规则**是否接近最优自适应信息半径，**仍是开放问题**。
4. 🔴 **一个关键的解读警告（论文自己写）**：
   > "Predictive accuracy also depends on the learned dynamics and the relationship between the CDE output and the target variable."
   ⇒ **识别集宽度度量的是"未观测演化对模型输出的影响"，不是"预测误差"**。若驱动只是离散数据的计算表示，证书度量的是"**替代连续重构之间的输出变异**"。
5. **正则性界必须反映真实信号**：若界排除了真路径，对应的识别保证失效。
6. **实验全在合成平面信号上**；评估在物理感测上需要"可额外测量的稠密记录过程"。

### 6.2 隐含局限（本仓视角）

- 🔴 **(a) 全部理论依赖 $V_j\in C_b^4$ 的光滑向量场** —— 神经网络的向量场在实践中**远不是** $C^4$，Lip 条件是 Lipschitz 而非 $C^4$。论文的定理在"训练好的 Neural CDE"上严格来说**不能直接套用**（实验部分确实用训练好的网络，但那是数值验证而非定理应用）⇒ **定理与实践之间有一道正则性缝隙**。
- 🔴 **(b) "预先给定 $L$"是外生的** —— $L$ 是使用者**假设**的，而非从数据估出的。若 $L$ 取得过紧，$X_L(S)=\emptyset$（相容类为空，识别保证失效）；取得过松，则 $A_i^\star$ 偏大、证书偏保守。**没有给出如何从数据选 $L$ 的方法**。
- 🔴 **(c) 面积独立可选取假设只在平面 + 速度界下成立**（Dido 定理的逐段独立），高维不成立（见 6.1(1)）。
- 🔴 **(d) 全篇与本仓主线（LTC/CfC）无技术交集** ⇒ **不能直接产生变体候选**；它的价值是**提供评估尺子**，而不是提供模型。
- 🔴 **(e) 复现门槛**：理论部分无法"复现"（只能核对推导），实验部分需要自建训练好的平面 CDE 与"可行连续补全"的数值搜索（论文用圆弧补全取到最大面积），本仓无对应代码 ⇒ **只能引用其定理与实验数字，不能声称已复现**。

### 6.3 未来方向

- 高维驱动下的识别集刻画（耦合面积分量的联合可行性）；
- 证书引导采集规则与最优自适应信息半径的距离；
- 更一般的正则性假设（放松 $C^4$）。

## 7. 对本仓的意义

🔴 **(1) 本篇给本仓提供了一把"衡量液态/连续时间模型"的尺子，而本仓此前完全缺少这种工具。** 本仓到 2026-11-19 为止的所有 benchmark（toy、r306/r307、bench_mdl_cfc、bench_cfc_vs_ssm、bench_cfc_temporal_dropout…）**全部是"点预测 + MSE"**。本篇指出：即便点预测的 MSE 很小，**在稀疏/不规则观测下，预测里仍可能有一个可定量的部分是观测根本没给的**。⇒ **一条新的、本仓可以立刻落地的评测轴：在稀疏观测任务上，除了 MSE，还要报告"点预测是否落在识别集 $[F(\bar X)\pm R_2(S)]$ 内"**。这比再加一个 MSE 数字昂贵得多，但它是**唯一能回答"连续时间机制是不是在编造"的问题**。

🔴 **(2) 它为本仓 11-18 建立的"真·不等间隔"纪律补上了理论侧的一块。** 11-18 的结论是："**本仓此前研读的多数论文处理的是'名义不规则、实际等间隔'，根本没给连续时间机制上场的机会。选论文时必须先核对采样是否真·不等间隔**"。本篇从**反面**给出同一条纪律的理论根据：**如果采样确实稀疏（间隔 $\delta$ 大），那么 CDE 的预测有 $O(\delta^2)$ 的不可辨识成分** ⇒ **在稀疏设定下，连续时间模型的表现有一个天花板，且这个天花板与"模型多好"无关，取决于"观测间隔多大"**。⇒ **本仓未来的任何"连续时间 vs 离散"的对比实验，都应该报告 $\delta$ 或 $N$，否则结论不可解释。**

🔴 **(3) 一条极便宜、可以立刻做、并且直接命中本仓核心命题的消融轴："把 $\Delta t$ 打乱 / 把观测顺序打乱，看输出的变化幅度"。** 本篇的机制说得很直白：**未观测段落的面积 $a_i$ 是自由变量，它以 $a_i[V_1,V_2]$ 的形式进入终态**。⇒ **本仓至今只有"打乱 $\Delta t$"这条轴（11-13 的 CT-GRU 消融），但从未测过"打乱路径几何"**。一个**只做两次前向**的诊断：在同一个稀疏观测序列上，构造两条相容但几何不同的插值路径（论文用圆弧补全），比较模型输出差 ⇒ **这个差就是本仓模型"对未观测几何的敏感度"的直接测量**。若本仓的 LTC/CfC 对此**完全不敏感**，那就直接支持"本仓液态模型没在用连续时间信息"（11-13 CT-GRU 负结果的加强版）；若**高度敏感**，则说明连续时间几何确实在起作用，但需要进一步问"敏感度是否超过观测所允许的 $R_2(S)$"。

🔴 **(4) 一条关于"插值器选择"的纪律性提醒，本仓的代码可能已经违反了它。** 本篇明说："**观测确定了最小可行速度界，但对未观测的补全路径不设任何有限上界**" ⇒ **"用哪个插值器"不是实现细节，而是**一个会直接改变输出的建模选择**。本仓 `lnn/core/ltc.py` / `cfc.py` 用的 $\Delta t$ 处理与任何连续时间实现都隐含了插值选择。⇒ **本仓若报告任何连续时间模型的结果，都应说明插值/离散化假设**（尤其在不等间隔数据上）。

🔴 **(5) 本篇对本仓主线的定位反而是"确认边界"。** LTC / CfC 是**自主式**模型（无控制路径），**根本不存在"插值自由度"这个自由度** ⇒ **本篇的所有结论对 LTC/CfC 不适用**。但这个"不适用"本身是一条有用信息：**本仓的液态模型没有插值歧义问题，所以它在稀疏观测下不会吃这个亏** —— 这是液态范式相对受控范式的一个**未被言明但真实存在**的优势，值得在本仓索引里单列一行。

**Verdict**：
- **"稀疏观测下 CDE 预测存在可量化的不可辨识成分，且其速率是 $\Theta(N^{-1})$（速度界）/ $\Theta(N^{-2})$（加速度界）" → TARGET-POSITIVE（本轮最有价值的一条理论贡献）**。引用时须标注：(a) preprint、作者自陈 "preliminary version"；(b) 只对**平面驱动**成立；(c) 依赖 $C_b^4$ 光滑向量场与**非退化敏感度**假设。
- **"目标特定的不确定性证书可由一次 adjoint 反传算得，并可作为观测设计的打分函数" → TARGET-POSITIVE（工程可用）**（Spearman 0.9987；最大局部歧义规则几乎追平一步前瞻 oracle，3.633 vs 3.626）。
- **"自适应选观测时刻能改善最坏情形信息半径" → TARGET-NEGATIVE（作者自证的典范反例）**（Prop 5.3：面积 CDE 上均匀网格已达最优 $\frac{L^2T^2}{2\pi N}$）。**这与"自适应主动感知总是更好"的直觉相反，值得本仓记录。**
- **"连续时间机制在稀疏观测下有结构性优势" → TARGET-DEPENDENT-WITH-NUANCE**。本篇**不证明**连续时间方法更准，它**证明的是"连续时间方法在稀疏观测下的预测有一个不可辨识的地板"** ⇒ **它给本仓提供了评价这类方法的必要条件，而不是支持它的证据。**
- **"本篇与液态命题（LTC/CfC）的关系" → TARGET-NEGATIVE（不适用）**。自主式模型无控制路径、无插值自由度 ⇒ 本篇结论**不可**用于支持或否定液态命题，引用时必须显式标注这一"族外"边界。
