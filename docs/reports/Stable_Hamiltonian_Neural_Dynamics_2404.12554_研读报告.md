---
title: "Learning Stable and Passive Neural Differential Equations"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-Differential-Equation, Stability-Theory, Port-Hamiltonian, Bi-Lipschitz, Structural-Guarantee, Preprint]
---

# 研读报告：SHND — 把稳定性做成**结构**而不是惩罚：向量场被参数化为 Lyapunov 函数的下降方向，耗散性由"反对称 + 半正定"的代数恒等式保证

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描，第八天）捞出来的。本仓**零覆盖**（`2404.12554` 在 183 篇报告中零命中；`Continuum Dropout` / `Polyak` / `bi-Lipschitz` 三个关键词同样零命中）。
> 🔴 **本篇对本仓的意义有两层**：(a) 它给出一条**本仓至今完全没有的"保证升级路径"** —— 本仓现有全部稳定性机制（`lyapunov_stable_cfc.py` / `iss_stable_cfc.py`）都是**采样残差经验正则**（惩罚项，11-24 已定级为 **B 级**），而 SHND 的保证**不依赖训练过程**，来自向量场的**参数化形式本身**（**A 级构造性**）⇒ 若移植，本仓的"稳定性证书"可以从"训练时没被违反"升级为"推理时恒成立"；(b) 🔴 **本仓逐条复算后发现论文自身有一处硬错误**（Theorem 1(b) 印刷的收敛速率 $\lambda=\epsilon$ 与其自身证明给出的 $\lambda=\epsilon\mu^2$ **相差 $\mu^{-2}=100$ 倍**，本仓用一个可容许实例数值证伪了印刷版）⇒ **引用本文的"全局指数稳定速率"时必须用证明版的 $\epsilon\mu^2$，不能引用定理陈述版的 $\epsilon$**。

## 1. 元数据

- **论文标题**：Learning Stable and Passive Neural Differential Equations
- **作者**：Jing Cheng, Ruigang Wang, Ian R. Manchester
- **机构**：Australian Centre for Robotics (ACFR) + School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney
- **发表时间**：2024-04-19 (v1) / 2024-12-10 (v2)（arXiv:2404.12554 [eess.SY]）
- **发表渠道**：**IEEE 控制类会议格式**（`ieeeconf.cls`），但**正文自标 "Preprint."，未标注会议/期刊名** ⇒ **未评审**（arXiv v2 仍是 preprint 状态）
- **代码**：https://github.com/ruigangwang7/StableNODE（论文正文 §IV 给出；本仓**未实测**，未克隆）
- **本地 PDF**：[papers/2404.12554/2404.12554.pdf](../../papers/2404.12554/2404.12554.pdf) ｜ 文本 [papers/2404.12554/2404.12554.txt](../../papers/2404.12554/2404.12554.txt)（6 页，25923 字符）
- **本仓复算脚本**：`/tmp/lnn_verify_final_2026-11-25.py`（§6.1 全部数字由该脚本实跑产出）
- **关联概念**：连续时间机制层 / 稳定性证书 / 端口哈密顿系统 / 双 Lipschitz 网络 / Polyak–Łojasiewicz 条件 / 结构先验

## 2. 核心问题

学一个向量场 $f:\mathbb{R}^n\to\mathbb{R}^n$ 去拟合数据，同时希望对应的 ODE $\dot x=f(x)$ **稳定**。这条要求在非线性系统上并不平凡，因为"稳定"有很多互不等价的层级（Lyapunov 稳定 / 指数稳定 / 收缩 / 无源），而每一层对应的**保证强度**与**表达能力代价**都不同。

已有路线各有硬伤：(a) 收缩类（Revay/Wang/Manchester 的 REN、Revay et al. 的 contracting RNN、Martinelli et al. 的 dissipative/contracting Neural ODE）给出的是**直接（无约束且满射）参数化**，训练容易，但**收缩性被绑定在"与状态无关的二次度量"上**，灵活性受限；(b) 神经 Lyapunov 函数类（Richards 2018 / Kolter & Manek 2019 / Dawson 2023 综述）依赖一个**学出来的 $V$**，而 $V$ 的正定性、强制性、以及"上二次界"很难同时保证 —— Kolter & Manek 的 ICNN 方案只能保证**下**二次界，作者自己写明*"it is still not clear how to enforce the quadratic upper bound"*，且当平衡点已知时无法强制 $V(x^\star)=0$，**结果动力学可能收敛到错误的平衡点**（这正是本文 Fig.5 复现出的 SD-ICNN 失败模式）。

论文把问题重新表述为三个可分开的子问题：**(i) 怎样构造一个同时具备二次上下界、且强制性的神经 Lyapunov 函数 $H$？**（答案：bi-Lipschitz 网络的平方范数）**(ii) 怎样在没有损失惩罚的情况下，让向量场自动成为 $H$ 的下降方向？**（答案：$f=[J-R]\nabla H$，$J$ 反对称、$R$ 半正定 ⇒ $\dot H=-\nabla^\top H R\nabla H\le 0$ **是代数恒等式，不是训练目标**）**(iii) 怎样把这种"耗散"扩展成"无源"（port-Hamiltonian）以便用于控制器学习？**

## 3. 方法论与核心思路

### 3.1 总体架构

```
bi-Lipschitz 网络 g_θ  ──►  H(x) = 0.5|g_θ(x)|²  ──►  ∇H(x)
                                                      │
                            ┌─────────────────────────┴──────────┐
                            ▼                                    ▼
              J(x) = S(x) − Sᵀ(x)   (反对称, 神经网络)   R(x) = L(x)ᵀL(x) + εI  (半正定)
                            └─────────────────────────┬──────────┘
                                                      ▼
                                    ẋ = [J(x) − R(x)] ∇H(x)      ← Eq.(11)
                                                      │
                          ┌───────────────────────────┴─────────────┐
                          ▼                                         ▼
              端口哈密顿扩展 (Eq.17):                        耗散性恒等式 (Eq.12):
              ẋ = [J−R]∇H + B(x)u,  y = B(x)ᵀ∇H           Ĥ = −∇ᵀHR∇H ≤ 0
```

### 3.2 模块一：bi-Lipschitz 网络作为 Lyapunov 函数的构造子

论文采用 Wang & Manchester (ICML 2024) 的 bi-Lipschitz 网络作为 $g$。若存在 $\nu\ge\mu>0$ 使

$$\mu|x-x'|\le |g(x)-g(x')|\le \nu|x-x'|,\qquad \forall x,x'\in\mathbb{R}^n \tag{Eq.6}$$

则 $g$ 可逆，且 $H(x)=0.5|g(x)|^2+c$ 满足 Polyak–Łojasiewicz 条件

$$\|\nabla_x H\|^2 \ge 2m\,|H(x)-H^\star|,\qquad H^\star=\min_y H(y),\quad m=\mu^2 \tag{Eq.7}$$

关键的三条性质（论文 §II-B）：

1. $H$ 可以非凸，但**只有一个极小点** $x^\star=g^{-1}(0)$，且 $\nabla H(x)\neq 0\ \forall x\neq x^\star$；
2. $H$ 被**二次函数上下夹住**：$\frac12\mu^2|x-x^\star|^2\le H(x)\le\frac12\nu^2|x-x^\star|^2$（**这是全篇的枢纽**）；
3. 水平集 $\{x: H(x)\le\alpha\}$ 与单位球同胚。

网络的实际形式（Eq.15）：$z_k=\sigma(W_k z_{k-1}+U_kx+b_k)$，$y=\mu x+\sum_k Y_k z_k+b_y$，并存在一个参数化映射 $M:\phi\mapsto\theta$ 使得**对任意** $\phi$ 生成的 $g_{M(\phi)}$ 都对给定的 $(\mu,\nu)$ 是 bi-Lipschitz 的 ⇒ 可直接用 SGD 训练。

### 3.3 模块二：Hamiltonian 型向量场 —— 稳定性来自代数恒等式

$$f(x) = [J(x)-R(x)]\nabla H(x),\qquad J(x)=-J^\top(x),\quad R(x)=R^\top(x) \tag{Eq.11}$$

其中 $J$（互联矩阵）与 $R$（阻尼矩阵）由神经网络参数化：

$$J(x)=S(x)-S^\top(x),\qquad R(x)=L(x)^\top L(x)+\epsilon I \tag{Eq.16}$$

**注意 $R$ 的参数化方式**：$L^\top L$ 保证半正定，$+\epsilon I$ 保证**一致正定**（这是指数稳定的前提）。⇒ **"耗散性由结构保证"不是一句口号，而是 $\nabla^\top H R\nabla H\ge\epsilon\|\nabla H\|^2$ 这个逐点的代数事实。**

Theorem 1（论文陈述）：
- (a) 若 $R(x)\succeq 0\ \forall x$，则在 $x^\star$ 处 Lyapunov 稳定，$\delta(\epsilon)=\epsilon\mu/\nu$；
- (b) 若存在 $\epsilon>0$ 使 $R(x)\succeq\epsilon I\ \forall x$，则**全局指数稳定**，$\kappa=\nu/\mu$，**$\lambda=\epsilon$**。

🔴 **(b) 的 $\lambda=\epsilon$ 与论文自身证明不一致** —— 见 §6.1。

### 3.4 模块三：端口哈密顿扩展（无源性）

$$\dot x=[J(x)-R(x)]\nabla H(x)+B(x)u,\qquad y=B(x)^\top\nabla H(x) \tag{Eq.17}$$

取 $H$ 为储能函数，耗散不等式为

$$\dot H-u^\top y=-\nabla^\top H R\nabla H\le 0 \tag{Eq.18}$$

（$B$ 项两两抵消）⇒ **无源性由结构保证**，与数据拟合无关。

### 3.5 与 LTC / CfC / NCP 的关系

- **与 LTC（Hasani 2021）**：LTC 的向量场 $f(x,I,t;\theta)$ 是**一般非线性**的，没有任何结构 ⇒ LTC **无法**给出任何稳定性保证。SHND 把 $f$ 换成 $[J-R]\nabla H$，是**结构受限**的子族。
- **与 CfC（Hasani 2022）**：🔴 **CfC 不是 ODE 流** —— 它的闭式解 $\sigma(-f\cdot t)\odot g + [1-\sigma(-f\cdot t)]\odot x$ 是对 LTC ODE 的**闭式近似**，不是任何向量场的精确积分。⇒ **把 SHND 的 $J,R,H$ 套到 CfC 的 cell 上，Theorem 1 的保证完全不成立**（它保证的是"连续积分 $\dot x=f(x)$ 沿下降方向走"，而 CfC 的更新是 $h_{t+1}=h_t+\Delta$ 形式的一步映射）。这与本仓 11-22/11-24 已确立的纪律一致（"CfC 的更新不是 ODE 流 ⇒ 投影只能保证状态落在流形上，而非沿流形积分"）。
- **与本仓 `ltc.py`**：本仓 `ltc.py` **是** ODE（走 `odeint`）⇒ **SHND 结构在 LTC 路线上是可移植的**，在 CfC 路线上不是。
- **与 NCP（Lechner 2020）**：NCP 是固定时间常数 + 稀疏离散接线，**没有稳定性内容**；SHND 与之无关。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq.(6) | $\mu\|x-x'\|\le\|g(x)-g(x')\|\le\nu\|x-x'\|$ | bi-Lipschitz：保证 $g$ 可逆、$H$ 强制、二次上下界 |
| Eq.(7) | $\|\nabla_xH\|^2\ge 2m|H-H^\star|,\ m=\mu^2$ | PL 条件，Theorem 1 的数学基础 |
| Eq.(10) | $\frac12\mu^2\|x-x^\star\|^2\le H(x)\le\frac12\nu^2\|x-x^\star\|^2$ | **二次上下界**（枢纽）：上界给指数稳定，下界给稳定半径 |
| Eq.(11) | $\dot x=[J(x)-R(x)]\nabla H(x)$ | SHND 主方程：$J$ 反对称、$R$ 半正定 |
| Eq.(12) | $\dot H=-\nabla^\top H(x)R(x)\nabla H(x)\le 0$ | **耗散恒等式**（保证来自结构，不是训练目标） |
| Eq.(13) | $\dot H\le-\epsilon\|\nabla H\|^2\le-2\epsilon\mu^2(H-H^\star)=-2\epsilon\mu^2H$ | 证明中的指数衰减推导（**速率 $\epsilon\mu^2$ 的来源**） |
| Eq.(16) | $J(x)=S(x)-S^\top(x),\quad R(x)=L(x)^\top L(x)+\epsilon I$ | 结构参数化：保证反对称 / 一致正定 |
| Eq.(17) | $\dot x=[J-R]\nabla H+B(x)u,\quad y=B(x)^\top\nabla H$ | 端口哈密顿扩展 |
| Eq.(18) | $\dot H-u^\top y=-\nabla^\top H R\nabla H\le 0$ | 无源性保证 |

## 5. 关键成果与贡献

### 5.1 主结果（本仓逐条复算，见 §6.1）

- **Theorem 1(a) 稳定**：$\delta(\epsilon)=\epsilon\mu/\nu$，由 Eq.(12) + Eq.(10) 直接推出，**本仓复核通过**。
- **Theorem 1(b) 指数稳定**：论文**陈述** $\lambda=\epsilon$，**其自身证明**（Eq.13–14）给出 $\lambda=\epsilon\mu^2$。**本仓数值证伪了陈述版**（§6.1）。
- **Theorem 2 无源性**：Eq.(18) 是逐点代数恒等式，**本仓复核通过**，无任何可调项。

### 5.2 实验结果（作者自报，本仓未复现）

- **数据集**：阻尼双摆（damped double pendulum），状态 $x=[\theta_1,\theta_2,\dot\theta_1,\dot\theta_2]^\top$，平衡点 $x^\star=0$（因每个关节有阻尼）。训练 2000 个样本（$|\theta_i|\le\pi$, $|\dot\theta_i|\le\pi$/s），测试 500 个样本；损失 = 均方 $\ell_2$（Eq.19）。
- **对比基线**：SD-MLP（Kolter & Manek 的投影式 Eq.20 + MLP 型 $V=|g(x)|^2$）、SD-ICNN（Eq.20 + ICNN 型 $V$）。
- **模型规模**：SHND 用 4-32-32-4 的 bi-Lipschitz 网络造 $H$（$(\mu,\nu)=(0.1,2)$），4-90-90-32 的 MLP 出 $L,S$ 后切成两个 $4\times4$ 矩阵，$\epsilon=0.01$；参数量 **≈15K**，与两个基线相当。
- **训练**：Adam，1000 epochs，batch 200，lr 从 0.01 按 cosine 衰减到 0。
- **仿真误差**：RK4，步长 0.01s，100 个初值（$|\theta_i|\le\pi/2$, $\dot\theta_i=0$）。Fig.2 显示 **SHND 的仿真误差随时间收敛到 0**，SD-ICNN 有**稳态误差**（收敛到别的平衡点），SD-MLP **持续振荡**。
- **超参消融**：Fig.4 显示 $\nu/\mu$ 有正则化作用（$\mu=0.1$ 固定，扫 $\nu$）；Fig.3 显示训练数据量增加时 SHND 性能持续改善而基线有波动。

### 5.3 贡献清单

1. **一个 $\mathcal{O}(1)$ 概念的稳定性保证范式**：把"学一个稳定的 $V$ + 用投影惩罚"换成"**参数化向量场使 $V$ 自动成为 Lyapunov 函数**"，保证不再依赖优化过程。
2. **可插拔的无源性扩展**：Eq.(17)–(18) 表明同一套 $J,R,H$ 结构直接给出端口哈密顿形式 ⇒ 面向"为未知但无源的系统学稳定控制器"的应用路径（作者 §V 列为未来工作）。
3. **对既有神经 Lyapunov 路线的能力补全**：明确指出 Kolter & Manek 的 ICNN 型 $V$ **无法保证二次上界**，这正是它们**无法给出指数稳定**的根因 —— 论文用"为什么别人做不到"的方式论证了自己为什么能做到。

## 6. 局限性与未来展望

### 6.1 🔴 本仓复算发现：Theorem 1(b) 的印刷速率 $\lambda=\epsilon$ 是错的（应为 $\epsilon\mu^2$）

**论文内部矛盾**：定理陈述（`hnn.tex:13`）写 *"holds with $\kappa=\nu/\mu$ and $\lambda=\epsilon$"*，而**其自身证明**（`hnn.tex:42`）写 *"where $\lambda=\epsilon\mu^2$, i.e., (11) is globally exponentially stable with rate $\epsilon\mu^2$"*。二者相差 $\mu^{-2}$ 倍。

**本仓数值证伪（可容许实例）**：取论文自身的超参 $\mu=0.1,\nu=2.0,\epsilon=0.01$，令 $g(x)=\mathrm{diag}(\mu,\nu)x$（**恰为 bi-Lipschitz 且上下界取等**），$J=0$，$R=\epsilon I$。这是完全满足 Theorem 1 假设的实例，且 Eq.(7) 的 PL 不等式**取等号**（最紧情形）。

```
  empirical decay rate over 600s = 1.000000e-04
  proof rate   eps*mu^2 = 1.000000e-04   <-- 与实测一致
  stated rate  eps       = 1.000000e-02   <-- 快 100 倍

  t= 300.0  |x(t)|=0.970446  stated-bound(kappa=nu/mu, lam=eps)=0.995741   -> holds
  t= 400.0  |x(t)|=0.960789  stated-bound(kappa=nu/mu, lam=eps)=0.366313   -> VIOLATED
  t= 600.0  |x(t)|=0.941765  stated-bound(kappa=nu/mu, lam=eps)=0.049575   -> VIOLATED

  analytic crossing time t* = ln(kappa)/(eps(1-mu^2)) = 302.6s
  observed first violation = 400.0s   (一致)
```

⇒ **定理的印刷版（$\lambda=\epsilon$、$\kappa=\nu/\mu$）对该实例是假的**：轨迹实际以 $\epsilon\mu^2$ 衰减，故所声称的包络最终必然被违反（解析交叉时刻 $t^*=302.6\text{s}$，数值观测首次违反在 400s，两者一致）。**证明版的 $\lambda=\epsilon\mu^2$ 是精确的。**

**判定**：这是**陈述—证明不一致的排版错误**，不是"定理被证伪" —— 定理的**内容（存在指数稳定）成立**，只是**速率常数写错了**。注意印刷版的界之所以在 $t<302\text{s}$ 内"看起来成立"，完全靠 $\kappa=\nu/\mu=20$ 这个**极松的前因子**撑着。

**🔴 实际后果（引用纪律，必须带）**：用论文自身的超参，指数稳定的**速率**因此差了 **100 倍**：

| | $\lambda$ | 衰减到 50% 所需 | 衰减到 1% 所需 |
|---|---|---|---|
| 印刷版 $\lambda=\epsilon$ | $1.0\times10^{-2}$ | 369 s | 760 s |
| **证明版 $\lambda=\epsilon\mu^2$** | $1.0\times10^{-4}$ | 36889 s (10.2 h) | 76009 s (21.1 h) |

⇒ **用证明版的速率，论文自己的 $\epsilon=0.01$ 在 60 秒时间尺度上给不出任何有用的 settling 保证**（要 1% 衰减需 $\epsilon\ge 12.67$）。论文 Fig.2 中"仿真误差收敛到 0"是**双摆短周期动力学本身的性质**（周期数秒量级），**不是指数稳定证书的产物** ⇒ **不可把 Fig.2 当作"证书起作用了"的证据**。

**本仓的复算脚本**：`/tmp/lnn_verify_final_2026-11-25.py` §(1)（含 600 秒 RK4 轨迹、解析交叉时刻、逐点界检查）。

### 6.2 🔴 Eq.(14) 的方向笔误

Eq.(14) 印作

$$|x(t)-x^\star|\le\frac{\sqrt{2H(x(t))}}{\nu}\le\frac{\sqrt{2e^{-2\lambda t}H(x(0))}}{\nu}\le\frac{\nu}{\mu}e^{-\lambda t}|x(0)-x^\star|$$

由 Eq.(10) $\frac12\mu^2|x-x^\star|^2\le H\le\frac12\nu^2|x-x^\star|^2$ 可知：**上界应是 $\sqrt{2H}/\mu$，下界才是 $\sqrt{2H}/\nu$**。印刷版把 $\mu$ 写成了 $\nu$ ⇒ 第一个不等号在 $H>0$ 时**为假**。本仓核对：把 $\nu$ 换成 $\mu$ 后整条链严格成立，且末行 $\frac{\nu}{\mu}e^{-\lambda t}|x(0)-x^\star|$ 与之一致 ⇒ **确为笔误**（`hnn.tex:38`）。

### 6.3 作者自陈局限

- §V 结论只说 *"Empirical results on a vector field fitting problem demonstrate its effectiveness"* —— **只有一个 toy 系统（阻尼双摆）、一个任务（向量场拟合）、一个维度（4 维）**。
- **完全没做"连续时间 vs 离散时间"的对照** ⇒ **不能被引用为"连续时间机制更优"的证据**。
- **没有做消融**：论文没有单独消融 $J$（反对称项）与 $R$（耗散项）各自的贡献；也没有消融 bi-Lipschitz 网络换成普通 MLP 的影响（只与 SD-MLP/SD-ICNN 这两类**不同范式**的稳定模型比，而那两类的 $V$ 本身就不满足二次界）。
- **没有稳定性证书的数值验证表**：论文**没有报告** $\lambda_{\max}$、$\epsilon$ 的实际值、$\mu,\nu$ 的实测值、也没有报"违反 $L_fV<0$ 的样本比例" ⇒ **"保证"全是理论陈述，没有实测审计**。
- §V 明说未来工作是"从时间序列数据学稳定模型"与"无源控制器" —— **即本文尚未做时间序列**。
- **未评审**（preprint），代码仓库本仓未实测。

### 6.4 隐含局限（本仓视角）

- **表达力代价未量化**：$f=[J-R]\nabla H$ 是 $H$ 的**下降方向**，其可达集合受 $H$ 的水平集形状限制（论文自己说水平集**可以非凸**）⇒ 能拟合的向量场集合比一般非线性小，但论文**没有给任何"表达力 vs 保证"的定量权衡**（例如在同一个数据集上比较 SHND 与无约束 MLP 的可达到的最低 loss）。这是本仓最关心的问题，却**完全缺失**。
- **依赖上游工作**：bi-Lipschitz 网络与 PLNet 全部来自 [18]（Wang & Manchester, ICML 2024），本文的**原创性集中在"如何把 PLNet 拼成 Hamiltonian 形式"**这一层。
- **假设很强**：Theorem 1 要求 $R(x)\succeq\epsilon I$ **全局成立**。若 $L(x)$ 训练后接近奇异，$\epsilon I$ 就是唯一的保底 —— 而 $\epsilon$ 是**用户手调的常数**，论文不讨论如何选 $\epsilon$（只给了固定值 0.01）。
- **平衡点**：若 $x^\star$ 未知，$x^\star=g^{-1}(0)$ 由 $g$ 决定 ⇒ **平衡点不是数据决定的，是结构决定的**；若 $x^\star$ 已知则用 $H(x)=0.5|g(x)-g(x^\star)|^2$。论文对这两种情形都给了方案，但**未讨论"学出来的 $x^\star$ 与真实平衡点不一致"时的行为**（那会导致收敛到错误平衡点 —— 正是它批评 SD-ICNN 的毛病）。

### 6.5 未来方向

- 从时间序列数据学稳定模型（作者 §V）。
- 无源/端口哈密顿控制器学习（作者 §V、Remark 3）。

## 7. 对本仓的意义

**(1) 一条"保证升级路径"（本仓至今没有）**：本仓 `lnn/core/lyapunov_stable_cfc.py` 的 `lyapunov_decay_loss` 是 `relu(V(h_next) - (1-alpha)V(h) + margin)` —— 一个**惩罚项**，其"零值"只意味着**在采样到的点上没被违反**（11-24 已定级为 **B 级：采样残差经验正则**）。SHND 的保证来自 $f$ 的**参数化形式**，不依赖训练 ⇒ 若在 **LTC（ODE）路线**上实现 `HamiltonianLTCCell`（$f_\theta(x)=[J(x)-R(x)]\nabla H(x)$，$J=S-S^\top$，$R=L^\top L+\epsilon I$，$H=0.5|g(x)|^2$，$g$ 为 bi-Lipschitz），则稳定性从 **B 级升到 A 级（构造性、推理时恒成立）**。⚠️ **CfC 路线不适用**（CfC 不是 ODE 流，见 §3.5）—— 这与 11-22 提出的 `PortHamiltonianCfCCell` 候选**方向一致但必须改名/降级**（11-22 已警告过"CfC 的投影只能保证状态落在流形上"）。

**(2) 两条新消融轴**：
① 🔴 **"保证的实现方式"轴**（承接 11-24 的"约束的实现方式"轴，但更上位）—— 在同一 toy 数据集上比较 (i) 无约束 CfC/LTC、(ii) **软惩罚**（本仓现状：`lyapunov_decay_loss`）、(iii) **硬投影**（11-24 的 PNDE 路线）、(iv) **结构参数化**（SHND 路线）⇒ 四者报同一个量：$\|V(h_{t+1})-V(h_t)\|$ 的 max/95 分位 + **违反 $L_fV<0$ 的时间点比例**（一次前向 + `torch.func.jvp`）⇒ 首次能回答"**本仓的稳定性保证，在'保证级别'这个维度上到底值多少**"。
② 🔴 **"保证代价"轴** —— SHND 的 $f$ 受 $H$ 水平集限制 ⇒ **表达力受限**。必须报：结构化模型的**可达到最低训练 loss** vs 无约束模型 ⇒ 若结构化版本 loss 明显更高，则"保证"是**用精度换来的**，这个交换比**必须显式报出**（本仓至今所有 bench 只报精度）。

**(3) 一条 bench 元数据纪律升级**：新增 **"证书速率 / 时间常数"列** —— SHND 的复算已经证明：**证书里写的速率常数和实际衰减速率可以差 100 倍**（$\epsilon$ vs $\epsilon\mu^2$）⇒ **任何稳定性报告必须同时给出"声称的速率"与"实测的衰减速率"两列**，且必须报 $\mu,\nu,\epsilon$ 的**实测值**（本仓 `lyapunov_stable_cfc.py` 报的 `lambda_min(P)` 属于另一类量，不能替代）。

**(4) 一条引用纪律（防污染证据链）**：本篇标题含 "Neural Differential Equations"，**极易被当成"连续时间有效"的证据** —— 但它**完全不做"连续时间 vs 离散"的对照**，**所有结论都关于"怎样构造一个稳定的连续时间模型"**，与"连续时间比离散好"无关 ⇒ 引用时必须显式标注这一点（延续 11-23/11-24 已确立的纪律）。

**Verdict**：
- **"结构化参数化可以把稳定性从 B 级经验正则升级到 A 级构造性保证" → TARGET-POSITIVE**（方法论级，本仓必须采纳；但仅限 **ODE 路线**，CfC 不适用）
- **"SHND 的全局指数稳定速率是 $\epsilon$" → TARGET-NEGATIVE-WITH-NUANCE**（🔴 论文自身错误；正确值是 $\epsilon\mu^2$，本仓已数值证伪印刷版 ⇒ 引用必须带这个折扣）
- **"SHND 证明了学稳定向量场比学稳定 $V$ + 投影更有效" → TARGET-DEPENDENT-WITH-NUANCE**（只在 4 维双摆的向量场拟合上验证过；**表达力代价完全未量化**，未评审，本仓未复现）
