---
title: Universal Approximation Property of Neural Ordinary Differential Equations
date: 2026-11
tags: [LNN, Continuous-Time, Neural-ODE, Universal-Approximation, Diffeomorphism, Invertible-Neural-Network, Theoretical-Foundation]
---

# 研读报告：Neural ODE 的 sup-通用逼近 — 从 $L_p$ 逼近升级到全输入域一致逼近

## 1. 元数据

- **论文标题**：Universal Approximation Property of Neural Ordinary Differential Equations
- **作者**：Takeshi Teshima (The University of Tokyo, RIKEN), Koichi Tojo (RIKEN), Masahiro Ikeda (RIKEN), Isao Ishikawa (Ehime University, RIKEN), Kenta Oono (The University of Tokyo)
- **发表时间**：2020-12-04 (v1)，NeurIPS 2020 Workshop on Differential Geometry meets Deep Learning
- **来源**：arXiv:2012.02414v1 [cs.LG]
- **代码**：N/A（纯理论工作，无代码发布）
- **本地 PDF**：[papers/2012.02414/2012.02414.pdf](../../papers/2012.02414/2012.02414.pdf) ／ 文本 [papers/2012.02414/2012.02414.txt](../../papers/2012.02414/2012.02414.txt)
- **关联概念**：Neural ODE / 可逆神经网络 (INN) / 微分同胚 / 通用逼近定理 / flow endpoint / 黎曼几何结构定理 / LTC / CfC

## 2. 核心问题

Neural ODE 作为**可逆神经网络（INN）**架构有两个突出优点：**自由形式的 Jacobian**（不限制 Jacobian 结构即可求逆）与**可用的无偏 Jacobian 行列式估计器**——这使它成为连续归一化流（CNF）的标准实现。

但它的**表示能力（representation power）此前只被部分揭开**：Li et al. 已证明 NODEs 在一定条件下是 $L_p$-通用逼近器（$L^p$-universal approximator），逼近目标是"全部连续映射"。论文指出这个结论**在实践中可能不够用**：

> "the $L_p$-universality may fail to guarantee an approximation for the entire input domain as it may still hold even if the approximator largely differs from the target function on a small region of the input space."

即：$L_p$ 范数允许在一个**小区域**上误差很大，只要总体积分占优。**对建模而言，"大部分输入域上准确"不等于"可用"**——一个在边界/少数区域偏差巨大的模型，在分类与密度估计里可能直接失效。

论文把问题重新表述为：**能否把保证从 $L_p$ 升级到 $\sup$（逐点一致）？如果能，覆盖的映射类是什么？**

答案的形状：**能——但只能覆盖一个受限的映射类**（$\mathcal{D}^2$，$C^2$ 微分同胚），而不是全部连续映射。

## 3. 方法论与核心思路

### 3.1 总体论证路线（"流终点分解"路线）

论证的核心是一个**三层结构**：

```
任意 f ∈ 𝒟²_c（紧支撑 C² 微分同胚）
        │  结构定理 (Herman–Thurston / Epstein / Mather)
        ▼
f = ψ_k ∘ ⋯ ∘ ψ_1   （有限个 flow endpoint 的复合）
        │  每个 flow endpoint 是一个"恒等映射的光滑变换"
        ▼
每个 ψ_i ≈ IVP[h_i]（NODE）——用 H 中的 h_i ∈ Lip(ℝ^d) 逼近
        ▼
INNH-NODE 对 𝒟² 是 sup-通用逼近器
```

关键在于**中间那步的分解定理**——它来自微分几何（不是机器学习），这是本篇方法论上的独特之处，也是它与 Li et al. 的 $L_p$ 结果在技术路径上的分野。

### 3.2 预备：模型类与逼近范数

**NODE 的基本对象**（Fact 1）：对 $f\in\mathrm{Lip}(\mathbb{R}^d)$，自洽 ODE

$$
\dot z(t) = f(z(t)),\quad z(0)=x
$$

有全局存在唯一解，定义为

$$
\mathrm{IVP}[f](x,t) := z(t)
$$

**流终点（flow endpoints）**（Def. 2）：$\Psi(F) := \{\mathrm{IVP}[f](\cdot,1)\mid f\in F\}$ ——即"走 1 个单位时间"的 NODE 变换。

**INNH-NODE**（Def. 3）：设 $\mathrm{Aff}$ 为所有可逆仿射映射的群，

$$
\mathrm{INNH}\text{-}\mathrm{NODE} := \{\, W\circ\psi_k\circ\cdots\circ\psi_1 \;\mid\; \psi_1,\dots,\psi_k\in\Psi(H),\ W\in\mathrm{Aff},\ k\in\mathbb{N}\,\}
$$

即"若干 NODE 复合后再接一个仿射层"。$H\subset\mathrm{Lip}(\mathbb{R}^d)$ 是**向量场族**（假设网络实现落在 $\mathrm{Lip}$ 内）。

**两种通用性**（Def. 4）：对映射 $f:K\to\mathbb{R}^n$，

$$
\|f\|_{\sup,K} := \sup_{x\in K}\|f(x)\|,\qquad
\|f\|_{p,K} := \Big(\int_K \|f(x)\|^p\,dx\Big)^{1/p}
$$

- **sup-通用性**：$\forall f,\varepsilon>0,\ \exists g\in M:\ \|f-g\|_{\sup,K}<\varepsilon$（逐点一致）
- **$L_p$-通用性**：同上但换成 $\|f-g\|_{p,K}$（积分意义）

**目标类**（Def. 5）：$\mathcal{D}^2$ = 所有 $C^2$-微分同胚 $f:U_f\to\mathrm{Im}(f)\subset\mathbb{R}^d$，其中 $U_f$ 开、与 $\mathbb{R}^d$ $C^2$-微分同胚（**允许随 $f$ 变化**）。论文强调这是一个"相当大"的类：包含整个 $\mathbb{R}^d$ 上的任意 $C^2$-微分同胚、开凸集上的、以及更一般的星形开集上的。

### 3.3 主定理（Theorem 1）

> **Theorem 1 (Universality of NODEs)**. Assume $H\subset\mathrm{Lip}(\mathbb{R}^d)$ is a sup-universal approximator for $\mathrm{Lip}(\mathbb{R}^d)$. Then, $\mathrm{INNH}$-NODE is a **sup-universal approximator for $\mathcal{D}^2$**.

即：**只要向量场族 $H$ 本身是 sup-通用逼近器，NODE 复合就对 $C^2$-微分同胚类是 sup-通用逼近器。**

论文列举的可用 $H$：**有限权重 + Lipschitz 连续激活**的多层感知机（ReLU 等），以及 Lipschitz Networks。

### 3.4 证明要点

论文的证明策略（自述）与 Teshima et al. 的 Theorem 1 类似但有"major modification"：

1. **降维目标**：用 Fact 2 把逼近目标从 $\mathcal{D}^2$ 降到**紧支撑**微分同胚类 $\mathrm{Diff}^2_c$（Fact 2 = Lemma 5 of Teshima et al. [9]）。
2. **分解**：由 Herman–Thurston [10] / Epstein [11] / Mather [12,13] 的结构定理，把每个 $f\in\mathrm{Diff}^2_c$ 表示为**有限个 flow endpoint 的复合**。
3. **逼近每个 flow endpoint**：每个 flow endpoint 可被 NODE 逼近（Lemma 2，依赖 Fact 3 = Proposition 6 of Teshima et al.，即"复合与逼近的相容性"）。
4. **拼接**：各因子用 $\sup$ 范数误差控制后复合。

**关键技术细节**：论文必须使用一个**与 Teshima et al. [9] 的 Corollary 2 不同的 flow endpoint 定义**（Def. 7），以便纳入足够的**光滑性**要求。附录中还给出了 Lemma 1（Modified Corollary 2 of Teshima et al.），证明被修改后的 flow endpoints 集合 $H^r$ 构成 $\mathrm{Diff}^c_r$ 的子群。

### 3.5 与既有结果的关系

论文自评三条相关工作：

| 结果 | 结论 | 与本文的关系 |
|---|---|---|
| Li et al. [5] Prop 3.8 | NODE + 终端层对**全部连续映射** $L_p$-通用 | **更大的目标类 + 更弱的范数**；本文是"更小类 + 更强范数" |
| Zhang et al. [14] Thm 1 | NODE **不是**通用逼近器（给出反例函数） | **不矛盾**：其目标类可含**不连续映射**，而 $\mathcal{D}^2$ 元素光滑可逆 |
| Zhang et al. [14] | **增广 NODE (ANODE)** 对 homeomorphism 是通用逼近器 | ANODE 的代价：除非训练到输出恒为 0 的补零维度，否则**破坏可逆性**；本文路线不引入这一代价 |

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 (Fact 1) | $z(0)=x,\ \dot z(t)=f(z(t)),\ t\in\mathbb{R}$ | Lipschitz 向场 ⇒ 全局存在唯一解（NODE 的基本对象） |
| Def. 1 | $\mathrm{IVP}[f](x,t):=z(t)$ | NODE 变换的定义 |
| Def. 2 | $\Psi(F):=\{\mathrm{IVP}[f](\cdot,1)\mid f\in F\}$ | **流终点**集合：走单位时间的 NODE |
| Def. 3 | $\mathrm{INNH}\text{-}\mathrm{NODE}:=\{W\circ\psi_k\circ\cdots\circ\psi_1\mid \psi_i\in\Psi(H), W\in\mathrm{Aff}\}$ | 模型类：NODE 复合 + 仿射尾 |
| Def. 4 | $\|f\|_{\sup,K}:=\sup_{x\in K}\|f(x)\|$；$\|f\|_{p,K}:=(\int_K\|f(x)\|^p dx)^{1/p}$ | 两种逼近范数 |
| Def. 5 | $\mathcal{D}^2$ := 全体 $C^2$-微分同胚 $f:U_f\to\mathrm{Im}(f)\subset\mathbb{R}^d$ | **目标类**（$U_f$ 开且 $C^2$-微分同胚于 $\mathbb{R}^d$） |
| **Thm 1** | $H$ 是 $\mathrm{Lip}(\mathbb{R}^d)$ 的 sup-通用逼近器 $\Rightarrow$ $\mathrm{INNH}$-NODE 是 $\mathcal{D}^2$ 的 sup-通用逼近器 | **主定理** |
| Lemma 1 | $H^r$ 构成 $\mathrm{Diff}^c_r$ 的子群（Modified Corollary 2） | 技术引理，保证复合结构 |
| Lemma 2 | 自治 ODE 流终点可被 NODE 逼近（假设 $H$ 为 sup-通用） | 分解后的逼近步 |

## 5. 关键成果与贡献

### 5.1 主结果

1. **把 NODE 的表示能力从 $L_p$ 升级到 $\sup$**：不再只保证"平均意义上接近"，而是保证**整个输入域上一致接近**。
2. **明确划定了"能保证什么"与"不能保证什么"**：能保证的是 $\mathcal{D}^2$（$C^2$ 微分同胚），**不能**保证覆盖全部连续映射（Zhang et al. 的反例依然有效）。
3. **给出可操作的条件**：$H$ 只需是"有限权重 + Lipschitz 激活"的 MLP —— **这正是绝大多数实践中的 NODE 配置**，所以该定理对工程实践的适用面很宽。
4. **不牺牲可逆性**：相对 ANODE 路线，INNH-NODE 保持"自由形式 Jacobian + 可用的行列式估计器"这一 CNF 关键性质。

### 5.2 贡献清单

- **方法论贡献**：把微分几何的**流终点分解结构定理**引入神经网络的表示能力分析——这是一条不同于"$L_p$ 积分估计"的技术路径。
- **边界刻画贡献**：用"更小的目标类换更强的范数"这一交换，明确了 NODE 表示能力讨论中的一条前沿线。
- **工程含义**：任何声称"ODE 家族是通用逼近器"的表述，都应注明是 $L_p$ 意义还是 $\sup$ 意义，以及目标类是否光滑可逆。

### 5.3 需要明确的性质边界

- 定理是**存在性**的：只保证"对任意 $\varepsilon>0$ 存在某个 NODE 复合"，**不给出层数/参数量的定量界**（论文把这一点明确列为未来工作）。
- 定理要求 $H$（向量场族）本身是 sup-通用逼近器 —— 这是**前提**而非结论。
- 目标类 $\mathcal{D}^2$ **不包含**一般连续映射，因此**不能**用它去论证 NODE 能拟合任意函数。

## 6. 局限性与未来展望

### 6.1 作者自陈局限 / 未来工作（论文第 5 节原文）

1. **开放问题**：sup-通用性是否能推广到**比 $\mathcal{D}^2$ 更大的映射类**？
2. **定量效率问题**：需要量化"逼近一个给定光滑度（如给定双 Lipschitz 常数）的微分同胚，需要**多少层 NODE**"。
3. **工作坊论文定位**：NeurIPS 2020 **Workshop** on Differential Geometry meets Deep Learning，非主会。

### 6.2 隐含局限（本仓视角）

1. **纯理论、无实验**：全文**没有一个数值实验**，没有对"$\sup$ 逼近在实践中值多少"的量化。对本仓而言，这意味着**不能直接引用它支撑任何性能论断**。
2. **"$\sup$ 逼近"是存在性而非可训练性**：定理保证存在某个 $k$ 层的复合，但**训练动态是否能找到它**，定理完全不管。对本仓的多变体比较（几十个 $\tau$ 变体）而言，这个 gap 很关键——**存在性与可训练性之间的鸿沟**。
3. **与 LTC/CfC 无直接关系**：本篇研究的是**自洽 ODE（autonomous）**作为可逆映射的表达能力，**完全没有涉及**输入依赖的向量场（$f(x,I;\theta)$）。因此**不能**用它论证"液态网络（输入依赖 ODE）的逼近能力"——这是一个**常见的误引**，本仓必须显式防住。
4. **$C^2$ 光滑性假设在实践中并非免费**：真实的 CfC/LTC 向量场（sigmoid/tanh 组合）确实光滑，但**含 ReLU 的复合可能只在弱意义下 Lipschitz**；论文假设 $H\subset\mathrm{Lip}$，**这依赖实现细节**（ReLU 是 1-Lipschitz，故可行，但复合的 Lipschitz 常数可能爆炸 → 逼近所需的 $k$ 可能极大）。
5. **与 2012.02414 同期的 2005.08926（Neural CDE）互补而非互替**：CDE 处理"数据驱动"，本文处理"自洽 + 可逆 + 逼近强度"，二者是正交的两条线。

### 6.3 未来方向

- 推广 $\sup$-通用性到更大的映射类（作者原话）。
- 定量层数界（作者原话）。
- 本仓可推进：**把本仓 `lnn/core/cfc.py` 的闭式解视作一个 flow endpoint**，检验其"表达力是否已饱和"——若 CfC 的闭式解 + 多层复合已经落在 $\mathcal{D}^2$ 内且实践中未饱和，则"再加强表达力"这条路的边际收益很低。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的什么变体**：**不直接产出变体**（纯理论），但给出一条**判据**：本仓若要新增"更强表达力"的变体（例如把 $f$ 从标量换成矩阵、把 $\sigma$ 换成可学习的有理函数、或引入 ANODE 式补零维度），**必须先回答"你落在哪个映射类、哪个范数下"**。实用价值在于**堵住一条错误路线**：本仓历史上出现过"加维度/加参数以增强表达力"的直觉式改动；本文提供的判据是——**先确认目标类是否在 $\mathcal{D}^2$ 内、范数是否已是 $\sup$**，否则增加的维度可能只是逼近了一个更小的类。

**可嫁接到哪个 bench 脚本**：不适用（无实验）。但可作为**理论背景章节**写入 `docs/research/`，为"表达力 vs 稳定性"的取舍提供引用依据。

**提供哪个新 ablation axis**：一个**理论性**的轴——**"逼近强度轴"**：
$$\text{轴：}\ \{\ L_p\text{ 逼近},\ \sup\text{ 逼近},\ \text{可训练性}\ \}\ \text{三者的分离}$
本仓目前的比较都是**经验性 MSE/准确率**，从未区分"表达力不足"与"优化不到"。本篇给了一个有用的区分工具：若一个变体在 $\sup$ 意义下**理论可逼近**（例如落在 $\mathcal{D}^2$ 内且向量场 Lipschitz）却仍然表现差，则**问题在优化/数值，不在架构容量** —— 这是一个能直接指导实验设计的判据。

**Verdict**：
- **TARGET-POSITIVE（对"ODE 家族的表达力有严格理论支撑"这一命题）**：在 $\mathcal{D}^2$（$C^2$ 微分同胚）+ $\sup$ 范数下，NODE 是 sup-通用逼近器，**且条件（MLP + Lipschitz 激活）正是工程常态**。这是对"连续时间建模不是一个随手包装的 trick"的一个**正面理论证据**。
- **TARGET-NEGATIVE-WITH-NUANCE（对"因此 LNN 更好"）**：本篇**不能**支撑任何"LNN 更好"的结论。理由有三：(a) 纯理论、无实验；(b) 存在性而非可训练性，**没有定量层数界**；(c) 目标类 $\mathcal{D}^2$ 不覆盖一般连续映射，且 **Zhang et al. 的"NODE 非通用逼近器"反例依然有效**。引用时**必须**成对出现（本文 vs Zhang et al. [14]），单引本文会被误读为"ODE 什么都能拟合"。

**引用纪律**：① 引用本文时**必须**同时声明目标是 $\mathcal{D}^2$（$C^2$ 微分同胚）而非一般连续映射，且范数是 $\sup$ 而非 $L_p$；② **不得**用本文论证"液态网络（LTC/CfC，输入依赖 ODE）的逼近能力"——本文只覆盖**自洽 ODE**；③ **不得**用本文论证任何性能结论（无实验）。
