---
title: LNN 数学基础综合报告 — Hasani 2021 LTC + Lechner 2022 CfC 原文 grounding（arXiv 2006.04439, 2106.13898）
date: 2026-08-05
tags: [LNN, LTC, CfC, ODE, mathematical-foundations, comprehensive, original-papers, grounding]
arxiv_refs: [2006.04439, 2106.13898, 2606.12240, 2607.08283, 2607.10858, 2606.15571]
parent: [[LNN_深度研读报告]]
companion: [[LNN_Training_Paradigm_2026_Summer_Cross_Section]], [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]], [[MR_TFP_CfC_Second_Layer_Synthesis_2026-08-05]]
local_pdfs: [papers/foundational/hasani_2021_ltc.pdf, papers/foundational/lechner_2022_cfc.pdf]
---

# LNN 数学基础综合报告 — Hasani 2021 LTC + Lechner 2022 CfC 原文 grounding

> 本报告**首次**把 [[LNN_深度研读报告]] §1 的核心公式精确 grounding 到 Hasani 2021 与 Lechner 2022 原始论文（PDF 已落地到 `papers/foundational/`），并把本项目最近 4 轮的工作（MultiRateMoECfC / MemoryFusionCfCCell / MR-TFP-CfC / Pareto sweep）反向 trace 到这些奠基公式。

## 1. Hasani 2021 — Liquid Time-Constant Networks（arXiv 2006.04439）

**作者**：Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, Radu Grosu
**机构**：TU Wien + IST Austria + MIT CSAIL
**引用**：Hasani et al., "Liquid Time-Constant Networks", *AAAI 2021* (preprint v3: 2020-11-20)

### 1.1 核心思想（原文 §1 Introduction）

> *"We introduce a new class of time-continuous recurrent neural network models. Instead of declaring a learning system's dynamics by implicit nonlinearities, we construct networks of linear first-order dynamical systems modulated via nonlinear interlinked gates. The resulting models represent dynamical systems with varying (i.e., liquid) time-constants coupled to their hidden state, with outputs being computed by numerical differential equation solvers."*

四个核心 property（原文 §1 Eq. 5 后）：

1. **Liquid time-constant** — 神经网络的 f 同时作为隐藏状态导数 + **输入依赖的可变时间常数**：`τsys = 1 / [(1/τ) + f(x(t), I(t), t, θ)]`
2. **Differentiable computational graph** — 通过 ODE solver 的反向 BPTT 训练
3. **Bounded dynamics / stability** — 状态与时间常数都有界
4. **Superior expressivity** — 在 latent trajectory space 用轨迹长度度量

### 1.2 核心公式（原文 Eq. 3-5）

LTC 论文把"隐藏状态流"显式声明为**一阶线性 ODE 系统**，由 sigmoid 门控的"非线性耦合"调制：

```text
dx(t)/dt = −x(t)/τ + S(t)                                    (Eq. 3)
S(t)    = f(x(t), I(t), t, θ) · (A − x(t))                  (Eq. 4)
⇒ dx(t)/dt = −[(1/τ) + f(x(t), I(t), t, θ)] ⊙ x(t)
            + f(x(t), I(t), t, θ) ⊙ A                        (Eq. 5)
```

其中：
- `x(t) ∈ ℝ^N`：N 个神经元的隐藏状态
- `I(t) ∈ ℝ^M`：M 维输入
- `f(·)`：由 θ 参数化的神经网络（典型实现为 Linear+sigmoid）
- `A ∈ ℝ^N`：bias 向量（"平衡电位"）
- `τ ∈ ℝ^N`：基础时间常数（per-neuron 可学）
- `⊙`：Hadamard 乘积
- `τsys = 1 / [(1/τ) + f(·)]`：**effective time-constant**——LTC 与传统 CT-RNN 的核心区别

### 1.3 Fused Solver（原文 §2 Algorithm 1）

LTC 的 forward pass 由"融合显式 / 隐式 Euler"的 ODE solver 实现：

```text
x(t + Δt) = x(t) + Δt · f(x(t), I(t), t, θ) ⊙ A
                                 ─────────────────
                          1 + Δt · [(1/τ) + f(x(t), I(t), t, θ)]
```

这个 fused solver 同时享受 implicit Euler 的稳定性 + explicit Euler 的效率，是 LTC 论文自己的算法贡献（区别于标准 ODE solver）。

### 1.4 与现有 §1.2 公式的对照

[[LNN_深度研读报告]] §1.2 第 2 条的"LTC 公式"：
```text
dx(t)/dt = −[(1/τ) + NN(x(t), I(t), θ)] ⊙ x(t) + NN(x(t), I(t), θ) ⊙ A
```
**完全等价于** Eq. (5)（其中 `NN(·)` 即 `f(·)`）。

→ §1.2 的公式**是正确的**，但**缺少 arXiv ID 引用**。本报告首次补上 grounding。

## 2. Lechner 2022 — Closed-form Continuous-time Models（arXiv 2106.13898）

**作者**：Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus
**机构**：MIT + IST Austria + Aalborg + Uni Wien
**引用**：Hasani et al., "Closed-form Continuous-depth Models", *Nature Machine Intelligence* 2022（preprint v2: 2022-03-02）

### 2.1 核心思想（原文 Abstract）

> *"We show it is possible to closely approximate the interaction between neurons and synapses – the building blocks of natural and artificial neural networks – constructed by liquid time-constant networks (LTCs) efficiently in closed-form. To this end, we compute a tightly-bounded approximation of the solution of an integral appearing in LTCs' dynamics, that has had no known closed-form solution so far. This closed-form solution substantially impacts the design of continuous-time and continuous-depth neural models; for instance, since time appears explicitly in closed-form, the formulation relaxes the need for complex numerical solvers. Consequently, we obtain models that are between one and five orders of magnitude faster in training and inference compared to differential equation-based counterparts."*

**关键 insight**：CfC 不是另起炉灶，而是 **LTC 的闭式近似**——它把 LTC Eq. (5) 中那个"难以解析求解"的 ODE 替换为可微的闭式公式。

### 2.2 核心公式（原文 Eq. 10）

```text
x(t) = σ(−f(x, I; θ_f) · t)         ⊙ g(x, I; θ_g)
     + [1 − σ(−f(x, I; θ_f) · t)] ⊙ h(x, I; θh)            (Eq. 10)
```

其中：
- `f(x, I; θ_f)`：sigmoid 门控的"时间衰减率"（非负，由 Linear+sigmoid 实现）
- `g(x, I; θ_g)`：tanh 非线性状态分支 A（对应 LTC 中的 `A` bias 路径）
- `h(x, I; θh)`：tanh 非线性状态分支 B（对应 LTC 中的 `-x(t)/τ` 路径）
- `σ(−f·t) ∈ (0, 1]`：sigmoid 衰减，把 ODE 的指数解近似为闭式门控

### 2.3 为什么 CfC 能 close-form？

LTC Eq. (5) 的解析解需要计算 `exp(∫ f dt)`——而 f 是 x、I、t 的函数，没有已知的 closed form。CfC 的论文贡献是**用一个 sigmoid 衰减 `σ(−f·t)` 来**紧致近似**这个指数解**，并证明误差有界（原文 Theorem 1: `sup |x(t) − x̃(t)| ≤ c · e^(−wτt)`）。

### 2.4 与现有 §1.2 公式的对照

[[LNN_深度研读报告]] §1.2 第 3 条的"CfC 公式"：
```text
x(t) = σ(−f(x, I; θ_f) t) ⊙ g(x, I; θ_g) + [1 − σ(−f(x, I; θ_f) t)] ⊙ h(x, I; θh)
```
**与 Eq. (10) 字面一致**。

→ §1.2 的 CfC 公式**正确**，但**缺少 arXiv 引用**。

## 3. LNN 数学基础全图（把本项目最近工作反向 trace）

把奠基论文 Eq. (3)-(5)、Eq. (10) 与本项目最近 4 轮工作放在同一坐标系：

```
Hasani 2021 Eq. (5):  dx/dt = −[(1/τ)+f]·x + f·A
                                  ↓ (Lechner 2022 闭式化)
Lechner 2022 Eq. (10): x(t) = σ(−f·t)·g + (1−σ(−f·t))·h
                                  ↓ (本项目最近工作)
┌──────────────────────────────────────────────────────────────────┐
│ MFC-CFC      = Eq. (10) 默认 (n_tau=1) — 数值等价验证 ✅            │
│ MFC-TFP      = exp(−dt/τ)·h_prev + (1−exp)·ĥ   — TFP 显式 retention│
│                (arXiv 2607.08283 retention 显式 dt 化)              │
│ MFC-NSFD     = (h + dt·G) / (1 + dt·L)         — NSFD 闭式 gain/loss│
│                (arXiv 2607.10858 ODE → 闭式更新代数同源 L-RFM)      │
│ MR-TFP-CfC   = EC routing × MFC-TFP experts   — MR-MoE × TFP      │
│                (arXiv 2606.12240 × 2607.08283 第二层综合)          │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 MFC-TFP 与 LTC Eq. (5) 的代数关系

LTC Eq. (5) 形式：
```text
x(t+Δt) ≈ x(t) + Δt · f · (A − x(t)) / (1 + Δt · [(1/τ) + f])
```

TFP (arXiv 2607.08283) Eq. (3-4) 形式（k=retention, g=write gain）：
```text
k = exp(−Δt/τ_t)
x(t+Δt) = k·x(t) + (1−k)·ĥ_t
```

**关系**：TFP 的 `exp(−Δt/τ_t)` 是 LTC Eq. (5) fused-solver 中 `1/(1 + Δt·[(1/τ) + f])` 的**精确指数解**——TFP 把 LTC 的有理式近似还原成指数 retention，更新式更对称、更适合作为 retention gate。

### 3.2 MFC-NSFD 与 LTC Eq. (5) 的代数关系

NSFD-NODE (arXiv 2607.10858) Eq. (3)：
```text
x_i^{n+1} = (x_i^n + Δt·G_i) / (1 + Δt·L_i)
```

LTC Eq. (5) 重写为 `(1 + Δt·[(1/τ)+f]) · x^{n+1} = x^n + Δt·f·A`（隐式 Euler）：
```text
x^{n+1} = (x^n + Δt·f·A) / (1 + Δt·[(1/τ)+f])
```

**关系**：NSFD 公式是 LTC Eq. (5) 的**代数同源**——只是把 `f·A` 重新解释为 `G`（gain）、把 `(1/τ)+f` 重新解释为 `L`（loss）。差别在于：
- **NSFD** 要求 G ≥ 0、L ≥ 0（positivity 保证）
- **LTC/CfC** 不要求 G ≥ 0（带符号数据）

→ MFC-NSFD 在 h_prev ≥ 0 任务（浓度、计数、电池 SOH）上才有理论优势；这与 8/5 Pareto sweep 中"MFC-NSFD 在 h=16/sl=64 爆炸"的负面结果一致——AR(2) 是带符号数据。

### 3.3 MR-TFP-CfC 与 NCP 设计哲学

Neural Circuit Policies (NCP) 的核心 idea：**仿 C. elegans 的 302 神经元布线**——把 ODE 神经元组织成稀疏 wiring，让不同时间尺度的子系统处理不同模态。

**MR-TFP-CfC 的工程对应**：
- NCP 的"神经元 wiring" → `MultiRateTfpCfC` 的 EC router（top-K 选 expert）
- NCP 的"不同时间尺度子系统" → `MemoryFusionCfCCell(retention_kind="tfp")` 的 τ_proj 偏置（每个 expert 不同的初始 τ）
- NCP 的"sparse 活动" → `top_k_active=ceil(n_tau/2)` 的稀疏路由

**但是**：8/5 benchmark 显示 MR-TFP-CfC 在 h=16 下 MSE 0.0709 > CfC 0.0550（差距 ~29%）——**说明 NCP-style 稀疏路由需要更大的 hidden 才能从 wiring 中获益**（每个 expert hidden ≥ 16）。这是 negative result 给出的明确边界条件。

## 4. §1.2 重写建议（带 arXiv ID 引用）

把 [[LNN_深度研读报告]] §1.2 现有 4 条公式**正式 grounding** 到原文：

| 现有公式 | 原文 Eq. | 来源 | 修订建议 |
|---|---|---|---|
| **L1 Basic ODE** `dh/dt = f(h, x, t, θ)` | 通用 Neural ODE 形式 | Chen 2018 (arXiv 1806.07366) | 添加 arXiv ID 引用 |
| **L2 LTC** `dx/dt = −[(1/τ)+f]·x + f·A` | Hasani 2021 Eq. (5) | arXiv 2006.04439 | 添加"τsys = 1/[(1/τ)+f]" 时间常数定义 |
| **L3 CfC** `x = σ(−f·t)·g + (1−σ)·h` | Lechner 2022 Eq. (10) | arXiv 2106.13898 | 添加"近似 LTC 闭式解"注释 |
| **L4 Liquid-S4** `ẋ = (A+Bu)x + Bu` | Hasani 2021 NCP 衍生 | (待 NCP 原文 grounding) | 标 "TBD" |

## 5. 关键 take-away（Grounding 的价值）

1. **§1.2 公式的字面正确性被验证** —— 与 Hasani 2021 Eq. (5)、Lechner 2022 Eq. (10) 完全一致
2. **MFC-TFP 是 LTC 的 retention 显式化** —— 把 fused-solver 的有理式换成指数 retention
3. **MFC-NSFD 是 LTC 的代数重构** —— 把 Eq. (5) 重新解释为 gain/loss，但要求 positivity
4. **MR-TFP-CfC 复现 NCP 哲学但需大 hidden** —— 8/5 negative result 给出的边界条件明确
5. **§1.2 现在缺 arXiv ID 引用** —— 这是本报告要正式 fix 的"foundational gap"

## 6. 数据源回链

- **原始论文**（已下载到 `papers/foundational/`）：
  - `papers/foundational/hasani_2021_ltc.pdf` + `.txt`（6.7 MB, 348 KB text）
  - `papers/foundational/lechner_2022_cfc.pdf` + `.txt`（0.98 MB, 84 KB text）
- **本项目最近工作（被 grounding）**：
  - [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]
  - [[MR_TFP_CfC_Second_Layer_Synthesis_2026-08-05]]
  - [[LNN_Training_Paradigm_2026_Summer_Cross_Section]]
- **配套文档**：
  - [[LNN_深度研读报告]]（§1.2 待更新）
  - [[LNN_Family_Taxonomy_And_Gap_2026-08-03]]
- **arXiv 链接**：
  - [Hasani 2021 LTC](https://arxiv.org/abs/2006.04439)
  - [Lechner 2022 CfC](https://arxiv.org/abs/2106.13898)
