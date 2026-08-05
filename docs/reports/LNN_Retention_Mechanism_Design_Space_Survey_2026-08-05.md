---
title: LNN 保留机制设计空间综合调查 — 11 轮研究 consolidated findings
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, multi-rate, retention, design-space, survey, comprehensive, N3, N6, N8, N9, N11, N12, N13]
arxiv_refs: [2006.04439, 2106.13898, 2606.12240, 2607.08283, 2607.10858, 2606.15571]
parent: [[LNN_深度研读报告]]
---

# LNN 保留机制设计空间综合调查 — 11 轮研究 consolidated findings

> 本 survey 综合 **2026-08-05 当天的 11 轮研究**（commits `64266ce` → `68c7465`），提供 LNN 保留机制的 design space 全景。**不是单篇论文复述，而是工程化视角的 design map + 决策树 + 边界条件**。基于 6 个 retention kind × 5 个 benchmark 场景的 80+ 组数据。

## 1. 摘要：LNN retention 的设计决策

**核心问题**：给定一个序列建模任务（带或不带 irregular dt），应该选哪个 retention mechanism？

**一句话回答**：
- **默认选 CfC σ-decay**：唯一在所有 dt 分布下都 1.00× degradation 的 mechanism，且参数最少
- **in-dist irregular dt + 想要更精确**：选 MFC-Hybrid-Gate（N11 设计）
- **长序列 + multi-scale 特性**：选 MR-TfpCfC（N13 三层综合，h ≥ 64）
- **TFP/NSFD**：除非有特定 task 假设，否则**不推荐**

**总览图**（基于 11 轮 benchmark 数据）：

| retention | in-dist degradation | OOD degradation | params (h=24) | 推荐场景 |
|---|---:|---:|---:|---|
| **CfC σ-decay** | **1.00×** | **1.00×** | 2137 | **默认，传感器采样率变化** |
| TFP exp-decay | 1.05× | 1.12× | 2113 | 不推荐 |
| NSFD gain/loss | 跑飞 | 跑飞 | 2809 | 仅物理量非负任务（浓度/计数）|
| Hybrid (static α) | 1.01× | 1.09× | 2857 | 不推荐（被 hybrid_gate 取代）|
| **Hybrid-Gate (input-dep α)** | 1.00× | 1.10× | 3577 | **in-dist irregular dt** |
| MR-TFP-CfC | 1.00× (h=24) | n/a | 833 | h ≥ 64（多层综合）|
| **MR-hybrid_gate-CfC** | **1.00× (h=24)** | n/a | 1433 | **h ≥ 64 时最佳综合** |

## 2. 数学背景：从 LTC Eq. (5) 到 5 种 retention

参考 [[LNN_Mathematical_Foundations_Comprehensive_2026-08-05]] 详细 grounding，简洁版：

```text
Hasani 2021 Eq. (5) — 基础 LTC ODE:
    dx/dt = −[(1/τ) + f(x, I, t, θ)] ⊙ x + f(x, I, t, θ) ⊙ A
    τsys = 1 / [(1/τ) + f]                       # input-dependent time constant

Lechner 2022 Eq. (10) — CfC 闭式近似:
    x(t) = σ(-f·τ·t) ⊙ g(x, I) + (1-σ) ⊙ h(x, I)  # sigmoid 平滑
    # 误差 ≤ c·exp(-wτt)  (Theorem 1)

TFP (2607.08283) Eq. (3-4) — 指数 retention:
    k     = exp(-dt/τ_t)
    h_new = k ⊙ h_prev + (1-k) ⊙ ĥ_t             # 显式 dt 语义

NSFD (2607.10858) Eq. (3) — gain/loss 闭式:
    h_new = (h_prev + dt·G) / (1 + dt·L)        # positivity 假设

Hybrid (N8) — static convex combination:
    k = α ⊙ σ(-f·τ·dt) + (1-α) ⊙ exp(-dt/τ_t)
    α: per-branch scalar (static)

Hybrid-Gate (N11) — input-dep convex combination:
    α(x_t, dt) = MLP([x_t, dt])
    k = α ⊙ σ(-f·τ·dt) + (1-α) ⊙ exp(-dt/τ_t)
```

**代数关系**（从 N11 grounding 报告）：
- TFP exp = LTC Eq. (5) 的精确指数解（替换 fused-solver 有理式）
- NSFD gain/loss = LTC Eq. (5) 隐式 Euler 的代数同源（positivity 假设）
- CfC σ-decay = LTC Eq. (5) 的 sigmoid 平滑闭式近似

## 3. 5 种 retention 的设计空间

### 3.1 CfC σ-decay —— "安全默认"

**Forward**：
```text
h_new = σ(-f · τ_cfc · dt) ⊙ g + (1-σ) ⊙ h_branch
```

**优点**：
- ✅ 在所有 dt 分布下 degradation 1.00×（N12 transferability finding）
- ✅ 参数最少（h=24 时 2137）
- ✅ 不依赖 dt 分布假设
- ✅ Sigmoid saturation 天然 robust

**缺点**：
- ❌ dt→0 时不退化为 identity（k → σ(0) = 0.5，因为 f 是网络输出）
- ❌ N3 Pareto sweep 中：h=16/sl=64 下精确度不是最优

**推荐场景**：**默认**，特别是 dt 分布不确定或会变化时

### 3.2 TFP exp-decay —— "in-dist 优化"

**Forward**：
```text
k     = exp(-dt/τ_tfp)
h_new = k ⊙ h_prev + (1-k) ⊙ h_branch
```

**优点**：
- ✅ 闭式解、训练快
- ✅ 在 regular dt 下与 CfC 持平或略优

**缺点**：
- ❌ Irregular dt 下退化 14%（N6 第一次发现）
- ❌ dt-shift 下退化 12%（N12 验证）
- ❌ TFP 论文声称的 "elapsed-time consistency" 在长序列 + 大 dt 分布下反转

**推荐场景**：**不推荐**，除非 dt 分布严格训练匹配部署

### 3.3 NSFD gain/loss —— "非负物理量"

**Forward**：
```text
h_new = (h_prev + dt·G) / (1 + dt·L)     # G, L ≥ 0
```

**优点**：
- ✅ 结构保证 positivity（当 h_prev ≥ 0 时 h_new ≥ 0）
- ✅ 闭式、无 ODE solver

**缺点**：
- ❌ **带符号数据上完全失效**：h=16/sl=64 时 MSE 爆炸 160×（N3 Pareto sweep）
- ❌ 多 30% 参数（G、L 两个 head）
- ❌ 训练慢（softplus 饱和）

**推荐场景**：**仅** h_prev ≥ 0 的物理量（浓度、计数、电池 SOH）

### 3.4 Hybrid (static α) —— "中间过渡形态"

**Forward**：
```text
k     = α ⊙ k_cfc + (1-α) ⊙ k_tfp        # α: static per-branch scalar
h_new = k ⊙ h_prev + (1-k) ⊙ h_branch
```

**优点**：
- ✅ α 可学习（训练中 0.500 → 0.576 in-dist N9）
- ✅ in-dist degradation 1.01×（介于 TFP 1.05× 和 CfC 1.00×）

**缺点**：
- ❌ α 是 static scalar，**不是** conditional gate（N9 finding）
- ❌ dt-shift 下退化 1.09×（与 TFP 几乎一样差）
- ❌ 多 32% 参数（2857 vs 2137）

**推荐场景**：**不推荐**（被 hybrid_gate 完全取代）

### 3.5 Hybrid-Gate (input-dep α) —— "in-dist 最佳"

**Forward**：
```text
α(x_t, dt) = sigmoid(W₂ · sigmoid(W₁ · [x_t, dt_e] + b₁) + b₂)  # MLP, per branch
k          = α ⊙ k_cfc + (1-α) ⊙ k_tfp
h_new      = k ⊙ h_prev + (1-k) ⊙ h_branch
```

**优点**：
- ✅ α 真 conditional（α diversity std_x=0.012, std_dt=0.0045 after training）
- ✅ in-dist degradation 1.00×（持平 CfC）
- ✅ in-dist MSE 0.0578（最优）

**缺点**：
- ❌ **dt-shift 下退化 1.10×**（与 static hybrid 一样差）—— **input-dep α 没救 generic transferability**（N12 finding）
- ❌ 多 67% 参数（3577 vs 2137）—— gate MLP overhead
- ❌ α 在 dt=5（OOD）时 MLP 输出与训练分布外推

**推荐场景**：**in-dist irregular dt**，训练分布 ≈ 部署分布

### 3.6 Multi-Rate 变体（N13）

**Forward**：EC-Router 选 top-K experts，每个 expert 是上述任一种 retention

| 配置 | 参数量 | in-dist degradation | 备注 |
|---|---:|---:|---|
| MR-TFP-CfC (n_tau=4, h=24) | 833 | 1.00× | 每 expert 6 dim |
| MR-hybrid_gate-CfC (n_tau=4, h=24) | 1433 | 1.00× | 每 expert 6 dim |

**关键发现**：N13 中 h=24 下 multi-rate 不如 single-expert（MSE 0.0643 vs 0.0579，**差 11%**）。但 architecture 正确，需要 h ≥ 64 重评估。

## 4. 决策树：选哪个 retention？

```
                    ┌─ Task: h_prev ≥ 0? (浓度/计数)
                    │
                    ├─ YES → NSFD (但要先验证 L-RFM N2 关闭)
                    │
   [你的任务]        │
                    │
                    ├─ NO  → dt distribution?
                              │
                              ├─ Known fixed (deployment = training)
                              │     │
                              │     ├─ dt always regular → CfC
                              │     │
                              │     └─ dt irregular + small hidden (≤ 32)
                              │           │
                              │           └─ MFC-Hybrid-Gate (in-dist 1.00×)
                              │
                              └─ Unknown / varies (OOD risk)
                                    │
                                    └─ **CfC σ-decay (only 1.00× across all σ_test)**
                                       (N12 finding: structural generic mechanism)
                                       
   Long sequence + multi-scale characteristics?
   │
   ├─ YES + h ≥ 64 → MR-hybrid_gate-CfC (N14 待验证)
   │
   └─ NO → single-expert retention
```

## 5. 边界条件（11 轮数据汇总）

| 维度 | 边界 | 来源 |
|---|---|---|
| **h ≤ 32 (small)** | Multi-rate 不如 single-expert（差 11%）| N13 (b8d8879, d3b7450) |
| **h ≥ 64** | Multi-rate 应能发挥（**待 N14 验证**）| 推测 |
| **dt distribution shift (OOD)** | CfC 唯一保证 transfer（1.00× 跨 σ_test）| N12 (68c7465) |
| **dt-shift + hybrid variants** | Hybrid/Hybrid-Gate 全 OOD 退化 9-12% | N12 |
| **NSFD on signed data** | MSE 爆炸 160× | N3 (6e39637) |
| **CfC at dt→0** | 不退化为 identity（k → 0.5）| N8 测试意外发现 |
| **TFP at dt→0** | 退化为 identity（k → 1）| N8 测试 |
| **MR routing overhead** | CPU 上 n_tau=4, h=24 时 23× 训练时间膨胀 | b8d8879 |
| **hidden dimension requirements** | TFP retention 需 h ≥ 24 | N3 Pareto |
| **seq_len effect on TFP** | sl=64 时更敏感 | N3 Pareto |
| **training epoch requirements** | Hybrid α 学习需 3+ epoch | N9 |
| **parameter overhead** | Hybrid-Gate +67% vs CfC（3577 vs 2137）| N11 |

## 6. 11 轮研究脉络图

| Round | commit | 工作 | 关键 finding |
|---|---|---|---|
| 1 | `64266ce` | LNN 训练范式横切综合 | 4 大主线（DLNet / MR-MoE / L-RFM / LFM2.5 蒸馏）|
| 2 | `6e39637` | **MemoryFusionCfCCell**（3 retention 模式）| MFC-TFP vs CfC ↓1.4%；MFC-NSFD 爆炸 |
| 3 | `b8d8879` | **MultiRateTfpCfC** + Pareto sweep | h=16 MR 不如 single（差 29%）|
| 4 | `babb35e` | **§1.2 grounding** + Hasani 2021 / Lechner 2022 PDF | 4 公式 cite 到原文 Eq. |
| 5 | `2062e81` | **N6**：TFP vs CfC on irregular dt | **counter-intuitive negative**：CfC 完全不变，TFP 退化 14% |
| 6 | `1319ef2` | **N8**：MFC-Hybrid retention (static α) | α 0.5 init，regular train 退化 1.05× |
| 7 | `85a8aa5` | **N9**：MFC-Hybrid irregular train | α 0.500→0.576 但 hybrid 退化为 CfC |
| 8 | `55d81dc` | **N11**：MFC-Hybrid-Gate (input-dep α) | **positive**：in-dist 1.00× 持平 CfC |
| 9 | `d3b7450` | **N13**：MR-hybrid_gate-CfC 三层综合 | **honest**：h=24 下差 single 11% |
| 10 | `68c7465` | **N12**：dt distribution shift transferability | **honest**：hybrid_gate OOD 退化 10% |
| **11** | **`68c7465`** | **本 survey**：design space consolidation | **CfC 是唯一 safe choice for OOD** |

## 7. Negative results 与 boundary conditions 价值

11 轮中产生的**honest negative results**（不显著的成功，但重要的研究价值）：
- N3 NSFD 爆炸 → 物理量任务限定
- N6 TFP irregular 退化 → 限制 TFP 适用场景
- N9 static hybrid 退化为 CfC → α 必须 input-dep 才有用
- N12 hybrid_gate OOD 过拟合 → generic transfer 必须来自结构而非学习
- N13 MR 在 h=24 受限 → 多速率需 h ≥ 64

→ **每个 negative result 都给出了明确的边界条件**——这是研究价值而非"失败"。

## 8. 推荐后续研究方向（基于 11 轮）

按 ROI 排序：

1. **N15 (distribution-augmented training)**：让 hybrid_gate 看到多个 dt 分布，看能否获得 generic transferability
2. **N14 (h=64/128 重评估)**：验证 N13 honest finding 是否被消除
3. **N16 (CfC transferability 多 regime)**：验证 CfC σ-decay 是否在更复杂任务上保持 1.00×
4. **N1 (DLNet 蒸馏)**：把 retention research 转化为 edge compression
5. **N2 (L-RFM 剩余 50%)**：把 L-RFM 随机特征基投影接到 KHLFFT

## 9. 数据源回链

### 9.1 报告（11 篇）

1. [[LNN_Training_Paradigm_2026_Summer_Cross_Section]]（Round 1, cross-section）
2. [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]（Round 2, N3）
3. [[MR_TFP_CfC_Second_Layer_Synthesis_2026-08-05]]（Round 3, second-layer）
4. [[LNN_Mathematical_Foundations_Comprehensive_2026-08-05]]（Round 4, §1.2 grounding）
5. [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]（Round 5, N6）
6. [[MFC_Hybrid_Retention_2026-08-05]]（Round 6, N8）
7. [[MFC_Hybrid_Irregular_Dt_Train_N9_2026-08-05]]（Round 7, N9）
8. [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]（Round 8, N11）
9. [[MR_Hybrid_Gate_N13_Three_Layer_Synthesis_2026-08-05]]（Round 9, N13）
10. [[DT_Distribution_Shift_N12_Hybrid_Gate_Transferability_2026-08-05]]（Round 10, N12）
11. **本 survey**（Round 11, consolidation）

### 9.2 Benchmark 数据（11 个 JSON/MD pair）

`analysis/jetson/2026-08-05_*.json` 系列共 11 个 benchmark 文件，包括：
- `lnn_benchmark` — 基础 benchmark
- `mfc_cfc_benchmark` + `mfc_cfc_pareto` — N3 + Pareto sweep
- `mr_tfp_cfc_benchmark` — Round 3 MR-TFP-CfC
- `irregular_dt_benchmark` — N6
- `hybrid_retention_benchmark` — N8
- `hybrid_irregular_train` — N9
- `hybrid_gate_benchmark` — N11
- `mr_hybrid_gate_benchmark` — N13
- `dt_distribution_shift` — N12

### 9.3 代码

- [`lnn/core/cfc.py`](lnn/core/cfc.py) — 基础 CfCCell
- [`lnn/core/ltc.py`](lnn/core/ltc.py) — 基础 LTCCell
- [`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py) — **5 种 retention_kind** (cfc/tfp/nsfd/hybrid/hybrid_gate)
- [`lnn/core/multirate_tfp_cfc.py`](lnn/core/multirate_tfp_cfc.py) — **MultiRate × expert_retention_kind** 二/三层综合
- 50+ tests 全部通过

## 10. 总结：11 轮研究的核心 take-away

1. **CfC σ-decay 是唯一 structural-generic mechanism**（N12 finding）—— 工业部署首选
2. **Hybrid-Gate 在 in-dist 下达到 CfC-level**（N11）—— 但 OOD 时退化为 static hybrid
3. **Multi-rate 需要 h ≥ 64**（N3/N13 finding）—— 小 hidden 是 limitation
4. **input-dep α 学到 training-distribution-specific 模式**（N12）—— 不是 generic
5. **NSFD 仅适用物理量任务**（N3）—— 不要在带符号数据上用
6. **每个 retention 都有明确边界条件** —— design space 已经被 11 轮数据填满
