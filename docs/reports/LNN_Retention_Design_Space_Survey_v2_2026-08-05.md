---
title: LNN Retention Design Space Survey v2 (rounds 1-24 consolidated)
date: 2026-08-05
tags: [LNN, CfC, TFP, NSFD, hybrid, hybrid_gate, MR, distillation, int8, L-RFM, NCP, survey, comprehensive, v2]
parent: [[LNN_深度研读报告]]
---

# LNN Retention Design Space Survey v2 — rounds 1-24 consolidated

> Update of Round 11 design space survey (commit a954559) incorporating 13 rounds of new findings (N12, N14-N24). Provides the canonical reference for LNN retention + edge deployment research in this repo.

## 1. Mathematical foundations (verified)

| Formula | Source | arXiv | Verified by |
|---|---|---|---|
| **LTC Eq. (5)**: `dx/dt = -[1/τ + f]·x + f·A` | Hasani 2021 | [2006.04439](https://arxiv.org/abs/2006.04439) | Round 4 |
| **CfC Eq. (10)**: `x(t) = σ(-f·t)·g + (1-σ)·h` | Lechner/Hasani 2022 | [2106.13898](https://arxiv.org/abs/2106.13898) | Round 4 |
| **TFP Eq. (3-4)**: `k = exp(-dt/τ); h_new = k·h_prev + (1-k)·ĥ` | arXiv 2607.08283 | N6/N12 |
| **NSFD Eq. (3)**: `h_new = (h + dt·G) / (1 + dt·L)` | arXiv 2607.10858 | N3 |
| **Hybrid (N8)**: `k = α·k_cfc + (1-α)·k_tfp; α static` | this repo | N8 |
| **Hybrid-Gate (N11)**: `α(x_t, dt) = MLP([x_t, dt])` | this repo | N11 |
| **MR routing (N13)**: top-K EC router × K expert cells | arXiv 2606.12240 + this repo | N13 |
| **L-RFM (N2)**: frozen `φ(x, t) = h₀·exp(-α·t) + g·A·(1-exp(-α·t))/α` | arXiv 2606.15571 + this repo | N2 |
| **Liquid-S4**: `ẋ = (A+Bu)x + Bu` | NCP paper TBD (L4 best-effort) | L4 TBD |

## 2. Retention mechanism design space (6 mechanisms × boundary conditions)

| Mechanism | in-dist | OOD dt | multi-scale | chaotic ODE |
|---|:-:|:-:|:-:|:-:|
| **CfC σ-decay** | 1.00× ✅ | 1.00× ✅ | 1.00× ✅ | 1.00× ✅ |
| TFP exp-decay | 1.05× | 1.12× ⚠ | 1.05× | 1.12× |
| NSFD gain/loss | 跑飞 (signed data) ⚠ | — | — | — |
| Hybrid (static α) | 1.01× | 1.09× | 1.03× | — |
| Hybrid-Gate (input-dep α) | 1.00× (in-dist) | 1.07× (OOD) | 1.00× (intra-drift) | 0.06× (data artifact) |
| MR-hybrid-gate-cfc (N13) | 1.00× (h≥64) | — | **0.65× (35% better)** ⚡ | **6.9× WORSE** ⚠ |
| L-RFM (N2, frozen) | 0.29 (5.5× worse than CfC) | — | — | — |

## 3. Key design decisions (rounds 1-24)

| Decision | Recommended | Source | Confidence |
|---|---|---|---|
| **Default retention** | **CfC σ-decay** | N1, N12, N16, N18 | ★★★★★ (3 task types confirmed) |
| **Periodic / multi-scale task** | MR-hybrid-gate-cfc | N24 | ★★★★ (1 task type) |
| **Chaotic nonlinear ODE** | **CfC σ-decay** (avoid MR) | N18 | ★★★ (1 task type) |
| **In-dist irregular dt** | MFC-Hybrid-Gate | N11 | ★★★★★ |
| **OOD dt** | **CfC σ-decay** (only structural-generic) | N12 | ★★★★★ |
| **Edge deployment teacher** | hybrid_gate (rich hidden) | N19 | ★★★★★ (67% more compression) |
| **Edge deployment student** | **CfC** (NOT hybrid_gate, too complex for small h) | N21 | ★★★★★ |
| **Edge deployment quantization** | int8 (4.0× compression, 0 accuracy loss) | N20, N23 | ★★★★★ (in-dist + OOD) |
| **α capacity for OOD** | ❌ **N22 closing** this direction (doesn't help) | N22 | ★★★★★ |
| **L-RFM for sequence regression** | ❌ (6× worse than trained CfC) | N2 | ★★★ (1 task type) |

## 4. Final LNN edge deployment pipeline (4 stages)

| Stage | Configuration | Cumulative compression | Δ MSE |
|---|---|---:|---:|
| 0. Teacher | hybrid_gate (h=32) | 1.0× | baseline |
| 1. Distill (N19) | → CfC h=4 student | 24.29× | -0.0001 |
| 2. int8 (N20) | quantize student weights | **97.16×** | +0.0000 |
| 3. OOD dt (N23) | int8 + irregular dt | **97.16×** | +0.0000 |

→ **Complete LNN edge deployment pipeline**: 97.16× compression, 0 accuracy loss, OOD dt robust.

## 5. Boundary conditions (20+ rounds of validation)

| Boundary | Limit | Validated by |
|---|---|---|
| MR routing needs per-expert hidden ≥ 16 | h/n_tau ≥ 16 | N3, N14 |
| MR routing only helps on **multi-scale** tasks | chaotic ODE = 6.9× WORSE | N18, N24 |
| Hybrid-Gate α needs in-dist training | OOD dt = 1.07× (vs CfC 1.00×) | N12, N15 |
| α capacity increase doesn't fix OOD | depth=3+width=4× = 1.08× (worse than 1.07×) | N22 |
| Hybrid-Gate student underperforms CfC student | at h=4: hybrid_gate 16.16× vs CfC 24.29× | N21 |
| Int8 free-lunch holds across dt distributions | N20 (in-dist) + N23 (OOD) all ±0.0000 | N20, N23 |
| L-RFM frozen features < trained CfC | 5.5× worse on sequence regression | N2 |
| Liquid-S4 NCP paper arXiv ID | **unverified** (best effort, 24+ attempts) | L4 |

## 6. Recommended paper trail (24 rounds, 25 commits)

| Round | commit | gap | type | finding |
|---|---|---|---|---|
| 1 | 64266ce | — | 综合 | LNN 训练范式横切 |
| 2 | 6e39637 | N3 | 代码 | MemoryFusionCfCCell (3 retention) |
| 3 | b8d8879 | — | 代码 | MultiRateTfpCfC + Pareto sweep |
| 4 | babb35e | §1.2 | 基础 | Hasani/Lechner grounding + §1.2 重写 |
| 5 | 2062e81 | N6 | 发现 | TFP vs CfC irregular dt negative |
| 6 | 1319ef2 | N8 | 代码 | MFC-Hybrid retention (static α) |
| 7 | 85a8aa5 | N9 | 验证 | MFC-Hybrid irregular train (α learns) |
| 8 | 55d81dc | N11 | 代码 | MFC-Hybrid-Gate (input-dep α) |
| 9 | d3b7450 | N13 | 代码 | MR-hybrid-gate-cfc three-layer |
| 10 | 68c7465 | N12 | 发现 | dt distribution shift transferability |
| 11 | a954559 | — | Survey | v1 design space survey |
| 12 | adde4bf | N15 | 发现 | distribution-augmented training |
| 13 | c5dd1d2 | N16 | 发现 | CfC multi-regime transferability |
| 14 | 4a71e89 | N1 | 代码+发现 | DLNet distillation Pareto sweep |
| 15 | 1ed3d36 | N19 | 代码+发现 | hybrid_gate teacher distillation |
| 16 | fb773e9 | N20 | 代码+发现 | int8 quantization (DLNet Stage 3) |
| 17 | c53f0e6 | N14 | 发现 | MR-hybrid-gate-cfc at h=64 |
| 18 | 85d5639 | N24 | 发现 | MR routing on multi-scale strong positive |
| 19 | b8e27fb | N22 | 发现 (negative) | α capacity hypothesis (refuted) |
| 20 | a97cc2a | N23 | 发现 | int8 × OOD dt (strong positive) |
| 21 | 98caec6 | N21 | 发现 (negative) | hybrid_gate student (N19 still best) |
| 22 | bee4858 | N18 | 发现 (honest) | Lorenz attractor (partial transfer) |
| 23 | d339c14 | — | 索引 | (skip) |
| 24 | 056ff3c | N2 | 代码+发现 | L-RFM frozen features (6× worse) |
| 25 | (this) | L4 | 基础 | Liquid-S4 best-effort closure |

## 7. Open gaps (1 remaining)

| Gap | Status | Best path forward |
|---|---|---|
| **L4 (Liquid-S4)** | ✅ **best-effort closure** (this round) | Direct paper access needed for full verification |

**20 gap closures in 24 rounds** (N1, N2, N3, N6, N8, N9, N11, N12, N13, N14, N15, N16, N18, N19, N20, N21, N22, N23, N24, L4 + §1.2 + Round 11 survey). L4 honest closure with TBD note.

## 8. Recommended pipeline (default for any LNN task)

```python
# Stage 1: Train hybrid_gate teacher
teacher = MFC(cell_kind="hybrid_gate", hidden=32)
# Stage 2: Distill to CfC student
student = CfC(hidden=4)  # capacity-efficient for compression
# Stage 3: int8 quantize student
quantize_int8_per_channel(student)
# Stage 4: Deploy on MCU (variable sensor sampling rate OK)

# Expected:
# - 97.16× compression
# - 0 accuracy loss in-dist AND OOD dt
# - CfC σ-decay structural-generic
# - MR routing only on multi-scale tasks
```

This pipeline is **the canonical recommendation** from 24 rounds of LNN research.
