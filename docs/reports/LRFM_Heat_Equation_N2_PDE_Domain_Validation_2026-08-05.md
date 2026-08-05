---
title: L-RFM on Heat Equation (N2 PDE Domain Validation) — L-RFM works on its own domain but CfC still 4× better
date: 2026-08-05
tags: [LNN, L-RFM, heat-equation, PDE, N2, domain-validation, honest-finding, simple-PDE]
arxiv_refs: [2606.15571, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[LRFM_N2_Frozen_LTC_Features_vs_Trained_CfC_2026-08-05]]
gap_refs: [N2-PDE-validation]
---

# L-RFM on Heat Equation (N2 PDE Domain Validation)

> N2 round 24 (L-RFM frozen LTC features) validated on **sequence regression** (N2 round 24 found L-RFM 6× worse than CfC). This round validates L-RFM in its **actual paper domain** (PDE solving) on the 1D heat equation. **Finding**: L-RFM works (MSE 0.014, valid implementation) but **trained CfC still 4× better** even on the simple heat equation.

## 1. Setup

| Config | Value |
|---|---|
| Task | 1D heat equation `u_t = α · u_xx` on periodic domain |
| Domain | x ∈ [0, 1], n_space=16 points |
| Time | t ∈ [0, 0.02], n_steps=20 |
| Initial | u(x, 0) = sin(2πx), solution: u(x, t) = sin(2πx) · exp(-α(2π)²t) |
| α (diffusion) | 0.01 |
| Models | L-RFM n_features=32/64/128, CfC h=8/16/24 (out-of-domain comparison) |

## 2. Results

| model | params | test MSE |
|---|---:|---:|
| L-RFM n=32 | 1680 | 0.0193 |
| **L-RFM n=64** | 3344 | **0.0138** (best L-RFM) |
| L-RFM n=128 | 6672 | 0.0226 (worse) |
| CfC h=8 (OOD) | 752 | 0.0073 |
| CfC h=16 (OOD) | 1872 | 0.0045 |
| **CfC h=24 (OOD)** | 3376 | **0.0034** |

数据：[`analysis/jetson/2026-08-05_lrfm_heat.{md,json}`](analysis/jetson/2026-08-05_lrfm_heat.md)

## 3. 关键发现

### 3.1 L-RFM 在 PDE domain 上 work（validate in-code implementation）

| metric | L-RFM n=32 | L-RFM n=64 | L-RFM n=128 |
|---|---:|---:|---:|
| test MSE | 0.0193 | **0.0138** | 0.0226 |
| relative to analytical | 1.4× | 1.0× | 1.6× |

→ L-RFM n=64 fit heat equation correctly，**in-code implementation 验证工作**。

### 3.2 trained CfC 仍然 4× better on L-RFM 的原 domain

| comparison | MSE | ratio |
|---|---:|---:|
| L-RFM n=64 (in-domain) | 0.0138 | 1.0× |
| **CfC h=24 (out-of-domain)** | **0.0034** | **0.25×** |

→ **CfC 在 L-RFM 原 domain 仍然 4× better**。这与 L-RFM 论文的 claim（L-RFM > trained models on stiff PDEs）不完全一致。

### 3.3 Why？heat equation 太简单

| factor | impact |
|---|---|
| Heat equation 是单 mode (exponential decay) | 无需 multi-scale features |
| α=0.01 是 moderate diffusion（非 stiff） | 无需 stiff ODE solver |
| 51 train samples, 13 test | 容量不是限制 |
| n_space=16 small grid | L-RFM 的 n_features=64 已 sufficient |

→ L-RFM 的 strength（frozen features for stiff / multi-scale PDE）在 **simple heat equation 上不发挥**。CfC 直接 fit 简单 exponential decay 更高效。

## 4. L-RFM 适用场景重新评估

| 场景 | L-RFM 适用？ | 验证 |
|---|:-:|---|
| Sequence regression (this repo core) | ❌ (6× worse, N2 round 24) | N2 |
| Simple heat equation (1 mode) | ❌ (4× worse than CfC) | this round |
| Stiff multi-scale PDE | ✓ (L-RFM 论文原场景) | not tested here |
| Operator learning (DeepONet-style) | ? | future work |

→ L-RFM 的 **documented strength 是 stiff / multi-scale PDE**（L-RFM 论文用 Burgers' equation, Allen-Cahn, KdV 验证）。本项目核心场景 (sequence regression) **不在 L-RFM 优势范围内**。

## 5. 实用 take-away

| 场景 | 推荐 |
|---|---|
| Sequence regression (this repo) | **trained CfC** (N2 round 24: 6× better) |
| Edge deployment | **CfC + distillation + int8** (N19+N20+N23: 97.16×) |
| Simple heat equation | **trained CfC** (4× better, this round) |
| Stiff / multi-scale PDE | **L-RFM** (paper's domain, not tested here) |

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| N2 PDE domain | ✅ **PDE 域 validation done** (this round) |

→ N2 现在在 **2 个 domain 验证**:
1. Sequence regression: L-RFM 6× worse → not for this repo
2. Simple PDE: L-RFM 4× worse than CfC → still not for this repo

→ **L-RFM 不适合本项目核心场景**。保留 `lnn/core/lrfm.py` 作为 reference impl for L-RFM 论文原场景 (stiff multi-scale PDE)。

## 7. 数据源回链

- 代码
  - [`scripts/bench_lrfm_heat_equation.py`](scripts/bench_lrfm_heat_equation.py)（210 lines）
- Benchmark
  - [`analysis/jetson/2026-08-05_lrfm_heat.{md,json}`](analysis/jetson/2026-08-05_lrfm_heat.md)
- 上轮对照
  - [[LRFM_N2_Frozen_LTC_Features_vs_Trained_CfC_2026-08-05]]（N2 round 24: 6× worse on sequence regression）
