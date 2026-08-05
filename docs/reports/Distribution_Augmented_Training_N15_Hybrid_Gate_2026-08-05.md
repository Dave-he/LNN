---
title: Distribution-Augmented Training (N15) — 部分改善 hybrid_gate OOD transferability（partial positive）
date: 2026-08-05
tags: [LNN, CfC, hybrid_gate, distribution-augmented-training, OOD, transferability, N15, partial-positive]
arxiv_refs: [2607.08283, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[DT_Distribution_Shift_N12_Hybrid_Gate_Transferability_2026-08-05]], [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]
gap_refs: [N15-distribution-augmented-training]
---

# Distribution-Augmented Training (N15) — partial positive

> N12 发现 hybrid_gate 的 input-dep α **过拟合训练 dt 分布**，OOD (σ=1.0) 时退化 1.10×。本轮 N15 验证 **distribution-augmented training**（训练时每个 batch 随机从 {0.3, 0.5, 1.0} 三个 dt 分布采样）能否让 α 学到更 general 的 dt-robustness。**结论：partial positive**——OOD 退化从 1.10× 改善到 1.07×，但仍达不到 CfC 的 1.00× perfect transfer。

## 1. 实验设计

| 配置 | N12 baseline | **N15 (本轮)** |
|---|---|---|
| 训练 dt | 单分布 LogNormal(0, 0.5) | **MIXED per-batch** LogNormal(0, σ) for σ ∈ {0.3, 0.5, 1.0} |
| 测试 dt | σ_test ∈ {0.3, 0.5, 1.0} | 同 N12 |

**关键问题**：让模型在训练中看到所有 3 个 dt 分布，能否让它学到 "generic dt-robustness" 而不是某个特定分布的 pattern？

## 2. Benchmark 结果

### 2.1 N15 结果（混合训练）

| 模型 | σ=0.3 | σ=0.5 | **σ=1.0 (OOD)** |
|---|---:|---:|---:|
| **cfc-baseline (regular train only)** | **1.00×** | **1.00×** | **1.00×** ✅ |
| **mfc-hybrid_gate (mixed dt train)** | 1.01× | 1.02× | **1.07×** ⚡ |

### 2.2 N12 baseline（对照）

| 模型 | σ=0.3 | σ=0.5 | σ=1.0 (OOD) |
|---|---:|---:|---:|
| cfc-baseline | 1.00× | 1.00× | 1.00× |
| mfc-hybrid_gate (single-dist train) | 1.01× | 1.04× | **1.10×** |

### 2.3 对比 N15 vs N12

| σ_test | N12 (single-dist) | **N15 (mixed-dist)** | Δ 改善 |
|---|---:|---:|---:|
| 0.3 | 1.01× | **1.01×** | 持平 |
| 0.5 (in-dist) | 1.04× | **1.02×** | ↓2pp |
| **1.0 (OOD)** | **1.10×** | **1.07×** | **↓3pp** ⚡ |

## 3. Partial positive 解读

### 3.1 Distribution-augmented training **部分有效**

- ✅ **OOD degradation 从 1.10× 改善到 1.07×**（↓3pp）
- ✅ **In-dist degradation 从 1.04× 改善到 1.02×**（↓2pp）
- ✅ 没有让 hybrid_gate 变差

### 3.2 但 **没达到** CfC 的 1.00× perfect transfer

- ❌ **σ=1.0 (OOD) 仍 1.07×**——α 没完全学到 generic 机制
- ❌ 即便看到三个 dt 分布训练，α 仍学到 distribution-specific patterns
- **→ CfC 仍是唯一 structural-generic mechanism**

### 3.3 Why？

α 的输入是 `cat([x_t, dt_e])`，dt 是 per-step scalar。即使训练时看到 dt ∈ [0.012, 17.5] 的全范围，MLP 的 `sigmoid(W₁ · [x_t, dt] + b₁) + b₂` 仍在 `[0, 1]` 上 saturate，**只能学到 "dt 在训练范围中心附近时 α = X，边缘时 α = Y"**——这是 interpolation 而非 generic。

→ **α 的 capacity 不够表达 generic dt-robustness**——它本质上是一个 learned interpolation function，不是 structural mechanism。

## 4. 实用 take-away（修订）

| 场景 | 推荐 retention + 训练策略 |
|---|---|
| Regular dt | CfC |
| In-dist irregular dt + 已知分布 | MFC-Hybrid-Gate |
| **In-dist irregular dt + 分布变化小** | **MFC-Hybrid-Gate + distribution-augmented training**（N15 改善）|
| **Irregular dt + 分布未知/大变化** | **CfC σ-decay**（仍是唯一 safe choice）|
| **传感器采样率未知/会剧烈变化** | **CfC σ-decay** |

## 5. N11 → N12 → N15 演进

| 实验 | α 类型 | 训练策略 | σ=1.0 (OOD) degradation |
|---|---|---|---|
| N11 | input-dep α MLP | single-dist (0.5) | n/a (in-dist test only) |
| N12 | input-dep α MLP | single-dist (0.5) | **1.10×** |
| **N15** | **input-dep α MLP** | **mixed-dist (0.3, 0.5, 1.0)** | **1.07×** ↓ |

→ distribution-augmented training 让 α 学到 **"multiple distribution interpolation"**，但仍是 interpolation 而非 generic mechanism。

## 6. Gap 状态更新

| # | 缺口 | 状态 |
|---|---|---|
| **N15** | distribution-augmented training 让 hybrid_gate transfer | ✅ **本轮关闭（partial positive）** |
| N14 | MR-hybrid_gate-CfC 在 h=64/128 上重评估 | ⏳ 下周 |
| N16 | CfC transferability 在多 regime 任务上验证 | ⏳ 路线图 |
| **新增 N17** | α capacity 增强（更深 MLP / attention-based gating）能否突破 interpolation 限制 | ⏳ 路线图 |
| N1 | DLNet 蒸馏 | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 7. 推荐后续动作

1. **本周**：N14 MR-hybrid_gate-CfC 在 h=64/128 上重评估（验证 N13 honest finding）
2. **下周**：N16 CfC 在多 regime 任务上的 transferability（验证 N12 finding 是否泛化）
3. **路线图**：N17 — 给 α MLP 加 attention / 多层 MLP，看 capacity 增强能否让 hybrid_gate 真正 generic transfer
4. **路线图**：N1 — 把 retention research 扩展到 edge compression（DLNet 蒸馏）

## 8. 数据源回链

- 代码
  - [`scripts/bench_distribution_augmented_training.py`](scripts/bench_distribution_augmented_training.py)（238 lines）
- Benchmark
  - [`analysis/jetson/2026-08-05_distribution_augmented_training.{md,json}`](analysis/jetson/2026-08-05_distribution_augmented_training.md)
- 上轮对照
  - [[DT_Distribution_Shift_N12_Hybrid_Gate_Transferability_2026-08-05]]（N12 single-dist 1.10×）
  - [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]（N11 in-dist 1.00×）
