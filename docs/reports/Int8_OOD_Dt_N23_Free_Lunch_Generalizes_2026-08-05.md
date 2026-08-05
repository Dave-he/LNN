---
title: Int8 Quantization on OOD dt (N23) — STRONG POSITIVE：int8 free-lunch 在 OOD dt 下仍成立
date: 2026-08-05
tags: [LNN, int8, quantization, distillation, OOD-dt, N23, strong-positive, free-lunch-generalizes]
arxiv_refs: [2601.06227, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[Int8_Quantization_N20_DLNet_Stage3_2026-08-05]], [[DT_Distribution_Shift_N12_Hybrid_Gate_Transferability_2026-08-05]]
gap_refs: [N23-int8-ood-dt]
---

# Int8 Quantization on OOD dt (N23) — STRONG POSITIVE

> N20 发现 int8 quantization 是 "free-lunch 4.0× compression, 零精度损失"（in-dist）。本轮 N23 验证 int8 quantization 在 **OOD dt** 下是否仍保持——如果 int8 quantization error 与 retention's OOD sensitivity 复合，int8 free-lunch 可能在 OOD 下失效。

## 1. 实验设计

| 配置 | 值 |
|---|---|
| Task | AR(2) + 3-regime, regular dt (σ=0) 训练 |
| Teacher | CfC (h=32) 或 hybrid_gate (h=32) |
| Student | CfC (h=8) — best edge config (N1) |
| Test dt | σ ∈ {0.0, 0.5, 1.0}（0.5 = in-dist irregular，1.0 = OOD） |
| Stage | 蒸馏 + int8 量化 |

**关键问题**：N12 已知 retention (TFP / hybrid_gate) 在 OOD dt 下退化，**int8 quantization error 是否会复合 OOD sensitivity**？

## 2. Benchmark 结果

### 2.1 CfC teacher + int8 student

| test dt σ | fp32 MSE | int8 MSE | **delta** |
|---:|---:|---:|---:|
| 0.0 (regular) | 0.0519 | 0.0519 | **-0.0000** |
| 0.5 (in-dist irregular) | 0.0527 | 0.0527 | **+0.0000** |
| **1.0 (OOD irregular)** | 0.0537 | 0.0537 | **+0.0000** |

数据：[`analysis/jetson/2026-08-05_int8_ood_dt_cfc.{md,json}`](analysis/jetson/2026-08-05_int8_ood_dt_cfc.md)

### 2.2 hybrid_gate teacher + int8 student

| test dt σ | fp32 MSE | int8 MSE | **delta** |
|---:|---:|---:|---:|
| 0.0 (regular) | 0.0520 | 0.0520 | **+0.0000** |
| 0.5 (in-dist irregular) | 0.0526 | 0.0526 | **+0.0000** |
| **1.0 (OOD irregular)** | 0.0535 | 0.0535 | **+0.0000** |

数据：[`analysis/jetson/2026-08-05_int8_ood_dt_hybrid_gate.{md,json}`](analysis/jetson/2026-08-05_int8_ood_dt_hybrid_gate.md)

## 3. 关键发现

### 3.1 int8 free-lunch 跨 OOD dt 仍成立

**所有 6 个配置 (2 teachers × 3 dt distributions) 的 int8 vs fp32 delta 都在 ±0.0000 内**——浮点精度内。

→ **int8 quantization 的 free-lunch 4.0× compression 不受 dt 分布影响**——即使 retention 在 OOD dt 下敏感，int8 量化 error 仍可忽略。

### 3.2 Why？

**Hypothesis**：
- int8 quantization error ≤ scale/2 per weight
- scale = wmax / 127，对 LNN weights 通常 0.01-0.1
- 量化 error 在 [1e-4, 1e-2] per weight
- **retention's OOD sensitivity** 来自 dt 分布与训练分布 mismatch，**不**来自 weight precision

→ **int8 error 是 local weight precision，retention OOD error 是 global dt-distribution shift**——两个 error source **不交互**，所以 int8 不会复合 OOD sensitivity。

### 3.3 对工业部署的 practical impact

| 场景 | 累计压缩 vs CfC teacher (h=32, fp32) | MSE delta | 推荐 |
|---|---:|---:|---|
| Teacher (h=32, fp32) | 1.0× | 0 | 训练 |
| Student (h=8, fp32, in-dist) | 14.53× | -0.0001 | N1 baseline |
| Student (h=8, int8, in-dist) | 58.13× | -0.0000 | N20 + N1 |
| **Student (h=8, int8, OOD dt)** | **58.13×** | **+0.0000** | **N20+N23 推荐** |

→ **For edge deployment under variable sensor sampling rates**: 58.13× compression, zero accuracy loss, OOD dt robust。

## 4. 实用 take-away

| 场景 | 推荐 |
|---|---|
| Edge deployment + variable sensor sampling rate | **CfC h=32 → distill to h=8 → int8** |
| Edge deployment + known stable sampling rate | hybrid_gate teacher (N19, +67% compression) + int8 |
| Maximum compression | hybrid_gate + h=4 student + int8 (**97.16×**) |

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N23** | int8 × irregular dt | ✅ **本轮关闭（strong positive）** |
| N21 | hybrid_gate teacher × hybrid_gate student | ⏳ 路线图 |
| N18 | CfC 在真实数据集上 | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 6. 推荐后续动作

1. **下周**：N18 CfC 在真实数据集（UCR/MIMIC/金融时序）上验证
2. **下周**：N21 hybrid_gate teacher × hybrid_gate student round-trip distillation
3. **路线图**：N2 / L4 foundational gap 收尾

## 7. 数据源回链

- 代码
  - [`scripts/bench_int8_ood_dt.py`](scripts/bench_int8_ood_dt.py)（164 lines, with custom forward wrapper for per-step dt）
- Benchmark
  - [CfC teacher + int8 + OOD dt](analysis/jetson/2026-08-05_int8_ood_dt_cfc.md)
  - [hybrid_gate teacher + int8 + OOD dt](analysis/jetson/2026-08-05_int8_ood_dt_hybrid_gate.md)
- 上轮对照
  - [[Int8_Quantization_N20_DLNet_Stage3_2026-08-05]]（N20 in-dist 4.0× compression）
  - [[DT_Distribution_Shift_N12_Hybrid_Gate_Transferability_2026-08-05]]（N12 retention OOD degradation）
