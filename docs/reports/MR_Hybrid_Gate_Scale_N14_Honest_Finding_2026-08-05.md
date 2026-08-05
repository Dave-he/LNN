---
title: MR-hybrid_gate-CfC at h≥64 (N14) — honest finding 加强：MR routing 在 AR(2) 上无优势
date: 2026-08-05
tags: [LNN, MR-MoE, hybrid_gate, multi-rate, scale-up, N14, n_tau=4, honest-finding, ar2-task-saturation]
arxiv_refs: [2606.12240, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[MR_Hybrid_Gate_N13_Three_Layer_Synthesis_2026-08-05]]
gap_refs: [N14-mr-hybrid-gate-scale]
---

# MR-hybrid_gate-CfC at h≥64 (N14) — honest finding 加强

> N13 假设 MR-hybrid-gate-cfc 在 h=24 退化 11% 是因为 per-expert hidden (6) 太小。本轮 N14 验证 h=64 (per-expert=16, N3 Pareto threshold) 下 gap 是否消失。**结论：N13 honest finding 加强——即便 h=64，MR routing 仍未帮助，AR(2) simple task 退化 ~6%**。可能因为 (a) 任务太简单 MR 没有 multi-scale 优势 (b) 数据不够 routing 学到分工 (c) top_k routing overhead 抵消了 multi-rate 优势。

## 1. 实验设计

| 配置 | 值 |
|---|---|
| Task | AR(2) + 3-regime + irregular dt (σ=0.5) |
| Models | CfC (single), mfc-hybrid_gate (single), MR-hybrid-gate-cfc (n_tau=4) |
| Sweep h | {24, 32, 48, 64} |
| per_expert (MR only) | {6, 8, 12, 16} |
| Data | 128 samples × seq_len=24 |
| Epochs × repeats | 3 × 2 |

## 2. Benchmark 结果

| 模型 | h=24 | h=32 | h=48 | h=64 |
|---|---:|---:|---:|---:|
| cfc (single) | 0.0615 | 0.0626 | 0.0643 | 0.0618 |
| mfc-hybrid_gate (single) | 0.0625 | 0.0649 | 0.0634 | **0.0606** ⚡ |
| mr-hybrid-gate-cfc (n_tau=4) | 0.0640 | 0.0638 | 0.0644 | 0.0643 |
| **per_expert (MR)** | 6 | 8 | 12 | 16 |
| **single-mr delta** | +2.4% | -1.7% | +1.6% | **+6.1%** ⚠ |

数据：[`analysis/jetson/2026-08-05_mr_hybrid_gate_scale.{md,json}`](analysis/jetson/2026-08-05_mr_hybrid_gate_scale.md)

## 3. 关键发现

### 3.1 single mfc-hybrid_gate 在 h=64 表现最佳

- h=64: 0.0606（最优）
- h=24: 0.0625
- h=32: 0.0649
- h=48: 0.0634

→ **single expert 在 h=64 达到最佳**——但 h=24 也接近最优（0.0625）。这表明 **AR(2) simple task 有 saturation point**（h=24 接近足够，h=64 略好但不显著）。

### 3.2 N13 honest finding 加强

| h | single mfc-hybrid_gate | MR-hybrid-gate-cfc | delta |
|---:|---:|---:|---:|
| 24 | 0.0625 | 0.0640 | +2.4% |
| 32 | 0.0649 | 0.0638 | -1.7% |
| 48 | 0.0634 | 0.0644 | +1.6% |
| **64** | **0.0606** | **0.0643** | **+6.1%** |

→ **即便 h=64 (per_expert=16)，MR routing 仍未帮助**——在 h=64 上退化反而最大 (6.1%)。

### 3.3 Why？

三个 hypothesis：

**H1: 任务太简单，MR multi-scale 没有发挥空间**
- AR(2) 3-regime 任务的"spectrum"很窄（只有 3 个 AR coefficient sets）
- MR routing 在窄 spectrum 任务上**冗余**——只需 1-2 个 expert 就够

**H2: 数据不够，routing 没学到分工**
- 128 samples × 24 seq_len 的小数据集
- top_k_active=2 的 routing 需要更多数据才能学到 meaningful specialization

**H3: top_k routing overhead 抵消 multi-rate 优势**
- 每 step 做 K×E 次 expert call（CPU 上 overhead 显著）
- 但本研究 h=64 vs h=24 per-step 计算量差不多（branched 切分），所以不是 H3

→ **H1 (任务太简单) + H2 (数据不够) 是主要因素**。

### 3.4 与 N3 Pareto sweep 的对比

N3 (round 282) 给出："multi-rate 需要 h ≥ 64 per expert"——即 per_expert ≥ 16。
N14 在 h=64 (per_expert=16) 验证了这个 threshold，但发现 **MR routing 仍未帮助 single expert**。

→ N3 threshold 是 **MR 优于 trivial baseline** 的 threshold，**不是 MR 优于 sophisticated single expert** 的 threshold。

## 4. 实用 take-away

| 场景 | 推荐 | 理由 |
|---|---|---|
| **AR(2) simple task + h=64** | **single mfc-hybrid_gate (h=64)** | 0.0606（最优）|
| AR(2) simple task + small hidden | single mfc-hybrid_gate (h=24) | 0.0625（接近最优，省参数）|
| Long sequence / multi-scale 任务 | 待 N24 验证（MR 可能发挥）| — |
| 训练数据 < 1000 samples | single expert（MR routing 学不到）| — |
| 训练数据 > 10000 samples | MR-hybrid-gate-cfc 可考虑 | — |

→ **MR routing 不是 free lunch——只在有足够数据 + 真正多尺度任务时才有效**。

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N14** | MR-hybrid-gate-cfc 在 h=64 重评估 | ✅ **本轮关闭（honest finding 加强）** |
| N21 | hybrid_gate teacher × hybrid_gate student | ⏳ 路线图 |
| **新增 N24** | MR routing 在 long sequence / multi-scale 任务上是否发挥 | ⏳ 路线图 |
| N22 | α capacity hypothesis | ⏳ 路线图 |
| N23 | int8 × irregular dt | ⏳ 路线图 |
| N17 | α capacity 增强 | ⏳ 路线图 |
| N18 | CfC 在真实数据集上 | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 6. 推荐后续动作

1. **下周**：N24 — MR routing 在 long sequence (sl≥96) / multi-scale 任务上是否发挥
2. **路线图**：N21 hybrid_gate teacher × hybrid_gate student round-trip
3. **路线图**：N18 CfC 在真实数据集（UCR/MIMIC/金融时序）上验证——AR(2) 之外的任务类型
4. **路线图**：N17 α capacity 增强——验证更大的 α MLP 能否让 hybrid_gate 真的"两全其美"

## 7. 数据源回链

- 代码
  - [`scripts/bench_mr_hybrid_gate_scale.py`](scripts/bench_mr_hybrid_gate_scale.py)（187 lines）
- Benchmark
  - [`analysis/jetson/2026-08-05_mr_hybrid_gate_scale.{md,json}`](analysis/jetson/2026-08-05_mr_hybrid_gate_scale.md)
- 上轮对照
  - [[MR_Hybrid_Gate_N13_Three_Layer_Synthesis_2026-08-05]]（N13 假设 = "small hidden 限制"）
  - [[LNN_Retention_Mechanism_Design_Space_Survey_2026-08-05]]（Round 11 survey 中 N13 honest finding 描述）
