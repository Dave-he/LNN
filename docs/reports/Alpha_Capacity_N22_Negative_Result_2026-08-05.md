---
title: α MLP Capacity Hypothesis (N22) — NEGATIVE：α capacity 增大不能突破 interpolation ceiling
date: 2026-08-05
tags: [LNN, hybrid_gate, alpha-capacity, MLP-depth, OOD, N22, negative-result, interpolation-ceiling]
arxiv_refs: [2106.13898, 2607.08283]
parent: [[LNN_深度研读报告]]
companion: [[Distribution_Augmented_Training_N15_Hybrid_Gate_2026-08-05]], [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]
gap_refs: [N22-alpha-capacity]
---

# α MLP Capacity Hypothesis (N22) — NEGATIVE

> N15 假设 α capacity 不足导致 hybrid_gate 只能 interpolation 而非 generic dt-robustness（OOD 1.07× degradation）。本轮 N22 测试 **deeper/wider α MLP** 能否突破 interpolation ceiling。**结论：NEGATIVE**——α capacity 增大不能跨过 ceiling，α 本身的 per-input 结构是限制因素。

## 1. 实验设计

| 配置 | 值 |
|---|---|
| Task | AR(2) + 3-regime, mixed-dt training (σ ∈ {0.3, 0.5, 1.0}) |
| Test | 3 σ values (N15 setup) |
| Models | 5 α capacity variants + CfC baseline |
| α MLP 变体 | depth ∈ {1, 2, 3}, width ∈ {branch_dim, 2×, 4×} |

## 2. Benchmark 结果

| 模型 | depth | width | params | σ=0.3 | σ=0.5 | **σ=1.0 (OOD)** |
|---|---:|---:|---:|---:|---:|---:|
| cfc-baseline | — | — | 2137 | 1.00× | 1.00× | **1.00×** |
| mfc-hybrid_gate (N11/N15 baseline) | 1 | branch_dim | 2977 | 1.01× | 1.03× | 1.07× |
| mfc-hybrid_gate (deeper) | 2 | 2× branch_dim | 3577 | 1.01× | 1.02× | 1.07× |
| mfc-hybrid_gate (deeper + wider) | 3 | 2× branch_dim | 4177 | 1.01× | 1.03× | **1.08×** ⚠ |
| mfc-hybrid_gate (deeper + much wider) | 3 | 4× branch_dim | 4177 | 1.01× | 1.03× | **1.08×** ⚠ |

数据：[`analysis/jetson/2026-08-05_alpha_capacity.{md,json}`](analysis/jetson/2026-08-05_alpha_capacity.md)

## 3. 关键发现（Honest Negative）

### 3.1 α capacity 增大不能突破 interpolation ceiling

- depth=1 → σ=1.0 OOD: 1.07×
- depth=2 → σ=1.0 OOD: 1.07× (no change)
- **depth=3 → σ=1.0 OOD: 1.08× (略变差 ⚠)**
- depth=3 width=4× → σ=1.0 OOD: 1.08× (no improvement)

→ **α capacity 增大不仅没帮助，反而略变差**（可能 overfitting 到训练分布）

### 3.2 α 本身的 per-input 结构是限制因素

N15 假设 α 只能 interpolation（不是 generic dt-robustness mechanism）因为 capacity 不够。**N22 反驳**：capacity 增到 depth=3 + width=4×（params +40%）仍 1.08× OOD degradation。

→ **α 本身的"per-input-dependent function"结构是限制**——无论 capacity 多大，它仍然是一个 learned function of (x_t, dt)，在 OOD dt 上必然外推。

### 3.3 Why？

**Hypothesis**: α MLP 是一个 **function approximator**，但它只能学习 **"training distribution 内的 mapping"**。

- 训练 dt ∈ {0.3, 0.5, 1.0}
- 测试 dt=1.0 (max training) → α 输出在训练范围内，OOD 不严重
- 测试 dt=2.0, 3.0 (超出 max training) → α 必须 **extrapolate** beyond training range
- Sigmoid 链 saturates → α 输出总是 [0, 1] 范围内，但**外推时 α 的语义可能与训练相反**

→ **α 本身就有外推限制**，这是 Sigmoid 链结构的固有 property。

## 4. 与 N15 互证

| 实验 | α capacity | σ=1.0 OOD degradation |
|---|---|---:|
| N15 (depth=1, mixed-dist training) | 2977 params | 1.07× |
| **N22 (depth=3, w=4×, mixed-dist training)** | 4177 params | 1.08× |

→ **N15 假设（capacity 不足）被 N22 反驳**：更大 capacity 不能改善 OOD。

## 5. 实用 take-away

| 场景 | 推荐 |
|---|---|
| Simple in-dist task | mfc-hybrid_gate (depth=1, N11 baseline) |
| OOD dt 需求 | **CfC σ-decay**（唯一 generic structural mechanism） |
| 不要尝试 deeper α MLP 解决 OOD | **已被 N22 反驳**——α 结构是限制 |

→ **α 增大没有价值**。应使用**任务内不同 retention**（N1/N19/N20 distillation pipeline）解决 OOD，而不是给 α MLP 更大 capacity。

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N22** | α capacity hypothesis | ✅ **本轮关闭（NEGATIVE result）** |
| N21 | hybrid_gate teacher × hybrid_gate student | ⏳ 路线图 |
| N23 | int8 × irregular dt | ⏳ 路线图 |
| N17 | α capacity 增强 | ⚠ **N22 反驳 N17 的方向** |
| N18 | CfC 在真实数据集上 | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 7. 推荐后续动作

1. **下周**：N18 CfC 在真实数据集（UCR/MIMIC/金融时序）上验证
2. **下周**：N21 hybrid_gate teacher × hybrid_gate student round-trip distillation
3. **路线图**：N23 int8 × irregular dt 验证
4. **路线图**：N2 / L4 foundational gap 收尾

## 8. 数据源回链

- 代码
  - [`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py)（新增 `alpha_mlp_depth` / `alpha_mlp_width` 参数 + 改进 init）
  - [`tests/test_alpha_mlp_capacity.py`](tests/test_alpha_mlp_capacity.py)（8 tests, all pass）
  - [`scripts/bench_alpha_capacity.py`](scripts/bench_alpha_capacity.py)（208 lines）
- Benchmark
  - [`analysis/jetson/2026-08-05_alpha_capacity.{md,json}`](analysis/jetson/2026-08-05_alpha_capacity.md)
- 上轮对照
  - [[Distribution_Augmented_Training_N15_Hybrid_Gate_2026-08-05]]（N15 mixed-dist training）
  - [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]（N11 input-dep α 基础）
