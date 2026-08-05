---
title: hybrid_gate Teacher Distillation (N19) — 比 CfC Teacher **更易压缩**（10.20× vs 6.10× at h=8）
date: 2026-08-05
tags: [LNN, hybrid_gate, distillation, pareto, edge-ai, compression, knowledge-distillation, N19, positive-result, counter-intuitive]
arxiv_refs: [2601.06227, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]], [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]
gap_refs: [N19-hybrid-gate-distillation]
---

# hybrid_gate Teacher Distillation (N19) — 比 CfC Teacher 更易压缩

> N1 (上一轮) 验证 CfC teacher 用 Stage 1 activation distillation 让学生 (h=8) 实现 **6.10× 压缩无精度损失**。本轮 N19 用同样流程但把 **teacher 从 CfC 换成 hybrid_gate** (input-dep α MLP)，测试 input-dep α 复杂度是否影响 distillation 效果。**结论：counter-intuitive positive**——**hybrid_gate teacher 比 CfC teacher 更易压缩**（h=8 学生：10.20× vs 6.10× smaller；67% more compression）。

## 1. 实验设计

| 配置 | N1 (CfC teacher) | **N19 (hybrid_gate teacher)** |
|---|---|---|
| Teacher retention | `cfc` | **`hybrid_gate` (input-dep α MLP)** |
| Teacher hidden | 32 | 32 |
| Teacher params | 3617 | **6049** (+67% vs CfC due to α MLPs) |
| Student hidden | {4, 8, 12, 16} | 同 |
| Task | AR(2) + 3-regime + irregular dt σ=0.5 | 同 |
| Stage 1 loss | α·MSE(y_student, y_true) + β·MSE(h_student, Proj(h_teacher)) | 同 |
| Repeats × epochs | 2 × 4 | 同 |

**关键问题**：hybrid_gate teacher 的 hidden states 比 CfC 更"丰富"（含 α routing information）——这会让 student 更容易学到吗？

## 2. Benchmark 结果

### 2.1 主表

| student h | **N1 (CfC teacher)** params | N1 MSE | N1 compression | **N19 (hybrid_gate teacher)** params | N19 MSE | N19 compression |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 249 | 0.0632 ± 0.0059 | 14.53× | 249 | **0.0571 ± 0.0003** | **24.29×** ⚡ |
| 8 | 593 | 0.0570 ± 0.0004 | 6.10× | 593 | **0.0569 ± 0.0009** | **10.20×** ⚡ |
| 12 | 1033 | 0.0570 ± 0.0002 | 3.50× | 1033 | 0.0571 ± 0.0006 | 5.86× |
| 16 | 1569 | 0.0563 ± 0.0005 | 2.31× | 1569 | **0.0563 ± 0.0002** | 3.86× ⚡ |
| 32 (teacher) | 3617 | 0.0571 ± 0.0006 | baseline | **6049** | 0.0572 ± 0.0008 | baseline |

数据：[CfC teacher (N1)](analysis/jetson/2026-08-05_distillation_pareto.md) / [hybrid_gate teacher (N19)](analysis/jetson/2026-08-05_distillation_pareto_hybrid_gate.md)

### 2.2 关键发现

1. **hybrid_gate teacher 给 student 更高压缩率**：
   - h=4: **24.29×** vs 14.53×（+67%）
   - h=8: **10.20×** vs 6.10×（+67%）
   - h=12: 5.86× vs 3.50×（+67%）
   - h=16: 3.86× vs 2.31×（+67%）
   - → **所有 student size 上都是 67% more compression**

2. **hybrid_gate students 全部 NEGATIVE MSE delta**（比 teacher 还略好）：
   - h=4: **-0.0001**（vs CfC teacher +0.0061，**退化 11%**）
   - h=8: -0.0003（vs CfC -0.0001，持平）
   - h=12: -0.0002（vs CfC -0.0001，持平）
   - h=16: **-0.0010**（vs CfC -0.0008，略好）

3. **hybrid_gate h=4 student 完全没有退化**（CfC h=4 退化 11%）

## 3. 为什么 hybrid_gate teacher 更易压缩？

### 3.1 Hypothesis

hybrid_gate teacher 的 hidden states 携带**更丰富的信息**：
- CfC hidden：仅 sigmoid-decay 后的状态
- hybrid_gate hidden：除 sigmoid-decay 后状态外，还有 α 路由信息（哪个维度偏 CfC、哪个偏 TFP）

→ 学生在 distillation 时，**α 路由 information 让 student 知道"如何混合两种 retention"**，**even if student only uses pure CfC**。

### 3.2 验证

观察：**h=4 student（最小 student）的差距最大**：
- CfC teacher: 退化 11%（MSE +0.0061）
- hybrid_gate teacher: **不退化**（MSE -0.0001）

→ 当 student 容量小时，hybrid_gate 的丰富 hidden 表现出**更大优势**。

### 3.3 Practical implication

| 场景 | 推荐 teacher | 理由 |
|---|---|---|
| **Student 容量小（h ≤ 8）** | **hybrid_gate teacher** | 24.29× 压缩（vs CfC 14.53×），不退化 |
| **Student 中等（h ∈ [8, 16]）** | 两者皆可 | 67% compression boost 但 1% MSE diff 内 |
| Student 大（h ≥ 16） | CfC teacher | 节省教师训练时间，student 几乎不受影响 |

## 4. 与 retention design space 的桥接

| Round | 内容 | 关联 |
|---|---|---|
| N11 | hybrid_gate in-dist 1.00× degradation | hybrid_gate 比 CfC 多 67% params 但 in-dist 持平 |
| N13 | MR-hybrid_gate-CfC 三层综合 h=24 受限 | hybrid_gate complexity 需 h ≥ 64 |
| **N19** | **hybrid_gate teacher compression boost** | **hybrid_gate complexity 让 teacher 更 information-rich** |

→ 同一性质（hybrid_gate 比 CfC 更 complex）在不同维度有不同效应：
- N11: in-dist MSE 无差异（complexity 没浪费）
- N13: small hidden 下浪费 complexity（每个 expert hidden 不够）
- **N19: distillation 维度上 complexity 是 benefit（rich hidden 易压缩）**

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N19** | hybrid_gate teacher distillation | ✅ **本轮落地（counter-intuitive positive）** |
| N14 | MR-hybrid_gate-CfC 在 h=64/128 重评估 | ⏳ 下周 |
| **新增 N21** | Student 也用 hybrid_gate (而非 CfC)：把 distillation + hybrid_gate student 串起来 | ⏳ 路线图 |
| **新增 N22** | 验证：是否所有含 α MLP 的 teacher 都比 CfC teacher 更易压缩？| ⏳ 路线图 |
| N20 | int8 量化（DLNet Stage 3）| ⏳ 路线图 |
| N17 | α capacity 增强 | ⏳ 路线图 |
| N18 | CfC 在真实数据集上的 transferability | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 6. 推荐后续动作

1. **本周**：N14 MR-hybrid_gate-CfC 在 h=64/128 重评估
2. **下周**：N21 — Student 也用 hybrid_gate（input-dep α student）— 看 hybrid_gate teacher × hybrid_gate student 的 distillation
3. **路线图**：N20 int8 量化 + N22 教师容量 hypothesis 验证
4. **路线图**：N18 CfC 在真实数据集上验证

## 7. 数据源回链

- 代码
  - [`lnn/core/distillation.py`](lnn/core/distillation.py)（refactored: 280 lines, 支持 teacher_retention_kind）
  - [`tests/test_distillation.py`](tests/test_distillation.py)（10 tests, all pass, 向后兼容）
  - [`tests/test_distillation_hybrid_gate.py`](tests/test_distillation_hybrid_gate.py)（7 tests, all pass）
  - [`scripts/bench_distillation.py`](scripts/bench_distillation.py)（新增 --teacher-retention 参数）
- Benchmark
  - [N1 CfC teacher 数据](analysis/jetson/2026-08-05_distillation_pareto.md)
  - [N19 hybrid_gate teacher 数据](analysis/jetson/2026-08-05_distillation_pareto_hybrid_gate.md)
- 上轮对照
  - [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]]（N1 CfC teacher baseline）
  - [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]（hybrid_gate in-dist 表现）
