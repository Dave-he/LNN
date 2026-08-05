---
title: hybrid_gate Student Distillation (N21) — N19 仍最优：CfC student 胜 hybrid_gate student
date: 2026-08-05
tags: [LNN, hybrid_gate, student, distillation, N21, negative-result-for-round-trip, N19-still-best]
arxiv_refs: [2601.06227, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[Hybrid_Gate_Teacher_Distillation_N19_2026-08-05]], [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]]
gap_refs: [N21-hybrid-gate-student]
---

# hybrid_gate Student Distillation (N21) — N19 仍最优

> N19 发现 hybrid_gate teacher 比 CfC teacher 更易压缩（h=4 student 24.29× vs 14.53×）。本轮 N21 验证 **hybrid_gate student** 是否进一步提升压缩比。**结论：NEGATIVE for round-trip**——**CfC student 仍是 best choice**。hybrid_gate student 的额外 complexity（α MLP + τ_proj + f_gate）在小 hidden 下是 overhead。

## 1. 实验设计

| Configuration | Teacher | Student | h=4 | h=8 | h=12 | h=16 |
|---|---|---|---:|---:|---:|---:|
| **N1 (N21 baseline)** | CfC | CfC | 14.53× | 6.10× | 3.50× | 2.31× |
| **N19 (N21 baseline)** | hybrid_gate | CfC | 24.29× | 10.20× | 5.86× | 3.86× |
| **N21 (本轮)** | hybrid_gate | **hybrid_gate** | 16.16× | 6.70× | 3.81× | 2.50× |
| N21 comparison | CfC | hybrid_gate | 11.71× | 4.86× | 2.76× | 1.81× |

## 2. Benchmark 结果

### 2.1 4 个 teacher-student 配置对比

| Teacher→Student | h=4 comp | h=4 MSE δ | h=8 comp | h=8 MSE δ | h=12 comp | h=12 MSE δ | h=16 comp | h=16 MSE δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CfC→CfC (N1)** | 14.53× | +0.0061 | 6.10× | -0.0001 | 3.50× | -0.0001 | 2.31× | -0.0008 |
| **hybrid_gate→CfC (N19)** | **24.29×** | **-0.0001** | **10.20×** | -0.0003 | **5.86×** | -0.0002 | **3.86×** | **-0.0010** |
| **hybrid_gate→hybrid_gate (N21)** | 16.16× | +0.0129 | 6.70× | +0.0030 | 3.81× | +0.0003 | 2.50× | -0.0000 |
| **CfC→hybrid_gate (N21 cmp)** | 11.71× | +0.0114 | 4.86× | +0.0053 | 2.76× | +0.0023 | 1.81× | +0.0001 |

数据：
- [N21 hybrid_gate→hybrid_gate](analysis/jetson/2026-08-05_distillation_hybrid_gate_to_hybrid_gate.md)
- [N21 CfC→hybrid_gate](analysis/jetson/2026-08-05_distillation_cfc_to_hybrid_gate.md)

## 3. 关键发现

### 3.1 N19 (hybrid_gate teacher → CfC student) 仍是 BEST

| h | **N19 compression** | N21 compression (round-trip) | Δ |
|---:|---:|---:|---:|
| 4 | **24.29×** | 16.16× | -33% |
| 8 | **10.20×** | 6.70× | -34% |
| 12 | **5.86×** | 3.81× | -35% |
| 16 | **3.86×** | 2.50× | -35% |

→ **N19 配置 (hybrid_gate teacher + CfC student) 压缩比 N21 (round-trip) 高 33-35% across all h**。

### 3.2 Why？hybrid_gate student 在 distillation 中退化

- hybrid_gate student 的额外 complexity（α MLP + τ_proj + f_gate）让 student **更大**（h=4 hybrid_gate: 416 params vs CfC: 249 params），但**不更准**（h=4 hybrid_gate delta +0.0129 vs CfC -0.0001）
- 小 hidden (h=4) hybrid_gate student **capacity 不足**：α MLP + τ_proj 需要 capacity，hidden 4 不够
- **CfC student 容量效率更高**：纯 CfC 路径不需要 routing 决策

### 3.3 N21 vs N1 (CfC→CfC) 对比

| h | N1 (CfC→CfC) compression | N21 (hybrid_gate→hybrid_gate) compression | Δ |
|---:|---:|---:|---:|
| 4 | 14.53× | 16.16× | +11% |
| 8 | 6.10× | 6.70× | +10% |
| 12 | 3.50× | 3.81× | +9% |
| 16 | 2.31× | 2.50× | +8% |

→ N21 比 N1 略好 (8-11% more compression) but **MSE delta worse** (N1 -0.0001, N21 +0.0129)

### 3.4 N21 的双面发现

1. **Teacher dimension** (N19): hybrid_gate teacher > CfC teacher
2. **Student dimension** (N21): CfC student > hybrid_gate student

→ **Best combination: hybrid_gate teacher → CfC student** (N19, 24.29× at h=4)

## 4. 实用 take-away（修订最终版）

| 场景 | Teacher | Student | 累计压缩 |
|---|---|---|---:|
| **In-dist distillation** | **hybrid_gate** (N19) | **CfC** | **24.29× at h=4** |
| 最大压缩（已知 distribution） | hybrid_gate (N19) | CfC + int8 (N20) | **97.16× at h=4** |
| Robust across dt distribution | CfC (N12+N16) | CfC + int8 | 58.13× at h=8 |
| **Default recommendation** | **CfC (universal)** | **CfC + int8** | **58.13× at h=8** |

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N21** | hybrid_gate student round-trip distillation | ✅ **本轮关闭（N19 仍 best）** |
| N18 | CfC 在真实数据集上 | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 6. 推荐后续动作

1. **下周**：N18 CfC 在真实数据集（UCR/MIMIC/金融时序）上验证——把 21 轮合成任务研究推到现实
2. **路线图**：N2 / L4 foundational gap 收尾

## 7. 数据源回链

- 代码
  - [`lnn/core/distillation.py`](lnn/core/distillation.py)（新增 `student_retention_kind` 参数 + init validation）
  - [`tests/test_distillation_round_trip.py`](tests/test_distillation_round_trip.py)（7 tests, all pass）
  - [`scripts/bench_distillation.py`](scripts/bench_distillation.py)（新增 --student-retention 参数）
- Benchmark
  - [N21 hybrid_gate→hybrid_gate](analysis/jetson/2026-08-05_distillation_hybrid_gate_to_hybrid_gate.md)
  - [N21 CfC→hybrid_gate](analysis/jetson/2026-08-05_distillation_cfc_to_hybrid_gate.md)
- 上轮对照
  - [[Hybrid_Gate_Teacher_Distillation_N19_2026-08-05]]（N19 仍 best）
  - [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]]（N1 baseline）
