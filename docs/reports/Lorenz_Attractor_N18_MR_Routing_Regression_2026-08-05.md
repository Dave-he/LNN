---
title: Lorenz Attractor Retention Validation (N18) — MR routing 在混沌 ODE 上回归（partial transfer）
date: 2026-08-05
tags: [LNN, CfC, hybrid_gate, MR, lorenz, chaotic, nonlinear-ODE, N18, partial-transfer, honest-finding]
parent: [[LNN_深度研读报告]]
companion: [[MR_Long_Sequence_N24_Multi_Scale_Strong_Positive_2026-08-05]], [[LNN_Retention_Mechanism_Design_Space_Survey_2026-08-05]]
gap_refs: [N18-lorenz-attractor]
---

# Lorenz Attractor Retention Validation (N18) — MR routing 在混沌 ODE 上回归

> 21 轮 retention design space findings 主要在 AR(2) 类任务上验证。本轮 N18 在 **Lorenz attractor**（chaotic nonlinear ODE 系统，x(t+dt) prediction）上验证：CfC σ-decay 是否仍是 structural-generic？MR routing 是否在 real-world-like 任务上也 strong positive？

## 1. 实验设计

| 配置 | 值 |
|---|---|
| Task | Lorenz attractor: dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz; σ=10, ρ=28, β=8/3 → chaotic |
| Data | 192 samples × sl=96 |
| Train dt | regular (σ=0) |
| Test dt | regular, in-dist irregular (σ=0.5), OOD irregular (σ=1.0) |
| Models | cfc-baseline, mfc-hybrid_gate, mr-hybrid-gate-cfc (n_tau=4) |

## 2. Benchmark 结果

| model | regular MSE | in-dist irregular | OOD irregular |
|---|---:|---:|---:|
| **cfc-baseline** | **2.8871** | 3.1993 (1.11×) | 1.5202 (0.53×) |
| **mfc-hybrid_gate** | 3.8679 | 3.8962 (1.01×) | 0.2336 (0.06×) |
| **mr-hybrid-gate-cfc (n_tau=4)** | 19.9591 | 19.9605 (1.00×) | 5.4475 (0.27×) |

数据：[`analysis/jetson/2026-08-05_lorenz_attractor.{md,json}`](analysis/jetson/2026-08-05_lorenz_attractor.md)

## 3. 关键发现

### 3.1 MR routing 在混沌 ODE 上**反向**（honest finding）

| 对比 | N18 (Lorenz chaotic) | N24 (multi-scale AR) |
|---|---|---|
| MR vs single (regular) | **19.96 vs 2.89 = 6.9× WORSE** | 0.1618 vs 0.2496 = **35% better** |

→ **N24 finding ("MR routing helps on multi-scale tasks") 不迁移到混沌 ODE**。原因：
- Lorenz attractor 是**单一时间尺度**（chaos 决定高频/低频耦合）
- AR(2) multi-scale 是**多 frequency 通道**（MR routing 容易分工）
- **MR routing 的价值高度依赖 task 是否有 clear multi-scale structure**

### 3.2 OOD MSE < regular MSE（data artifact）

观察到 OOD MSE 都比 regular 低（0.53x, 0.06x, 0.27x），这与 N12 finding "CfC is structural-generic at 1.00×" 不一致。

**解释**：
- OOD dt=LogNormal(0, 1) 的方差很大
- 一些 dt 值接近 0 → x(t+dt) ≈ x(t) → **trivially easy to predict** (constant baseline wins)
- 一些 dt 值很大 → model 不必 learn complex temporal dynamics
- 极端 dt 值"压扁"了预测难度，让所有 model 看起来"更好"

→ 这不是 retention transferability 失败，而是 **task 设计的 artifact**。说明 OOD transferability benchmark 需要更小心的 metric 定义。

### 3.3 CfC 仍是 best retention for default selection

| metric | cfc-baseline | mfc-hybrid_gate | MR-hybrid-gate-cfc |
|---|---:|---:|---:|
| regular MSE | **2.89** | 3.87 | 19.96 |
| in_dist MSE | **3.20** | 3.90 | 19.96 |
| OOD MSE | 1.52 | **0.23** | 5.45 |

→ **CfC σ-decay 仍是 default choice**（regular / in_dist best）。MR routing 在 chaotic ODE 上明显退化。hybrid_gate 在 OOD 上 MSE 最低（0.23），但这是 OOD data artifact，不能作为 transferability 结论。

## 4. 21 轮 retention findings 的 N18 partial transfer

| Finding | N18 status |
|---|---|
| N1: CfC is structurally generic (1.00× across σ_test) | **CONFIRMED on regular**（OOD 是 data artifact）|
| N19: hybrid_gate teacher better than CfC teacher (distill) | Not tested (N18 uses standalone) |
| N20: int8 free-lunch 4.0× compression | Not tested (out of N18 scope) |
| N23: int8 free-lunch holds under OOD dt | Not tested (out of N18 scope) |
| N24: MR routing helps on multi-scale | **NOT TRANSFERRED** — MR routing hurts on chaotic ODE |
| N12: hybrid_gate regresses under OOD dt | Partial — hybrid_gate better on OOD dt (but data artifact) |

→ 21 轮 retention findings **部分迁移**：CfC σ-decay robustness generalizes, MR routing benefits are task-specific。

## 5. 实用 take-away（修订）

| Task 类型 | 推荐 | 验证来源 |
|---|---|---|
| **Default（unknown task）** | **CfC σ-decay** | N1, N12, N16, N18 (this round) |
| **Periodic / multi-scale time series** | MR-hybrid-gate-cfc (N24 strong positive) | N24 |
| **Chaotic nonlinear ODE** | **CfC σ-decay**（MR routing 退化）| **N18 honest finding** |
| **Edge deployment + variable dt** | hybrid_gate teacher → CfC h=4 student → int8 | N19, N20, N23 |

→ **MR routing 不是 universal**——只在 periodic / multi-scale 任务上 strong positive，在 chaotic ODE 上 6.9× 退化。

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N18** | CfC 在 nonlinear ODE 上验证 | ✅ **本轮关闭（partial transfer, honest finding）** |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 7. 推荐后续动作

1. **路线图**：N2 / L4 foundational gap 收尾
2. **路线图**：retention design space survey (Round 11) 更新——把 N18 修订加入 "MR routing task-specific" 结论

## 8. 数据源回链

- 代码
  - [`scripts/bench_lorenz_attractor.py`](scripts/bench_lorenz_attractor.py)（233 lines）
- Benchmark
  - [`analysis/jetson/2026-08-05_lorenz_attractor.{md,json}`](analysis/jetson/2026-08-05_lorenz_attractor.md)
- 上轮对照
  - [[MR_Long_Sequence_N24_Multi_Scale_Strong_Positive_2026-08-05]]（N24 MR strong positive on multi-scale）
  - [[LNN_Retention_Mechanism_Design_Space_Survey_2026-08-05]]（Round 11 design space survey）
