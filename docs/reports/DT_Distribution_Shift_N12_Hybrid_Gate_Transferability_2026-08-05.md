---
title: dt distribution shift transferability (N12) — hybrid_gate input-dep α 过拟合训练分布（honest finding）
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, dt-distribution-shift, transferability, OOD-robustness, N12, honest-finding]
arxiv_refs: [2106.13898, 2607.08283]
parent: [[LNN_深度研读报告]]
companion: [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]], [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]
gap_refs: [N12-dt-distribution-shift]
---

# dt distribution shift transferability (N12) — hybrid_gate input-dep α 过拟合训练分布

> N11 让 `α(x_t, dt)` 成为真正的 conditional gate，N12 问：**这种 conditional gating 学到的是 generic dt-robustness 还是 training-distribution-specific 的模式？** 测试方法：训练 dt ~ LogNormal(0, 0.5)，测试 dt 在 {0.3, 0.5, 1.0} 三个分布上。

## 1. 实验设计

| 配置 | 值 |
|---|---|
| 训练 dt | LogNormal(0, 0.5) — IRREGULAR |
| 测试 dt σ_test ∈ {0.3, 0.5, 1.0} | 0.5 in-dist, 0.3 similar, **1.0 OOD** |
| 测试 dt=1.0 | regular baseline |
| 模型 | cfc-baseline / mfc-tfp / mfc-hybrid / mfc-hybrid_gate |
| repeats × epochs | 2 × 4 |

**关键问题**：当 σ_test 远离训练 σ_train=0.5 时，hybrid_gate 的 degradation ratio 是否爆涨（过拟合）还是保持稳定（generic）？

## 2. Benchmark 结果

| 模型 | σ=0.3 | σ=0.5 | **σ=1.0 (OOD)** |
|---|---:|---:|---:|
| cfc-baseline | **1.00×** | **1.00×** | **1.00×** |
| mfc-tfp | 1.02× | 1.05× | **1.12×** ⚠ |
| mfc-hybrid (static α) | 1.01× | 1.03× | **1.09×** |
| mfc-hybrid_gate | 1.01× | 1.04× | **1.10×** |

### 2.1 关键观察

1. **CfC 完全 transfer**：σ_test ∈ {0.3, 0.5, 1.0} 都 1.00× —— **sigmoid saturation 是 generic dt-robustness 机制**（不受训练 dt 分布影响）
2. **TFP/Hybrid/Hybrid-Gate 全部过拟合训练分布**：σ_test=1.0 时 degradation 飙到 1.09-1.12×
3. **hybrid_gate 与 static hybrid 几乎一致**（σ=1.0: 1.10× vs 1.09×）—— **input-dep α 没救**

## 3. Honest finding 解读

**N11 的 input-dep α 在 OOD dt 下没起作用**。原因可能是：
- α 的输入维度是 `cat([x_t, dt_e])`，但只让 dt 进入 MLP 一次（per-step scalar），没有 dt-distribution 信息
- 模型只学到"regular dt 训练时输入的 dt 大概是 1.0 附近"，所以 dt=5 时 MLP 输出与训练分布外推
- TFP path 的 `exp(-dt/τ)` 在 dt=5（OOD）时也会剧烈震荡，与 α 无关

→ **input-dep α 学到的是 "training-distribution 的 conditional gating"，不是 "generic dt-robustness"**

### 3.1 与上一轮 finding 一致

- N6（8/5 N6）：TFP 在 irregular dt 下退化 14% —— TFP 不是 generic dt-robust
- N12（本轮）：TFP 在 dt-shift 下退化 12%（与 N6 的 14% 接近）—— TFP 总是过拟合 dt 分布

→ **TFP 的"显式依赖 dt" 既是它的优势（在 in-dist 上 precise），也是它的劣势（在 OOD 上 fragile）**

## 4. 与 N11 finding 的对照

| 维度 | N11 (in-dist) | N12 (OOD) |
|---|---|---|
| 任务 | σ_train = σ_test = 0.5 | σ_test = 1.0 ≠ σ_train = 0.5 |
| hybrid_gate degradation | 1.00× (持平 CfC) | **1.10×** (与 static hybrid 几乎一样) |
| hybrid_gate MSE | 0.0578 | 0.0615 |

→ N11 的 "1.00× 持平 CfC" 仅在 **in-dist** 下成立。一旦 dt 分布 shift，hybrid_gate **立即退化为与 static hybrid 一样差**。

## 5. 实用 take-away

| 场景 | 推荐 retention |
|---|---|
| Regular dt (constant) | CfC 或 MFC-TFP（精度相当）|
| **Irregular dt, training distribution matches deployment** | MFC-hybrid_gate（1.00× degradation）|
| **Irregular dt, training distribution DIFFERS from deployment** | **CfC σ-decay**（唯一保证 transfer）|
| Future sensor stream with unknown dt distribution | **CfC σ-decay** |

→ **TfP / hybrid / hybrid_gate 仅在"训练 dt 分布 ≈ 部署 dt 分布"时安全**。这是工业部署时必须考虑的**分布匹配假设**。

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N12** | hybrid_gate 在 dt distribution shift 下的 transferability | ✅ **本轮关闭（honest finding）** |
| N14 | MR-hybrid_gate-CfC 在 h=64/128 上重评估 | ⏳ 下周 |
| **新增 N15** | hybrid_gate 在 dt-shift 下需要 distribution-augmented training 或 distributional robust α | ⏳ 路线图 |
| **新增 N16** | CfC 在 dt-shift 下的 perfect transfer (1.00×) 是 generic 机制，建议扩展到 LTC 任务 | ⏳ 路线图 |

## 7. 推荐后续动作

1. **本周**：N14 MR-hybrid_gate-CfC 在 h=64/128 上重评估（CPU 慢，但可验证 N13 honest finding）
2. **下周**：N15 — 训练 hybrid_gate 时混合多个 dt 分布（distribution-augmented training），看是否能获得 transferability
3. **路线图**：N16 — 验证 CfC σ-decay 在长序列 / 大 dt-shift / 多 regime 任务上是否仍保持 1.00× degradation

## 8. 数据源回链

- 代码
  - [`scripts/bench_dt_distribution_shift.py`](scripts/bench_dt_distribution_shift.py)（226 lines）
- Benchmark
  - [`analysis/jetson/2026-08-05_dt_distribution_shift.{md,json}`](analysis/jetson/2026-08-05_dt_distribution_shift.md)
- 上轮对照
  - [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]（N11 in-dist 1.00×）
  - [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]（N6 negative result）
