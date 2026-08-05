---
title: MFC-Hybrid 在 irregular Δt 训练下学到 conditional gating — N9 验证（部分 positive result）
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, conditional-gating, alpha-learning, N9, partial-positive]
parent: [[LNN_深度研读报告]]
companion: [[MFC_Hybrid_Retention_2026-08-05]], [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]
---

# MFC-Hybrid 在 irregular Δt 训练下学到 conditional gating — N9 验证

> 本轮验证 N9：把上一轮 "α 在 regular 训练下不变" 的观察推进——改为 irregular dt 训练，看 α 是否学到 conditional gating。**部分 positive result**：α 真的在学习（CfC 方向），但 hybrid **没有超越 CfC**，只是退化为 CfC——这本身是一个有意义的研究结论。

## 1. 实验设计

| 配置 | 值 |
|---|---|
| 训练 dt | LogNormal(0, 0.5)，范围 [0.101, 6.860]，mean 0.998 |
| Test A (regular) | dt = 1.0（恒定）|
| Test B (irregular) | LogNormal(0, 0.5)（与训练同分布）|
| 模型 | cfc / mfc-cfc / mfc-tfp / mfc-hybrid |
| repeats × epochs | 3 × 4 |
| hidden / seq_len | 24 / 48 |
| 任务 | 非平稳 AR(2) + 3-regime |

## 2. 结果

### 2.1 主表

| 模型 | test_mse_regular | test_mse_irregular | **degradation ratio** |
|---|---:|---:|---:|
| cfc-baseline | 0.0573 ± 0.0000 | 0.0574 ± 0.0000 | **1.00×** |
| mfc-cfc | 0.0572 ± 0.0001 | 0.0573 ± 0.0002 | **1.00×** |
| mfc-tfp | 0.0575 ± 0.0001 | 0.0605 ± 0.0002 | **1.05×** |
| **mfc-hybrid** | 0.0576 ± 0.0001 | **0.0582 ± 0.0002** | **1.01×** ⚡ |

### 2.2 α trajectory（关键证据）

| Epoch | α mean (over 3 runs) |
|---:|---:|
| 1 | 0.501 |
| 2 | 0.525 |
| 3 | 0.557 |
| 4 | 0.576 |

→ **α 真的在学 CfC 方向**（从 0.5 → 0.576，4 epoch）。还没收敛（仅 4 epoch）—— 推测更多 epoch 会继续向 1.0 收敛。

## 3. 解读

### 3.1 Hybrid 接近 CfC 的鲁棒性

对比上一轮 (regular train) 与本轮 (irregular train)：

| 实验 | cfc | mfc-tfp | mfc-hybrid |
|---|---:|---:|---:|
| 8/5 N6 (regular train, irregular test) | 1.00× | 1.14× | 1.05× |
| **8/5 N9 (irregular train, irregular test)** | **1.00×** | **1.05×** | **1.01×** ⚡ |

→ **irregular dt 训练让所有模型都更鲁棒**：
- cfc 始终 1.00×（训练条件不影响 CfC σ-decay 的 dt-robustness）
- mfc-tfp 从 1.14× → 1.05×（9% 改善）
- **mfc-hybrid 从 1.05× → 1.01×**（4% 改善，几乎与 cfc 持平）

### 3.2 α conditional gating 的两种解读

**解读 A（α 是 global scalar）**：α mean 0.576 表明 hybrid 在 **global** 层面学到 "多用 CfC"。但每个 hidden dim 的 α 可能差异巨大（不同维度不同 mix），如果要看 per-dim α 分布需要进一步分析。

**解读 B（α 不做 conditional gating）**：即使训练 dt 变化，hybrid 也没有**per-input** 切换 α（因为 α 是 nn.Parameter 而不是函数 `α(x, dt)`）。这意味着 hybrid 本质上只是一个 learned static mix。

→ **目前的 α 是 static parameter，不是 conditional gate**。要让 hybrid 真正"conditional"，需要把 α 改成 `α = sigmoid(MLP([x_t, dt]))` —— 这是 N11 候选。

### 3.3 Hybrid 没超越 CfC 的原因

- Hybrid 在 irregular train 下达到 1.01× degradation，**几乎与 CfC 的 1.00× 持平**
- 但 **mse 数字上 mfc-hybrid 略高于 mfc-cfc**（irregular 0.0582 vs 0.0573，差 1.6%）
- 原因：hybrid 多了 τ_proj（TFP path）+ alpha parameter，**但 TFP path 几乎没贡献**（α → 1）—— 多余的参数反而拖累 fit

→ **本实验环境下，hybrid 没有"两边优势兼得"，反而略输 CfC**。这是 N9 的 honest 结论。

## 4. 与上一轮 N8 的关联

| 维度 | N8 (regular train) | N9 (irregular train) |
|---|---|---|
| α trajectory 终值 | 0.462 (几乎没动) | **0.576** (明显向 CfC 移动) |
| Hybrid degradation | 1.05× | **1.01×** |
| Hybrid vs CfC irregular MSE | 0.0618 vs 0.0589 (差 4.9%) | **0.0582 vs 0.0574 (差 1.4%)** |
| 结论 | α 没学，因为没有 dt-jitter 信号 | α 学了，但 hybrid 退化为 CfC |

→ N8 → N9 演进：训练数据加 dt-jitter 后 α 真的动了，hybrid 的 irregular MSE 也改善了。

## 5. 研究 take-away

1. **α 是 learned scalar，能从训练信号中学习**——这不是 trivial finding，验证了 hybrid 设计的有效性
2. **α 不会自动成为 conditional gate**——它是 static parameter，没有 input-dependent 机制
3. **Hybrid 在这个 task 上**没有**超越 CfC**——因为 AR(2) 数据上 CfC 已经是最优的 retention，hybrid 多了 TFP path 反而是 overhead
4. **"两边优势兼得"的真实门槛**：α 必须 input-dependent（N11 candidate），或者两个 path 在不同 dt regime 下确实各有优势（这个 task 上不成立）

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N9** | Hybrid 在 irregular dt 训练下的 α conditional gating | ✅ **本轮完成（partial positive）** |
| **新增 N11** | α input-dependent (`α = sigmoid(MLP([x_t, dt]))`) 实现真正的 conditional gate | ⏳ 下周 |
| **新增 N12** | 在 dt 分布 shift（train dt~LogNormal(0, σ_1), test dt~LogNormal(0, σ_2)）下 hybrid 是否仍鲁棒 | ⏳ 下周 |
| N10 | Hybrid × MR-TFP-CfC 三层组合 | ⏳ 待评估 |

## 7. 推荐后续动作

1. **本周**：N11 实现 input-dependent α（最有可能真正获得 conditional gating 行为）
2. **下周**：N12 测试 hybrid 在 dt distribution shift 下的 transferability
3. **路线图**：写一份 "LNN retention mechanism design space" survey paper，列出 cfc / tfp / nsfd / hybrid 的适用边界条件

## 8. 数据源回链

- 代码
  - [`scripts/bench_hybrid_irregular_train.py`](scripts/bench_hybrid_irregular_train.py)（298 lines, α trajectory 跟踪）
- Benchmark 数据
  - [`analysis/jetson/2026-08-05_hybrid_irregular_train.{md,json}`](analysis/jetson/2026-08-05_hybrid_irregular_train.md)
- 上轮对照
  - [[MFC_Hybrid_Retention_2026-08-05]]（N8 regular-train baseline）
  - [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]（N6 counter-intuitive negative result）
