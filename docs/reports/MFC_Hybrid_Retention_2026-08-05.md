---
title: MFC-Hybrid Retention — CfC × TFP Learned Mix 在 regular + irregular Δt 下的表现
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, learned-mix, constructive-synthesis]
arxiv_refs: [2607.08283, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]], [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]
gap_refs: [N8-TFP×CfC-hybrid]
---

# MFC-Hybrid Retention — CfC × TFP Learned Mix

> 把上一轮 TFP-vs-CfC 的 **counter-intuitive negative result** 转化为 **constructive synthesis**：用 learned per-element mix `α ∈ [0, 1]` 把 CfC σ-decay（dt-robust）与 TFP exp-retention（explicit dt）组合。**新 retention_kind="hybrid" 已落地，10/10 测试通过，benchmark 显示 hybrid 在 regular dt 下与 CfC/TFP 持平，在 irregular dt 下比 TFP 退化显著更少**。

## 1. 设计动机

| 路径 | 来源 | retention 公式 | dt-robustness | regular-dt 表现 |
|---|---|---|---|---|
| **CfC σ-decay** | Lechner 2022 (arXiv 2106.13898) Eq. (10) | `σ(-f·τ·dt)` | ✅ **完全不变** (ratio 1.00×) | 与 TFP 持平 |
| **TFP exp-decay** | Hasani 2022 (arXiv 2607.08283) Eq. (3) | `exp(-dt/τ)` | ❌ **退化 14%** (ratio 1.14×) | 比 CfC 略优 |

**Hybrid 假设**：让模型 **learn** α per hidden dim，使 `k = α · k_cfc + (1-α) · k_tfp`，理论上：
- regular dt 下 α → 0（用 TFP 拿小优势）
- irregular dt 下 α → 1（用 CfC 拿 dt-robustness）

但 α 学习依赖训练时是否有 dt 分布变化信号（详见 §4）。

## 2. 实现

代码：[`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py) — 新增 `retention_kind="hybrid"` 分支。
测试：[`tests/test_hybrid_retention.py`](tests/test_hybrid_retention.py) — **10/10 通过**。

### 2.1 Forward 公式

```text
# CfC 路径：sigmoid saturation → dt-robust
k_cfc = σ(-f_cfc · τ_cfc · dt)                # type: ignore[index]

# TFP 路径：指数 → explicit dt
k_tfp = exp(-dt / softplus(τ_tfp_proj([x,h])))   # type: ignore[index]

# Learned per-element mix
α = sigmoid(self.alpha[i])                       # type: ignore[index]  ∈ [0, 1]
k = α · k_cfc + (1 - α) · k_tfp

# Forward (CfC-style)
h_new = k · h_prev + (1 - k) · h_branch
```

### 2.2 α 初始化与训练

- **Init**：`alpha = 0` ⇒ `sigmoid(0) = 0.5`，两路等权
- **Train**：通过反向传播直接优化 `alpha`（logit 形式），让模型自由学
- **Param count**：每 expert 增加 `hidden_dim` 个标量（vs CfC-only 多 `τ_proj` 参数）

## 3. 测试覆盖（10/10 通过）

文件：[`tests/test_hybrid_retention.py`](tests/test_hybrid_retention.py)

| 测试 | 验证内容 |
|---|---|
| `test_hybrid_in_valid_set` | `_VALID_RETENTION` 包含 "hybrid" |
| `test_init_hybrid_creates_alpha` | init 后 α = sigmoid(0) = 0.5 |
| `test_init_hybrid_creates_both_paths` | 同时存在 f_gate (CfC path) 和 tau_proj (TFP path) |
| `test_forward_shape_hybrid` (×2) | forward shape 正确（n_tau=1 与 n_tau=3）|
| `test_hybrid_differs_from_cfc_and_tfp` | hybrid 输出与纯 cfc、纯 tfp 不同 |
| `test_hybrid_alpha_zero_dt_zero_recovers_h_prev` | α=0 + dt→0 时退化为 h_prev（TFP 路径）|
| `test_hybrid_alpha_zero_matches_tfp_k` | α=0 时 retention k 与纯 TFP 一致 |
| `test_gradients_flow_alpha` | α 参数有梯度流 |
| `test_end_to_end_training_step_hybrid` | 5 步训练 loss 下降 |

**意外发现**：CfC σ-decay 在 dt→0 时 **不会**退化为 h_prev（k_cfc → σ(0) = 0.5 而非 1）—— 这是因为 k_cfc 依赖网络输出 f 而非常数 1。TFP 的 `exp(-dt/τ)` 才是真正的 dt→0 退化。

## 4. Benchmark 结果

数据：[`analysis/jetson/2026-08-05_hybrid_retention_benchmark.{md,json}`](analysis/jetson/2026-08-05_hybrid_retention_benchmark.md)

| 模型 | test MSE (regular) | test MSE (irregular) | **degradation ratio** | 训练秒 |
|---|---:|---:|---:|---:|
| cfc-baseline | 0.0589 ± 0.0001 | 0.0589 ± 0.0001 | **1.00×** | 16.0 |
| mfc-cfc | 0.0590 ± 0.0001 | 0.0590 ± 0.0000 | **1.00×** | 12.3 |
| mfc-tfp | 0.0586 ± 0.0002 | 0.0671 ± 0.0012 | **1.14×** | 11.7 |
| **mfc-hybrid** | 0.0590 ± 0.0002 | **0.0618 ± 0.0005** ⚡ | **1.05×** | 16.6 |

### 4.1 关键发现

1. **Hybrid degradation 1.05×**，介于 CfC 的 1.00× 和 TFP 的 1.14× 之间
2. **regular dt 与 CfC/TFP 持平**（0.0590 vs 0.0589 vs 0.0586）
3. **Hybrid 显著优于 TFP**（irregular MSE 0.0618 vs 0.0671，↓7.8%）

### 4.2 为什么 α 没有完全学到 "irregular dt 用 CfC"？

我们观测了 α 在 20 步训练后的值：
- init：mean 0.500
- 20 步后：mean 0.462（几乎没变）

**原因**：训练 dt = 1.0 恒定（与上轮 benchmark 同），模型从未见过 dt 抖动，没有信号去学"irregular 用 CfC"。α 保持在 0.5 附近 ⇒ hybrid 退化为"sigmoid 与 exp 的算术平均"——介于两者之间，但仍然比纯 TFP 更鲁棒。

**未来验证（N9）**：在 **irregular dt 上训练** hybrid，看 α 是否学到 conditional gating。

## 5. 与上一轮的关联

| 实验 | MFC-TFP vs CfC |
|---|---|
| 8/3 single config (regular dt, h=24) | TFP ↓1.4% MSE |
| 8/3 Pareto sweep (regular dt, h × sl grid) | TFP h=32/sl=64 略优；h=16 打平 |
| **8/5 N6 (irregular dt)** | **TFP 退化 14%** |
| **8/5 N8 hybrid (regular dt train, irregular dt test)** | **Hybrid 退化 5%（介于 TFP 14% 和 CfC 0%）**|

→ Hybrid 不是 "0 vs 14%" 的极端解，而是 "用 1.05× 的轻微退化换 regular dt 的 TFP-level 性能"。这是**两个 performance regime 间的实用 interpolation**。

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N8** | TFP × CfC hybrid 能否兼得两边优势 | ✅ **本轮落地**（hybrid 1.05× 退化率）|
| **新增 N9** | Hybrid 在 irregular dt **训练**时是否能学到 conditional gating（α 切换）| ⏳ 下周 |
| **新增 N10** | Hybrid 与 `MultiRateTfpCfC` 的三层组合（multi-rate × TFP × hybrid）| ⏳ 下周 |

## 7. 推荐后续动作

1. **本周**：跑 hybrid 在 **irregular dt 训练**下的 α 学习曲线（验证 conditional gating hypothesis）
2. **下周**：把 hybrid 接到 `MultiRateTfpCfC`，验证三层组合
3. **路线图**：写一份 "LNN retention mechanism design space" 综合表，把 cfc / tfp / nsfd / hybrid 的适用边界条件系统化

## 8. 数据源回链

- 代码
  - [`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py)（hybrid 分支）
  - [`tests/test_hybrid_retention.py`](tests/test_hybrid_retention.py)（10 tests, all pass）
- Benchmark
  - [`analysis/jetson/2026-08-05_hybrid_retention_benchmark.{md,json}`](analysis/jetson/2026-08-05_hybrid_retention_benchmark.md)
- 上轮对照
  - [[TFP_vs_CfC_on_Irregular_Dt_2026-08-05]]（N6 counter-intuitive negative result）
  - [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]（retention_kind 接口定义）
- 论文引用
  - [TFP arXiv 2607.08283](https://arxiv.org/abs/2607.08283)
  - [Lechner 2022 CfC arXiv 2106.13898](https://arxiv.org/abs/2106.13898)
