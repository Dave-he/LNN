---
title: MFC-Hybrid retention (CfC × TFP) on regular AND irregular Δt - 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, robustness, learned-mix, second-order-synthesis]
---

# MFC-Hybrid retention (CfC × TFP) on regular AND irregular Δt - 2026-08-05

## 任务
合成 **非平稳 AR(2) + 3-regime** 时间序列。训练 dt=1.0（恒定），测试 dt 分两种：
- **regular**：dt=1.0（恒定）
- **irregular**：dt ~ LogNormal(0, 0.5)，范围 [0.123, 4.742]

## 结果（3 次重复 mean±std）

| 模型 | test MSE (regular) | test MSE (irregular) | **degradation ratio** | 训练秒 |
|---|---:|---:|---:|---:|

| cfc-baseline | 0.0589 ± 0.0001 | 0.0589 ± 0.0001 | **1.00×** | 16.05 |

| mfc-cfc | 0.0590 ± 0.0001 | 0.0590 ± 0.0000 | **1.00×** | 12.29 |

| mfc-tfp | 0.0586 ± 0.0002 | 0.0671 ± 0.0012 | **1.14×** | 11.67 |

| mfc-hybrid | 0.0590 ± 0.0002 | 0.0618 ± 0.0005 | **1.05×** | 16.56 |


## 解读

**Hybrid 设计动机**（来自上一轮 negative result）：
TFP 在 irregular dt 下退化 14%，CfC σ-decay 完全不变。本轮设计 hybrid 让模型
*learn* CfC 和 TFP 路径的混合权重 α ∈ [0, 1]，理论上可以学到"regular dt 用
TFP、irregular dt 用 CfC" 的 conditional gating。

**Hybrid forward 公式**：
```
k_cfc = sigmoid(-f · τ_cfc · dt)            ← sigmoid saturation, dt-robust
k_tfp = exp(-dt / softplus(τ_tfp))           ← exponential, explicit dt
k     = α · k_cfc + (1-α) · k_tfp           ← learned mix per hidden dim
h_new = k · h_prev + (1-k) · h_branch
```
