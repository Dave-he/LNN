---
title: TFP retention vs CfC σ-decay on irregular Δt — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, retention, irregular-dt, robustness, dt-explicit]
---

# TFP retention vs CfC σ-decay on irregular Δt — 2026-08-05

## 任务
合成 **非平稳 AR(2) + 3-regime** 时间序列（与上轮 benchmark 同 task）。
**关键差异**：训练 dt = 1.0（恒定），测试 dt ~ LogNormal(0, 0.5)（jittered）。
验证 TFP 论文 (arXiv 2607.08283) 的核心 claim："retention 显式依赖 dt → 对 dt 分布变化更鲁棒"。

## 结果（3 次重复 mean±std）

| 模型 | 测试 MSE (regular dt) | 测试 MSE (irregular dt) | **degradation ratio** | 训练秒 |
|---|---:|---:|---:|---:|
| cfc | 0.0589 ± 0.0001 | 0.0589 ± 0.0001 | **1.00×** | 21.48 |
| mfc-cfc | 0.0590 ± 0.0001 | 0.0590 ± 0.0000 | **1.00×** | 37.61 |
| mfc-tfp | 0.0586 ± 0.0002 | 0.0671 ± 0.0012 | **1.14×** | 37.76 |

## 解读
- **degradation ratio < 1** = 不规则 dt 下 MSE 比 regular 更低（噪声帮助泛化）
- **degradation ratio ≈ 1** = 不规则 dt 下 MSE 几乎不变（理想鲁棒）
- **degradation ratio >> 1** = 不规则 dt 下 MSE 显著上升（dt 分布依赖）
- TFP 的 retention 显式依赖 dt ⇒ 在 irregular dt 下应当比 CfC σ-decay **更鲁棒**（ratio 更接近 1）。
## Verdict
TBD — see report.
