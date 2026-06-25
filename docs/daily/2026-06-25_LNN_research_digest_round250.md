---
title: LNN 每日研究追踪 - 2026-06-25 (round 250)
date: 2026-06-25
tags: [LNN, daily, automation, frozen-basis, L-RFM, honest-negative]
---

# LNN 每日研究追踪 - 2026-06-25 (round 250, session #85, hourly loop #11)

> 自动生成：聚合 arXiv 2026-06 LNN / CfC / LTC 相关更新。

## 本轮论文 (backlog)

| arXiv ID     | 标题                                                                | 关联             |
|--------------|---------------------------------------------------------------------|------------------|
| 2606.19579  | FlowFake: Liquid Networks for Audio Deepfake Detection             | round 245 backlog |
| 2606.15807  | Memory-Augmented Graph LTC for Traffic Prediction                  | round 248 backlog |

## 选定方向 — Frozen Random Basin CfC

### 背景
- **Round 246**: Frozen random τ — strict win all 3 (-65/-37/-55%)
- **L-RFM paper** (arXiv:2606.15571): random features outperform learned
- **Round 248**: Per-branch LEARNED basin centers — strict win all 3

### 关键问题
**Random features 不需要学，那 random basin centers 呢？**
- Round 246 证明: τ 可以 frozen random
- Round 248 证明: per-branch basin geometry 提供 strict gain
- **Round 250 问题**: 几何中心是否也需要 learned？还是 frozen random 即可？

### 实现 — FrozenRandomBasinCfCCell
- K frozen random τ branches (r246)
- K' × K frozen random basin centers (NEW: r250)
- mix_param learned (作为唯一的结构参数)
- 推论: 如果 frozen basins ≈ learned basins，结构就完全随机 — 全部学习在 mix + output

### Bench 结果 (2026-06-25, 27 cells: 3 ds × 3 modes × 3 seeds, 100 epochs)

| dataset   | baseline | r248 | **r250 frozen_random** | Δ% vs base | Δ% vs r248 | mean_H | H1 | H2 | H3 |
|-----------|----------|------|------------------------|------------|------------|---------|----|----|-----|
| toy_sin   | 0.0060   | 0.0020 | 0.0020              | **-65.7%** | +4.6%      | 0.520   | ✓  | ✓  | ✗   |
| structured| 0.0021   | 0.0011 | 0.0013              | **-37.2%** | **+27.7%** | 0.678   | ✓  | ✗  | ✓   |
| random    | 0.0115   | 0.0048 | 0.0052              | **-54.7%** | +7.9%      | 0.625   | ✓  | ✓  | ✓   |

**注意**: r250 frozen_random_basin 的 test_mse 数字 EXACTLY matches round 246 frozen_sampled (因为 basin centers 是 buffer，从不进入 gradient)

### 结论 — **HONEST TARGET-DEPENDENT-WITH-NUANCE**

- **H1 (safe vs baseline) 3/3 ✓** — strict win all 3
- **H2 (parity with r248) 2/3 ✓** — structured regresses +27.7%, others parity
- **H3 (basins used) 2/3 ✓** — toy_sin collapses to 0.520 < 0.55 threshold

### 关键 insight
**Frozen random basins = frozen_sampled (round 246)** — they reduce to the same model because basin geometry doesn't connect to gradient:
- basin_centers 是 buffer (no gradient)
- V value 是 basin_centers 和 h 的函数，但 lyap_lambda=0.0 所以 V 不进 loss
- alpha_mix 只依赖 mix_param (和 r246 一样)
- → r250 ≈ r246 numerically (test_mse 数字 EXACTLY match)

**Discovery**: frozen random features work for τ (r246) because τ is a structural hyperparameter that DIRECTLY affects cell dynamics. Frozen random features DON'T work for basin geometry because basin geometry only enters the model through the loss function — if the loss doesn't use it, it's dead weight.

**The geometry MUST be connected to gradient to help task loss** — frozen random basins work as **diagnostic** (V_mean, basin_H) but not as **task-loss mechanism**.

### 建议动作 (后续 round)
- **Round 251**: 输入条件 gate on r250 (input + frozen-V gate) — 测试 frozen-V 是否能给 input gating 提供 signal
- **Round 252**: aux loss ENABLE — 启用 lyap_lambda > 0 让 frozen basins 通过 aux 影响梯度
- **下游候选**: r248 的 learned basins + r249 的 gate + r250 frozen 用于诊断

## PR 候选
`lnn/core/frozen_random_basin_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`