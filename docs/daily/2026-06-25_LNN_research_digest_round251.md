---
title: LNN 每日研究追踪 - 2026-06-25 (round 251)
date: 2026-06-25
tags: [LNN, daily, automation, aux-supervision, contraction, frozen-basin, negative-with-nuance]
---

# LNN 每日研究追踪 - 2026-06-25 (round 251, session #86, hourly loop #12)

> 自动生成：聚合 arXiv 2026-06 LNN / CfC / LTC 相关更新。

## 本轮论文 (backlog)

| arXiv ID     | 标题                                                                | 关联             |
|--------------|---------------------------------------------------------------------|------------------|
| 2606.19579  | FlowFake: Liquid Networks for Audio Deepfake Detection             | round 245 backlog |
| 2606.15807  | Memory-Augmented Graph LTC for Traffic Prediction                  | round 248 backlog |

## 选定方向 — Aux-Supervised Frozen Random Basin CfC

### 背景
- **Round 250**: Frozen random basins + lyap_lambda=0 → reduces to r246 (basin geometry never enters gradient)
- **关键 gap**: round 250 数字 EXACTLY matches r246 因为 basin_centers 是 buffer
- **Round 251 修复**: enable lyap_lambda=0.1 让 frozen basins 通过 aux loss 进入 gradient

### 实现 — AuxSupervisedFrozenRandomBasinCfCCell
- 继承 FrozenRandomBasinCfCCell (round 250)
- `forward_with_aux` 默认使用 `default_lyap_lambda=0.1`
- basin_centers 仍是 buffer (no direct grad) — 但 aux loss 通过 cell_k params 进入 gradient

### Bench 结果 (2026-06-25, 36 cells: 3 ds × 4 modes × 3 seeds, 100 epochs)

| dataset   | r248 | r250 (no aux) | **r251 (aux=0.1)** | Δ% vs r248 | Δ% vs r250 | aux first→last | H1 | H2 | H3 |
|-----------|------|---------------|---------------------|------------|------------|----------------|----|----|-----|
| toy_sin   | 0.0020 | 0.0020       | 0.0045              | +128.5%    | +118.6%    | 0.181→0.136    | ✗  | ✓  | ✓   |
| structured| 0.0011 | 0.0013       | 0.0017              | +62.4%     | +27.2%     | 0.193→0.099    | ✗  | ✓  | ✓   |
| random    | 0.0048 | 0.0052       | 0.0049              | **+1.6%**  | **-5.8%**  | 0.714→0.335    | ✓  | ✓  | ✓   |

### 结论 — **HONEST TARGET-DEPENDENT-WITH-NUANCE**

- **H1 (parity with r248) 1/3 ✓** — random 仅 dataset 通过
- **H2 (aux decreasing) 3/3 ✓** — aux loss 在所有 dataset 上都下降
- **H3 (V contraction) 3/3 ✓** — V_next ≤ V_prev × (1-α) 满足

### 关键 insight
**Contraction aux 是 task-loss 杀手 (在 smooth/structured data 上)**:
- toy_sin: +128% vs r248, +119% vs r250 — 强制 sinusoidal data contract 杀死自然周期性
- structured: +62% vs r248, +27% vs r250 — 类似问题
- random: -5.8% vs r250 — random data 没有内在 temporal 结构，contraction 与数据兼容

**Aux supervision 在 DIRECTIONAL sense 上工作 (aux decreases, V contracts) 但 TASK-LOSS sense 上失败**:
- Lyap loss gradient 流向 cell_k params (而非 mix_param，因为 mix 不在 Lyap graph)
- Cell params 调整以满足 contraction，但偏离 task objective
- 这是 "aux prior vs task prior" 的冲突

### Round 250 → 251 Insight
- **r250 (no aux)**: frozen basins 是 dead weight — 与 r246 等价
- **r251 (aux ON)**: frozen basins 通过 aux 路径影响 gradient — 但 aux prior 与 task prior 冲突
- **结论**: frozen basins 不能 task-loss-improvement 路径达成；只有 LEARNED basins (r248) 才能

### 建议动作 (后续 round)
- **Round 252**: 输入条件 gate on r251 (input + aux-V gate) — 测试 aux-V 是否能给 gating 提供 signal
- **Round 253**: learned basins + aux supervision (r248 + lyap_lambda=0.1) — 测试 aux 是否能提升 r248
- **下游候选**: r248 + aux 用于 random 数据（contractable），r249 用于 smooth 数据

## PR 候选
`lnn/core/aux_supervised_frozen_basin_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`