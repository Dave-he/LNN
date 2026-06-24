---
title: LNN 每日研究追踪 - 2026-06-25 (round 249)
date: 2026-06-25
tags: [LNN, daily, automation, gating, per-branch, geometry, routing]
---

# LNN 每日研究追踪 - 2026-06-25 (round 249, session #84, hourly loop #10)

> 自动生成：聚合 arXiv 2026-06 LNN / CfC / LTC 相关更新。

## 本轮论文 (backlog 复用)

| arXiv ID     | 标题                                                                | 关联             |
|--------------|---------------------------------------------------------------------|------------------|
| 2606.19579  | FlowFake: Liquid Networks for Audio Deepfake Detection             | 已在 round 245 backlog |
| 2606.15807  | Memory-Augmented Graph LTC for Traffic Prediction                  | 已在 round 248 backlog |

## 选定方向 — Input + Geometry-Conditioned Per-Branch Gate

### 背景
- **Round 243** (AdaptiveGatedMultiTauCfCCell): input-only gate — **FULL NEGATIVE** (H1 ✗ all 3 datasets)
- **Round 246** (FrozenSampledMultiTauCfCCell): input-blind learned mix — strict win all 3
- **Round 248** (PerBranchMultiBasinLyapunovCfCCell): per-branch basins + input-blind mix — strict win all 3

### Round 249 = Geometry-coupled gating
**InputGeometryGatedPerBranchCfCCell**: gate input = `[x_t, V_1(h_1), ..., V_K(h_K)]`
- 路由取决于 **input + 几何证据** (per-branch Lyapunov value)
- 几何证据来自 round 244/248 的 multi-basin V
- 这是 round 243 input-only gate 的修复版 — 加 geometric evidence 提供 context

### Bench 结果 (2026-06-25, 27 cells: 3 ds × 3 modes × 3 seeds, 100 epochs)

| dataset   | baseline | r248 | **r249 input_geom** | Δ% vs base | Δ% vs r248 | alpha_H | H1 | H2 | H3 |
|-----------|----------|------|---------------------|------------|------------|---------|----|----|-----|
| toy_sin   | 0.0060   | 0.0020 | 0.0018           | -69.5%     | -7.0%      | 0.630   | ✓  | ✓  | ✗   |
| structured| 0.0021   | 0.0011 | **0.0009**      | -58.1%     | **-14.8%**| 1.127   | ✓  | ✓  | ✓   |
| random    | 0.0115   | 0.0048 | 0.0044           | -61.1%     | -7.4%      | 0.934   | ✓  | ✓  | ✗   |

### 结论 — **STRICTLY POSITIVE FIRST-ROUND STRICT WIN (H1+H2)**

- **H1 (safe vs baseline) 3/3 ✓** — strict win all 3 (-58 to -69%)
- **H2 (gate beats r248) 3/3 ✓** — strict improvement over r248 all 3 datasets
- **H3 (gate entropy diverse) 1/3 ✓** — structured passes, toy_sin/random partial collapse
- **NEW BEST on structured**: 0.0009 vs r248 0.0011 (-14.8%)

### 关键 insight
**Geometry-coupled gate 比 input-only gate 工作得好得多**:
- Round 243 (input-only): full negative — input 没有 geometric context 时变成 noise source
- Round 249 (input + V per-branch): strict win — V 提供 geometric evidence 让 gate 有 signal

**Gate entropy collapse 是 target-dependent**:
- structured (multi-frequency): gate uses 3-4 branches (alpha_H 1.13, near log 4 = 1.39)
- toy_sin (single-frequency): gate collapses to 1-2 branches (alpha_H 0.63)
- random (high-entropy): gate uses 2-3 branches (alpha_H 0.93)

**On smooth data, ONE branch dominates** — gate correctly identifies that only one τ fits the smooth signal. On complex data, multiple branches compete.

### PR 候选
`lnn/core/input_geometry_gated_per_branch_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`

## 建议动作 (后续 round)
- **Round 250**: 启用 Lyap aux loss 训练 (r247-249 都关了) — 测试 geometric supervision 是否进一步改善
- **Round 251**: 给 gate 加 entropy regularization (鼓励 diverse routing)
- **下游候选**: 与 QuITE embedding 组合 (round 102-103) for PhysioNet gap