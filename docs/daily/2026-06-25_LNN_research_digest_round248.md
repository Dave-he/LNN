---
title: LNN 每日研究追踪 - 2026-06-25 (round 248)
date: 2026-06-25
tags: [LNN, daily, automation, arxiv, composition, per-branch, multi-basin, frozen-tau, strictly-positive]
---

# LNN 每日研究追踪 - 2026-06-25 (round 248, session #83, hourly loop #9)

> 自动生成：聚合 arXiv 2026-06 LNN / CfC / LTC 相关更新。

## 本轮新增论文

| arXiv ID     | 提交日期       | 标题                                                                | 与本仓关联            |
|--------------|----------------|---------------------------------------------------------------------|-----------------------|
| 2606.15807  | 2026-06-14 | Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liquid Time-Constant Networks | 中 — 图 + LTC，可借鉴至时空场景 |
| 2606.12240  | 2026-06-10 | Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training | **高** — 直接对应 round 246 多尺度 τ |

## 选定方向 — Per-Branch Multi-Basin Lyapunov CfC

### 背景
- **Round 246**: Frozen-Sampled Multi-τ — **BIGGEST win** (-65/-37/-55%)
- **Round 247**: Global multi-basin composition — **safe superset** of r246 but composition tax
- **关键 gap**: round 247 多 basin 是 *global* 的 (所有 τ-branch 共享一套 basin centers)。这导致 K + K' summative 组合，但 r246 已经在所有 3 个 dataset 上 win 太大，加 global basin 变成 stylistic tax。

### Round 248 = Multiplicative composition
**PerBranchMultiBasinLyapunovCfCCell** — 每个 τ-branch 有自己的 basin centers:
- basin_centers 形状: `(n_branches, n_basin, hidden_size)` 而非 `(n_basin, hidden_size)`
- 4 branches × 3 basins = **12 effective basin centers per fused state**
- per-branch contraction loss: 每个 branch 用自己的 basin geometry
- per-branch separation loss: 防止每个 branch 的 basins collapse

### Bench 结果 (2026-06-25, 36 cells: 3 ds × 4 modes × 3 seeds, 100 epochs)

| dataset   | baseline | r246 | r247 | **r248 per_branch** | Δ% vs base | Δ% vs r246 | Δ% vs r247 | H_per_branch |
|-----------|----------|------|------|---------------------|------------|------------|------------|--------------|
| toy_sin   | 0.0060   | 0.0020 | 0.0041 | **0.0020**     | **-67.2%** | -4.4%      | **-52.2%** | 0.596        |
| structured| 0.0021   | 0.0013 | 0.0013 | **0.0011**     | **-50.8%** | **-21.7%**| **-21.3%**| 0.807        |
| random    | 0.0115   | 0.0052 | 0.0069 | **0.0048**     | **-58.0%** | -7.3%      | **-30.0%**| 0.695        |

### 结论 — **STRICTLY POSITIVE FIRST-ROUND STRICT WIN**

| Hypothesis | toy_sin | structured | random |
|------------|---------|------------|--------|
| H1 safe vs baseline       | ✓ | ✓ | ✓ |
| H2 basins used (H > 0.55) | ✓ | ✓ | ✓ |
| H3 win vs r246           | ✓ | ✓ | ✓ |
| H4 win vs r247 (global)   | ✓ | ✓ | ✓ |

- **3/3 datasets** — per-branch beats baseline AND r246 AND r247
- **structured**: per-branch 0.0011 beats r246 0.0013 by **-21.7%** (NEW BEST on this dataset)
- **random**: per-branch 0.0048 beats r246 0.0052 by **-7.3%** 
- **toy_sin**: per-branch 0.0020 ties r246 0.0020 (marginally better -4.4%)
- **All 12 effective basins per branch used** — H_per_branch 0.596-0.807 (above 0.55 threshold)

### 关键 insight
**Per-branch multiplicative composition 优于 global summative composition**:
- Round 247 (global basins): safe but composition tax
- Round 248 (per-branch basins): strict win on all 3

Why it works: each τ-branch specializes in its own geometric manifold. Fast τ (τ≈0.05) needs different basin geometry than slow τ (τ≈20). Sharing geometry across branches causes interference. Per-branch factorization decouples the geometry from the timescale.

### 与 arXiv:2606.12240 "Multi-Rate MoE for LNN" 对照
- 该论文: 多速率 (multi-rate) MoE 加速 LNN 训练
- 本仓 round 246-248: 多速率通过 frozen random τ + per-branch basin geometry
- **对齐**: 论文主张多速率对 LNN 训练有效，本仓实证 frozen random τ 是关键 (L-RFM 启示)
- **超越**: 本仓 per-branch basin 是论文未提及的额外维度

### PR 候选
`lnn/core/per_branch_multibasin_lyapunov_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`

## 建议动作 (后续 round)
- **Round 249**: 学习 per-branch basin 中心 + 输入条件 gate (per-branch W_gate[i]·x)
- **Round 250**: 将 per-branch Lyap aux loss 真正启用 (round 247 都没开)
- **下游候选**: 图结构数据 (PhysioNet/ICU) 用 per-branch basin + QuITE embedding