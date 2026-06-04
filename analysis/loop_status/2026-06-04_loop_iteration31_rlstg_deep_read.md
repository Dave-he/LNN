---
title: 2026-06-04 Loop iteration 31 — RLSTG deep read (arXiv 2601.14115, WWW '26)
date: 2026-06-04
tags: [LNN, loop, paper-deep-read, RLSTG, riemannian-LTC, hyperbolic, tangent-space, WWW-2026, ltc-extension, prd-c-level]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 31 — RLSTG deep read

> `/loop 1h` 第 31 次触发。
> 用户提示 "搜索LNN相关论文代码,补充PRD和论文报告" — 紧接 iter#30 (PRD 工具 3 件套) 后,
> 本轮回到论文深读,挑 daily digest 里**仍未覆盖**的 LTC 扩展方向:
> **Riemannian Liquid Spatio-Temporal Graph Network (RLSTG, arXiv 2601.14115,
> Lu et al. WWW '26)**。
>
> 1. **新论文深读** `docs/reports/Riemannian_Liquid_Spatio-Temporal_Graph_Network_RLSTG_研读报告.md` (~200 行)
> 2. **公式对齐**: LTC + tangent space ODE + exp/log wrapper
> 3. **PRD C-level 表** +1 行 (RLSTG)
> 4. **零回归** pytest 117/117 + verify 9/9
> 5. **commit + rebase + push origin/master**

## 1. 一句话定位

> **LTC 从欧几里得搬到黎曼流形**: 局部 tangent space 求 ODE,通过 `exp` map
> 推回流形;用 hyperbolic (双曲) 嵌入树状层级 / spherical (球面) 嵌入环状结构;
> **理论推广** LTC stability / universal approximation 到黎曼域。

## 2. 关键公式

```
d/dt h = f(h, x, t; θ)              # tangent space T_{h} M 上
h_{t+Δt} = exp_{h_t}(Δt · d/dt h)   # 沿流形 pushforward
```

`tangent space 局部欧几里得` → LTC 的 `f(·; θ)` **不需要修改** — 只需外层 `exp` 包装。

## 3. 仓库 6 套 LNN backbone 与 RLSTG 对齐

| Backbone | 起源 | 空间 | 状态 |
|---|---|---|---|
| LTC | Hasani 2021 | 欧几里得 | ✅ 已有 |
| CfC | Hasani 2022 | 欧几里得 (闭式解) | ✅ 已有 |
| CT-LTC | 仓内 | 欧几里得 | ✅ 已有 |
| PDNA-pulse | iter#19 | 欧几里得 (aug) | ✅ 已有 |
| SVAF-τ-blend | iter#22 | 欧几里得 (aug) | ✅ 已有 |
| DynPMNN-FHN | iter#23 | 欧几里得 (FHN ODE) | ✅ 已有 |
| **RLSTG-LTC** (NEW) | iter#31 | **黎曼 (tangent space)** | **调研中** (C-level) |

## 4. 评级 + ROI

| 维度 | 评级 |
|---|---|
| 学术新意 | **A** (WWW '26 accepted) |
| 工程价值 | B+ (需新依赖 `geoopt`) |
| 代码可获取 | C (无官方代码) |
| 本仓优先级 | **B** (ROI 低于 DynPMNN,但与 graph_lnn 姐妹) |

## 5. pytest 套件(117/117, 49.77s)

```
117 passed, 1 warning in 49.77s
```

vs iter#30: 117 → 117 = **0 变动,0 回归**(纯论文改动)。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 提交与推送

iter#31 改动:
- 增 `docs/reports/Riemannian_Liquid_Spatio-Temporal_Graph_Network_RLSTG_研读报告.md` (~200 行, 9 节)
- 改 `docs/PRD_LNN_Edge_Research.md` (C-level 表 +1 行)
- 增 `analysis/jetson/2026-06-04_loop_iteration17_rlstg_deep_read.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-04_loop_iteration31_rlstg_deep_read.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 总体**: 12/16 = 75.0%(无变化)
**C-level 表累计**: 6 → 7 (RLSTG 加入)

## 8. 下轮 (iter#32) 候选

按 §10 next-up + iter#31 评估:
1. **§10 #4 (HierarchicalDecayLiquidTADHead in graph_lnn)**: 综合,代码改动
2. **§10 #3 (Comparative phase-D)**: 需空载 RAM
3. **§10 #2 (DynPMNN stage B PRD status stale)**: 仅 admin 更新
4. **RLSTG stage A** (调研 + design): 0.5 loop,与下一轮 graph_lnn 工作接
5. **§10 #7 (LFM2.5 INT8)**: RAM blocker
