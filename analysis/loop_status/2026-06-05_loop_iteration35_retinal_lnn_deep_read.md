---
title: 2026-06-05 Loop iteration 35 — Retinal Ganglion LNN deep read (arXiv 2511.18014, AAAI-26)
date: 2026-06-05
tags: [LNN, loop, paper-deep-read, retinal-lnn, LTC-CfC, vision-prosthetics, AAAI-2026, edge-ai, prd-c-level, iter35]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-05 Loop iteration 35 — Retinal Ganglion LNN deep read

> `/loop 1h` 第 35 次触发。
> 用户提示 "搜索LNN相关论文代码, 补充PRD和论文报告" — 紧接 iter#34 (EntroLnn) 后,
> 本轮挑 daily digest 里**仍未覆盖**的 LTC 生物应用方向:
> **Modeling Retinal Ganglion Cells with Neural Differential Equations**
> (arXiv 2511.18014, Dobek et al. AAAI-26 Student Abstract)。
>
> 1. **新论文深读** `docs/reports/Modeling_Retinal_Ganglion_Cells_with_Neural_Differential_Equations_研读报告.md` (~200 行, 9 节)
> 2. **关键发现**: 论文 = 本仓 LTC/CfC 在生物 RGC 数据上的实证, 公式同构
> 3. **PRD C-level 表** +1 行 (Retinal LNN)
> 4. **零回归** pytest 102/102 + verify 9/9
> 5. **commit + rebase + push origin/master**

## 1. 一句话定位

> **LTC + CfC 应用到虎蝾螈 RGC 神经活动预测**: 4 架构 (ConvNet / LSTM / LTC / CfC)
> 共享同一卷积栈,NODEs 赢 MAE 2.73/2.86 vs ConvNet 4.07 + 5-8× 少参数 + 更快收敛;
> 输 Pearson ρ (0.480 vs 0.569) — 时序对位差。**Jetson 边缘 视觉假体**应用场景。

## 2. 关键结果(论文 Table 1, rgc9 数据集)

| 模型 | Pearson ρ ↑ | MAE ↓ | #params | Time [s] |
|---|---:|---:|---:|---:|
| ConvNet | **0.569** | 4.07 | ~3-5k | ~30 |
| LSTM | 0.421 | 3.18 | ~10k | **最慢** |
| **LTC** | 0.480 | **2.73** | **5-8× 少** | ~30 |
| **CfC** | 0.474 | 2.86 | **5-8× 少** | ~30 |

**ANOVA**: 3 数据集全部 p<0.05 (NODEs 显著优于 ConvNet/LSTM on MAE)。

## 3. 仓库 8 套 LNN 概念扩展 + 1 个新评估维度

| Backbone | 起源 | 状态 |
|---|---|---|
| LTC / CfC | Hasani 2021/2022 | ✅ 已有 (本论文直接对应) |
| CT-LTC | 仓内 | ✅ 已有 |
| PDNA-pulse | iter#19 | ✅ 已有 |
| SVAF-τ-blend | iter#22 | ✅ 已有 |
| DynPMNN-FHN | iter#23 | ✅ 已有 |
| RLSTG-LTC | iter#31 | 调研中 (黎曼) |
| transformable-LTC | iter#34 | 调研中 (部署精化) |
| (新) **retinal-eval** | iter#35 | **新评估维度候选** (Jetson 边缘) |

## 4. 评级 + ROI

| 维度 | 评级 |
|---|---|
| 学术新意 | B+ (AAAI-26 Student Abstract, 实证) |
| 工程价值 | **A** (Jetson 边缘 视觉假体, 5-8× 少参数) |
| 代码可获取 | C (Student Abstract 性质) |
| 本仓优先级 | **A-** (公式复用 0 障碍) |

## 5. pytest 套件(102/102, 29.67s)

```
102 passed, 1 warning in 29.67s
```

vs iter#34: 102 → 102 = **0 变动,0 回归**(纯论文改动)。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 提交与推送

iter#35 改动:
- 增 `docs/reports/Modeling_Retinal_Ganglion_Cells_with_Neural_Differential_Equations_研读报告.md` (~200 行, 9 节)
- 改 `docs/PRD_LNN_Edge_Research.md` (C-level 表 +1 行)
- 增 `analysis/jetson/2026-06-05_loop_iteration21_retinal_lnn_deep_read.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-05_loop_iteration35_retinal_lnn_deep_read.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 总体**: 14/16 = 87.5%(无变化)
**C-level 表累计**: 8 → 9 (Retinal LNN 加入)

## 8. 下轮 (iter#36) 候选

按 §10 next-up(剩 2 个 pending):
1. **§10 #3 (Comparative phase-D)**: 需空载 RAM
2. **§10 #7 (LFM2.5 INT8)**: RAM blocker

其他候选:
- **RLSTG stage A** (调研 + design): 0.5 loop
- **EntroLnn stage A** (调研 + design): 0.5 loop
- **Retinal LNN stage A** (调研 + design): 0.5 loop (新候选)
- **paper deep-read** (下一个未覆盖)
