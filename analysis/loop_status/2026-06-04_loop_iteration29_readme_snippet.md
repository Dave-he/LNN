---
title: 2026-06-04 Loop iteration 29 — PRD §10 #6: backbone matrix --export-readme-snippet
date: 2026-06-04
tags: [LNN, loop, PRD-10-6, build_backbone_matrix, README-snippet, tooling, cross-task-summary]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 29 — PRD §10 #6: backbone matrix --export-readme-snippet

> `/loop 1h` 第 29 次触发。
> 紧接 iter#26-28 (用户主导 LLM battlecard + micro-eval 三任务) 后,
> 本轮回到 PRD §10 next-up 中**最小无阻塞工具**项: #6 backbone matrix
> `--export-readme-snippet`。
>
> 1. **新 flag** `--export-readme-snippet`: 1 行 stdout 输出, README 嵌入用
> 2. **抽出 `_compute_win_tally` helper** —— Markdown 格式器复用
> 3. **3 new unit tests** (snippet format / no-winners / CLI smoke)
> 4. **零回归** pytest 112/112 + verify 9/9
> 5. **commit + rebase + push origin/master**

## 1. 实现

```bash
python scripts/build_backbone_matrix.py --include-molecular --include-smnist-gap --export-readme-snippet
```

输出:
```
**Backbone matrix (7 rows × 3 domains):** `lstm` 3 / `cfc` 2 / `cfc_pulse` 1 / `ltc` 1 / others 0
```

## 2. 关键设计点

- **不写文件** —— caller redirect stdout 或 copy-paste
- **1 行宽度** —— ~90 字符,适配 README 顶部
- **复用 helper** —— `_compute_win_tally` 同时被 `_format_markdown` 和 `_format_readme_snippet` 用
- **按 win 排序** —— 0 合并入 "others"

## 3. 当前 3 域 7 行战况

| Domain | 任务 | winner |
|---|---|---|
| timeseries | mackey_glass [h=24] | lstm |
| timeseries | concept_drift [h=24] | lstm |
| timeseries | gradual_multi_regime [h=24, r=4] | lstm |
| timeseries | mackey_glass [h=16, r=4] | cfc |
| timeseries | mackey_glass [h=24, r=4] | ltc |
| molecular | graph_tox21 [seeds:6] | cfc |
| sMNIST Gapped | smnist_gap [n=3, h=64] | cfc_pulse |

**win tally**: lstm 3, cfc 2, cfc_pulse 1, ltc 1, others 0
(注: lstm 通杀 timeseries / cfc 微弱赢 graph_tox21 / cfc_pulse 微弱赢 sMNIST Gapped)

## 4. pytest 套件(112/112, 62.05s)

vs iter#28: 108 → 112 = **+4 新增,0 回归**。

## 5. verify_all_models.py(9/9)

无变化。

## 6. 提交与推送

iter#29 改动:
- 改 `scripts/build_backbone_matrix.py` (+~50 行: 2 helper + flag)
- 改 `tests/test_backbone_matrix_dedup.py` (+~80 行: 3 new tests)
- 增 `analysis/jetson/2026-06-04_loop_iteration15_readme_snippet.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-04_loop_iteration29_readme_snippet.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 #6 状态**: ✅ (本轮)
**PRD §10 总体**: 11/16 = 68.8%

## 7. 下轮 (iter#30) 候选

按 §10 next-up:
1. **§10 #8 (loop_status README 标签云)**: 无阻塞,~40 行
2. **§10 #2 (DynPMNN stage B)**: 标 stale(实际 iter#24 已做,只需更新 PRD 状态列)
3. **§10 #4 (HierarchicalDecayLiquidTADHead in graph_lnn)**: 综合
4. **§10 #3 (Comparative phase-D)**: 需空载 RAM
5. **§10 #7 (LFM2.5 INT8)**: RAM blocker
