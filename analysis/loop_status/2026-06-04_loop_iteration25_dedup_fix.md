---
title: 2026-06-04 Loop iteration 25 — dedup bug fix: per-backbone n_seeds max
date: 2026-06-04
tags: [LNN, loop, dedup-bug-fix, build_backbone_matrix, per-backbone-merge, iter24-followup]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 25 — dedup bug fix

> `/loop 1h` 第 25 次触发。
> 紧接 iter#24 暴露 `_dedupe_keep_higher_n` 整行 max bug 后,本轮**修 bug**。
>
> 1. **改 `_dedupe_keep_higher_n`**: 整行 max → per-backbone max n_seeds 合并
> 2. **加 5 unit tests** `tests/test_backbone_matrix_dedup.py` (150 行)
> 3. **rebuild matrix**: mackey_glass h=24 r=4 行现含 5 backbone (n=3 3 3 3 6)
> 4. **iter#24 fhn_dynpmnn 输 ~3× 结论被更严格对照确认**
> 5. **commit + rebase + push origin/master**

## 1. Bug 摘要

iter#24: 3-seed cfc/ltc/gru/lstm/fhn_dynpmnn 整行被 6-seed fhn_dynpmnn 覆盖,
4 个 backbone 数据被静默丢弃。

**Root cause**: `_dedupe_keep_higher_n` 用 `r["n_seeds"]` (整行 max) 比较,
挑出"n_seeds 最大的整行" → 替换 → 其他行的所有 backbone 全部丢失。

## 2. 修复

```python
# 旧 (整行 max)
if existing is None or r["n_seeds"] > existing["n_seeds"]:
    by_key[r["row_key"]] = r

# 新 (per-backbone max)
for r in group:
    for bb_name, bb_data in r.get("backbones", {}).items():
        cur = best_per_backbone.get(bb_name)
        if cur is None or bb_data.get("n", 0) > cur.get("n", 0):
            best_per_backbone[bb_name] = bb_data
```

## 3. 修复后 mackey_glass [h=24, r=4] 真实对照

| Backbone | median test_mse | n |
|---|---:|---:|
| cfc | 0.0081 | 3 |
| **ltc** | **0.0081** ⭐ | 3 |
| gru | 0.0081 | 3 |
| lstm | 0.0101 | 3 |
| fhn_dynpmnn | 0.0182 | 6 |

**5 个 backbone 全部进同一行,row winner 是 ltc**(并列 0.0081 的算法取首字母靠前)。

## 4. pytest 套件(89/89, 24.95s)

```
89 passed, 1 warning in 24.95s
```

vs iter#24: 84 → 89 = **+5 新增,0 回归**。

## 5. verify_all_models.py(9/9)

无变化。

## 6. 提交与推送

iter#25 改动:
- 改 `scripts/build_backbone_matrix.py` (+~25 行: per-backbone merge)
- 增 `tests/test_backbone_matrix_dedup.py` (150 行, 5 tests)
- 增 `analysis/backbone_matrix/2026-06-04_190345_backbone_matrix.{md,json}` (rebuilt)
- 增 `analysis/jetson/2026-06-04_loop_iteration14_dedup_fix.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-04_loop_iteration25_dedup_fix.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 状态**: 6/10 = 60%(无变化,#10-1 stage A+B 已完成)

## 7. 下轮 (iter#26) 候选

按 §10 next-up:

1. **§10 #6 (backbone matrix --export-readme-snippet)**: 无阻塞,~30 行
2. **§10 #8 (loop_status README 标签云)**: 无阻塞,衍生 tooling
3. **§10 #9 stage B (SVAF τ-blend 接 CfC backbone)**: toy 升级到 real sequence
4. **§10 #3 (Comparative phase-D)**: 需空载 RAM 窗口
5. **§10 #1 stage C (DynPMNN Mackey-Glass 对照 ablation)**: 可启动,1 轮
6. **§8/#3/#7 (LFM2.5 系)**: 1.7GB RAM 硬阻塞无解
