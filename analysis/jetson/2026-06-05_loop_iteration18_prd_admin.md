---
title: Jetson validation summary — iter#32 admin: stale DynPMNN stage B status fix
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, prd-admin, stale-status-fix, iter32
---

# Jetson validation summary — iter#32 admin: stale DynPMNN stage B status fix

> 本轮执行 **iter#32 admin cleanup** —— 修 iter#24 暴露的 PRD stale status。
> iter#24 实际完成了 PRD #10-2 (DynPMNN stage B) 的所有工作, 但当时只更新了 #10-1 状态列。

## 1. 改动量

```
docs/PRD_LNN_Edge_Research.md   1 行状态更新 (#10-2: pending → ✅ (iter#24))
```

## 2. 修复细节

| 字段 | 旧 | 新 |
|---|---|---|
| #10-2 状态 | `pending` | **✅ (iter#24)**: 6-seed mackey_glass fhn_dynpmnn median MSE 0.0182, backbone matrix ingest 把 fhn_dynpmnn 加进 mackey_glass h=24 r=4 行 (诚实负面: 输 ~3× vs cfc/ltc/gru) |

## 3. prd-status 修复对比

```
旧: §10: 12/16 done (75.0%), 4 pending
新: §10: 13/16 done (81.2%), 3 pending  ← #10-2 修了
```

## 4. 已知 limitation 候选(待 iter#33+)

- §10 #3 (Comparative phase-D): 需空载 RAM
- §10 #4 (HierarchicalDecayLiquidTADHead in graph_lnn): 综合,代码改动
- §10 #7 (LFM2.5 INT8): RAM blocker
- §10 #8: 已 ✅
- §10 #6: 已 ✅

## 5. pytest 套件(97/97, 43.02s)

```
97 passed, 1 warning in 43.02s
```

vs iter#31: 117 → 97 = **-20**(本轮没跑全部 117 个测试,因为 admin-only 改动不需要;为快速验证跑了 7 个核心文件 97 个测试)。0 回归。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 关键 takeaway

1. **PRD 状态维护是 ongoing 责任** —— iter#24 当时只改 #10-1 status,漏改 #10-2;iter#32 admin cleanup 修复
2. **prd-status 工具是 single source of truth** —— `--prd-status` 12 → 13 立即反映
3. **admin cleanup 也是 valid iter# 工作** —— 不必每轮都做大幅 code change,小修复累积起来很重要
