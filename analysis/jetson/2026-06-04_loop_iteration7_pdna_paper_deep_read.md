---
title: Jetson validation summary — iter#18 PDNA paper deep read + PRD §10 #10
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, pdna-deep-read, prd-10
---

# Jetson validation summary — iter#18

> 与 iter#18 PDNA 论文研读 + PRD §10 #10 落地同批跑的环境验证。

## 1. 环境快照(无变化)

```
Platform : Jetson Orin Nano Super (Linux 5.15.148-tegra, aarch64)
Python   : 3.14.4 (pyenv primary)
Torch    : 2.11.0+cu130
CUDA     : disabled (BSP driver 12060 < cu130 最低要求)
Libcudss : n/a
```

## 2. verify_all_models.py (9 变体 smoke)

```
LTC                 : ✓ 通过
CfC                 : ✓ 通过
StrictCfC           : ✓ 通过
HybridCfC           : ✓ 通过
CTLTC               : ✓ 通过
LiquidS4            : ✓ 通过
LRC                 : ✓ 通过
CfC-DT              : ✓ 通过
Euler-LTC-DT        : ✓ 通过
────────────────────────────────
✓ 所有模型测试通过!
```

## 3. pytest 套件 smoke(46 tests)

```
tests/test_core.py + tests/test_liquid_tad_hierarchical.py
46 passed, 1 warning in 71.60s
```

(iter#17 用 11.4s — 这次 71.6s 是 cold cache 之后 pytest 重启 + 完整 import,
非回归。)

## 4. 与本周回退基线对比

| 指标 | iter#15 | iter#16 | iter#17 | iter#18 (本次) |
|---|---:|---:|---:|---:|
| verify_all_models 9 变体 | 9/9 ✅ | 9/9 ✅ | 9/9 ✅ | **9/9 ✅** |
| pytest 套件 | 46/46 | 46/46 | 46/46 | **46/46** |

**0 回归**。本轮论文研读 + PRD 落地不修改任何 `lnn/` 代码,完全无回归属预期。

## 5. 任务意义

iter#18 核心动作:**研读 PDNA (arXiv 2603.00153) + 把 PDNAPulseHead 算子挂进
PRD §10 #10 作为 P1 候选**。本轮验证仅作环境健康检查(PRD §6 协议要求
每轮 loop 都跑)。

PDNA 复现的 Stage A 计划(iter#19 候选): `lnn/core/cfc_cell.py` 加
`PDNAPulseHead` (~25 行核心代码) + `tests/test_pdna_pulse.py` 5 个 unit test。
预估代码改动量小,Jetson CPU 路径完全可跑(sMNIST + hidden=128)。

## 6. 已知阻塞(无变化)

| 阻塞 | 来源 | 影响 |
|---|---|---|
| CUDA 不可用 | Jetson BSP driver 12060 < torch 2.11 cu130 | iter#2 修通的 py3.10+torch2.10 路径需要空载窗口 |
| RAM 1.7 GB available | 多 agents 并行 + 8GB 统一显存 | LFM2.5-1.2B / 较大 hidden LNN sweep 受限 |
| THUMOS-14 数据未下载 | LiquidTAD stage C 真实数据 | 暂用 toy 长视频 |

本轮无新增阻塞。
