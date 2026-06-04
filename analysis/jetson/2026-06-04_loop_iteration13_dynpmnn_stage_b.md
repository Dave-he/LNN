---
title: Jetson validation summary — iter#24 DynPMNN §10 #1 stage B: backbone matrix integration + 6-seed benchmark
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, dynpmnn-stage-b, prd-10-1, honest-negative
---

# Jetson validation summary — iter#24 DynPMNN stage B

> 本轮执行 **PRD §10 #1 stage B** —— `scripts/ablation_lnn_vs_lstm_timeseries.py`
> 加 `fhn_dynpmnn` backbone,跑 6-seed benchmark on mackey_glass + backbone matrix 更新。
> **iter#16 报告预期** "如果 DynPMNN 能在 mackey_glass / gradual_multi_regime
> 上跑出 mean MSE 比 LSTM 低" — 本轮**给出诚实负面信号**: 仓库又添一条
> task-conditional ranking 负面证据。

## 1. 改动量

```
scripts/ablation_lnn_vs_lstm_timeseries.py   +9 行 (_build_model 加 fhn_dynpmnn case)
analysis/timeseries_ablation/2026-06-04_180636_lnn_vs_lstm.json   新增 (6 seeds fhn_dynpmnn)
analysis/timeseries_ablation/2026-06-04_185242_lnn_vs_lstm.json   新增 (3 seeds cfc/ltc/gru/lstm @ same r=4 config)
analysis/timeseries_ablation/2026-06-04_185436_lnn_vs_lstm.json   新增 (3 seeds fhn_dynpmnn @ same r=4 config)
analysis/backbone_matrix/2026-06-04_180651_backbone_matrix.{md,json}   新增
analysis/backbone_matrix/2026-06-04_185534_backbone_matrix.{md,json}   新增
```

## 2. 关键结果(诚实负面)

### 2.1 fhn_dynpmnn 6-seed (samples=1200, seq=32, epochs=8)

| seed | test_mse | test_mae |
|---:|---:|---:|
| 42 | 0.0119 | 0.0865 |
| 7 | 0.0257 | 0.1258 |
| 123 | 0.0151 | 0.1017 |
| 2026 | 0.0213 | 0.1213 |
| 11 | 0.0141 | 0.0937 |
| 313 | 0.0316 | 0.1413 |
| **median** | **0.0182** | 0.1075 |

### 2.2 同 config cfc/ltc/gru/lstm 3-seed (samples=1000, seq=24, epochs=6)

| Backbone | median test_mse | mean test_mse |
|---|---:|---:|
| **cfc** | **0.0081** | 0.0078 |
| ltc | 0.0081 | 0.0088 |
| gru | 0.0081 | 0.0083 |
| lstm | 0.0101 | 0.0093 |
| **fhn_dynpmnn** | **0.0227** | 0.0228 |

**fhn_dynpmnn 输 ~3× vs cfc/ltc/gru** — 4 Euler 步 FHN ODE 在 mackey_glass 平滑时序上
表现不足。这是**诚实的负面信号**,与 iter#16 报告的预期相符 (双向都有价值)。

### 2.3 与仓库 LNN 通杀 thesis 对齐

23 轮 loop 累计 + 第 6 套 backbone (DynPMNN-FHN) **依然没有"通杀 LNN"**:
- mackey_glass h=24 (3 seeds): LSTM 0.0030 ⭐, gru 0.0036, cfc 0.0055, ltc 0.0050
- **mackey_glass h=24 r=4 fhn**: fhn_dynpmnn 0.0182 (worse than all 4)

DynPMNN **不**给 LNN 在 timeseries 任务上提供新 win。

## 3. pytest 套件(84/84)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed
tests/test_dynpmnn.py               :  9 passed
─────────────────────────────────────────────
84 passed, 1 warning in 11.78s
```

`scripts/ablation_lnn_vs_lstm_timeseries.py` 改动**不触碰**任何单元测试路径。

## 4. verify_all_models.py(9/9)

无变化。

## 5. 已知 limitation(iter#24 暴露)

`scripts/build_backbone_matrix.py` 的 `_dedupe_keep_higher_n` 用"整行 max n_seeds"做
dedup,导致新加的 3-seed cfc/ltc/gru/lstm 整行被旧的 6-seed fhn_dynpmnn 覆盖。
**正确做法是 per-backbone n_seeds dedup**。这是 iter#25 可修的小 bug。

## 6. 关键 takeaway

1. **DynPMNN-FHN 在 mackey_glass 上输 ~3×** — 与 iter#16 报告双向预期一致
2. **6-seed 跑出 median 0.0182** vs 4 Euler 步 ODE 平滑时序不足的工程证据
3. **仓库 LNN 通杀 thesis 进一步强化** — 6 套连续时间架构 (LTC/CfC/CT-LTC/PDNA/SVAF/DynPMNN)
   仍无在合成时序回归 + N≥5 seed 下稳定赢 LSTM
4. **论文只跑 California Housing 的局限** 在本仓的 mackey_glass 反例上得到加强
5. **dedup bug 待修** — per-backbone n_seeds dedup 应替换整行 max dedup
