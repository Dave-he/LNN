---
title: 2026-06-04 Loop iteration 12 — backbone matrix + ablation runner v2
date: 2026-06-04
tags: [LNN, loop, PRD-9, meta-tooling, backbone-matrix, robust-stats, automation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 12 — backbone matrix automation + ablation runner v2

> `/loop 1h` 第 12 次触发。
> Iter#11 把 "CfC 赢 LSTM" 撤回时,留了两条工程债:
>   (1) ablation runner 默认只报 mean ± std,没有 median / std-over-mean 警报;
>   (2) 没有自动方式把多个 ablation JSON 透视成一张 task × backbone 矩阵。
>
> 本轮一次性还掉两条债,**完成 PRD §9 #7 (cross-data backbone matrix)**:
>
> 1. **`scripts/ablation_lnn_vs_lstm_timeseries.py` v2 输出**:
>    自动报 median + std/mean ratio + N 警报 + 双视图 (mean+median) Δ-vs-LSTM 表
>    + 解读模板从 N=3 思维改成 mean/median 一致性 + std/mean 阈值。
> 2. **新增 `scripts/build_backbone_matrix.py` (~280 行)**:
>    扫 `analysis/timeseries_ablation/*_lnn_vs_lstm.json` +
>    可选 `analysis/molecular/*_tox21_styled_graph_lnn.json`,
>    pivot 成 task × backbone 矩阵;
>    dedup 规则: 同 dataset+config 重复时**保留 n_seeds 更高的那个**
>    (机械执行 iter#11 教训)。
> 3. **首跑结果**: 4 task × 4 backbone,**LSTM 3 wins / GRU 1 win / CfC/LTC 0 wins**,
>    机械确认 11 轮 loop 累计的"无通杀 backbone"结论。

## 1. ablation runner v2

### 1.1 patch 内容

**`_aggregate(per_run)`** 加了:
- `test_mse_median` (outlier-resistant 中位数)
- `test_mse_min`, `test_mse_max` (区间感)
- `test_mse_std_over_mean` (变异系数,用于 ⚠️ 标记)
- `n_seeds`

**两个新工具函数**:
- `_variance_flag(s/m)` — std/mean > 1 → " ⚠️";> 0.5 → " ⚠"
- `_small_n_flag(n)` — N<3 → " ⚠️ N<3";N<5 → " ⚠ N<5"

**`_format_markdown` 改写**:
- 表头从 `Test MSE` 改成 `mean MSE | **median MSE** | std/mean | min/max | n`
- 配置块带 N 警报: `seeds (8): [...]` vs `seeds (3): [...] ⚠ N<5`
- N<5 时自动 append iter#11 lesson 一行
- Δ-vs-LSTM 双视图: Δmean_mse 和 Δmedian_mse 并列,✅/❌ 标记
- 解读模板从"std>mean 是 seed 敏感"升级到"mean 和 median 一致才是真信号"

向后兼容: 旧 JSON 仍能被 `_format_markdown` 处理(median/n_seeds 现场算)。

### 1.2 v2 示例输出预览(下次跑 ablation 自动产生)

```text
## 跨 seed 鲁棒统计
| Backbone | params | mean MSE | median MSE | std/mean | min/max | n |
| `cfc`    | 1,921  | 0.21116 ± 0.31558 | 0.05136 | 1.49 ⚠️ | 0.019/0.734 | 8 |
| `lstm`   | 2,617  | 0.18346 ± 0.38329 | 0.02376 | 2.09 ⚠️ | 0.008/1.119 | 8 |

## 相对 LSTM baseline (mean + median 双视图)
| Backbone | Δparams | Δmean_mse  | Δmedian_mse | Δtrain_s | Δinf_throughput |
| `cfc`    | -26.60% | +15.10% ❌ | +116.16% ❌ | +113.77% | -45.62%         |
```

## 2. backbone matrix runner (PRD §9 #7)

### 2.1 设计

| 维度 | 决定 |
|---|---|
| 输入 | `analysis/timeseries_ablation/*_lnn_vs_lstm.json` (默认) + `--include-molecular` 加 `analysis/molecular/*_tox21_styled_graph_lnn.json` |
| Row key | `dataset [warmup=X,h=Y,r=Z]` — 让同 dataset 不同 config 各占一行 |
| Dedup | **保留 n_seeds 更大的那个** — 直接执行 iter#11 教训,iter#10 N=3 被 iter#11 N=8 自动覆盖 |
| Metric | 时序: median test_mse (iter#11);分子: median val_auc_roc |
| Winner | 时序 = argmin median MSE;分子 = argmax median AUC |
| 输出 | JSON + Markdown (含 win tally) 到 `analysis/backbone_matrix/` |

### 2.2 首跑结果

```text
=== Backbone matrix (4 rows, 4 backbones) ===
  mackey_glass [h=24]                       (n= 3)  winner: lstm
  concept_drift [h=24]                      (n= 3)  winner: lstm
  gradual_multi_regime [warmup=0.1,h=24,r=4] (n= 8) winner: lstm  ← 自动取 iter#11 over iter#10
  graph_tox21 [seeds:3]                     (n= 3)  winner: gru
```

**Win tally**: LSTM 3 / GRU 1 / CfC 0 / LTC 0

### 2.3 完整矩阵 (来自 `analysis/backbone_matrix/2026-06-04_073357_backbone_matrix.md`)

#### Timeseries (lower median test_mse better)

| Task / config | n | cfc | ltc | gru | lstm |
|---|---:|---:|---:|---:|---:|
| mackey_glass [h=24] | 3 | 0.0055 | 0.0050 | 0.0036 | **0.0030** ⭐ |
| concept_drift [h=24] | 3 | 0.0121 | 0.0893 | 0.0218 | **0.0050** ⭐ |
| gradual_multi_regime [warmup=0.1,h=24,r=4] | 8 | 0.0514 | 0.1944 | 0.0347 | **0.0238** ⭐ |

#### Molecular (higher median val_auc_roc better)

| Task / config | n | cfc | ltc | gru | lstm |
|---|---:|---:|---:|---:|---:|
| graph_tox21 [seeds:3] | 3 | 0.7361 | 0.7431 | **0.7461** ⭐ | — |

### 2.4 关键发现 — 机械确认 iter#11 retraction

仓库 11+ 轮 loop 累计的 task-conditional ranking 第一次被 **机械工具** 透视:
- 时序回归 (3 个不同非平稳协议) 全部 LSTM 赢;
- 静态分子图 GRU 微弱赢(median, 与 iter#6 mean-tie 结论略差);
- **CfC 和 LTC 在任何任务上的 median 都不是第一**;
- iter#10 (warmup=0.1, N=3) 的"CfC 赢"行被 iter#11 (N=8) 自动覆盖,
  下次自动报告里只能看到 8-seed 版本 — 工具防止旧错误结论复活。

## 3. PRD §9 进展更新

| # | 状态 | 备注 |
|---:|:---:|---|
| 9-1 | ⏳ pending | LFM2.5 — 等 RAM 空 |
| 9-2 | ✅ iter#10/#11 | gradual_multi_regime + warmup + 8-seed (撤回 + 真结论) |
| 9-3 | ⏳ pending | LiquidTAD Stage C-true (THUMOS-14 真数据) |
| 9-4 | ⏳ pending | `experiment_graph_lnn_molecule.py --frozen-encoder` |
| 9-5 | ⏳ pending | `loop_status.py --since-last-loop` |
| 9-6 | ⏳ pending | ONNX + TensorRT INT8 (待 CUDA 稳定 + RAM) |
| **9-7** | **✅ iter#12** | **`scripts/build_backbone_matrix.py`** |
| 9-8 | ⏳ pending | GitHub Actions weekly CI |

## 4. 衍生工作

| 任务 | 推入 |
|---|---|
| 在新 ablation 跑完后自动 `build_backbone_matrix.py` 加进 retry | NEXT_STEPS / 加进 PRD §10 |
| 加 `--rows-filter dataset=mackey_glass` 看时间序列内对比 | NEXT_STEPS |
| 让 win tally 区分 "median 第一" vs "在 1 std 内 tied" | NEXT_STEPS |
| 用 matrix 数据生成 README badge 行 (LSTM 3/4 / GRU 1/4) | docs |

## 5. 参考产物

- 代码增量:
  - `scripts/ablation_lnn_vs_lstm_timeseries.py` (+ median / std-over-mean / N 警报 / 双视图 Δ)
  - `scripts/build_backbone_matrix.py` (新增 ~280 行)
- Matrix 首跑: `analysis/backbone_matrix/2026-06-04_073357_backbone_matrix.{json,md}`
- 上一轮: [[2026-06-04_loop_iteration11_phaseC_8seed_retraction]]
- 累积 iter reports: [[2026-06-04_loop_iteration10_gradual_warmup_cfc_wins]] /
  [[2026-06-04_loop_iteration9_prd9_and_concept_drift]] /
  [[2026-06-04_loop_iteration8_loop_status_tooling]] /
  [[2026-06-04_loop_iteration7_lnn_vs_lstm_v2]] /
  [[2026-06-04_loop_iteration6_graph_lnn_tox21_smoke]] /
  [[2026-06-04_loop_iteration5_gcn_cfc_repo_survey]]
- PRD: [[PRD_LNN_Edge_Research]] §9 #7
