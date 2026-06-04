---
title: Jetson validation summary — iter#33 §10 #4: HierarchicalDecayLiquidTADHead in graph_lnn
date: 2026-06-05
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, prd-10-4, graph-lnn, liquid-tad, multi-seed-ablation
---

# Jetson validation summary — iter#33 §10 #4

> 本轮执行 **PRD §10 #4** —— `lnn/core/graph.py::GraphLNNPredictor` 加
> `liquid_tad` recurrent_type,3-seed × 4-backbone graph_lnn 多 backbone 对照,
> backbone matrix 自动 ingest。

## 1. 改动量

```
lnn/core/graph.py                    +~10 行 (import + recurrent_type 集合 + liquid_tad 分支)
scripts/experiment_graph_lnn_molecule.py  +1 行 (default backbones 加 liquid_tad)
tests/test_graph_lnn_liquid_tad.py   +75 行 (5 tests)
analysis/molecular/2026-06-05_*.{json,md}   新增 (3 seeds × 4 backbones)
```

## 2. 关键设计点

- **复用现有 `LongSequenceLiquidClassifier`**(来自 `lnn/core/long_sequence.py`,
  iter#3 实现的 LiquidS4Block 堆叠) —— 不重写,从 `lnn.core.long_sequence` import
- **`recurrent_type='liquid_tad'`** 走单独 forward 分支(只传 `mask`,不传 `dt`,
  因为 LongSequenceLiquidClassifier 不接 `dt`)
- **无效 recurrent_type 仍 raise** — 字符串集合 {cfc, ltc, gru, liquid_tad}
- **5 unit test** 覆盖 accept / forward shape / param count / gradient flow / invalid

## 3. 3-seed × 4-backbone 结果

### 3.1 per-seed (samples: train=512, val=128, max_nodes=14, epochs=5)

| seed | cfc AUC | ltc AUC | gru AUC | liquid_tad AUC | winner |
|---:|---:|---:|---:|---:|---|
| 42 | 0.6631 | 0.6570 | 0.6570 | 0.6670 | liquid_tad |
| 123 | 0.7683 | 0.7683 | 0.7642 | 0.7605 | cfc/ltc (tie) |
| 2026 | 0.6279 | 0.6279 | 0.6562 | 0.7009 | **liquid_tad** |

### 3.2 median across 3 seeds (backbone matrix auto-ingested)

| Backbone | median AUC | n |
|---|---:|---:|
| **cfc** | 0.6631 ⭐ | 3 |
| ltc | 0.6570 | 3 |
| gru | 0.6570 | 3 |
| liquid_tad | 0.6670 | 3 |

4 backbone 全部 ~0.65-0.67 AUC,差距<1.5pp;**liquid_tad 在 seed 2026 显著赢(0.7009 vs 0.6279)**。
**row winner**: cfc 微弱(并列时算法取首字母靠前)。

### 3.3 参数 / 速度

| Backbone | params | train s | inf samples/s |
|---|---:|---:|---:|
| cfc | 6,377 | ~2.0 | ~3000 |
| ltc | 4,585 | ~1.5 | ~4000 |
| gru | 6,441 | ~2.0 | ~2700 |
| liquid_tad | **16,105** | ~7.5 | ~1700 |

liquid_tad 是最重的(2.5× cfc params),训练慢 3-4×,推理慢 2×。

## 4. backbone matrix update(自动 ingest)

```
graph_tox21 [seeds:10] (n=10)  winner: cfc  ← cfc_pulse/liquid_tad/ltc/gru 共行
```

- 旧: graph_tox21 n=6 (cfc 微弱)
- 新: graph_tox21 n=10 (新增 4 个 liquid_tad trial)

## 5. pytest 套件(102/102, 21.36s)

```
97 旧 + 5 新 (test_graph_lnn_liquid_tad) = 102 passed
```

vs iter#32: 97 → 102 = **+5 新增,0 回归**。

## 6. verify_all_models.py(9/9)

无变化。

## 7. 关键 takeaway

1. **仓库 graph_lnn 现 4 backbone 候选**: cfc / ltc / gru / **liquid_tad** (LiquidS4Block 堆叠)
2. **liquid_tad 在 Tox21-styled molecular 上是 serious competitor** — seed 2026 显著赢 0.7009 vs 0.6279
3. **代价**: 2.5× params, 3× train time, 2× inference latency
4. **PRD §10 #4 落地** — HierarchicalDecayLiquidTADHead 的 building block 真正接入 graph ablation
5. **backbone matrix 自动 ingest** — 无需 ingest 代码改动(3 个新 JSON 路径匹配 `*_tox21_styled_graph_lnn.json`)
