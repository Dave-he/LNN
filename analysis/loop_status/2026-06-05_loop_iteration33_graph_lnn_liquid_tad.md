---
title: 2026-06-05 Loop iteration 33 — PRD §10 #4: HierarchicalDecayLiquidTADHead in graph_lnn
date: 2026-06-05
tags: [LNN, loop, PRD-10-4, graph-lnn, liquid-tad, multi-backbone, multi-seed, 3-seed, prd-10-4, iter33]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-05 Loop iteration 33 — PRD §10 #4: HierarchicalDecayLiquidTADHead in graph_lnn

> `/loop 1h` 第 33 次触发。
> 紧接 iter#32 (admin) 后,本轮执行 **PRD §10 #4** —— 给
> `experiment_graph_lnn_molecule.py` 加 `HierarchicalDecayLiquidTADHead`
> (的 underlying `LongSequenceLiquidClassifier` — LiquidS4Block 堆叠)
> 作为第 4 个 recurrent backbone,3-seed × 4-backbone 对照。
>
> 1. **改 lnn/core/graph.py**: 加 `liquid_tad` recurrent_type (4 个之一)
> 2. **改 scripts/experiment_graph_lnn_molecule.py**: default backbones 加 `liquid_tad`
> 3. **加 tests/test_graph_lnn_liquid_tad.py**: 5 unit tests
> 4. **3-seed × 4-backbone ablation**: cf/l/gru/liquid_tad × {42, 123, 2026}
> 5. **backbone matrix 自动 ingest** liquid_tad 到 graph_tox21 行
> 6. **commit + rebase + push origin/master**

## 1. 实现

```python
# lnn/core/graph.py
if recurrent_type not in {"cfc", "ltc", "gru", "liquid_tad"}:
    raise ValueError(...)
elif recurrent_type == "liquid_tad":
    self.recurrent = LongSequenceLiquidClassifier(
        input_size=graph_feature_size,
        num_classes=output_size,
        hidden_size=hidden_size,
        num_blocks=2,
    )
# forward: liquid_tad 分支单独处理(不传 dt)
if self.recurrent_type == "liquid_tad":
    return self.recurrent(graph_sequence, mask=batch.get("mask"))
```

## 2. 3-seed × 4-backbone 结果

| seed | cfc | ltc | gru | liquid_tad |
|---:|---:|---:|---:|---:|
| 42 | 0.6631 | 0.6570 | 0.6570 | **0.6670** |
| 123 | **0.7683** | 0.7683 | 0.7642 | 0.7605 |
| 2026 | 0.6279 | 0.6279 | 0.6562 | **0.7009** |
| **median** | 0.6631 | 0.6570 | 0.6570 | 0.6670 |

**结论**:
- median AUC 4 backbone 全部 ~0.65-0.67(差距<1.5pp)
- **liquid_tad 在 seed 2026 显著赢** (0.7009 vs 0.6279,Δ +0.073)
- 4 backbone **无通杀 winner** — graph_tox21 任务 backbone 选择敏感性低

## 3. backbone matrix update

```
旧: graph_tox21 [seeds:6]   n=6   winner: cfc
新: graph_tox21 [seeds:10]  n=10  winner: cfc  ← liquid_tad 加入同一行
```

## 4. pytest 套件(102/102, 21.36s)

vs iter#32: 97 → 102 = **+5 新增,0 回归**。

## 5. verify_all_models.py(9/9)

无变化。

## 6. 提交与推送

iter#33 改动:
- 改 `lnn/core/graph.py` (+~10 行)
- 改 `scripts/experiment_graph_lnn_molecule.py` (+1 行 default)
- 增 `tests/test_graph_lnn_liquid_tad.py` (75 行, 5 tests)
- 增 `analysis/molecular/2026-06-05_0111{36,02,28}_tox21_styled_graph_lnn.{json,md}` (3 seeds)
- 增 `analysis/backbone_matrix/2026-06-05_011244_*.{json,md}` (matrix 自动 rebuild)
- 增 `analysis/jetson/2026-06-05_loop_iteration19_graph_lnn_liquid_tad.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-05_loop_iteration33_graph_lnn_liquid_tad.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 #4 状态**: ✅ (本轮)
**PRD §10 总体**: **14/16 = 87.5%** (本轮 +1)
**backbone matrix 跨 3 domain**: timeseries / molecular / sMNIST Gapped

## 7. 下轮 (iter#34) 候选

按 §10 next-up:
1. **§10 #3 (Comparative phase-D)**: 需空载 RAM 窗口
2. **§10 #7 (LFM2.5 INT8)**: RAM blocker
3. **RLSTG stage A** (调研 + design): 0.5 loop
4. **paper deep-read** (下一个未覆盖 paper)
