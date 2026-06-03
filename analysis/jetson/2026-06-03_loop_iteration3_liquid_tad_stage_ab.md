---
title: 2026-06-03 Loop iteration 3 - LiquidTAD Hierarchical Decay 落地 (Stage A+B)
date: 2026-06-03
tags: [LNN, LiquidTAD, paper-replication, loop, validation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-03 Loop iteration 3 — LiquidTAD Hierarchical Decay 落地 (Stage A + B)

> 本轮(`/loop 1h` 第三次触发)推进 PRD §8 任务 #2 LiquidTAD 复现的
> Stage A(算子实现 + 单测)与 Stage B(smoke 集成)。
> Stage C–E(THUMOS-14 真实数据 + Jetson 量化)留给后续 loop,
> 取决于 CUDA 路径稳定 + 数据准备时间窗口。

## 1. 上游观察

iter#1 写 LiquidTAD 研读报告时,我以为 `parallel_liquid_relaxation` /
`LiquidTADHead` 还不存在;实际 `lnn/core/long_sequence.py` 早已实现了
"并行指数松弛 + depthwise conv + FFN" 三件套(已被 `lnn/core/__init__.py`
导出)。**所以本轮的差异化贡献是论文专属的 Hierarchical Decay-Rate
Sharing Strategy**,这是现有 `LiquidS4Block` 没有的:

- `LiquidS4Block`: retain 门由数据驱动(`sigmoid(W·h)`)— 每个时刻、
  每个样本各不相同,本质是软门控 RNN 的并行近似。
- **HierarchicalDecayLiquidBlock(本轮新增)**: 每个 channel 只学一个
  共享 retain 标量,与时间、样本无关 — 这才是论文里
  "the exponential relaxation prior of liquid neural dynamics" 的精确写法,
  也是参数效率高的根本原因。
- **HierarchicalDecayLiquidTADHead(本轮新增)**: 把多个 block 串成 FPN-like
  金字塔,初始 decay 沿 `init_decay × decay_growth^index` 几何级数增长
  (深层时间感受野大);`--tad-share-decay` 可强制层间 retain 完全共享。

## 2. Stage A 产出 — 算子 + 单测

### 2.1 代码改动

| 文件 | 改动 | 行数 |
|---|---|---:|
| `lnn/core/long_sequence.py` | 新增 `HierarchicalDecayLiquidBlock` 与 `HierarchicalDecayLiquidTADHead` | +138 |
| `lnn/core/__init__.py` | 导出新类 | +4 |
| `tests/test_liquid_tad_hierarchical.py` | 6 个新测试 | +106 |

### 2.2 单测覆盖

```text
tests/test_liquid_tad_hierarchical.py
├── test_hierarchical_decay_block_shapes_and_grad         ✓
├── test_hierarchical_decay_block_retain_in_unit_interval ✓  (init_decay round-trip)
├── test_hierarchical_decay_block_mask_zeroes_padding     ✓
├── test_hierarchical_decay_tad_head_outputs_and_decay_growth ✓
├── test_hierarchical_decay_tad_head_share_decay_ties_params  ✓
└── test_parallel_liquid_relaxation_matches_recurrent_reference ✓
```

`6 passed in 4.97s`(pyenv 3.14.4 + torch 2.11.0+cu130)。
最后一个测试比较关键 — 它用 16 步显式递推 reference 验证仓库已有的
`parallel_liquid_relaxation` 与 `h_t = retain * h_{t-1} + (1-retain) * value`
的并行展开一致(`atol=1e-3, rtol=1e-3` — 因为 cumsum-log 路径的
`prefix.clamp_min(1e-8)` 保护会引入微小数值漂移)。

## 3. Stage B 产出 — smoke 集成 + 对比

### 3.1 CLI 改动

`scripts/experiment_long_sequence.py` 新增 4 个 flag:

```text
--tad-head {data_dependent, hierarchical_decay}    选 head
--tad-init-decay 0.80                             第 0 块 retain
--tad-decay-growth 1.05                           跨块几何增长
--tad-share-decay                                 跨块共享 retain 参数
```

### 3.2 同配置对照(samples=64, seq=48, hidden=16, 3 blocks, 3 epochs, seed=42, cpu)

| Head | 参数量 | Test loss | Test acc |
|---|---:|---:|---:|
| **HierarchicalDecayLiquidTADHead** (新增,论文真原型) | **5,382** | 1.078 | 78.79% |
| LiquidTADHead (data_dependent,默认) | 6,218 | 0.664 | **80.87%** |

### 3.3 读数

1. **参数效率**: 新 head 比 data_dependent 少 13.4% 参数
   (每 block 砍掉了 `retain_proj: Linear(H, H)` 这层)。
2. **smoke 上 acc 反而低 2.1 pp**: 不意外 — 在 64 样本 / 3 epochs 这种极小
   smoke 上,数据驱动 retain 更容易快速收敛;论文的精度优势在 THUMOS-14
   规模(10.82M 参数、大 batch、多 epoch)才能浮现。
3. **复现路线**: Stage C(THUMOS-14 子集)将提供决定性数据。
   本轮 stage A/B 完成意味着 stage C 不再被代码缺失阻塞,
   只剩 (i) 数据准备 (ii) Jetson 空载窗口跑 cuda。

### 3.4 产物

```text
analysis/long_sequence/2026-06-03_223418_long_sequence.{json,md}  # hierarchical_decay
analysis/long_sequence/2026-06-03_223440_long_sequence.{json,md}  # data_dependent baseline
```

## 4. 顺手刷新今日 daily digest

`python3 scripts/daily_lnn_research.py --date 2026-06-03 --max-results 30 --per-query 10`
本时段 API 已恢复(iter#1 / iter#2 都遇到 arXiv 429 + GitHub 403):

```text
papers/repos/models: 25 / 51 / 23
```

仓库数量从 iter#1 的 32 涨到 51,arXiv 仍卡在 25(已是 max-results 上限)。

## 5. PRD §8 进展更新

| # | 任务 | 状态 |
|---|---|---|
| 1 | Jetson CUDA wheel | ✅ iter#2 done(scripts/jetson_cuda_env.sh) |
| 2 | LiquidTAD 复现 | **stage A ✅ + stage B ✅ (本轮)**;stage C-E 待 CUDA 空载窗口 |
| 3 | LFM2.5-1.2B INT4 推理 | pending(等 8GB 显存空 ≥ 2GB) |
| 4 | EMMA 多模态验证 | pending(EMMA agent 远程在做,避免冲突) |
| 5 | Comparative Analysis LNN vs LSTM v2 | pending |
| 6 | GCN-CfC smoke | pending |
| 7 | Pareto sweep PRD 集成 | ✅ iter#2 partial done |
| 8 | Loop 去重 | pending |

## 6. 后续 loop 待办

1. **stage C 启动**: 复用 `scripts/replicate_temporal_dropout.py` 的数据加载模板,
   做 THUMOS-14 50-video 子集 → run hierarchical_decay vs data_dependent
   (epoch=20, hidden=64, num_blocks=4)。
2. **HierarchicalDecayLiquidTADHead 加 ablation flag**: 让 hierarchical/share/data_dependent
   能跑成同一脚本的 3-way 比较。
3. **把 `tests/test_liquid_tad_hierarchical.py` 加入 CI**(如有 `.github/workflows`)。

## 7. 参考

- [[LiquidTAD_Efficient_Temporal_Action_Detection_研读报告]] — iter#2 研读
- [[2026-06-03_loop_iteration2_cuda_fix_pareto]] — iter#2 CUDA 修复
- [[2026-06-03_loop_validation_summary]] — iter#1 验证总结
- [[PRD_LNN_Edge_Research]] — PRD 主文件
- 算子源码: `lnn/core/long_sequence.py` (`HierarchicalDecayLiquidBlock` +
  `HierarchicalDecayLiquidTADHead`)
- 单测: `tests/test_liquid_tad_hierarchical.py`
- smoke 数据: `analysis/long_sequence/2026-06-03_223*_long_sequence.{json,md}`
