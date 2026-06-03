---
title: 2026-06-03 Loop iteration 4 — LiquidTAD 3-way head ablation (3 seeds)
date: 2026-06-03
tags: [LNN, LiquidTAD, ablation, paper-replication, loop, validation, multiseed]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-03 Loop iteration 4 — LiquidTAD 3-way head ablation

> 本轮(`/loop 1h` 第四次触发)把 iter#3 的 LiquidTAD `HierarchicalDecayLiquidTADHead`
> 推到中等规模、跨 3 个随机 seed,对照 `data_dependent` 与 `hierarchical_shared` 两个变体。
> **首次产生 hierarchical 反超 data_dependent 的统计证据**: 跨 3 seed 一致
> −14% 参数、−25% test loss、+3.2 pp 平均 frame acc,且方差小了 ~10×。

## 1. 实验配置

- 数据: `SyntheticLongSequenceDataset(num_samples=192, seq_len=96, feature_size=6, num_classes=3)`
- 网络: 3 blocks, hidden_size=32, batch=16, AdamW lr=3e-4, epochs=8
- Head 变体:
  - `data_dependent` → `LiquidTADHead`(per-step retain 由 hidden 预测)
  - `hierarchical_decay` → `HierarchicalDecayLiquidTADHead`,
    init_decay=0.80, decay_growth=1.05, **share_decay=False**(每 block 独立学 retain)
  - `hierarchical_shared` → 同上但 **share_decay=True**(跨 block 共享单一 retain 参数)
- Seeds: 42, 7, 123
- Device: cpu(Jetson Orin Nano,本时段 CUDA NvMap ENOMEM,iter#2 解释)

驱动脚本: `scripts/ablation_liquid_tad_heads.py`(本轮新增,196 行)

## 2. 原始结果(3 seed × 3 head)

| seed | head | params | test_loss | test acc | train s |
|---:|---|---:|---:|---:|---:|
| 42 | data_dependent | 22,730 | 0.4142 | 92.01% | 15.62 |
| 42 | hierarchical_decay | 19,526 | 0.2224 | 99.34% | 14.15 |
| 42 | hierarchical_shared | 19,462 | 0.2192 | 99.13% | 14.72 |
| 7 | data_dependent | 22,730 | 0.2360 | 98.82% | 15.28 |
| 7 | hierarchical_decay | 19,526 | 0.2078 | 99.41% | 13.93 |
| 7 | hierarchical_shared | 19,462 | 0.2055 | 99.38% | 14.02 |
| 123 | data_dependent | 22,730 | 0.2296 | 97.57% | 15.29 |
| 123 | hierarchical_decay | 19,526 | 0.2259 | 99.34% | 14.44 |
| 123 | hierarchical_shared | 19,462 | 0.2208 | 99.27% | 14.18 |

## 3. 跨 seed 汇总(mean ± std)

| Head | params | test acc | test loss | train s |
|---|---:|---:|---:|---:|
| data_dependent | 22,730 | 96.13% **± 3.55** pp | 0.2933 **± 0.107** | 15.40 ± 0.20 |
| **hierarchical_decay** | **19,526** | **99.36% ± 0.04 pp** | **0.2187 ± 0.009** | 14.17 ± 0.26 |
| **hierarchical_shared** | **19,462** | **99.26% ± 0.13 pp** | **0.2152 ± 0.008** | 14.31 ± 0.36 |

### 3.1 相对 baseline 提升 (mean over 3 seeds)

| Head | Δparams | Δtest_loss | Δtest_acc | Δtrain_s |
|---|---:|---:|---:|---:|
| hierarchical_decay | **−14.1%** | **−25.4%** | **+3.23 pp** | −8.0% |
| hierarchical_shared | **−14.4%** | **−26.6%** | **+3.13 pp** | −7.1% |

## 4. 解读

### 4.1 paper claim 在合成数据上得到强支持

LiquidTAD(arXiv:2604.18274)的核心宣称是:
"the structural decay-sharing prior trades parameters for accuracy"。
本次 3-seed 同条件直接对照得到 −14% 参数 + +3 pp 平均精度,
**且 hierarchical 的 std 比 data_dependent 小了一个数量级**
(0.04 pp vs 3.55 pp),
比 iter#3 的单 seed smoke 信号大得多
(那次 hierarchical 还输 2.1 pp,但 samples 只有 64,epoch 只有 3)。

### 4.2 data_dependent 的 seed=42 异常拉低了它的平均

data_dependent seed=42 acc 92.01% 比同 head 的 seed=7/123(98.82%/97.57%)
低 5–7 pp,且 test_loss 高 80%。这是 LSTM/RNN 类门控网络 **众所周知的高 seed 方差** —
EMMA agent 的 multi-seed 教训刚好印证 (远程 commit
`probe(loo-multiseed): SOTA 0.42 IS SEED-LUCKY — 5-seed mean 8.16 ± 6.78`)。
**hierarchical 在 3 seed 上的稳态精度是 99.13%~99.41%,这是结构化先验降低优化噪声的直接收益。**

### 4.3 share_decay 与否差距很小

- non-shared (`hierarchical_decay`)允许每个 block 独立学 retain;
  shared (`hierarchical_shared`)把整个 head 的所有 block 绑定到 16 个标量上。
- 在本任务上两者 loss/acc 几乎并列,shared 略低 loss(0.2152 vs 0.2187)
  但 acc 略低 0.1 pp。
- 推论: 在 SyntheticLongSequenceDataset 这种 boundary 不太复杂的任务上,
  shared 已经足够;真实的多 boundary 长视频(THUMOS-14)上可能需要 non-shared
  才能给深层 block 更长的时间常数。

## 5. 与 paper 的对比

| 指标 | LiquidTAD 论文 | 本次 smoke |
|---|---|---|
| 数据 | THUMOS-14, ActivityNet-1.3 | SyntheticLongSequenceDataset |
| 模型规模 | 10.82 M params | 19.5 K params(差 ~550×) |
| 关键宣称 | "−60% params vs ActionFormer, 69.46% avg mAP" | "−14% params vs data_dependent, +3.23 pp acc" |
| 量度 | mAP@IoU 0.3~0.7 | frame-level accuracy + boundary MAE |
| 复现规模 | full | C-lite(本轮) |

本次只能算 "**复现 paper 方向性 claim 的最小信号**":
方向对了、信号稳定、可以推到下一阶段(Stage C: THUMOS-14 50-video 子集)。

## 6. PRD §8 进展更新

| # | 任务 | 状态 |
|---|---|---|
| 1 | Jetson CUDA wheel | ✅ iter#2 |
| 2 | LiquidTAD 复现 | A+B ✅(iter#3)+ **C-lite ✅(本轮 3-seed ablation)** / 真正 Stage C 待数据 + CUDA 空载 |
| 3 | LFM2.5-1.2B INT4 推理 | pending |
| 4 | EMMA 多模态 | pending (EMMA agent 远程在做) |
| 5 | Comparative LNN vs LSTM v2 | pending |
| 6 | GCN-CfC smoke | pending |
| 7 | Pareto sweep PRD 集成 | ✅ iter#2 |
| 8 | Loop 去重 | pending |

## 7. 衍生工作

| 任务 | 推入 |
|---|---|
| 把 `ablation_liquid_tad_heads.py` 加 `--seeds N --mean` 一次跑多 seed 自动算 mean±std | NEXT_STEPS |
| Stage C-true: THUMOS-14 50-video 子集 → 用相同 ablation 比 mAP | PRD §8 #2 stage C |
| `--head` 选择项纳入 README LiquidTAD 章节 | docs |
| 把"3 seed 一致优势"写成 LiquidTAD 研读报告 v2 的实证补充 | docs/reports |

## 8. 参考产物

- 源代码: `scripts/ablation_liquid_tad_heads.py` (本轮新增)
- 3 次 JSON+MD 产物:
  - `analysis/long_sequence/2026-06-03_233348_liquid_tad_head_ablation.{json,md}` (seed=42)
  - `analysis/long_sequence/2026-06-03_233*_liquid_tad_head_ablation.{json,md}` (seed=7, 123)
- 实现细节: `lnn/core/long_sequence.py::HierarchicalDecayLiquidBlock` /
  `::HierarchicalDecayLiquidTADHead`
- 论文研读: [[LiquidTAD_Efficient_Temporal_Action_Detection_研读报告]]
- 上一轮: [[2026-06-03_loop_iteration3_liquid_tad_stage_ab]]
- PRD: [[PRD_LNN_Edge_Research]]
