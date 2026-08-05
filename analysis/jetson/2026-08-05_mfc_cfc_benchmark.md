---
title: MemoryFusionCfC 三模式对比 vs CfC/LTC/GRU - 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, NSFD, cross-paper, retention, memory-fusion, benchmark]
---

# MemoryFusionCfC 三模式对比 vs CfC/LTC/GRU - 2026-08-05

## 任务
合成 **非平稳 AR(2) + 3-regime** 时间序列。模型需要在一阶/二阶系数不规则切换的条件下做下一步回归。

## 结果（3 次重复 mean±std）

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| CfC | 2137 | 0.0589 ± 0.0000 | 4698.3 | 15.51 |
| MFC-CFC | 2137 | 0.0590 ± 0.0001 | 4737.9 | 16.16 |
| MFC-TFP | 2113 | 0.0581 ± 0.0006 | 4470.9 | 15.71 |
| MFC-NSFD | 2809 | 0.0707 ± 0.0093 | 2945.8 | 18.90 |
| LTC | 1465 | 0.0617 ± 0.0093 | 1283.5 | 53.72 |
| GRU | 2185 | 0.0575 ± 0.0023 | 12516.1 | 7.94 |

## 解读

- 同一 `MemoryFusionCfCCell` 仅切换 `retention_kind`，权重初始化相同（同 seed），因此 MSE 差异直接反映 *retention 机制* 的优劣。

- **MFC-CFC** 应与原 **CfC** 在统计误差范围内一致（数值等价声明）。

- **MFC-TFP** 与 **MFC-NSFD** 是 N3 / N2 论文机制的 CfC 化重参数化。

- **LTC** / **GRU** 作为传统 baseline。
