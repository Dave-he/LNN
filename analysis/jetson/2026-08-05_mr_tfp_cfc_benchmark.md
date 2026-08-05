---
title: MR-TFP-CfC vs MR-MoE-CfC vs 单 expert baseline - 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, multi-rate, MR-MoE, cross-paper, retention, second-layer-synthesis]
---

# MR-TFP-CfC vs MR-MoE-CfC vs 单 expert baseline - 2026-08-05

## 任务
合成 **非平稳 AR(2) + 3-regime** 时间序列 (与上轮 benchmark 同 task 配置)。

## 结果（3 次重复 mean±std）

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|

| CfC-baseline | 1041 | 0.0550 ± 0.0000 | 2621.1 | 4.74 |

| MFC-TFP | 1025 | 0.0571 ± 0.0006 | 2690.3 | 5.33 |

| MR-MoE-CfC (n_tau=4) | 481 | 0.0639 ± 0.0037 | 933.9 | 23.29 |

| MR-TFP-CfC (n_tau=4) | 465 | 0.0709 ± 0.0018 | 409.4 | 108.33 |

| MR-TFP-CfC (n_tau=4, k=1) | 465 | 0.0714 ± 0.0013 | 373.6 | 131.71 |


## 解读
- **CfC-baseline** 与 **MFC-TFP** 是单 expert 上轮 benchmark 的对照。
- **MR-MoE-CfC (n_tau=4)** 是上轮已落地的多速率 MoE (2606.12240)。
- **MR-TFP-CfC (n_tau=4)** 是本轮新落地的第二层综合（2606.12240 × 2607.08283）。
- **MR-TFP-CfC (n_tau=4, k=1)** 强制 top-K=1，验证最稀疏 routing 下的表现。
