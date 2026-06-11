---
title: Jetson LNN 基准验证 - 2026-06-11
date: 2026-06-11
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-06-11

## 环境
- 平台：macOS-15.7.7-x86_64-i386-64bit
- 设备树型号：unknown
- PyTorch：2.2.2
- CUDA：False (None)
- Jetson BSP：unknown

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 452917.6 | 0.45 |
| LTC | 1321 | 0.464351 | 104187.4 | 1.15 |
| PDNAPulse | 3170 | 0.284479 | 475660.1 | 0.38 |
| GRU | 1969 | 0.394308 | 975704.0 | 0.16 |

## Benchmark 图
![Jetson LNN Benchmark](2026-06-11_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `GRU` 是同等隐藏维度的传统循环网络基线，便于比较参数量、误差和吞吐。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
