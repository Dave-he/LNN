---
title: Jetson LNN 基准验证 - 2026-10-03
date: 2026-10-03
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-03

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 85 MB / available 797 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.78 MB

## 功耗与温度
- 功耗采样：可用 (195 samples @ 100ms)
- 采样窗口时长：20.76s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2041/2312 mW
  - VDD_IN: 6851/7268 mW
  - VDD_SOC: 1518/1597 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 42357 mJ (42.357 J)
  - VDD_IN: 142193 mJ (142.193 J)
  - VDD_SOC: 31517 mJ (31.517 J)
- 温度：
  - cpu: 49.7°C (peak 50.3°C)
  - gpu: 50.0°C (peak 50.4°C)
  - soc0: 49.2°C (peak 49.6°C)
  - soc1: 49.5°C (peak 49.8°C)
  - soc2: 48.5°C (peak 48.8°C)
  - tj: 50.0°C (peak 50.4°C)
- GPU 利用率：mean 20%, peak 32%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.26s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2113/2113 mW
  - VDD_IN: 6908/6908 mW
  - VDD_SOC: 1517/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 548 mJ (0.548 J)
  - VDD_IN: 1792 mJ (1.792 J)
  - VDD_SOC: 394 mJ (0.394 J)
- 温度：
  - cpu: 49.5°C (peak 49.6°C)
  - gpu: 50.1°C (peak 50.2°C)
  - soc0: 49.1°C (peak 49.2°C)
  - soc1: 49.4°C (peak 49.6°C)
  - soc2: 48.4°C (peak 48.5°C)
  - tj: 50.1°C (peak 50.2°C)
- GPU 利用率：mean 26%, peak 26%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (13 samples @ 100ms)
- 采样窗口时长：1.39s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1960/2073 mW
  - VDD_IN: 6709/6868 mW
  - VDD_SOC: 1505/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2726 mJ (2.726 J)
  - VDD_IN: 9330 mJ (9.330 J)
  - VDD_SOC: 2092 mJ (2.092 J)
- 温度：
  - cpu: 49.8°C (peak 50.1°C)
  - gpu: 50.1°C (peak 50.4°C)
  - soc0: 49.4°C (peak 49.5°C)
  - soc1: 49.6°C (peak 49.8°C)
  - soc2: 48.6°C (peak 48.7°C)
  - tj: 50.2°C (peak 50.4°C)
- GPU 利用率：mean 20%, peak 20%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.29s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2053/2073 mW
  - VDD_IN: 6848/6868 mW
  - VDD_SOC: 1517/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 587 mJ (0.587 J)
  - VDD_IN: 1957 mJ (1.957 J)
  - VDD_SOC: 434 mJ (0.433 J)
- 温度：
  - cpu: 49.7°C (peak 49.8°C)
  - gpu: 50.1°C (peak 50.1°C)
  - soc0: 49.4°C (peak 49.5°C)
  - soc1: 49.6°C (peak 49.6°C)
  - soc2: 48.8°C (peak 48.8°C)
  - tj: 50.2°C (peak 50.4°C)
- GPU 利用率：mean 18%, peak 26%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.405097 | 73254.8 | 3.62 | 0.12 |
| LTC | 1321 | 0.358536 | 13470.9 | 10.14 | 0.63 |
| PDNAPulse | 3170 | 0.302686 | 76648.6 | 2.39 | 0.13 |
| GRU | 1969 | 0.388883 | 4637957.6 | 0.77 | n/a |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
