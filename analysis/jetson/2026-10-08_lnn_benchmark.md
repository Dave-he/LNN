---
title: Jetson LNN 基准验证 - 2026-10-08
date: 2026-10-08
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-08

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 187 MB / available 4795 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (192 samples @ 100ms)
- 采样窗口时长：20.34s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1398/1717 mW
  - VDD_IN: 6026/6720 mW
  - VDD_SOC: 1449/1560 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 28447 mJ (28.447 J)
  - VDD_IN: 122595 mJ (122.595 J)
  - VDD_SOC: 29476 mJ (29.476 J)
- 温度：
  - cpu: 46.9°C (peak 47.7°C)
  - gpu: 47.7°C (peak 48.2°C)
  - soc0: 46.9°C (peak 47.3°C)
  - soc1: 47.0°C (peak 47.4°C)
  - soc2: 46.1°C (peak 46.3°C)
  - tj: 47.7°C (peak 48.2°C)
- GPU 利用率：mean 20%, peak 45%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.30s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1397/1397 mW
  - VDD_IN: 6040/6040 mW
  - VDD_SOC: 1460/1480 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 423 mJ (0.423 J)
  - VDD_IN: 1829 mJ (1.829 J)
  - VDD_SOC: 442 mJ (0.442 J)
- 温度：
  - cpu: 46.6°C (peak 46.8°C)
  - gpu: 47.8°C (peak 47.9°C)
  - soc0: 46.8°C (peak 46.9°C)
  - soc1: 47.0°C (peak 47.1°C)
  - soc2: 46.0°C (peak 46.1°C)
  - tj: 47.8°C (peak 47.9°C)
- GPU 利用率：mean 22%, peak 26%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (13 samples @ 100ms)
- 采样窗口时长：1.39s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1462/1477 mW
  - VDD_IN: 6046/6080 mW
  - VDD_SOC: 1440/1440 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2037 mJ (2.037 J)
  - VDD_IN: 8426 mJ (8.427 J)
  - VDD_SOC: 2007 mJ (2.007 J)
- 温度：
  - cpu: 47.1°C (peak 47.3°C)
  - gpu: 47.9°C (peak 48.1°C)
  - soc0: 47.1°C (peak 47.2°C)
  - soc1: 47.2°C (peak 47.3°C)
  - soc2: 46.2°C (peak 46.3°C)
  - tj: 47.8°C (peak 48.1°C)
- GPU 利用率：mean 19%, peak 20%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.24s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1437/1437 mW
  - VDD_IN: 6040/6040 mW
  - VDD_SOC: 1440/1440 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 346 mJ (0.346 J)
  - VDD_IN: 1456 mJ (1.456 J)
  - VDD_SOC: 347 mJ (0.347 J)
- 温度：
  - cpu: 47.4°C (peak 47.6°C)
  - gpu: 47.9°C (peak 48.2°C)
  - soc0: 47.2°C (peak 47.2°C)
  - soc1: 47.3°C (peak 47.3°C)
  - soc2: 46.3°C (peak 46.4°C)
  - tj: 47.9°C (peak 48.2°C)
- GPU 利用率：mean 26%, peak 27%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.405097 | 68382.7 | 3.61 | 0.12 |
| LTC | 1321 | 0.358536 | 13369.4 | 9.91 | 0.57 |
| PDNAPulse | 3170 | 0.302686 | 78638.3 | 2.29 | 0.10 |
| GRU | 1969 | 0.388883 | 2820768.4 | 0.71 | n/a |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
