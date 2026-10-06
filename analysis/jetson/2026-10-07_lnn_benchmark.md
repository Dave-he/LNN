---
title: Jetson LNN 基准验证 - 2026-10-07
date: 2026-10-07
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-07

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 94 MB / available 5212 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (188 samples @ 100ms)
- 采样窗口时长：19.89s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1379/1717 mW
  - VDD_IN: 5992/6600 mW
  - VDD_SOC: 1445/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 27432 mJ (27.432 J)
  - VDD_IN: 119168 mJ (119.168 J)
  - VDD_SOC: 28744 mJ (28.744 J)
- 温度：
  - cpu: 47.0°C (peak 47.7°C)
  - gpu: 47.6°C (peak 48.2°C)
  - soc0: 46.9°C (peak 47.2°C)
  - soc1: 47.0°C (peak 47.4°C)
  - soc2: 46.1°C (peak 46.4°C)
  - tj: 47.7°C (peak 48.2°C)
- GPU 利用率：mean 20%, peak 33%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.28s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1340/1360 mW
  - VDD_IN: 5940/5960 mW
  - VDD_SOC: 1440/1440 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 376 mJ (0.376 J)
  - VDD_IN: 1669 mJ (1.669 J)
  - VDD_SOC: 404 mJ (0.405 J)
- 温度：
  - cpu: 46.8°C (peak 46.9°C)
  - gpu: 47.5°C (peak 47.5°C)
  - soc0: 46.8°C (peak 46.8°C)
  - soc1: 47.0°C (peak 47.1°C)
  - soc2: 46.0°C (peak 46.1°C)
  - tj: 47.5°C (peak 47.5°C)
- GPU 利用率：mean 24%, peak 26%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (12 samples @ 100ms)
- 采样窗口时长：1.36s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1450/1477 mW
  - VDD_IN: 6027/6080 mW
  - VDD_SOC: 1440/1440 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1980 mJ (1.980 J)
  - VDD_IN: 8225 mJ (8.225 J)
  - VDD_SOC: 1965 mJ (1.965 J)
- 温度：
  - cpu: 47.0°C (peak 47.1°C)
  - gpu: 47.8°C (peak 48.0°C)
  - soc0: 47.0°C (peak 47.2°C)
  - soc1: 47.2°C (peak 47.3°C)
  - soc2: 46.2°C (peak 46.3°C)
  - tj: 47.8°C (peak 48.0°C)
- GPU 利用率：mean 20%, peak 21%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.25s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1378/1397 mW
  - VDD_IN: 5940/5960 mW
  - VDD_SOC: 1440/1440 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 349 mJ (0.349 J)
  - VDD_IN: 1503 mJ (1.503 J)
  - VDD_SOC: 364 mJ (0.364 J)
- 温度：
  - cpu: 46.9°C (peak 46.9°C)
  - gpu: 48.0°C (peak 48.2°C)
  - soc0: 47.2°C (peak 47.2°C)
  - soc1: 47.2°C (peak 47.2°C)
  - soc2: 46.3°C (peak 46.3°C)
  - tj: 48.0°C (peak 48.2°C)
- GPU 利用率：mean 27%, peak 28%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.405097 | 71426.2 | 3.51 | 0.11 |
| LTC | 1321 | 0.358536 | 13843.1 | 9.61 | 0.56 |
| PDNAPulse | 3170 | 0.302686 | 81041.4 | 2.31 | 0.10 |
| GRU | 1969 | 0.388883 | 2904838.8 | 0.72 | n/a |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
