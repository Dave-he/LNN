---
title: Jetson LNN 基准验证 - 2026-10-24
date: 2026-10-24
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-24

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 145 MB / available 1016 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (2213 samples @ 100ms)
- 采样窗口时长：242.96s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3260/3968 mW
  - VDD_IN: 8454/9116 mW
  - VDD_SOC: 1588/1711 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 792008 mJ (792.008 J)
  - VDD_IN: 2053965 mJ (2053.965 J)
  - VDD_SOC: 385844 mJ (385.844 J)
- 温度：
  - cpu: 54.0°C (peak 54.8°C)
  - gpu: 54.1°C (peak 54.6°C)
  - soc0: 53.1°C (peak 53.4°C)
  - soc1: 53.6°C (peak 54.0°C)
  - soc2: 52.7°C (peak 53.1°C)
  - tj: 54.2°C (peak 54.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (21 samples @ 100ms)
- 采样窗口时长：2.43s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3314/3412 mW
  - VDD_IN: 8753/8956 mW
  - VDD_SOC: 1674/1708 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 8053 mJ (8.053 J)
  - VDD_IN: 21272 mJ (21.272 J)
  - VDD_SOC: 4069 mJ (4.069 J)
- 温度：
  - cpu: 53.9°C (peak 54.3°C)
  - gpu: 53.9°C (peak 54.2°C)
  - soc0: 52.8°C (peak 52.9°C)
  - soc1: 53.3°C (peak 53.6°C)
  - soc2: 52.4°C (peak 52.5°C)
  - tj: 54.0°C (peak 54.3°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：0.97s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3705/3769 mW
  - VDD_IN: 8877/8916 mW
  - VDD_SOC: 1576/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 3588 mJ (3.588 J)
  - VDD_IN: 8598 mJ (8.598 J)
  - VDD_SOC: 1526 mJ (1.526 J)
- 温度：
  - cpu: 54.4°C (peak 54.8°C)
  - gpu: 54.2°C (peak 54.5°C)
  - soc0: 53.2°C (peak 53.3°C)
  - soc1: 53.7°C (peak 53.9°C)
  - soc2: 52.9°C (peak 53.0°C)
  - tj: 54.5°C (peak 54.8°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (28 samples @ 100ms)
- 采样窗口时长：3.10s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3178/3457 mW
  - VDD_IN: 8291/8558 mW
  - VDD_SOC: 1568/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 9866 mJ (9.866 J)
  - VDD_IN: 25742 mJ (25.742 J)
  - VDD_SOC: 4867 mJ (4.867 J)
- 温度：
  - cpu: 54.0°C (peak 54.3°C)
  - gpu: 54.1°C (peak 54.5°C)
  - soc0: 53.1°C (peak 53.2°C)
  - soc1: 53.6°C (peak 53.8°C)
  - soc2: 52.7°C (peak 52.8°C)
  - tj: 54.1°C (peak 54.5°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (32 samples @ 100ms)
- 采样窗口时长：3.52s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3035/3338 mW
  - VDD_IN: 8204/8678 mW
  - VDD_SOC: 1571/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 10669 mJ (10.669 J)
  - VDD_IN: 28839 mJ (28.839 J)
  - VDD_SOC: 5522 mJ (5.522 J)
- 温度：
  - cpu: 54.0°C (peak 54.2°C)
  - gpu: 54.1°C (peak 54.4°C)
  - soc0: 53.2°C (peak 53.3°C)
  - soc1: 53.6°C (peak 53.8°C)
  - soc2: 52.7°C (peak 52.9°C)
  - tj: 54.1°C (peak 54.4°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (161 samples @ 100ms)
- 采样窗口时长：18.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3280/3610 mW
  - VDD_IN: 8536/8956 mW
  - VDD_SOC: 1608/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 59482 mJ (59.482 J)
  - VDD_IN: 154769 mJ (154.769 J)
  - VDD_SOC: 29148 mJ (29.148 J)
- 温度：
  - cpu: 54.1°C (peak 54.5°C)
  - gpu: 54.1°C (peak 54.5°C)
  - soc0: 53.2°C (peak 53.4°C)
  - soc1: 53.7°C (peak 53.9°C)
  - soc2: 52.8°C (peak 53.0°C)
  - tj: 54.2°C (peak 54.6°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (20 samples @ 100ms)
- 采样窗口时长：2.25s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3455/3729 mW
  - VDD_IN: 8676/8956 mW
  - VDD_SOC: 1574/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 7760 mJ (7.760 J)
  - VDD_IN: 19488 mJ (19.488 J)
  - VDD_SOC: 3535 mJ (3.535 J)
- 温度：
  - cpu: 54.2°C (peak 54.5°C)
  - gpu: 54.1°C (peak 54.4°C)
  - soc0: 53.2°C (peak 53.3°C)
  - soc1: 53.6°C (peak 53.8°C)
  - soc2: 52.8°C (peak 52.9°C)
  - tj: 54.3°C (peak 54.5°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 7595.8 | 18.27 | 1.44 |
| LTC | 1321 | 0.465366 | 6369.6 | 42.64 | 1.74 |
| PDNAPulse | 3170 | 0.286117 | 7934.6 | 14.80 | 1.32 |
| GRU | 1969 | 0.393358 | 21039.0 | 7.10 | 0.58 |
| NCPS-LTC | 2547 | 0.621254 | 1100.5 | 102.76 | 10.47 |
| NCPS-CfC | 15737 | 0.106344 | 5176.8 | 23.74 | 1.95 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
