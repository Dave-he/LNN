---
title: Jetson LNN 基准验证 - 2026-10-23
date: 2026-10-23
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-23

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 94 MB / available 1752 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (2404 samples @ 100ms)
- 采样窗口时长：280.00s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3858/4364 mW
  - VDD_IN: 9042/9474 mW
  - VDD_SOC: 1580/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1080157 mJ (1080.157 J)
  - VDD_IN: 2531818 mJ (2531.818 J)
  - VDD_SOC: 442352 mJ (442.352 J)
- 温度：
  - cpu: 55.5°C (peak 56.1°C)
  - gpu: 55.3°C (peak 55.8°C)
  - soc0: 54.3°C (peak 54.7°C)
  - soc1: 54.8°C (peak 55.2°C)
  - soc2: 54.0°C (peak 54.4°C)
  - tj: 55.5°C (peak 56.1°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (18 samples @ 100ms)
- 采样窗口时长：2.11s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 4018/4047 mW
  - VDD_IN: 9072/9116 mW
  - VDD_SOC: 1545/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 8498 mJ (8.498 J)
  - VDD_IN: 19185 mJ (19.185 J)
  - VDD_SOC: 3268 mJ (3.268 J)
- 温度：
  - cpu: 55.6°C (peak 55.9°C)
  - gpu: 55.4°C (peak 55.6°C)
  - soc0: 54.4°C (peak 54.5°C)
  - soc1: 54.9°C (peak 55.0°C)
  - soc2: 54.1°C (peak 54.2°C)
  - tj: 55.6°C (peak 55.8°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：1.06s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3908/3968 mW
  - VDD_IN: 9110/9155 mW
  - VDD_SOC: 1591/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4136 mJ (4.136 J)
  - VDD_IN: 9642 mJ (9.642 J)
  - VDD_SOC: 1684 mJ (1.684 J)
- 温度：
  - cpu: 55.7°C (peak 56.0°C)
  - gpu: 55.3°C (peak 55.5°C)
  - soc0: 54.3°C (peak 54.4°C)
  - soc1: 54.8°C (peak 55.0°C)
  - soc2: 54.0°C (peak 54.1°C)
  - tj: 55.7°C (peak 56.0°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (26 samples @ 100ms)
- 采样窗口时长：3.21s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3878/4007 mW
  - VDD_IN: 8974/9076 mW
  - VDD_SOC: 1563/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 12442 mJ (12.442 J)
  - VDD_IN: 28796 mJ (28.796 J)
  - VDD_SOC: 5014 mJ (5.014 J)
- 温度：
  - cpu: 55.6°C (peak 55.8°C)
  - gpu: 55.4°C (peak 55.7°C)
  - soc0: 54.4°C (peak 54.6°C)
  - soc1: 54.9°C (peak 55.1°C)
  - soc2: 54.1°C (peak 54.2°C)
  - tj: 55.6°C (peak 55.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (48 samples @ 100ms)
- 采样窗口时长：5.63s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3716/3888 mW
  - VDD_IN: 8936/9116 mW
  - VDD_SOC: 1604/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 20922 mJ (20.922 J)
  - VDD_IN: 50316 mJ (50.316 J)
  - VDD_SOC: 9030 mJ (9.030 J)
- 温度：
  - cpu: 55.4°C (peak 55.8°C)
  - gpu: 55.2°C (peak 55.6°C)
  - soc0: 54.2°C (peak 54.3°C)
  - soc1: 54.6°C (peak 54.9°C)
  - soc2: 53.9°C (peak 54.0°C)
  - tj: 55.4°C (peak 55.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (120 samples @ 100ms)
- 采样窗口时长：14.59s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3824/4047 mW
  - VDD_IN: 8987/9155 mW
  - VDD_SOC: 1579/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 55788 mJ (55.788 J)
  - VDD_IN: 131100 mJ (131.100 J)
  - VDD_SOC: 23034 mJ (23.034 J)
- 温度：
  - cpu: 55.4°C (peak 55.8°C)
  - gpu: 55.2°C (peak 55.5°C)
  - soc0: 54.2°C (peak 54.4°C)
  - soc1: 54.6°C (peak 54.9°C)
  - soc2: 53.9°C (peak 54.0°C)
  - tj: 55.4°C (peak 55.8°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (19 samples @ 100ms)
- 采样窗口时长：2.36s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3758/3968 mW
  - VDD_IN: 9107/9394 mW
  - VDD_SOC: 1610/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 8886 mJ (8.886 J)
  - VDD_IN: 21532 mJ (21.532 J)
  - VDD_SOC: 3806 mJ (3.806 J)
- 温度：
  - cpu: 55.5°C (peak 55.9°C)
  - gpu: 55.3°C (peak 55.4°C)
  - soc0: 54.3°C (peak 54.4°C)
  - soc1: 54.8°C (peak 55.0°C)
  - soc2: 54.0°C (peak 54.1°C)
  - tj: 55.5°C (peak 55.9°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 8517.4 | 18.57 | 1.30 |
| LTC | 1321 | 0.465366 | 5619.9 | 43.32 | 1.95 |
| PDNAPulse | 3170 | 0.286117 | 8395.3 | 27.49 | 1.46 |
| GRU | 1969 | 0.393358 | 17849.4 | 9.42 | 0.65 |
| NCPS-LTC | 2547 | 0.621254 | 1212.8 | 121.89 | 8.87 |
| NCPS-CfC | 15737 | 0.106344 | 3450.7 | 27.66 | 3.40 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
