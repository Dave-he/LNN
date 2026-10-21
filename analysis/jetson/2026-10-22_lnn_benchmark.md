---
title: Jetson LNN 基准验证 - 2026-10-22
date: 2026-10-22
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-22

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 154 MB / available 1779 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (1246 samples @ 100ms)
- 采样窗口时长：137.04s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3484/3934 mW
  - VDD_IN: 8451/9036 mW
  - VDD_SOC: 1551/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 477463 mJ (477.463 J)
  - VDD_IN: 1158107 mJ (1158.107 J)
  - VDD_SOC: 212556 mJ (212.556 J)
- 温度：
  - cpu: 53.5°C (peak 54.3°C)
  - gpu: 53.5°C (peak 54.1°C)
  - soc0: 52.4°C (peak 52.9°C)
  - soc1: 52.9°C (peak 53.4°C)
  - soc2: 52.1°C (peak 52.5°C)
  - tj: 53.6°C (peak 54.3°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (9 samples @ 100ms)
- 采样窗口时长：1.09s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3550/3696 mW
  - VDD_IN: 8600/8717 mW
  - VDD_SOC: 1592/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 3869 mJ (3.869 J)
  - VDD_IN: 9373 mJ (9.373 J)
  - VDD_SOC: 1735 mJ (1.735 J)
- 温度：
  - cpu: 53.4°C (peak 53.6°C)
  - gpu: 53.1°C (peak 53.3°C)
  - soc0: 52.1°C (peak 52.2°C)
  - soc1: 52.5°C (peak 52.6°C)
  - soc2: 51.8°C (peak 51.8°C)
  - tj: 53.4°C (peak 53.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.41s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3616/3656 mW
  - VDD_IN: 8621/8678 mW
  - VDD_SOC: 1552/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1484 mJ (1.484 J)
  - VDD_IN: 3537 mJ (3.537 J)
  - VDD_SOC: 637 mJ (0.637 J)
- 温度：
  - cpu: 53.7°C (peak 53.9°C)
  - gpu: 53.5°C (peak 53.6°C)
  - soc0: 52.4°C (peak 52.5°C)
  - soc1: 53.0°C (peak 53.1°C)
  - soc2: 52.1°C (peak 52.2°C)
  - tj: 53.6°C (peak 53.9°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (20 samples @ 100ms)
- 采样窗口时长：2.41s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3541/3696 mW
  - VDD_IN: 8460/8572 mW
  - VDD_SOC: 1534/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 8518 mJ (8.518 J)
  - VDD_IN: 20352 mJ (20.352 J)
  - VDD_SOC: 3690 mJ (3.690 J)
- 温度：
  - cpu: 53.4°C (peak 53.7°C)
  - gpu: 53.3°C (peak 53.5°C)
  - soc0: 52.3°C (peak 52.4°C)
  - soc1: 52.7°C (peak 52.9°C)
  - soc2: 52.0°C (peak 52.1°C)
  - tj: 53.5°C (peak 53.7°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (17 samples @ 100ms)
- 采样窗口时长：1.89s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3628/3775 mW
  - VDD_IN: 8604/8717 mW
  - VDD_SOC: 1550/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 6865 mJ (6.865 J)
  - VDD_IN: 16281 mJ (16.281 J)
  - VDD_SOC: 2932 mJ (2.933 J)
- 温度：
  - cpu: 53.9°C (peak 54.1°C)
  - gpu: 53.7°C (peak 54.0°C)
  - soc0: 52.7°C (peak 52.8°C)
  - soc1: 53.2°C (peak 53.3°C)
  - soc2: 52.3°C (peak 52.4°C)
  - tj: 53.9°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (70 samples @ 100ms)
- 采样窗口时长：7.52s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3394/3537 mW
  - VDD_IN: 8304/8598 mW
  - VDD_SOC: 1534/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 25522 mJ (25.521 J)
  - VDD_IN: 62451 mJ (62.451 J)
  - VDD_SOC: 11540 mJ (11.540 J)
- 温度：
  - cpu: 53.7°C (peak 54.1°C)
  - gpu: 53.6°C (peak 54.0°C)
  - soc0: 52.6°C (peak 52.8°C)
  - soc1: 53.1°C (peak 53.3°C)
  - soc2: 52.2°C (peak 52.4°C)
  - tj: 53.7°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (10 samples @ 100ms)
- 采样窗口时长：1.18s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3445/3537 mW
  - VDD_IN: 8412/8452 mW
  - VDD_SOC: 1556/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4077 mJ (4.077 J)
  - VDD_IN: 9954 mJ (9.954 J)
  - VDD_SOC: 1841 mJ (1.841 J)
- 温度：
  - cpu: 53.5°C (peak 53.8°C)
  - gpu: 53.4°C (peak 53.7°C)
  - soc0: 52.4°C (peak 52.6°C)
  - soc1: 52.8°C (peak 53.1°C)
  - soc2: 52.0°C (peak 52.1°C)
  - tj: 53.6°C (peak 53.8°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 16767.6 | 10.27 | 0.63 |
| LTC | 1321 | 0.465366 | 7872.4 | 23.68 | 1.38 |
| PDNAPulse | 3170 | 0.286117 | 15379.8 | 9.06 | 0.67 |
| GRU | 1969 | 0.393358 | 51735.3 | 4.16 | 0.24 |
| NCPS-LTC | 2547 | 0.621254 | 2497.8 | 59.80 | 4.22 |
| NCPS-CfC | 15737 | 0.106344 | 9628.5 | 13.07 | 1.10 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
