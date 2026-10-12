---
title: Jetson LNN 基准验证 - 2026-10-13
date: 2026-10-13
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-13

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 285 MB / available 4149 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (898 samples @ 100ms)
- 采样窗口时长：95.90s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2908/3144 mW
  - VDD_IN: 7648/8094 mW
  - VDD_SOC: 1481/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 278909 mJ (278.909 J)
  - VDD_IN: 733408 mJ (733.408 J)
  - VDD_SOC: 142030 mJ (142.030 J)
- 温度：
  - cpu: 52.0°C (peak 52.9°C)
  - gpu: 52.2°C (peak 53.2°C)
  - soc0: 51.2°C (peak 52.1°C)
  - soc1: 51.7°C (peak 52.6°C)
  - soc2: 50.8°C (peak 51.6°C)
  - tj: 52.2°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 11%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.56s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2993/3025 mW
  - VDD_IN: 7724/7747 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1684 mJ (1.684 J)
  - VDD_IN: 4345 mJ (4.345 J)
  - VDD_SOC: 830 mJ (0.830 J)
- 温度：
  - cpu: 51.4°C (peak 51.6°C)
  - gpu: 51.4°C (peak 51.6°C)
  - soc0: 50.3°C (peak 50.4°C)
  - soc1: 50.8°C (peak 50.8°C)
  - soc2: 50.0°C (peak 50.1°C)
  - tj: 51.5°C (peak 51.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.27s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2886/2905 mW
  - VDD_IN: 7607/7627 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 770 mJ (0.770 J)
  - VDD_IN: 2029 mJ (2.029 J)
  - VDD_SOC: 394 mJ (0.394 J)
- 温度：
  - cpu: 51.8°C (peak 52.0°C)
  - gpu: 52.2°C (peak 52.4°C)
  - soc0: 51.0°C (peak 51.0°C)
  - soc1: 51.4°C (peak 51.4°C)
  - soc2: 50.6°C (peak 50.7°C)
  - tj: 52.2°C (peak 52.4°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (17 samples @ 100ms)
- 采样窗口时长：1.86s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2772/2866 mW
  - VDD_IN: 7503/7587 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5144 mJ (5.144 J)
  - VDD_IN: 13924 mJ (13.924 J)
  - VDD_SOC: 2738 mJ (2.737 J)
- 温度：
  - cpu: 51.6°C (peak 51.8°C)
  - gpu: 51.7°C (peak 52.0°C)
  - soc0: 50.8°C (peak 51.0°C)
  - soc1: 51.2°C (peak 51.6°C)
  - soc2: 50.3°C (peak 50.4°C)
  - tj: 51.8°C (peak 52.0°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (15 samples @ 100ms)
- 采样窗口时长：1.69s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2953/3025 mW
  - VDD_IN: 7712/7854 mW
  - VDD_SOC: 1499/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5003 mJ (5.003 J)
  - VDD_IN: 13064 mJ (13.064 J)
  - VDD_SOC: 2539 mJ (2.539 J)
- 温度：
  - cpu: 52.7°C (peak 52.9°C)
  - gpu: 52.9°C (peak 53.2°C)
  - soc0: 51.9°C (peak 52.1°C)
  - soc1: 52.3°C (peak 52.5°C)
  - soc2: 51.4°C (peak 51.6°C)
  - tj: 52.9°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (55 samples @ 100ms)
- 采样窗口时长：5.87s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2939/3025 mW
  - VDD_IN: 7660/7814 mW
  - VDD_SOC: 1480/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 17242 mJ (17.242 J)
  - VDD_IN: 44936 mJ (44.936 J)
  - VDD_SOC: 8682 mJ (8.682 J)
- 温度：
  - cpu: 52.6°C (peak 52.9°C)
  - gpu: 52.8°C (peak 53.2°C)
  - soc0: 51.8°C (peak 52.0°C)
  - soc1: 52.2°C (peak 52.5°C)
  - soc2: 51.3°C (peak 51.4°C)
  - tj: 52.8°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：0.89s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2870/3025 mW
  - VDD_IN: 7626/7814 mW
  - VDD_SOC: 1490/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2548 mJ (2.548 J)
  - VDD_IN: 6768 mJ (6.768 J)
  - VDD_SOC: 1322 mJ (1.322 J)
- 温度：
  - cpu: 51.7°C (peak 51.9°C)
  - gpu: 51.9°C (peak 52.1°C)
  - soc0: 51.0°C (peak 51.2°C)
  - soc1: 51.5°C (peak 51.6°C)
  - soc2: 50.5°C (peak 50.6°C)
  - tj: 51.9°C (peak 52.1°C)
- GPU 利用率：mean 0%, peak 3%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 33433.6 | 5.46 | 0.29 |
| LTC | 1321 | 0.465366 | 10419.0 | 13.79 | 0.94 |
| PDNAPulse | 3170 | 0.286117 | 20452.0 | 6.26 | 0.46 |
| GRU | 1969 | 0.393358 | 70679.1 | 2.16 | 0.14 |
| NCPS-LTC | 2547 | 0.621254 | 3151.8 | 45.31 | 3.04 |
| NCPS-CfC | 15737 | 0.106344 | 10851.0 | 9.48 | 0.88 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
