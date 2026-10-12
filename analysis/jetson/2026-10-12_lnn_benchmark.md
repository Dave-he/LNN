---
title: Jetson LNN 基准验证 - 2026-10-12
date: 2026-10-12
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-12

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 239 MB / available 3268 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (1006 samples @ 100ms)
- 采样窗口时长：107.49s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2881/3105 mW
  - VDD_IN: 7662/8094 mW
  - VDD_SOC: 1505/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 309712 mJ (309.712 J)
  - VDD_IN: 823561 mJ (823.561 J)
  - VDD_SOC: 161755 mJ (161.755 J)
- 温度：
  - cpu: 53.1°C (peak 53.9°C)
  - gpu: 53.4°C (peak 54.1°C)
  - soc0: 52.5°C (peak 53.0°C)
  - soc1: 52.9°C (peak 53.5°C)
  - soc2: 51.9°C (peak 52.5°C)
  - tj: 53.4°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.68s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2886/2985 mW
  - VDD_IN: 7659/7775 mW
  - VDD_SOC: 1508/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1972 mJ (1.972 J)
  - VDD_IN: 5235 mJ (5.235 J)
  - VDD_SOC: 1031 mJ (1.031 J)
- 温度：
  - cpu: 52.6°C (peak 52.8°C)
  - gpu: 52.7°C (peak 53.1°C)
  - soc0: 51.9°C (peak 51.9°C)
  - soc1: 52.3°C (peak 52.4°C)
  - soc2: 51.4°C (peak 51.4°C)
  - tj: 52.8°C (peak 53.1°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.29s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2905/2905 mW
  - VDD_IN: 7675/7695 mW
  - VDD_SOC: 1495/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 851 mJ (0.851 J)
  - VDD_IN: 2247 mJ (2.247 J)
  - VDD_SOC: 438 mJ (0.438 J)
- 温度：
  - cpu: 52.9°C (peak 52.9°C)
  - gpu: 53.2°C (peak 53.3°C)
  - soc0: 52.2°C (peak 52.2°C)
  - soc1: 52.8°C (peak 52.9°C)
  - soc2: 51.9°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (16 samples @ 100ms)
- 采样窗口时长：1.80s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2763/2945 mW
  - VDD_IN: 7555/7735 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4964 mJ (4.964 J)
  - VDD_IN: 13572 mJ (13.572 J)
  - VDD_SOC: 2722 mJ (2.721 J)
- 温度：
  - cpu: 52.8°C (peak 53.1°C)
  - gpu: 53.1°C (peak 53.3°C)
  - soc0: 52.2°C (peak 52.2°C)
  - soc1: 52.6°C (peak 52.7°C)
  - soc2: 51.6°C (peak 51.8°C)
  - tj: 53.1°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (16 samples @ 100ms)
- 采样窗口时长：1.78s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2930/2985 mW
  - VDD_IN: 7747/7814 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5214 mJ (5.214 J)
  - VDD_IN: 13785 mJ (13.785 J)
  - VDD_SOC: 2696 mJ (2.696 J)
- 温度：
  - cpu: 53.5°C (peak 53.7°C)
  - gpu: 53.7°C (peak 54.1°C)
  - soc0: 52.8°C (peak 53.0°C)
  - soc1: 53.3°C (peak 53.4°C)
  - soc2: 52.3°C (peak 52.4°C)
  - tj: 53.8°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (66 samples @ 100ms)
- 采样窗口时长：7.16s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2860/2985 mW
  - VDD_IN: 7611/7695 mW
  - VDD_SOC: 1494/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 20486 mJ (20.486 J)
  - VDD_IN: 54524 mJ (54.525 J)
  - VDD_SOC: 10701 mJ (10.701 J)
- 温度：
  - cpu: 53.4°C (peak 53.8°C)
  - gpu: 53.7°C (peak 54.0°C)
  - soc0: 52.8°C (peak 52.9°C)
  - soc1: 53.2°C (peak 53.4°C)
  - soc2: 52.3°C (peak 52.4°C)
  - tj: 53.7°C (peak 54.0°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.72s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2945/2985 mW
  - VDD_IN: 7722/7775 mW
  - VDD_SOC: 1508/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2128 mJ (2.128 J)
  - VDD_IN: 5580 mJ (5.580 J)
  - VDD_SOC: 1090 mJ (1.090 J)
- 温度：
  - cpu: 53.0°C (peak 53.2°C)
  - gpu: 53.2°C (peak 53.3°C)
  - soc0: 52.3°C (peak 52.4°C)
  - soc1: 52.7°C (peak 52.8°C)
  - soc2: 51.8°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

## 状态
- ok_cpu: None
