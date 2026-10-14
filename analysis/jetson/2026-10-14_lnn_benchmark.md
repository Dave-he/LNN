---
title: Jetson LNN 基准验证 - 2026-10-14
date: 2026-10-14
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-14

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 230 MB / available 4729 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (896 samples @ 100ms)
- 采样窗口时长：95.57s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2879/3105 mW
  - VDD_IN: 7954/8452 mW
  - VDD_SOC: 1557/1634 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 275148 mJ (275.149 J)
  - VDD_IN: 760099 mJ (760.099 J)
  - VDD_SOC: 148800 mJ (148.800 J)
- 温度：
  - cpu: 51.4°C (peak 52.6°C)
  - gpu: 51.5°C (peak 52.6°C)
  - soc0: 50.7°C (peak 51.7°C)
  - soc1: 50.9°C (peak 52.0°C)
  - soc2: 50.1°C (peak 51.1°C)
  - tj: 51.6°C (peak 52.6°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.69s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2892/2945 mW
  - VDD_IN: 7947/7974 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2001 mJ (2.001 J)
  - VDD_IN: 5499 mJ (5.499 J)
  - VDD_SOC: 1076 mJ (1.076 J)
- 温度：
  - cpu: 50.5°C (peak 50.7°C)
  - gpu: 50.7°C (peak 50.9°C)
  - soc0: 49.8°C (peak 49.9°C)
  - soc1: 50.0°C (peak 50.2°C)
  - soc2: 49.2°C (peak 49.3°C)
  - tj: 50.7°C (peak 50.9°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.25s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2945/2985 mW
  - VDD_IN: 8014/8054 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 739 mJ (0.739 J)
  - VDD_IN: 2010 mJ (2.010 J)
  - VDD_SOC: 390 mJ (0.390 J)
- 温度：
  - cpu: 51.2°C (peak 51.2°C)
  - gpu: 51.4°C (peak 51.4°C)
  - soc0: 50.5°C (peak 50.5°C)
  - soc1: 50.8°C (peak 50.9°C)
  - soc2: 50.0°C (peak 50.0°C)
  - tj: 51.4°C (peak 51.4°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (17 samples @ 100ms)
- 采样窗口时长：1.85s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2802/2866 mW
  - VDD_IN: 7887/7934 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5173 mJ (5.173 J)
  - VDD_IN: 14559 mJ (14.559 J)
  - VDD_SOC: 2870 mJ (2.870 J)
- 温度：
  - cpu: 51.0°C (peak 51.5°C)
  - gpu: 51.1°C (peak 51.4°C)
  - soc0: 50.3°C (peak 50.4°C)
  - soc1: 50.5°C (peak 50.7°C)
  - soc2: 49.6°C (peak 49.8°C)
  - tj: 51.1°C (peak 51.5°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (12 samples @ 100ms)
- 采样窗口时长：1.37s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2849/2905 mW
  - VDD_IN: 7904/7934 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 3899 mJ (3.899 J)
  - VDD_IN: 10816 mJ (10.816 J)
  - VDD_SOC: 2128 mJ (2.128 J)
- 温度：
  - cpu: 52.1°C (peak 52.4°C)
  - gpu: 52.4°C (peak 52.6°C)
  - soc0: 51.5°C (peak 51.6°C)
  - soc1: 51.8°C (peak 52.0°C)
  - soc2: 50.9°C (peak 51.0°C)
  - tj: 52.3°C (peak 52.6°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (59 samples @ 100ms)
- 采样窗口时长：6.30s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2903/3065 mW
  - VDD_IN: 7958/8133 mW
  - VDD_SOC: 1556/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 18284 mJ (18.284 J)
  - VDD_IN: 50113 mJ (50.113 J)
  - VDD_SOC: 9797 mJ (9.797 J)
- 温度：
  - cpu: 52.0°C (peak 52.4°C)
  - gpu: 52.2°C (peak 52.6°C)
  - soc0: 51.3°C (peak 51.5°C)
  - soc1: 51.6°C (peak 51.8°C)
  - soc2: 50.8°C (peak 50.9°C)
  - tj: 52.2°C (peak 52.6°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.61s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2921/3025 mW
  - VDD_IN: 7974/8054 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1773 mJ (1.773 J)
  - VDD_IN: 4839 mJ (4.839 J)
  - VDD_SOC: 944 mJ (0.944 J)
- 温度：
  - cpu: 51.1°C (peak 51.2°C)
  - gpu: 51.3°C (peak 51.5°C)
  - soc0: 50.4°C (peak 50.5°C)
  - soc1: 50.7°C (peak 50.9°C)
  - soc2: 49.9°C (peak 50.0°C)
  - tj: 51.3°C (peak 51.5°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 26793.0 | 5.72 | 0.37 |
| LTC | 1321 | 0.465366 | 9900.8 | 14.52 | 0.98 |
| PDNAPulse | 3170 | 0.286117 | 31170.6 | 6.19 | 0.33 |
| GRU | 1969 | 0.393358 | 73421.0 | 2.04 | 0.14 |
| NCPS-LTC | 2547 | 0.621254 | 3060.0 | 44.63 | 3.39 |
| NCPS-CfC | 15737 | 0.106344 | 13305.1 | 9.11 | 0.73 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
