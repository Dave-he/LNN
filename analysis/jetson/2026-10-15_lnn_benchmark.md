---
title: Jetson LNN 基准验证 - 2026-10-15
date: 2026-10-15
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-15

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 253 MB / available 4680 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (898 samples @ 100ms)
- 采样窗口时长：95.81s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2871/3065 mW
  - VDD_IN: 7591/8014 mW
  - VDD_SOC: 1477/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 275083 mJ (275.083 J)
  - VDD_IN: 727285 mJ (727.285 J)
  - VDD_SOC: 141560 mJ (141.560 J)
- 温度：
  - cpu: 51.4°C (peak 52.2°C)
  - gpu: 51.5°C (peak 52.5°C)
  - soc0: 50.6°C (peak 51.4°C)
  - soc1: 50.9°C (peak 51.8°C)
  - soc2: 50.1°C (peak 50.9°C)
  - tj: 51.5°C (peak 52.5°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.59s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2905/2945 mW
  - VDD_IN: 7603/7627 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1706 mJ (1.706 J)
  - VDD_IN: 4465 mJ (4.465 J)
  - VDD_SOC: 866 mJ (0.866 J)
- 温度：
  - cpu: 50.7°C (peak 50.8°C)
  - gpu: 50.6°C (peak 50.7°C)
  - soc0: 49.7°C (peak 49.8°C)
  - soc1: 50.1°C (peak 50.2°C)
  - soc2: 49.3°C (peak 49.3°C)
  - tj: 50.7°C (peak 50.8°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.26s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2846/2866 mW
  - VDD_IN: 7547/7547 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 731 mJ (0.731 J)
  - VDD_IN: 1937 mJ (1.937 J)
  - VDD_SOC: 379 mJ (0.379 J)
- 温度：
  - cpu: 51.2°C (peak 51.3°C)
  - gpu: 51.4°C (peak 51.4°C)
  - soc0: 50.5°C (peak 50.5°C)
  - soc1: 50.7°C (peak 50.8°C)
  - soc2: 49.9°C (peak 49.9°C)
  - tj: 51.4°C (peak 51.4°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (17 samples @ 100ms)
- 采样窗口时长：1.84s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2729/2826 mW
  - VDD_IN: 7444/7547 mW
  - VDD_SOC: 1477/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5029 mJ (5.029 J)
  - VDD_IN: 13718 mJ (13.718 J)
  - VDD_SOC: 2722 mJ (2.722 J)
- 温度：
  - cpu: 50.9°C (peak 51.3°C)
  - gpu: 51.1°C (peak 51.3°C)
  - soc0: 50.2°C (peak 50.3°C)
  - soc1: 50.5°C (peak 50.7°C)
  - soc2: 49.6°C (peak 49.7°C)
  - tj: 51.1°C (peak 51.3°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (13 samples @ 100ms)
- 采样窗口时长：1.47s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2884/2945 mW
  - VDD_IN: 7618/7707 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4247 mJ (4.247 J)
  - VDD_IN: 11217 mJ (11.217 J)
  - VDD_SOC: 2172 mJ (2.172 J)
- 温度：
  - cpu: 52.0°C (peak 52.2°C)
  - gpu: 52.1°C (peak 52.5°C)
  - soc0: 51.2°C (peak 51.3°C)
  - soc1: 51.6°C (peak 51.7°C)
  - soc2: 50.8°C (peak 50.8°C)
  - tj: 52.1°C (peak 52.5°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (59 samples @ 100ms)
- 采样窗口时长：6.39s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2884/3025 mW
  - VDD_IN: 7582/7747 mW
  - VDD_SOC: 1476/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 18433 mJ (18.433 J)
  - VDD_IN: 48460 mJ (48.460 J)
  - VDD_SOC: 9431 mJ (9.431 J)
- 温度：
  - cpu: 51.8°C (peak 52.2°C)
  - gpu: 52.0°C (peak 52.3°C)
  - soc0: 51.1°C (peak 51.3°C)
  - soc1: 51.5°C (peak 51.7°C)
  - soc2: 50.6°C (peak 50.8°C)
  - tj: 52.0°C (peak 52.3°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.72s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2892/2905 mW
  - VDD_IN: 7614/7667 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2082 mJ (2.082 J)
  - VDD_IN: 5480 mJ (5.480 J)
  - VDD_SOC: 1062 mJ (1.062 J)
- 温度：
  - cpu: 51.2°C (peak 51.4°C)
  - gpu: 51.4°C (peak 51.7°C)
  - soc0: 50.4°C (peak 50.5°C)
  - soc1: 50.7°C (peak 50.8°C)
  - soc2: 49.9°C (peak 50.0°C)
  - tj: 51.4°C (peak 51.7°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 32709.3 | 5.48 | 0.30 |
| LTC | 1321 | 0.465366 | 10147.9 | 13.56 | 0.93 |
| PDNAPulse | 3170 | 0.286117 | 25633.0 | 5.93 | 0.37 |
| GRU | 1969 | 0.393358 | 73836.6 | 1.82 | 0.13 |
| NCPS-LTC | 2547 | 0.621254 | 3006.0 | 45.83 | 3.28 |
| NCPS-CfC | 15737 | 0.106344 | 12649.9 | 9.61 | 0.76 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
