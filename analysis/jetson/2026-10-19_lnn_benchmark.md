---
title: Jetson LNN 基准验证 - 2026-10-19
date: 2026-10-19
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-19

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 228 MB / available 3146 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (294 samples @ 100ms)
- 采样窗口时长：31.22s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2473/3105 mW
  - VDD_IN: 7168/7747 mW
  - VDD_SOC: 1472/1557 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 77206 mJ (77.206 J)
  - VDD_IN: 223819 mJ (223.819 J)
  - VDD_SOC: 45963 mJ (45.963 J)
- 温度：
  - cpu: 48.6°C (peak 49.8°C)
  - gpu: 48.7°C (peak 49.6°C)
  - soc0: 47.9°C (peak 48.5°C)
  - soc1: 48.1°C (peak 48.8°C)
  - soc2: 47.3°C (peak 48.0°C)
  - tj: 48.8°C (peak 49.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.32s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2631/2631 mW
  - VDD_IN: 7268/7268 mW
  - VDD_SOC: 1437/1437 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 855 mJ (0.855 J)
  - VDD_IN: 2361 mJ (2.361 J)
  - VDD_SOC: 467 mJ (0.467 J)
- 温度：
  - cpu: 48.5°C (peak 48.6°C)
  - gpu: 48.3°C (peak 48.3°C)
  - soc0: 47.5°C (peak 47.6°C)
  - soc1: 47.6°C (peak 47.7°C)
  - soc2: 47.0°C (peak 47.0°C)
  - tj: 48.5°C (peak 48.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (10 samples @ 100ms)
- 采样窗口时长：1.17s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2296/2392 mW
  - VDD_IN: 6948/7068 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2686 mJ (2.687 J)
  - VDD_IN: 8130 mJ (8.130 J)
  - VDD_SOC: 1728 mJ (1.728 J)
- 温度：
  - cpu: 48.3°C (peak 48.5°C)
  - gpu: 48.5°C (peak 48.8°C)
  - soc0: 47.8°C (peak 47.9°C)
  - soc1: 48.0°C (peak 48.1°C)
  - soc2: 47.1°C (peak 47.2°C)
  - tj: 48.5°C (peak 48.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.40s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2853/2866 mW
  - VDD_IN: 7547/7587 mW
  - VDD_SOC: 1476/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1139 mJ (1.139 J)
  - VDD_IN: 3012 mJ (3.013 J)
  - VDD_SOC: 589 mJ (0.589 J)
- 温度：
  - cpu: 49.5°C (peak 49.6°C)
  - gpu: 49.4°C (peak 49.5°C)
  - soc0: 48.4°C (peak 48.5°C)
  - soc1: 48.6°C (peak 48.8°C)
  - soc2: 47.9°C (peak 48.0°C)
  - tj: 49.5°C (peak 49.6°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (10 samples @ 100ms)
- 采样窗口时长：1.10s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2709/3065 mW
  - VDD_IN: 7372/7747 mW
  - VDD_SOC: 1476/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2989 mJ (2.989 J)
  - VDD_IN: 8133 mJ (8.133 J)
  - VDD_SOC: 1629 mJ (1.629 J)
- 温度：
  - cpu: 49.4°C (peak 49.7°C)
  - gpu: 49.2°C (peak 49.4°C)
  - soc0: 48.3°C (peak 48.4°C)
  - soc1: 48.5°C (peak 48.7°C)
  - soc2: 47.8°C (peak 47.9°C)
  - tj: 49.4°C (peak 49.6°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.32s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2751/2751 mW
  - VDD_IN: 7388/7388 mW
  - VDD_SOC: 1457/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 880 mJ (0.880 J)
  - VDD_IN: 2364 mJ (2.364 J)
  - VDD_SOC: 466 mJ (0.466 J)
- 温度：
  - cpu: 48.5°C (peak 48.6°C)
  - gpu: 48.8°C (peak 49.1°C)
  - soc0: 47.9°C (peak 47.9°C)
  - soc1: 48.1°C (peak 48.1°C)
  - soc2: 47.4°C (peak 47.4°C)
  - tj: 48.8°C (peak 49.1°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 56203.2 | 2.51 | 0.16 |
| LTC | 1321 | 0.465366 | 16315.1 | 6.97 | 0.55 |
| PDNAPulse | 3170 | 0.286117 | 58112.9 | 1.86 | 0.16 |
| GRU | 1969 | 0.393358 | 247583.9 | 0.78 | n/a |
| NCPS-LTC | 2547 | 0.621254 | 16577.4 | 11.05 | 0.55 |
| NCPS-CfC | 15737 | 0.106344 | 47278.2 | 2.43 | 0.20 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
