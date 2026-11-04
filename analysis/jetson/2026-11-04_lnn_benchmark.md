---
title: Jetson LNN 基准验证 - 2026-11-04
date: 2026-11-04
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-11-04

> ⚠️ **运行状态：`ok`**
> Real GPU measurement: the model ran on CUDA on the Jetson, so the steps/sec below are edge/GPU performance evidence. Still a smoke benchmark (synthetic data, one seed) — not a paper-level reproduction.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 86 MB / available 1234 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (69 samples @ 100ms)
- 采样窗口时长：7.45s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3661/4436 mW
  - VDD_IN: 8958/9737 mW
  - VDD_SOC: 1603/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 27264 mJ (27.264 J)
  - VDD_IN: 66707 mJ (66.707 J)
  - VDD_SOC: 11939 mJ (11.939 J)
- 温度：
  - cpu: 55.9°C (peak 56.7°C)
  - gpu: 55.8°C (peak 56.2°C)
  - soc0: 54.9°C (peak 55.0°C)
  - soc1: 55.4°C (peak 55.7°C)
  - soc2: 54.5°C (peak 54.8°C)
  - tj: 56.0°C (peak 56.8°C)
- GPU 利用率：mean 8%, peak 25%

## 各模型独立功耗

### CfCStyle
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.34s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3358/3378 mW
  - VDD_IN: 8518/8558 mW
  - VDD_SOC: 1592/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1133 mJ (1.133 J)
  - VDD_IN: 2875 mJ (2.875 J)
  - VDD_SOC: 537 mJ (0.537 J)
- 温度：
  - cpu: 55.9°C (peak 56.2°C)
  - gpu: 55.9°C (peak 56.0°C)
  - soc0: 54.9°C (peak 54.9°C)
  - soc1: 55.3°C (peak 55.3°C)
  - soc2: 54.5°C (peak 54.5°C)
  - tj: 56.1°C (peak 56.2°C)
- GPU 利用率：mean 14%, peak 16%

### PDNAPulse
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：64 / 16
- 隐藏维度 / Epoch：8 / 1
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 329 | 0.691654 | 11750.8 | 1.54 | n/a |
| LTC | 185 | 0.670937 | 1958.1 | 1.80 | 6.91 |
| PDNAPulse | 418 | 0.607059 | 12093.4 | 0.43 | n/a |
| GRU | 273 | 0.609695 | 291277.4 | 0.55 | n/a |

## Benchmark 图
![Jetson LNN Benchmark](2026-11-04_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
