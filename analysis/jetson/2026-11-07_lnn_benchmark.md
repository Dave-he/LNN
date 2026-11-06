---
title: Jetson LNN 基准验证 - 2026-11-07
date: 2026-11-07
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-11-07

> ⚠️ **运行状态：`ok`**
> Real GPU measurement: the model ran on CUDA on the Jetson, so the steps/sec below are edge/GPU performance evidence. Still a smoke benchmark (synthetic data, one seed) — not a paper-level reproduction.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 202 MB / available 3734 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (61 samples @ 100ms)
- 采样窗口时长：6.54s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2021/2272 mW
  - VDD_IN: 6831/7228 mW
  - VDD_SOC: 1487/1557 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 13222 mJ (13.222 J)
  - VDD_IN: 44703 mJ (44.703 J)
  - VDD_SOC: 9729 mJ (9.729 J)
- 温度：
  - cpu: 48.6°C (peak 49.2°C)
  - gpu: 48.9°C (peak 49.3°C)
  - soc0: 48.1°C (peak 48.3°C)
  - soc1: 48.3°C (peak 48.6°C)
  - soc2: 47.4°C (peak 47.6°C)
  - tj: 49.0°C (peak 49.3°C)
- GPU 利用率：mean 9%, peak 26%

## 各模型独立功耗

### CfCStyle
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.30s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2113/2113 mW
  - VDD_IN: 6749/6749 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 630 mJ (0.630 J)
  - VDD_IN: 2013 mJ (2.013 J)
  - VDD_SOC: 440 mJ (0.441 J)
- 温度：
  - cpu: 48.6°C (peak 48.7°C)
  - gpu: 48.9°C (peak 49.0°C)
  - soc0: 48.2°C (peak 48.3°C)
  - soc1: 48.4°C (peak 48.5°C)
  - soc2: 47.5°C (peak 47.5°C)
  - tj: 48.9°C (peak 49.0°C)
- GPU 利用率：mean 16%, peak 16%

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
| CfCStyle | 329 | 0.691654 | 11914.6 | 1.41 | n/a |
| LTC | 185 | 0.670937 | 2196.8 | 1.74 | 4.84 |
| PDNAPulse | 418 | 0.607059 | 11766.6 | 0.41 | n/a |
| GRU | 273 | 0.609695 | 236415.9 | 0.52 | n/a |

## Benchmark 图
![Jetson LNN Benchmark](2026-11-07_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
