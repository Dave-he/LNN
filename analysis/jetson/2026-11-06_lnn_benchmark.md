---
title: Jetson LNN 基准验证 - 2026-11-06
date: 2026-11-06
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-11-06

> ⚠️ **运行状态：`ok`**
> Real GPU measurement: the model ran on CUDA on the Jetson, so the steps/sec below are edge/GPU performance evidence. Still a smoke benchmark (synthetic data, one seed) — not a paper-level reproduction.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 170 MB / available 1298 MB
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
- 采样窗口时长：7.39s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3370/4278 mW
  - VDD_IN: 9083/10174 mW
  - VDD_SOC: 1839/1944 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 24917 mJ (24.917 J)
  - VDD_IN: 67165 mJ (67.165 J)
  - VDD_SOC: 13596 mJ (13.596 J)
- 温度：
  - cpu: 53.0°C (peak 53.6°C)
  - gpu: 54.1°C (peak 54.8°C)
  - soc0: 52.7°C (peak 52.9°C)
  - soc1: 53.0°C (peak 53.2°C)
  - soc2: 52.1°C (peak 52.3°C)
  - tj: 54.1°C (peak 54.8°C)
- GPU 利用率：mean 56%, peak 99%

## 各模型独立功耗

### CfCStyle
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.41s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 4082/4120 mW
  - VDD_IN: 9909/9975 mW
  - VDD_SOC: 1907/1907 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1657 mJ (1.657 J)
  - VDD_IN: 4023 mJ (4.023 J)
  - VDD_SOC: 774 mJ (0.774 J)
- 温度：
  - cpu: 53.1°C (peak 53.2°C)
  - gpu: 54.4°C (peak 54.4°C)
  - soc0: 52.7°C (peak 52.8°C)
  - soc1: 53.0°C (peak 53.1°C)
  - soc2: 52.2°C (peak 52.3°C)
  - tj: 54.4°C (peak 54.4°C)
- GPU 利用率：mean 36%, peak 97%

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
| CfCStyle | 329 | 0.691654 | 11184.3 | 1.29 | n/a |
| LTC | 185 | 0.670937 | 1480.0 | 2.22 | 9.67 |
| PDNAPulse | 418 | 0.607059 | 11604.2 | 0.42 | n/a |
| GRU | 273 | 0.609695 | 234466.9 | 0.57 | n/a |

## Benchmark 图
![Jetson LNN Benchmark](2026-11-06_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
