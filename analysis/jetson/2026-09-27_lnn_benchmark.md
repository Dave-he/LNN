---
title: Jetson LNN 基准验证 - 2026-09-27
date: 2026-09-27
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-27

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (82 samples @ 100ms)
- 采样窗口时长：8.75s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2937/3968 mW
  - VDD_IN: 8044/8916 mW
  - VDD_SOC: 1611/1674 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 25714 mJ (25.714 J)
  - VDD_IN: 70414 mJ (70.414 J)
  - VDD_SOC: 14102 mJ (14.102 J)
- 温度：
  - cpu: 53.2°C (peak 54.1°C)
  - gpu: 53.3°C (peak 53.8°C)
  - soc0: 52.3°C (peak 52.6°C)
  - soc1: 52.7°C (peak 53.1°C)
  - soc2: 51.8°C (peak 52.2°C)
  - tj: 53.4°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2348/2348 mW
  - VDD_IN: 7388/7388 mW
  - VDD_SOC: 1594/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 314 mJ (0.314 J)
  - VDD_IN: 988 mJ (0.988 J)
  - VDD_SOC: 213 mJ (0.213 J)
- 温度：
  - cpu: 52.8°C (peak 52.8°C)
  - gpu: 53.4°C (peak 53.4°C)
  - soc0: 52.3°C (peak 52.3°C)
  - soc1: 52.6°C (peak 52.6°C)
  - soc2: 51.6°C (peak 51.6°C)
  - tj: 53.4°C (peak 53.4°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.51s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2785/2941 mW
  - VDD_IN: 7974/8173 mW
  - VDD_SOC: 1643/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1415 mJ (1.415 J)
  - VDD_IN: 4051 mJ (4.051 J)
  - VDD_SOC: 835 mJ (0.835 J)
- 温度：
  - cpu: 53.1°C (peak 53.3°C)
  - gpu: 53.2°C (peak 53.2°C)
  - soc0: 52.4°C (peak 52.4°C)
  - soc1: 52.7°C (peak 52.8°C)
  - soc2: 51.8°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.22s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3934/3934 mW
  - VDD_IN: 8877/8877 mW
  - VDD_SOC: 1552/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 877 mJ (0.877 J)
  - VDD_IN: 1979 mJ (1.979 J)
  - VDD_SOC: 346 mJ (0.346 J)
- 温度：
  - cpu: 53.9°C (peak 53.9°C)
  - gpu: 53.5°C (peak 53.5°C)
  - soc0: 52.4°C (peak 52.4°C)
  - soc1: 52.8°C (peak 52.8°C)
  - soc2: 52.1°C (peak 52.1°C)
  - tj: 53.9°C (peak 53.9°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 139868.2 | 1.31 | 0.07 |
| LTC | 1321 | 0.464351 | 36745.0 | 4.05 | 0.27 |
| PDNAPulse | 3170 | 0.284479 | 80906.0 | 1.82 | 0.13 |
| GRU | 1969 | 0.394308 | 274990.9 | 0.59 | n/a |

## CUDA 回退
- 本次优先尝试 Jetson CUDA 路径，但 CUDA 运行时返回内存/加速器错误，已自动回退到 CPU smoke benchmark。
- 回退原因：

```text
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1154, please report a bug to PyTorch.
```

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
