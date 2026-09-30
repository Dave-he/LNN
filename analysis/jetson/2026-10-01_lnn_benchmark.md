---
title: Jetson LNN 基准验证 - 2026-10-01
date: 2026-10-01
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-01

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 109 MB / available 4496 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.78 MB

## 功耗与温度
- 功耗采样：可用 (74 samples @ 100ms)
- 采样窗口时长：7.96s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2019/3775 mW
  - VDD_IN: 6704/8492 mW
  - VDD_SOC: 1480/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 16073 mJ (16.073 J)
  - VDD_IN: 53371 mJ (53.371 J)
  - VDD_SOC: 11783 mJ (11.783 J)
- 温度：
  - cpu: 48.4°C (peak 49.7°C)
  - gpu: 48.8°C (peak 49.3°C)
  - soc0: 47.9°C (peak 48.1°C)
  - soc1: 48.1°C (peak 48.5°C)
  - soc2: 47.2°C (peak 47.8°C)
  - tj: 48.8°C (peak 49.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1597/1597 mW
  - VDD_IN: 6240/6240 mW
  - VDD_SOC: 1480/1480 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 210 mJ (0.210 J)
  - VDD_IN: 821 mJ (0.821 J)
  - VDD_SOC: 195 mJ (0.195 J)
- 温度：
  - cpu: 47.8°C (peak 47.8°C)
  - gpu: 48.6°C (peak 48.6°C)
  - soc0: 47.8°C (peak 47.8°C)
  - soc1: 47.9°C (peak 47.9°C)
  - soc2: 47.1°C (peak 47.1°C)
  - tj: 48.6°C (peak 48.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.49s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1587/1597 mW
  - VDD_IN: 6240/6240 mW
  - VDD_SOC: 1480/1480 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 782 mJ (0.782 J)
  - VDD_IN: 3076 mJ (3.076 J)
  - VDD_SOC: 730 mJ (0.730 J)
- 温度：
  - cpu: 48.5°C (peak 48.8°C)
  - gpu: 48.7°C (peak 48.8°C)
  - soc0: 48.0°C (peak 48.1°C)
  - soc1: 48.0°C (peak 48.2°C)
  - soc2: 47.2°C (peak 47.2°C)
  - tj: 48.7°C (peak 48.8°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.21s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3775/3775 mW
  - VDD_IN: 8452/8452 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 784 mJ (0.784 J)
  - VDD_IN: 1755 mJ (1.755 J)
  - VDD_SOC: 306 mJ (0.306 J)
- 温度：
  - cpu: 49.6°C (peak 49.6°C)
  - gpu: 49.2°C (peak 49.2°C)
  - soc0: 48.2°C (peak 48.2°C)
  - soc1: 48.3°C (peak 48.3°C)
  - soc2: 47.8°C (peak 47.8°C)
  - tj: 49.6°C (peak 49.6°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 141676.3 | 1.14 | 0.06 |
| LTC | 1321 | 0.464351 | 37744.2 | 3.79 | 0.21 |
| PDNAPulse | 3170 | 0.284479 | 91625.1 | 1.62 | 0.12 |
| GRU | 1969 | 0.394308 | 400946.9 | 0.50 | n/a |

## CUDA 回退
- 本次优先尝试 Jetson CUDA 路径，但 CUDA 运行时返回内存/加速器错误，已自动回退到 CPU smoke benchmark。
- ⚠️ 下表的**速度与精度不是边缘推理性能证据**，只是 CPU smoke；上方功耗/温度采样仍来自真实 Jetson 传感器，有效。
- 回退原因：

```text
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
