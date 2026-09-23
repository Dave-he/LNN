---
title: Jetson LNN 基准验证 - 2026-09-23
date: 2026-09-23
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-23

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
- CUDA 设备：Orin，显存 7619.78 MB

## 功耗与温度
- 功耗采样：可用 (115 samples @ 100ms)
- 采样窗口时长：12.80s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3620/3888 mW
  - VDD_IN: 8821/8996 mW
  - VDD_SOC: 1598/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 46342 mJ (46.342 J)
  - VDD_IN: 112928 mJ (112.928 J)
  - VDD_SOC: 20460 mJ (20.460 J)
- 温度：
  - cpu: 56.7°C (peak 57.1°C)
  - gpu: 56.6°C (peak 57.1°C)
  - soc0: 55.7°C (peak 55.9°C)
  - soc1: 56.2°C (peak 56.5°C)
  - soc2: 55.4°C (peak 55.6°C)
  - tj: 56.8°C (peak 57.2°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.14s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3650/3650 mW
  - VDD_IN: 8797/8797 mW
  - VDD_SOC: 1592/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 511 mJ (0.511 J)
  - VDD_IN: 1231 mJ (1.231 J)
  - VDD_SOC: 223 mJ (0.223 J)
- 温度：
  - cpu: 56.8°C (peak 56.8°C)
  - gpu: 56.8°C (peak 56.8°C)
  - soc0: 55.7°C (peak 55.7°C)
  - soc1: 56.2°C (peak 56.2°C)
  - soc2: 55.3°C (peak 55.3°C)
  - tj: 56.8°C (peak 56.8°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.58s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3610/3650 mW
  - VDD_IN: 8869/8916 mW
  - VDD_SOC: 1630/1632 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2110 mJ (2.110 J)
  - VDD_IN: 5184 mJ (5.184 J)
  - VDD_SOC: 952 mJ (0.953 J)
- 温度：
  - cpu: 56.8°C (peak 56.9°C)
  - gpu: 56.6°C (peak 56.7°C)
  - soc0: 55.7°C (peak 55.8°C)
  - soc1: 56.3°C (peak 56.5°C)
  - soc2: 55.4°C (peak 55.5°C)
  - tj: 56.8°C (peak 57.2°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.41s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3861/3888 mW
  - VDD_IN: 8916/8916 mW
  - VDD_SOC: 1550/1550 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1581 mJ (1.581 J)
  - VDD_IN: 3651 mJ (3.651 J)
  - VDD_SOC: 635 mJ (0.635 J)
- 温度：
  - cpu: 56.8°C (peak 56.8°C)
  - gpu: 56.7°C (peak 57.0°C)
  - soc0: 55.7°C (peak 55.8°C)
  - soc1: 56.3°C (peak 56.4°C)
  - soc2: 55.5°C (peak 55.6°C)
  - tj: 56.9°C (peak 57.0°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 132965.3 | 1.42 | 0.08 |
| LTC | 1321 | 0.464351 | 32090.4 | 4.76 | 0.35 |
| PDNAPulse | 3170 | 0.284479 | 43009.0 | 2.27 | 0.25 |
| GRU | 1969 | 0.394308 | 150723.9 | 0.96 | n/a |

## CUDA 回退
- 本次优先尝试 Jetson CUDA 路径，但 CUDA 运行时返回内存/加速器错误，已自动回退到 CPU smoke benchmark。
- 回退原因：

```text
torch.AcceleratorError: CUDA error: out of memory Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
