---
title: Jetson LNN 基准验证 - 2026-09-29
date: 2026-09-29
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-29

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
- 功耗采样：可用 (103 samples @ 100ms)
- 采样窗口时长：11.15s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2921/3599 mW
  - VDD_IN: 10271/10991 mW
  - VDD_SOC: 3392/3491 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 32570 mJ (32.569 J)
  - VDD_IN: 114535 mJ (114.535 J)
  - VDD_SOC: 37822 mJ (37.822 J)
- 温度：
  - cpu: 57.1°C (peak 57.7°C)
  - gpu: 57.2°C (peak 57.6°C)
  - soc0: 56.6°C (peak 56.8°C)
  - soc1: 57.0°C (peak 57.2°C)
  - soc2: 55.9°C (peak 56.2°C)
  - tj: 57.3°C (peak 57.7°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2812/2812 mW
  - VDD_IN: 10174/10174 mW
  - VDD_SOC: 3412/3412 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 378 mJ (0.378 J)
  - VDD_IN: 1367 mJ (1.367 J)
  - VDD_SOC: 458 mJ (0.459 J)
- 温度：
  - cpu: 57.0°C (peak 57.0°C)
  - gpu: 57.2°C (peak 57.2°C)
  - soc0: 56.6°C (peak 56.6°C)
  - soc1: 57.0°C (peak 57.0°C)
  - soc2: 55.8°C (peak 55.8°C)
  - tj: 57.2°C (peak 57.2°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.55s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2921/2971 mW
  - VDD_IN: 10458/10554 mW
  - VDD_SOC: 3481/3491 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1598 mJ (1.598 J)
  - VDD_IN: 5721 mJ (5.721 J)
  - VDD_SOC: 1904 mJ (1.905 J)
- 温度：
  - cpu: 57.2°C (peak 57.4°C)
  - gpu: 57.2°C (peak 57.3°C)
  - soc0: 56.7°C (peak 56.8°C)
  - soc1: 57.0°C (peak 57.1°C)
  - soc2: 55.9°C (peak 56.0°C)
  - tj: 57.3°C (peak 57.4°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.22s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3525/3525 mW
  - VDD_IN: 10792/10792 mW
  - VDD_SOC: 3372/3372 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 777 mJ (0.777 J)
  - VDD_IN: 2379 mJ (2.379 J)
  - VDD_SOC: 743 mJ (0.743 J)
- 温度：
  - cpu: 57.3°C (peak 57.3°C)
  - gpu: 57.4°C (peak 57.4°C)
  - soc0: 56.7°C (peak 56.7°C)
  - soc1: 57.0°C (peak 57.0°C)
  - soc2: 56.2°C (peak 56.2°C)
  - tj: 57.6°C (peak 57.6°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 139440.2 | 1.26 | 0.09 |
| LTC | 1321 | 0.464351 | 34248.8 | 4.13 | 0.39 |
| PDNAPulse | 3170 | 0.284479 | 83500.0 | 2.21 | 0.16 |
| GRU | 1969 | 0.394308 | 307119.5 | 0.53 | n/a |

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
