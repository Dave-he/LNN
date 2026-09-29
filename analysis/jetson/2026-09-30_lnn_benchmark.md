---
title: Jetson LNN 基准验证 - 2026-09-30
date: 2026-09-30
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-30

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 339 MB / available 1549 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (101 samples @ 100ms)
- 采样窗口时长：10.89s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2942/3605 mW
  - VDD_IN: 10206/10951 mW
  - VDD_SOC: 3353/3531 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 32034 mJ (32.034 J)
  - VDD_IN: 111121 mJ (111.121 J)
  - VDD_SOC: 36506 mJ (36.506 J)
- 温度：
  - cpu: 55.2°C (peak 55.8°C)
  - gpu: 55.3°C (peak 55.8°C)
  - soc0: 54.6°C (peak 54.9°C)
  - soc1: 55.0°C (peak 55.3°C)
  - soc2: 54.0°C (peak 54.3°C)
  - tj: 55.4°C (peak 55.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2579/2579 mW
  - VDD_IN: 9697/9697 mW
  - VDD_SOC: 3298/3298 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 347 mJ (0.347 J)
  - VDD_IN: 1305 mJ (1.305 J)
  - VDD_SOC: 444 mJ (0.444 J)
- 温度：
  - cpu: 54.9°C (peak 54.9°C)
  - gpu: 55.1°C (peak 55.1°C)
  - soc0: 54.6°C (peak 54.6°C)
  - soc1: 55.3°C (peak 55.3°C)
  - soc2: 53.9°C (peak 53.9°C)
  - tj: 55.3°C (peak 55.3°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.50s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2648/2658 mW
  - VDD_IN: 9896/9936 mW
  - VDD_SOC: 3355/3372 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1313 mJ (1.313 J)
  - VDD_IN: 4906 mJ (4.906 J)
  - VDD_SOC: 1663 mJ (1.663 J)
- 温度：
  - cpu: 55.0°C (peak 55.1°C)
  - gpu: 55.3°C (peak 55.5°C)
  - soc0: 54.7°C (peak 54.7°C)
  - soc1: 55.1°C (peak 55.2°C)
  - soc2: 54.0°C (peak 54.0°C)
  - tj: 55.3°C (peak 55.5°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.24s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3546/3605 mW
  - VDD_IN: 10694/10753 mW
  - VDD_SOC: 3293/3293 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 866 mJ (0.867 J)
  - VDD_IN: 2614 mJ (2.614 J)
  - VDD_SOC: 805 mJ (0.805 J)
- 温度：
  - cpu: 55.5°C (peak 55.8°C)
  - gpu: 55.5°C (peak 55.5°C)
  - soc0: 54.8°C (peak 54.8°C)
  - soc1: 55.1°C (peak 55.2°C)
  - soc2: 54.2°C (peak 54.2°C)
  - tj: 55.6°C (peak 55.8°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 139411.0 | 1.20 | 0.09 |
| LTC | 1321 | 0.464351 | 37640.4 | 4.09 | 0.33 |
| PDNAPulse | 3170 | 0.284479 | 75914.3 | 1.94 | 0.18 |
| GRU | 1969 | 0.394308 | 254540.8 | 0.61 | n/a |

## CUDA 回退
- 本次优先尝试 Jetson CUDA 路径，但 CUDA 运行时返回内存/加速器错误，已自动回退到 CPU smoke benchmark。
- ⚠️ 下表的**速度与精度不是边缘推理性能证据**，只是 CPU smoke；上方功耗/温度采样仍来自真实 Jetson 传感器，有效。
- 回退原因：

```text
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable Search for `cudaErrorDevicesUnavailable' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
