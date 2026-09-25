---
title: Jetson LNN 基准验证 - 2026-09-26
date: 2026-09-26
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-26

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
- 功耗采样：可用 (104 samples @ 100ms)
- 采样窗口时长：11.22s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2753/3735 mW
  - VDD_IN: 7777/8797 mW
  - VDD_SOC: 1583/1634 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 30894 mJ (30.895 J)
  - VDD_IN: 87275 mJ (87.275 J)
  - VDD_SOC: 17764 mJ (17.764 J)
- 温度：
  - cpu: 53.3°C (peak 54.1°C)
  - gpu: 53.5°C (peak 54.0°C)
  - soc0: 52.5°C (peak 52.8°C)
  - soc1: 53.0°C (peak 53.3°C)
  - soc2: 52.0°C (peak 52.4°C)
  - tj: 53.5°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.14s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2547/2547 mW
  - VDD_IN: 7535/7535 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 353 mJ (0.353 J)
  - VDD_IN: 1045 mJ (1.045 J)
  - VDD_SOC: 216 mJ (0.216 J)
- 温度：
  - cpu: 53.2°C (peak 53.2°C)
  - gpu: 53.6°C (peak 53.6°C)
  - soc0: 52.5°C (peak 52.5°C)
  - soc1: 53.0°C (peak 53.0°C)
  - soc2: 51.9°C (peak 51.9°C)
  - tj: 53.6°C (peak 53.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.53s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2517/2587 mW
  - VDD_IN: 7525/7615 mW
  - VDD_SOC: 1594/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1343 mJ (1.343 J)
  - VDD_IN: 4016 mJ (4.016 J)
  - VDD_SOC: 851 mJ (0.851 J)
- 温度：
  - cpu: 53.5°C (peak 53.6°C)
  - gpu: 53.5°C (peak 53.7°C)
  - soc0: 52.5°C (peak 52.6°C)
  - soc1: 53.0°C (peak 53.1°C)
  - soc2: 52.0°C (peak 52.1°C)
  - tj: 53.5°C (peak 53.7°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.26s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3576/3576 mW
  - VDD_IN: 8538/8558 mW
  - VDD_SOC: 1552/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 947 mJ (0.947 J)
  - VDD_IN: 2261 mJ (2.261 J)
  - VDD_SOC: 411 mJ (0.411 J)
- 温度：
  - cpu: 54.0°C (peak 54.1°C)
  - gpu: 53.7°C (peak 53.8°C)
  - soc0: 52.7°C (peak 52.8°C)
  - soc1: 53.1°C (peak 53.1°C)
  - soc2: 52.3°C (peak 52.3°C)
  - tj: 54.0°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 134686.6 | 1.27 | 0.07 |
| LTC | 1321 | 0.464351 | 34830.5 | 4.29 | 0.27 |
| PDNAPulse | 3170 | 0.284479 | 67046.4 | 2.09 | 0.15 |
| GRU | 1969 | 0.394308 | 360779.6 | 0.55 | n/a |

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
