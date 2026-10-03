---
title: Jetson LNN 基准验证 - 2026-10-04
date: 2026-10-04
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-04

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 119 MB / available 1422 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (94 samples @ 100ms)
- 采样窗口时长：10.02s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2325/3576 mW
  - VDD_IN: 7150/8373 mW
  - VDD_SOC: 1525/1597 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 23306 mJ (23.306 J)
  - VDD_IN: 71668 mJ (71.668 J)
  - VDD_SOC: 15290 mJ (15.290 J)
- 温度：
  - cpu: 49.6°C (peak 50.6°C)
  - gpu: 49.9°C (peak 50.5°C)
  - soc0: 49.0°C (peak 49.4°C)
  - soc1: 49.2°C (peak 49.7°C)
  - soc2: 48.4°C (peak 48.9°C)
  - tj: 49.9°C (peak 50.6°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.12s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2073/2073 mW
  - VDD_IN: 6829/6829 mW
  - VDD_SOC: 1517/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 259 mJ (0.259 J)
  - VDD_IN: 852 mJ (0.852 J)
  - VDD_SOC: 189 mJ (0.189 J)
- 温度：
  - cpu: 49.5°C (peak 49.5°C)
  - gpu: 49.6°C (peak 49.6°C)
  - soc0: 49.0°C (peak 49.0°C)
  - soc1: 49.2°C (peak 49.2°C)
  - soc2: 48.3°C (peak 48.3°C)
  - tj: 49.6°C (peak 49.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.49s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2192/2272 mW
  - VDD_IN: 7158/7268 mW
  - VDD_SOC: 1567/1597 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1070 mJ (1.071 J)
  - VDD_IN: 3495 mJ (3.495 J)
  - VDD_SOC: 765 mJ (0.765 J)
- 温度：
  - cpu: 49.5°C (peak 49.8°C)
  - gpu: 49.9°C (peak 50.1°C)
  - soc0: 49.0°C (peak 49.1°C)
  - soc1: 49.3°C (peak 49.4°C)
  - soc2: 48.4°C (peak 48.4°C)
  - tj: 49.9°C (peak 50.1°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.18s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3537/3537 mW
  - VDD_IN: 8293/8293 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 646 mJ (0.646 J)
  - VDD_IN: 1514 mJ (1.514 J)
  - VDD_SOC: 277 mJ (0.277 J)
- 温度：
  - cpu: 50.4°C (peak 50.4°C)
  - gpu: 50.2°C (peak 50.2°C)
  - soc0: 49.1°C (peak 49.1°C)
  - soc1: 49.6°C (peak 49.6°C)
  - soc2: 48.8°C (peak 48.8°C)
  - tj: 50.4°C (peak 50.4°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 149854.8 | 1.18 | 0.06 |
| LTC | 1321 | 0.464351 | 38046.2 | 3.82 | 0.24 |
| PDNAPulse | 3170 | 0.284479 | 109278.4 | 1.74 | 0.10 |
| GRU | 1969 | 0.394308 | 163887.4 | 0.49 | n/a |

## CUDA 回退
- 本次优先尝试 Jetson CUDA 路径，但 CUDA 运行时返回内存/加速器错误，已自动回退到 CPU smoke benchmark。
- ⚠️ 下表的**速度与精度不是边缘推理性能证据**，只是 CPU smoke；上方功耗/温度采样仍来自真实 Jetson 传感器，有效。
- 回退原因：

```text
torch.AcceleratorError: CUDA error: out of memory Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1 Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
