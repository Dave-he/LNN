---
title: Jetson LNN 基准验证 - 2026-10-02
date: 2026-10-02
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-02

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 89 MB / available 4962 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.78 MB

## 功耗与温度
- 功耗采样：可用 (94 samples @ 100ms)
- 采样窗口时长：10.01s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1976/3815 mW
  - VDD_IN: 6720/8971 mW
  - VDD_SOC: 1484/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 19778 mJ (19.778 J)
  - VDD_IN: 67268 mJ (67.268 J)
  - VDD_SOC: 14860 mJ (14.861 J)
- 温度：
  - cpu: 47.4°C (peak 48.9°C)
  - gpu: 47.6°C (peak 48.2°C)
  - soc0: 46.8°C (peak 47.1°C)
  - soc1: 46.9°C (peak 47.3°C)
  - soc2: 46.1°C (peak 46.8°C)
  - tj: 47.7°C (peak 48.9°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.12s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1437/1437 mW
  - VDD_IN: 6040/6040 mW
  - VDD_SOC: 1440/1440 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 179 mJ (0.179 J)
  - VDD_IN: 751 mJ (0.751 J)
  - VDD_SOC: 179 mJ (0.179 J)
- 温度：
  - cpu: 47.2°C (peak 47.2°C)
  - gpu: 47.3°C (peak 47.3°C)
  - soc0: 46.8°C (peak 46.8°C)
  - soc1: 46.8°C (peak 46.8°C)
  - soc2: 46.0°C (peak 46.0°C)
  - tj: 47.3°C (peak 47.3°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.47s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1487/1517 mW
  - VDD_IN: 6130/6160 mW
  - VDD_SOC: 1470/1480 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 696 mJ (0.696 J)
  - VDD_IN: 2868 mJ (2.868 J)
  - VDD_SOC: 688 mJ (0.688 J)
- 温度：
  - cpu: 47.0°C (peak 47.1°C)
  - gpu: 47.6°C (peak 47.8°C)
  - soc0: 46.7°C (peak 46.8°C)
  - soc1: 46.9°C (peak 47.0°C)
  - soc2: 46.0°C (peak 46.0°C)
  - tj: 47.6°C (peak 47.8°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.34s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3762/3815 mW
  - VDD_IN: 8891/8971 mW
  - VDD_SOC: 1592/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1288 mJ (1.288 J)
  - VDD_IN: 3043 mJ (3.043 J)
  - VDD_SOC: 545 mJ (0.545 J)
- 温度：
  - cpu: 48.6°C (peak 48.8°C)
  - gpu: 48.0°C (peak 48.2°C)
  - soc0: 47.1°C (peak 47.1°C)
  - soc1: 47.2°C (peak 47.3°C)
  - soc2: 46.7°C (peak 46.8°C)
  - tj: 48.5°C (peak 48.6°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 150344.8 | 1.11 | 0.05 |
| LTC | 1321 | 0.464351 | 39705.8 | 3.68 | 0.19 |
| PDNAPulse | 3170 | 0.284479 | 51205.5 | 1.77 | 0.21 |
| GRU | 1969 | 0.394308 | 236043.4 | 0.60 | n/a |

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
