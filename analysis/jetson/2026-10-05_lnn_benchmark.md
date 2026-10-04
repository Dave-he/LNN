---
title: Jetson LNN 基准验证 - 2026-10-05
date: 2026-10-05
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-05

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 83 MB / available 1464 MB
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
- 采样窗口时长：10.05s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2444/3775 mW
  - VDD_IN: 7268/8572 mW
  - VDD_SOC: 1520/1557 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 24552 mJ (24.552 J)
  - VDD_IN: 73019 mJ (73.019 J)
  - VDD_SOC: 15276 mJ (15.276 J)
- 温度：
  - cpu: 49.7°C (peak 50.8°C)
  - gpu: 49.9°C (peak 50.4°C)
  - soc0: 49.0°C (peak 49.2°C)
  - soc1: 49.2°C (peak 49.6°C)
  - soc2: 48.4°C (peak 49.0°C)
  - tj: 50.0°C (peak 50.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2073/2073 mW
  - VDD_IN: 6868/6868 mW
  - VDD_SOC: 1517/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 265 mJ (0.265 J)
  - VDD_IN: 879 mJ (0.879 J)
  - VDD_SOC: 194 mJ (0.194 J)
- 温度：
  - cpu: 49.3°C (peak 49.3°C)
  - gpu: 49.7°C (peak 49.7°C)
  - soc0: 48.9°C (peak 48.9°C)
  - soc1: 49.2°C (peak 49.2°C)
  - soc2: 48.3°C (peak 48.3°C)
  - tj: 49.7°C (peak 49.7°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.48s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2182/2192 mW
  - VDD_IN: 6998/7028 mW
  - VDD_SOC: 1517/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1041 mJ (1.041 J)
  - VDD_IN: 3340 mJ (3.340 J)
  - VDD_SOC: 724 mJ (0.724 J)
- 温度：
  - cpu: 49.6°C (peak 49.6°C)
  - gpu: 49.8°C (peak 49.9°C)
  - soc0: 49.0°C (peak 49.0°C)
  - soc1: 49.2°C (peak 49.3°C)
  - soc2: 48.3°C (peak 48.4°C)
  - tj: 49.8°C (peak 49.9°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.21s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3735/3735 mW
  - VDD_IN: 8572/8572 mW
  - VDD_SOC: 1512/1512 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 785 mJ (0.785 J)
  - VDD_IN: 1803 mJ (1.803 J)
  - VDD_SOC: 318 mJ (0.318 J)
- 温度：
  - cpu: 50.5°C (peak 50.5°C)
  - gpu: 50.2°C (peak 50.2°C)
  - soc0: 49.1°C (peak 49.1°C)
  - soc1: 49.4°C (peak 49.4°C)
  - soc2: 48.8°C (peak 48.8°C)
  - tj: 50.5°C (peak 50.5°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 144892.9 | 1.16 | 0.06 |
| LTC | 1321 | 0.464351 | 39019.6 | 3.73 | 0.23 |
| PDNAPulse | 3170 | 0.284479 | 92146.0 | 1.84 | 0.12 |
| GRU | 1969 | 0.394308 | 456170.9 | 0.55 | n/a |

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
