---
title: Jetson LNN 基准验证 - 2026-09-24
date: 2026-09-24
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-24

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
- 功耗采样：可用 (151 samples @ 100ms)
- 采样窗口时长：17.06s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 7591/9361 mW
  - VDD_IN: 13131/15035 mW
  - VDD_SOC: 1728/1813 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 129479 mJ (129.479 J)
  - VDD_IN: 223977 mJ (223.977 J)
  - VDD_SOC: 29477 mJ (29.477 J)
- 温度：
  - cpu: 64.4°C (peak 64.9°C)
  - gpu: 65.2°C (peak 65.8°C)
  - soc0: 63.2°C (peak 63.6°C)
  - soc1: 63.2°C (peak 63.4°C)
  - soc2: 63.1°C (peak 63.4°C)
  - tj: 65.2°C (peak 65.8°C)
- GPU 利用率：mean 66%, peak 94%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.17s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 7163/7163 mW
  - VDD_IN: 12735/12735 mW
  - VDD_SOC: 1737/1737 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1250 mJ (1.250 J)
  - VDD_IN: 2222 mJ (2.222 J)
  - VDD_SOC: 303 mJ (0.303 J)
- 温度：
  - cpu: 64.5°C (peak 64.5°C)
  - gpu: 65.2°C (peak 65.2°C)
  - soc0: 63.2°C (peak 63.2°C)
  - soc1: 63.2°C (peak 63.2°C)
  - soc2: 63.2°C (peak 63.2°C)
  - tj: 65.2°C (peak 65.2°C)
- GPU 利用率：mean 58%, peak 58%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.60s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 9110/9243 mW
  - VDD_IN: 14761/14917 mW
  - VDD_SOC: 1771/1771 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5425 mJ (5.425 J)
  - VDD_IN: 8790 mJ (8.790 J)
  - VDD_SOC: 1055 mJ (1.055 J)
- 温度：
  - cpu: 64.6°C (peak 64.8°C)
  - gpu: 65.6°C (peak 65.8°C)
  - soc0: 63.3°C (peak 63.3°C)
  - soc1: 63.2°C (peak 63.2°C)
  - soc2: 63.3°C (peak 63.3°C)
  - tj: 65.6°C (peak 65.8°C)
- GPU 利用率：mean 77%, peak 90%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.43s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 6426/6426 mW
  - VDD_IN: 11746/11746 mW
  - VDD_SOC: 1661/1661 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2790 mJ (2.790 J)
  - VDD_IN: 5100 mJ (5.100 J)
  - VDD_SOC: 721 mJ (0.721 J)
- 温度：
  - cpu: 64.2°C (peak 64.2°C)
  - gpu: 64.4°C (peak 64.4°C)
  - soc0: 63.3°C (peak 63.3°C)
  - soc1: 63.1°C (peak 63.1°C)
  - soc2: 63.0°C (peak 63.0°C)
  - tj: 64.5°C (peak 64.5°C)
- GPU 利用率：mean 51%, peak 51%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312975 | 102960.8 | 1.63 | 0.15 |
| LTC | 1321 | 0.464351 | 30263.6 | 4.67 | 0.59 |
| PDNAPulse | 3170 | 0.284479 | 39498.2 | 3.55 | 0.34 |
| GRU | 1969 | 0.394308 | 148387.5 | 0.89 | n/a |

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
