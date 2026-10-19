---
title: Jetson LNN 基准验证 - 2026-10-20
date: 2026-10-20
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-20

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 1474 MB / available 2966 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (1409 samples @ 100ms)
- 采样窗口时长：151.81s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3239/4483 mW
  - VDD_IN: 8190/9394 mW
  - VDD_SOC: 1544/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 491771 mJ (491.771 J)
  - VDD_IN: 1243392 mJ (1243.392 J)
  - VDD_SOC: 234320 mJ (234.320 J)
- 温度：
  - cpu: 53.4°C (peak 55.2°C)
  - gpu: 53.5°C (peak 54.8°C)
  - soc0: 52.5°C (peak 53.5°C)
  - soc1: 53.0°C (peak 54.1°C)
  - soc2: 52.1°C (peak 53.4°C)
  - tj: 53.5°C (peak 55.2°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (11 samples @ 100ms)
- 采样窗口时长：1.19s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3104/3144 mW
  - VDD_IN: 8047/8094 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 3702 mJ (3.702 J)
  - VDD_IN: 9595 mJ (9.595 J)
  - VDD_SOC: 1854 mJ (1.854 J)
- 温度：
  - cpu: 53.0°C (peak 53.3°C)
  - gpu: 53.0°C (peak 53.3°C)
  - soc0: 52.1°C (peak 52.3°C)
  - soc1: 52.5°C (peak 52.6°C)
  - soc2: 51.6°C (peak 51.7°C)
  - tj: 53.1°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.35s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3118/3184 mW
  - VDD_IN: 7987/8054 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1087 mJ (1.087 J)
  - VDD_IN: 2784 mJ (2.784 J)
  - VDD_SOC: 528 mJ (0.528 J)
- 温度：
  - cpu: 53.0°C (peak 53.1°C)
  - gpu: 53.2°C (peak 53.3°C)
  - soc0: 52.4°C (peak 52.5°C)
  - soc1: 52.7°C (peak 52.8°C)
  - soc2: 51.8°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (19 samples @ 100ms)
- 采样窗口时长：2.11s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2878/3144 mW
  - VDD_IN: 7731/8094 mW
  - VDD_SOC: 1523/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 6083 mJ (6.083 J)
  - VDD_IN: 16338 mJ (16.338 J)
  - VDD_SOC: 3220 mJ (3.220 J)
- 温度：
  - cpu: 53.0°C (peak 53.4°C)
  - gpu: 53.2°C (peak 53.5°C)
  - soc0: 52.2°C (peak 52.4°C)
  - soc1: 52.7°C (peak 52.9°C)
  - soc2: 51.7°C (peak 51.8°C)
  - tj: 53.2°C (peak 53.5°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (45 samples @ 100ms)
- 采样窗口时长：4.99s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 4350/4483 mW
  - VDD_IN: 9160/9315 mW
  - VDD_SOC: 1466/1472 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 21686 mJ (21.686 J)
  - VDD_IN: 45663 mJ (45.663 J)
  - VDD_SOC: 7310 mJ (7.310 J)
- 温度：
  - cpu: 54.8°C (peak 55.2°C)
  - gpu: 54.4°C (peak 54.7°C)
  - soc0: 53.3°C (peak 53.5°C)
  - soc1: 53.9°C (peak 54.1°C)
  - soc2: 53.2°C (peak 53.4°C)
  - tj: 54.8°C (peak 55.2°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (73 samples @ 100ms)
- 采样窗口时长：7.90s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3090/3219 mW
  - VDD_IN: 8159/8452 mW
  - VDD_SOC: 1586/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 24423 mJ (24.423 J)
  - VDD_IN: 64486 mJ (64.486 J)
  - VDD_SOC: 12538 mJ (12.537 J)
- 温度：
  - cpu: 53.6°C (peak 53.9°C)
  - gpu: 53.7°C (peak 54.1°C)
  - soc0: 52.8°C (peak 52.9°C)
  - soc1: 53.2°C (peak 53.5°C)
  - soc2: 52.3°C (peak 52.4°C)
  - tj: 53.7°C (peak 54.1°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：0.92s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2886/2985 mW
  - VDD_IN: 7680/7775 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2641 mJ (2.641 J)
  - VDD_IN: 7028 mJ (7.028 J)
  - VDD_SOC: 1386 mJ (1.386 J)
- 温度：
  - cpu: 52.9°C (peak 53.2°C)
  - gpu: 53.2°C (peak 53.5°C)
  - soc0: 52.3°C (peak 52.4°C)
  - soc1: 52.7°C (peak 52.9°C)
  - soc2: 51.8°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.5°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 14522.2 | 14.84 | 0.65 |
| LTC | 1321 | 0.465366 | 8737.8 | 19.24 | 1.11 |
| PDNAPulse | 3170 | 0.286117 | 20046.8 | 7.68 | 0.48 |
| GRU | 1969 | 0.393358 | 63184.9 | 2.49 | 0.19 |
| NCPS-LTC | 2547 | 0.621254 | 2397.1 | 63.97 | 4.36 |
| NCPS-CfC | 15737 | 0.106344 | 3694.5 | 23.51 | 3.09 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
