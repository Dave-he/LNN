---
title: Jetson LNN 基准验证 - 2026-10-21
date: 2026-10-21
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-21

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 210 MB / available 2020 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (1136 samples @ 100ms)
- 采样窗口时长：123.22s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3099/3616 mW
  - VDD_IN: 7955/8612 mW
  - VDD_SOC: 1519/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 381866 mJ (381.866 J)
  - VDD_IN: 980137 mJ (980.137 J)
  - VDD_SOC: 187216 mJ (187.216 J)
- 温度：
  - cpu: 52.0°C (peak 53.2°C)
  - gpu: 52.0°C (peak 53.0°C)
  - soc0: 51.1°C (peak 51.9°C)
  - soc1: 51.5°C (peak 52.4°C)
  - soc2: 50.6°C (peak 51.4°C)
  - tj: 52.1°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (11 samples @ 100ms)
- 采样窗口时长：1.32s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3148/3224 mW
  - VDD_IN: 8115/8133 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4168 mJ (4.168 J)
  - VDD_IN: 10744 mJ (10.744 J)
  - VDD_SOC: 2059 mJ (2.059 J)
- 温度：
  - cpu: 51.3°C (peak 51.4°C)
  - gpu: 51.2°C (peak 51.5°C)
  - soc0: 50.2°C (peak 50.4°C)
  - soc1: 50.6°C (peak 50.7°C)
  - soc2: 49.8°C (peak 49.9°C)
  - tj: 51.3°C (peak 51.5°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.28s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3025/3025 mW
  - VDD_IN: 7775/7775 mW
  - VDD_SOC: 1495/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 838 mJ (0.838 J)
  - VDD_IN: 2153 mJ (2.153 J)
  - VDD_SOC: 414 mJ (0.414 J)
- 温度：
  - cpu: 52.3°C (peak 52.3°C)
  - gpu: 52.3°C (peak 52.4°C)
  - soc0: 51.2°C (peak 51.3°C)
  - soc1: 51.6°C (peak 51.6°C)
  - soc2: 50.8°C (peak 50.9°C)
  - tj: 52.4°C (peak 52.4°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (26 samples @ 100ms)
- 采样窗口时长：2.92s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3120/3338 mW
  - VDD_IN: 8084/8333 mW
  - VDD_SOC: 1555/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 9123 mJ (9.123 J)
  - VDD_IN: 23642 mJ (23.642 J)
  - VDD_SOC: 4547 mJ (4.547 J)
- 温度：
  - cpu: 52.0°C (peak 52.3°C)
  - gpu: 52.0°C (peak 52.3°C)
  - soc0: 51.0°C (peak 51.2°C)
  - soc1: 51.4°C (peak 51.7°C)
  - soc2: 50.6°C (peak 50.7°C)
  - tj: 52.1°C (peak 52.3°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (14 samples @ 100ms)
- 采样窗口时长：1.58s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2951/2985 mW
  - VDD_IN: 7711/7747 mW
  - VDD_SOC: 1492/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4657 mJ (4.657 J)
  - VDD_IN: 12169 mJ (12.169 J)
  - VDD_SOC: 2355 mJ (2.355 J)
- 温度：
  - cpu: 52.4°C (peak 52.6°C)
  - gpu: 52.6°C (peak 52.9°C)
  - soc0: 51.7°C (peak 51.8°C)
  - soc1: 52.1°C (peak 52.2°C)
  - soc2: 51.2°C (peak 51.3°C)
  - tj: 52.6°C (peak 52.9°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (55 samples @ 100ms)
- 采样窗口时长：5.93s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2930/3105 mW
  - VDD_IN: 7678/7894 mW
  - VDD_SOC: 1489/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 17368 mJ (17.368 J)
  - VDD_IN: 45514 mJ (45.514 J)
  - VDD_SOC: 8825 mJ (8.825 J)
- 温度：
  - cpu: 52.4°C (peak 52.7°C)
  - gpu: 52.5°C (peak 52.9°C)
  - soc0: 51.6°C (peak 51.8°C)
  - soc1: 52.0°C (peak 52.3°C)
  - soc2: 51.1°C (peak 51.3°C)
  - tj: 52.5°C (peak 52.9°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.68s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2978/3025 mW
  - VDD_IN: 7718/7814 mW
  - VDD_SOC: 1488/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2029 mJ (2.029 J)
  - VDD_IN: 5258 mJ (5.258 J)
  - VDD_SOC: 1014 mJ (1.014 J)
- 温度：
  - cpu: 52.0°C (peak 52.3°C)
  - gpu: 52.2°C (peak 52.3°C)
  - soc0: 51.2°C (peak 51.3°C)
  - soc1: 51.7°C (peak 51.8°C)
  - soc2: 50.8°C (peak 50.8°C)
  - tj: 52.2°C (peak 52.3°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 14071.1 | 10.23 | 0.73 |
| LTC | 1321 | 0.465366 | 6032.6 | 28.06 | 1.60 |
| PDNAPulse | 3170 | 0.286117 | 26041.5 | 9.43 | 0.36 |
| GRU | 1969 | 0.393358 | 72638.1 | 2.03 | 0.15 |
| NCPS-LTC | 2547 | 0.621254 | 3090.2 | 47.79 | 3.08 |
| NCPS-CfC | 15737 | 0.106344 | 11856.6 | 10.34 | 0.82 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
