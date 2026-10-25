---
title: Jetson LNN 基准验证 - 2026-10-26
date: 2026-10-26
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-26

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 490 MB / available 1781 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (1258 samples @ 100ms)
- 采样窗口时长：138.09s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3200/3848 mW
  - VDD_IN: 8183/9155 mW
  - VDD_SOC: 1566/1711 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 441820 mJ (441.820 J)
  - VDD_IN: 1129956 mJ (1129.956 J)
  - VDD_SOC: 216258 mJ (216.258 J)
- 温度：
  - cpu: 52.8°C (peak 53.9°C)
  - gpu: 52.9°C (peak 53.9°C)
  - soc0: 51.9°C (peak 52.8°C)
  - soc1: 52.3°C (peak 53.3°C)
  - soc2: 51.5°C (peak 52.2°C)
  - tj: 53.0°C (peak 53.9°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (7 samples @ 100ms)
- 采样窗口时长：0.80s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3200/3298 mW
  - VDD_IN: 7957/8054 mW
  - VDD_SOC: 1504/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2562 mJ (2.562 J)
  - VDD_IN: 6369 mJ (6.369 J)
  - VDD_SOC: 1204 mJ (1.204 J)
- 温度：
  - cpu: 51.9°C (peak 52.1°C)
  - gpu: 52.0°C (peak 52.2°C)
  - soc0: 51.1°C (peak 51.2°C)
  - soc1: 51.4°C (peak 51.6°C)
  - soc2: 50.6°C (peak 50.7°C)
  - tj: 52.0°C (peak 52.2°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.37s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3272/3298 mW
  - VDD_IN: 8120/8133 mW
  - VDD_SOC: 1513/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1212 mJ (1.212 J)
  - VDD_IN: 3007 mJ (3.007 J)
  - VDD_SOC: 560 mJ (0.560 J)
- 温度：
  - cpu: 52.5°C (peak 52.6°C)
  - gpu: 52.7°C (peak 52.8°C)
  - soc0: 51.7°C (peak 51.7°C)
  - soc1: 52.1°C (peak 52.2°C)
  - soc2: 51.3°C (peak 51.4°C)
  - tj: 52.7°C (peak 52.8°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (24 samples @ 100ms)
- 采样窗口时长：2.68s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3052/3219 mW
  - VDD_IN: 8109/8293 mW
  - VDD_SOC: 1590/1634 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 8193 mJ (8.193 J)
  - VDD_IN: 21765 mJ (21.765 J)
  - VDD_SOC: 4267 mJ (4.267 J)
- 温度：
  - cpu: 52.4°C (peak 52.8°C)
  - gpu: 52.4°C (peak 52.8°C)
  - soc0: 51.4°C (peak 51.5°C)
  - soc1: 51.8°C (peak 52.0°C)
  - soc2: 51.0°C (peak 51.1°C)
  - tj: 52.5°C (peak 52.9°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (17 samples @ 100ms)
- 采样窗口时长：1.96s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3280/3457 mW
  - VDD_IN: 8194/8373 mW
  - VDD_SOC: 1552/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 6424 mJ (6.424 J)
  - VDD_IN: 16046 mJ (16.046 J)
  - VDD_SOC: 3040 mJ (3.040 J)
- 温度：
  - cpu: 53.5°C (peak 53.8°C)
  - gpu: 53.6°C (peak 53.8°C)
  - soc0: 52.5°C (peak 52.7°C)
  - soc1: 53.0°C (peak 53.2°C)
  - soc2: 52.1°C (peak 52.2°C)
  - tj: 53.6°C (peak 53.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (71 samples @ 100ms)
- 采样窗口时长：7.99s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3153/3338 mW
  - VDD_IN: 8124/8333 mW
  - VDD_SOC: 1564/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 25185 mJ (25.185 J)
  - VDD_IN: 64896 mJ (64.896 J)
  - VDD_SOC: 12490 mJ (12.490 J)
- 温度：
  - cpu: 53.3°C (peak 53.8°C)
  - gpu: 53.5°C (peak 53.8°C)
  - soc0: 52.5°C (peak 52.8°C)
  - soc1: 53.0°C (peak 53.2°C)
  - soc2: 52.1°C (peak 52.2°C)
  - tj: 53.5°C (peak 53.8°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (7 samples @ 100ms)
- 采样窗口时长：0.83s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3019/3144 mW
  - VDD_IN: 7900/8054 mW
  - VDD_SOC: 1538/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2495 mJ (2.495 J)
  - VDD_IN: 6527 mJ (6.527 J)
  - VDD_SOC: 1271 mJ (1.271 J)
- 温度：
  - cpu: 52.4°C (peak 52.6°C)
  - gpu: 52.6°C (peak 52.8°C)
  - soc0: 51.6°C (peak 51.8°C)
  - soc1: 52.0°C (peak 52.2°C)
  - soc2: 51.2°C (peak 51.2°C)
  - tj: 52.6°C (peak 52.8°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 22471.9 | 7.05 | 0.43 |
| LTC | 1321 | 0.465366 | 6764.7 | 19.23 | 1.47 |
| PDNAPulse | 3170 | 0.286117 | 22518.7 | 11.93 | 0.44 |
| GRU | 1969 | 0.393358 | 52659.1 | 2.98 | 0.20 |
| NCPS-LTC | 2547 | 0.621254 | 2424.6 | 66.75 | 4.39 |
| NCPS-CfC | 15737 | 0.106344 | 9929.2 | 13.07 | 1.09 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
