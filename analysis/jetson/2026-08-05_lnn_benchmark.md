---
title: Jetson LNN 基准验证 - 2026-08-05
date: 2026-08-05
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-05

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (1320 samples @ 100ms)
- 采样窗口时长：141.39s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2784/3537 mW
  - VDD_IN: 7828/8757 mW
  - VDD_SOC: 1521/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 393573 mJ (393.573 J)
  - VDD_IN: 1106838 mJ (1106.838 J)
  - VDD_SOC: 215094 mJ (215.094 J)
- 温度：
  - cpu: 53.6°C (peak 54.6°C)
  - gpu: 53.9°C (peak 54.8°C)
  - soc0: 53.0°C (peak 53.8°C)
  - soc1: 53.4°C (peak 54.2°C)
  - soc2: 52.5°C (peak 53.2°C)
  - tj: 53.9°C (peak 54.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (14 samples @ 100ms)
- 采样窗口时长：1.55s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3032/3100 mW
  - VDD_IN: 8210/8293 mW
  - VDD_SOC: 1584/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4693 mJ (4.693 J)
  - VDD_IN: 12708 mJ (12.708 J)
  - VDD_SOC: 2452 mJ (2.452 J)
- 温度：
  - cpu: 53.2°C (peak 53.5°C)
  - gpu: 53.4°C (peak 53.7°C)
  - soc0: 52.4°C (peak 52.5°C)
  - soc1: 52.8°C (peak 53.0°C)
  - soc2: 51.9°C (peak 52.0°C)
  - tj: 53.4°C (peak 53.7°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.25s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2980/2980 mW
  - VDD_IN: 8233/8253 mW
  - VDD_SOC: 1552/1552 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 732 mJ (0.732 J)
  - VDD_IN: 2022 mJ (2.022 J)
  - VDD_SOC: 381 mJ (0.381 J)
- 温度：
  - cpu: 53.6°C (peak 53.8°C)
  - gpu: 53.7°C (peak 53.7°C)
  - soc0: 52.8°C (peak 52.9°C)
  - soc1: 53.3°C (peak 53.4°C)
  - soc2: 52.4°C (peak 52.4°C)
  - tj: 53.7°C (peak 53.8°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (18 samples @ 100ms)
- 采样窗口时长：2.01s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2461/2587 mW
  - VDD_IN: 7539/7775 mW
  - VDD_SOC: 1541/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4942 mJ (4.942 J)
  - VDD_IN: 15139 mJ (15.139 J)
  - VDD_SOC: 3095 mJ (3.095 J)
- 温度：
  - cpu: 53.2°C (peak 53.7°C)
  - gpu: 53.7°C (peak 54.0°C)
  - soc0: 52.7°C (peak 52.8°C)
  - soc1: 53.2°C (peak 53.3°C)
  - soc2: 52.2°C (peak 52.3°C)
  - tj: 53.7°C (peak 54.0°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (15 samples @ 100ms)
- 采样窗口时长：1.68s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2627/2706 mW
  - VDD_IN: 7514/7575 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4414 mJ (4.413 J)
  - VDD_IN: 12623 mJ (12.623 J)
  - VDD_SOC: 2478 mJ (2.478 J)
- 温度：
  - cpu: 54.0°C (peak 54.2°C)
  - gpu: 54.4°C (peak 54.7°C)
  - soc0: 53.5°C (peak 53.6°C)
  - soc1: 54.0°C (peak 54.2°C)
  - soc2: 53.0°C (peak 53.1°C)
  - tj: 54.4°C (peak 54.7°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (84 samples @ 100ms)
- 采样窗口时长：9.07s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2616/2941 mW
  - VDD_IN: 7575/8133 mW
  - VDD_SOC: 1485/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 23734 mJ (23.734 J)
  - VDD_IN: 68723 mJ (68.723 J)
  - VDD_SOC: 13472 mJ (13.473 J)
- 温度：
  - cpu: 53.9°C (peak 54.4°C)
  - gpu: 54.3°C (peak 54.6°C)
  - soc0: 53.4°C (peak 53.6°C)
  - soc1: 53.9°C (peak 54.0°C)
  - soc2: 52.9°C (peak 53.1°C)
  - tj: 54.3°C (peak 54.6°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.75s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2786/2866 mW
  - VDD_IN: 7768/7814 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2094 mJ (2.094 J)
  - VDD_IN: 5840 mJ (5.840 J)
  - VDD_SOC: 1139 mJ (1.139 J)
- 温度：
  - cpu: 53.4°C (peak 53.6°C)
  - gpu: 53.7°C (peak 53.7°C)
  - soc0: 52.8°C (peak 52.9°C)
  - soc1: 53.2°C (peak 53.3°C)
  - soc2: 52.3°C (peak 52.3°C)
  - tj: 53.7°C (peak 53.7°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 11439.7 | 10.71 | 0.86 |
| LTC | 1321 | 0.465366 | 9191.7 | 24.36 | 1.02 |
| PDNAPulse | 3170 | 0.286117 | 23388.9 | 4.33 | 0.40 |
| GRU | 1969 | 0.393358 | 76627.1 | 3.79 | 0.14 |
| NCPS-LTC | 2547 | 0.621254 | 2151.8 | 66.00 | 4.65 |
| NCPS-CfC | 15737 | 0.106344 | 11026.7 | 11.98 | 0.85 |

## Benchmark 图
![Jetson LNN Benchmark](2026-08-05_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
