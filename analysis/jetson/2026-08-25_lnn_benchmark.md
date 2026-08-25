---
title: Jetson LNN 基准验证 - 2026-08-25
date: 2026-08-25
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-25

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
- 功耗采样：可用 (394 samples @ 100ms)
- 采样窗口时长：41.86s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2430/2985 mW
  - VDD_IN: 7185/7854 mW
  - VDD_SOC: 1491/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 101722 mJ (101.722 J)
  - VDD_IN: 300736 mJ (300.736 J)
  - VDD_SOC: 62416 mJ (62.416 J)
- 温度：
  - cpu: 50.8°C (peak 51.7°C)
  - gpu: 51.0°C (peak 51.8°C)
  - soc0: 50.1°C (peak 50.8°C)
  - soc1: 50.5°C (peak 51.1°C)
  - soc2: 49.6°C (peak 50.2°C)
  - tj: 51.0°C (peak 51.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.36s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2711/2711 mW
  - VDD_IN: 7388/7388 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 985 mJ (0.985 J)
  - VDD_IN: 2684 mJ (2.684 J)
  - VDD_SOC: 536 mJ (0.536 J)
- 温度：
  - cpu: 50.6°C (peak 50.8°C)
  - gpu: 50.7°C (peak 50.7°C)
  - soc0: 49.9°C (peak 49.9°C)
  - soc1: 50.1°C (peak 50.1°C)
  - soc2: 49.3°C (peak 49.3°C)
  - tj: 50.7°C (peak 50.7°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.13s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2627/2627 mW
  - VDD_IN: 7388/7388 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 351 mJ (0.351 J)
  - VDD_IN: 987 mJ (0.987 J)
  - VDD_SOC: 197 mJ (0.197 J)
- 温度：
  - cpu: 51.2°C (peak 51.2°C)
  - gpu: 51.2°C (peak 51.2°C)
  - soc0: 50.1°C (peak 50.1°C)
  - soc1: 50.3°C (peak 50.3°C)
  - soc2: 49.5°C (peak 49.5°C)
  - tj: 51.2°C (peak 51.2°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (10 samples @ 100ms)
- 采样窗口时长：1.12s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2276/2352 mW
  - VDD_IN: 7016/7068 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2552 mJ (2.552 J)
  - VDD_IN: 7867 mJ (7.867 J)
  - VDD_SOC: 1656 mJ (1.656 J)
- 温度：
  - cpu: 50.6°C (peak 50.8°C)
  - gpu: 50.9°C (peak 51.2°C)
  - soc0: 50.0°C (peak 50.1°C)
  - soc1: 50.3°C (peak 50.5°C)
  - soc2: 49.4°C (peak 49.5°C)
  - tj: 50.9°C (peak 51.2°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.50s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2886/2905 mW
  - VDD_IN: 7628/7655 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1449 mJ (1.449 J)
  - VDD_IN: 3830 mJ (3.830 J)
  - VDD_SOC: 741 mJ (0.741 J)
- 温度：
  - cpu: 51.4°C (peak 51.6°C)
  - gpu: 51.5°C (peak 51.6°C)
  - soc0: 50.6°C (peak 50.6°C)
  - soc1: 50.9°C (peak 51.0°C)
  - soc2: 50.1°C (peak 50.2°C)
  - tj: 51.5°C (peak 51.6°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (15 samples @ 100ms)
- 采样窗口时长：1.65s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2702/2985 mW
  - VDD_IN: 7428/7695 mW
  - VDD_SOC: 1486/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4461 mJ (4.461 J)
  - VDD_IN: 12263 mJ (12.263 J)
  - VDD_SOC: 2453 mJ (2.453 J)
- 温度：
  - cpu: 51.3°C (peak 51.7°C)
  - gpu: 51.4°C (peak 51.8°C)
  - soc0: 50.5°C (peak 50.7°C)
  - soc1: 50.8°C (peak 51.0°C)
  - soc2: 50.0°C (peak 50.1°C)
  - tj: 51.4°C (peak 51.8°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.48s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2688/2706 mW
  - VDD_IN: 7398/7428 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1292 mJ (1.292 J)
  - VDD_IN: 3557 mJ (3.557 J)
  - VDD_SOC: 709 mJ (0.709 J)
- 温度：
  - cpu: 50.9°C (peak 51.1°C)
  - gpu: 51.2°C (peak 51.3°C)
  - soc0: 50.2°C (peak 50.2°C)
  - soc1: 50.4°C (peak 50.5°C)
  - soc2: 49.6°C (peak 49.6°C)
  - tj: 51.2°C (peak 51.3°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 48908.1 | 3.45 | 0.18 |
| LTC | 1321 | 0.465366 | 17197.0 | 8.10 | 0.53 |
| PDNAPulse | 3170 | 0.286117 | 37379.0 | 2.64 | 0.24 |
| GRU | 1969 | 0.393358 | 129418.1 | 0.78 | 0.07 |
| NCPS-LTC | 2547 | 0.621254 | 11785.6 | 14.73 | 0.83 |
| NCPS-CfC | 15737 | 0.106344 | 37852.8 | 3.41 | 0.26 |

## Benchmark 图
![Jetson LNN Benchmark](2026-08-25_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
