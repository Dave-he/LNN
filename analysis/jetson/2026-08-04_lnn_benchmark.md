---
title: Jetson LNN 基准验证 - 2026-08-04
date: 2026-08-04
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-04

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
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (253 samples @ 100ms)
- 采样窗口时长：26.93s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1696/2507 mW
  - VDD_IN: 6479/7468 mW
  - VDD_SOC: 1498/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 45677 mJ (45.677 J)
  - VDD_IN: 174463 mJ (174.463 J)
  - VDD_SOC: 40341 mJ (40.341 J)
- 温度：
  - cpu: 51.1°C (peak 52.2°C)
  - gpu: 51.7°C (peak 52.2°C)
  - soc0: 50.9°C (peak 51.2°C)
  - soc1: 51.2°C (peak 51.5°C)
  - soc2: 50.2°C (peak 50.4°C)
  - tj: 51.7°C (peak 52.2°C)
- GPU 利用率：mean 21%, peak 32%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.18s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1557/1557 mW
  - VDD_IN: 6320/6320 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 283 mJ (0.283 J)
  - VDD_IN: 1149 mJ (1.149 J)
  - VDD_SOC: 268 mJ (0.269 J)
- 温度：
  - cpu: 50.8°C (peak 50.8°C)
  - gpu: 51.6°C (peak 51.6°C)
  - soc0: 51.0°C (peak 51.0°C)
  - soc1: 51.2°C (peak 51.2°C)
  - soc2: 50.2°C (peak 50.2°C)
  - tj: 51.6°C (peak 51.6°C)
- GPU 利用率：mean 25%, peak 25%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：0.90s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1637/1677 mW
  - VDD_IN: 6385/6440 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1468 mJ (1.468 J)
  - VDD_IN: 5726 mJ (5.726 J)
  - VDD_SOC: 1325 mJ (1.325 J)
- 温度：
  - cpu: 50.9°C (peak 51.2°C)
  - gpu: 51.8°C (peak 52.0°C)
  - soc0: 50.9°C (peak 51.0°C)
  - soc1: 51.2°C (peak 51.4°C)
  - soc2: 50.1°C (peak 50.2°C)
  - tj: 51.8°C (peak 52.0°C)
- GPU 利用率：mean 20%, peak 20%

### PDNAPulse
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.17s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1517/1517 mW
  - VDD_IN: 6240/6240 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 264 mJ (0.264 J)
  - VDD_IN: 1084 mJ (1.084 J)
  - VDD_SOC: 257 mJ (0.257 J)
- 温度：
  - cpu: 51.1°C (peak 51.1°C)
  - gpu: 51.7°C (peak 51.7°C)
  - soc0: 51.1°C (peak 51.1°C)
  - soc1: 51.3°C (peak 51.3°C)
  - soc2: 50.1°C (peak 50.1°C)
  - tj: 51.7°C (peak 51.7°C)
- GPU 利用率：mean 25%, peak 25%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：256 / 32
- 隐藏维度 / Epoch：16 / 5
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 1169 | 0.313733 | 49042.2 | 4.99 | 0.17 |
| LTC | 625 | 0.400177 | 9410.7 | 14.81 | 0.86 |
| PDNAPulse | 1474 | 0.291902 | 52169.2 | 3.80 | 0.16 |
| GRU | 929 | 0.353614 | 1931749.1 | 0.43 | n/a |

## Benchmark 图
![Jetson LNN Benchmark](2026-08-04_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
