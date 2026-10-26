---
title: Jetson LNN 基准验证 - 2026-10-26
date: 2026-10-26
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-26

> ⚠️ **运行状态：`ok`**
> Real GPU measurement: the model ran on CUDA on the Jetson, so the steps/sec below are edge/GPU performance evidence. Still a smoke benchmark (synthetic data, one seed) — not a paper-level reproduction.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 91 MB / available 1802 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (71 samples @ 100ms)
- 采样窗口时长：7.70s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2014/2428 mW
  - VDD_IN: 7046/7615 mW
  - VDD_SOC: 1562/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 15510 mJ (15.511 J)
  - VDD_IN: 54275 mJ (54.275 J)
  - VDD_SOC: 12036 mJ (12.036 J)
- 温度：
  - cpu: 52.4°C (peak 53.2°C)
  - gpu: 53.0°C (peak 53.4°C)
  - soc0: 52.2°C (peak 52.4°C)
  - soc1: 52.6°C (peak 52.8°C)
  - soc2: 51.5°C (peak 51.7°C)
  - tj: 53.0°C (peak 53.4°C)
- GPU 利用率：mean 8%, peak 27%

## 各模型独立功耗

### CfCStyle
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.26s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2229/2269 mW
  - VDD_IN: 7176/7216 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 588 mJ (0.588 J)
  - VDD_IN: 1893 mJ (1.893 J)
  - VDD_SOC: 410 mJ (0.410 J)
- 温度：
  - cpu: 52.5°C (peak 52.5°C)
  - gpu: 53.0°C (peak 53.2°C)
  - soc0: 52.2°C (peak 52.2°C)
  - soc1: 52.7°C (peak 52.8°C)
  - soc2: 51.5°C (peak 51.5°C)
  - tj: 53.0°C (peak 53.2°C)
- GPU 利用率：mean 17%, peak 17%

### PDNAPulse
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：64 / 16
- 隐藏维度 / Epoch：8 / 1
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 329 | 0.691654 | 11064.5 | 1.91 | n/a |
| LTC | 185 | 0.670937 | 2378.4 | 1.78 | 4.55 |
| PDNAPulse | 418 | 0.607059 | 12276.1 | 0.45 | n/a |
| GRU | 273 | 0.609695 | 204975.5 | 0.67 | n/a |

## Benchmark 图
![Jetson LNN Benchmark](2026-10-26_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
