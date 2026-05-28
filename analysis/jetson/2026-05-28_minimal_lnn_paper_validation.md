---
title: Jetson 最小 LNN 论文思路验证 - 2026-05-28
date: 2026-05-28
tags: [LNN, Jetson, CfC, LTC, simulation]
---

# Jetson 最小 LNN 论文思路验证 - 2026-05-28

## 验证目标
- 用最小非平稳不规则采样序列验证 LTC/CfC 论文里的连续时间动态建模思路。
- 比较 `CfC-DT`、`Euler-LTC-DT` 与传统 `GRU+dt` 在 ID/OOD 误差、参数量、训练时间和推理吞吐上的差异。
- 本实验是本机 Jetson smoke validation，不等同于正式论文复现。

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 机器架构：aarch64
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False，device_count=1，torch CUDA=13.0
- CUDA 说明：CUDA device is visible, but torch.cuda.is_available() is false.
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 配置
- 设备：cpu
- 样本数 / 序列长度：384 / 40
- 隐藏维度 / Epoch：16 / 5
- Batch / LR：64 / 0.003
- Seed：42

## 结果
| 模型 | 参数量 | ID MSE | OOD MSE | OOD 退化 | ID MAE | OOD MAE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CfC-DT | 1169 | 0.291490 | 1.155476 | 296.4% | 0.438160 | 0.887749 | 59148.2 | 4.28 |
| Euler-LTC-DT | 625 | 0.283694 | 0.863879 | 204.5% | 0.431628 | 0.753289 | 82867.2 | 3.03 |
| GRU+dt | 977 | 0.397782 | 1.282772 | 222.5% | 0.530609 | 0.949078 | 321832.5 | 1.32 |

## 图
![Minimal LNN Paper Validation](2026-05-28_minimal_lnn_paper_validation.png)

## 结论
- `CfC-DT` 对应闭式连续时间思想，重点观察是否以较小参数和较高吞吐跑通不规则 `dt` 输入。
- `Euler-LTC-DT` 对应 LTC 的输入依赖时间常数，用固定步 Euler 做 Jetson 友好的最小模拟。
- `GRU+dt` 是同等输入信息的传统循环基线；若 LNN 方案没有在误差、退化率或延迟上占优，需要继续调 `hidden_size`、`seq_len`、学习率和数据难度。

## 复现命令

```bash
python scripts/minimal_lnn_paper_validation.py --cpu --samples 384 --seq-len 40 --hidden-size 16 --epochs 5 --batch-size 64 --lr 0.003 --weight-decay 0.0001 --grad-clip 1.0 --seed 42 --inference-repeats 6
```
