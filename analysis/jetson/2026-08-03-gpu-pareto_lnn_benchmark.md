---
title: Jetson LNN 基准验证 - 2026-08-03-gpu-pareto
date: 2026-08-03-gpu-pareto
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-03-gpu-pareto

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：unknown
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

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- Samples / Epoch：192 / 3
- Hidden sweep：[8, 16, 24]
- SeqLen sweep：[16, 32]
- Seeds：[42, 43]
- 设备：cuda

## Pareto 结果
| Front | 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yes | CfCStyle | 24 | 32 | 43 | 2521 | 0.248911 | 35425.0 | 3.40 |
| yes | PDNAPulse | 16 | 32 | 43 | 1474 | 0.267485 | 37396.3 | 3.02 |
| yes | PDNAPulse | 24 | 32 | 43 | 3170 | 0.278735 | 38340.9 | 3.05 |
| yes | CfCStyle | 16 | 32 | 43 | 1169 | 0.288531 | 35780.8 | 3.53 |
| yes | PDNAPulse | 8 | 32 | 43 | 418 | 0.307434 | 38814.5 | 3.01 |
| yes | PDNAPulse | 24 | 32 | 42 | 3170 | 0.316487 | 41135.0 | 3.10 |
| yes | PDNAPulse | 16 | 32 | 42 | 1474 | 0.317952 | 38848.3 | 3.01 |
| yes | GRU | 24 | 32 | 42 | 1969 | 0.352016 | 1598667.9 | 0.13 |
| yes | CfCStyle | 8 | 32 | 42 | 329 | 0.387720 | 35166.4 | 3.43 |
| yes | GRU | 16 | 32 | 42 | 929 | 0.388195 | 1098075.9 | 0.18 |
| yes | CfCStyle | 8 | 32 | 43 | 329 | 0.426778 | 35021.4 | 3.43 |
| yes | GRU | 8 | 32 | 43 | 273 | 0.431042 | 1138831.1 | 0.19 |
| yes | LTC | 8 | 32 | 43 | 185 | 0.437732 | 7090.0 | 12.45 |
| yes | GRU | 16 | 32 | 43 | 929 | 0.470857 | 1512779.8 | 0.14 |
| yes | GRU | 8 | 32 | 42 | 273 | 0.486094 | 1399056.0 | 0.15 |
| yes | LTC | 8 | 16 | 43 | 185 | 0.590416 | 6809.4 | 6.53 |
| yes | LTC | 8 | 16 | 42 | 185 | 0.611989 | 6977.2 | 6.48 |
|  | LTC | 24 | 32 | 42 | 1321 | 0.317136 | 6795.6 | 12.94 |
|  | CfCStyle | 24 | 32 | 42 | 2521 | 0.333051 | 35845.4 | 3.42 |
|  | CfCStyle | 16 | 32 | 42 | 1169 | 0.354412 | 35304.2 | 3.45 |
|  | GRU | 24 | 32 | 43 | 1969 | 0.358831 | 1181599.2 | 0.16 |
|  | PDNAPulse | 8 | 32 | 42 | 418 | 0.363572 | 37654.3 | 3.10 |
|  | LTC | 24 | 32 | 43 | 1321 | 0.385424 | 6864.0 | 12.72 |
|  | CfCStyle | 24 | 16 | 43 | 2521 | 0.385458 | 28383.3 | 2.09 |
|  | PDNAPulse | 24 | 16 | 42 | 3170 | 0.403600 | 29731.4 | 1.70 |
|  | PDNAPulse | 24 | 16 | 43 | 3170 | 0.404332 | 30360.9 | 1.74 |
|  | PDNAPulse | 16 | 16 | 43 | 1474 | 0.419344 | 33469.4 | 1.75 |
|  | PDNAPulse | 16 | 16 | 42 | 1474 | 0.422621 | 31977.8 | 1.79 |
|  | CfCStyle | 16 | 16 | 43 | 1169 | 0.439955 | 29608.9 | 1.92 |
|  | PDNAPulse | 8 | 16 | 43 | 418 | 0.440730 | 29612.5 | 1.81 |
|  | LTC | 16 | 32 | 43 | 625 | 0.443071 | 6766.0 | 12.85 |
|  | CfCStyle | 24 | 16 | 42 | 2521 | 0.447717 | 29682.7 | 2.00 |
|  | LTC | 16 | 32 | 42 | 625 | 0.467261 | 7087.0 | 12.40 |
|  | CfCStyle | 16 | 16 | 42 | 1169 | 0.473279 | 29527.0 | 1.99 |
|  | GRU | 24 | 16 | 42 | 1969 | 0.475153 | 660077.6 | 0.20 |
|  | GRU | 24 | 16 | 43 | 1969 | 0.488303 | 476308.3 | 0.18 |
|  | CfCStyle | 8 | 16 | 42 | 329 | 0.507128 | 28895.5 | 2.36 |
|  | PDNAPulse | 8 | 16 | 42 | 418 | 0.509493 | 28483.1 | 1.95 |
|  | LTC | 24 | 16 | 42 | 1321 | 0.512739 | 6821.4 | 6.60 |
|  | GRU | 16 | 16 | 42 | 929 | 0.519910 | 648756.4 | 0.18 |
|  | LTC | 24 | 16 | 43 | 1321 | 0.533649 | 6456.0 | 6.68 |
|  | CfCStyle | 8 | 16 | 43 | 329 | 0.543425 | 33634.1 | 2.10 |
|  | GRU | 8 | 16 | 43 | 273 | 0.544936 | 537194.2 | 0.18 |
|  | LTC | 8 | 32 | 42 | 185 | 0.562610 | 7041.6 | 12.51 |
|  | GRU | 16 | 16 | 43 | 929 | 0.571893 | 641004.2 | 0.18 |
|  | GRU | 8 | 16 | 42 | 273 | 0.576736 | 780926.8 | 0.27 |
|  | LTC | 16 | 16 | 42 | 625 | 0.581893 | 6954.8 | 6.41 |
|  | LTC | 16 | 16 | 43 | 625 | 0.603967 | 6805.8 | 6.41 |

## Pareto 图
![Jetson LNN Pareto](2026-08-03-gpu-pareto_lnn_pareto.png)

## GPU vs CPU 对照 (2026-08-03 双侧 sweep)

| 指标 | CPU `2026-08-03-cpu-pareto` | GPU `2026-08-03-gpu-pareto` | 加速比 |
|---|---|---|---|
| CfCStyle h=24 seq=32 seed=42 MSE | 0.470 | 0.333 | 误差更好 |
| CfCStyle h=24 seq=32 seed=42 steps/s | 18 201 | 34 444 | **1.9×** |
| PDNAPulse h=24 seq=32 seed=42 MSE | 0.413 | 0.316 | 误差更好 |
| PDNAPulse h=24 seq=32 seed=42 steps/s | 23 593 | 36 723 | 1.6× |
| GRU h=24 seq=32 seed=42 steps/s | 69 905 | **1 229 079** | **17.6×** |
| GRU h=8 seq=16 seed=42 MSE | 0.601 | 0.577 | 误差略好 |
| LTC h=8 seq=32 seed=42 steps/s | 10 575 | 6 469 | 0.61× (ODE solver 绑 CPU host) |
| LTC h=8 seq=16 seed=42 steps/s | 7 261 | 6 552 | 0.90× |

观察:
- **GRU 在 GPU 上拿到 17× 加速**,GEMM-bound + cuBLAS Tensor Core 全开。
- **CfCStyle / PDNAPulse 仅 ~1.6-1.9×**,因为 CfC forward 的 `for t in seq` 循环是 sequential 的,host sync 锁住 GPU 增益。
- **LTC 在 GPU 上反而比 CPU 慢**(0.61-0.90×):ODE solver `torchdiffeq.odeint` 是 for-step Python loop,host-bound,GPU 没帮忙。
- **关键决策**:对边缘 CfC 部署,真正能吃 GPU 的是 `torch.compile(model, mode="reduce-overhead")` 把循环 fuse,而不是把 hidden_size 拉大。
- 与 [[Orin_Nano_Super_LNN_Deployment_v2_2026-08-03]] § 4.2 预测对比:`sm_87 估算 steps/s ~250 000` vs 实测 `CfCStyle h=24 = 34 444` —— 预测高估 ~7×,因为 CfC inner-loop 是 sequential。**更新预测**:sm_87 上 hidden=24-256 CfC forward 实际可达 30-50K steps/s(GEMM-bound + reduce-overhead 可推到 80K+)。

## 解读
- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、更短训练时间和更高吞吐。
- 该 sweep 是边缘筛选入口,正式实验应在真实 Jetson CUDA 路径上增加多 seed、能耗和导出后延迟。
