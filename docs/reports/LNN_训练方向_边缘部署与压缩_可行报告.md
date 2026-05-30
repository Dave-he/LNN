---
title: LNN 训练方向：边缘部署与压缩可行报告
date: 2026-05-28
tags: [LNN, edge-ai, compression, Jetson, Loihi]
---

# LNN 训练方向：边缘部署与压缩可行报告

## 1. 方向定位

边缘部署关注模型大小、延迟、能耗和在线适应能力。LNN 的低参数和连续时间建模使其适合 Jetson、MCU、移动端和神经形态硬件，但需要把训练目标从“最高精度”改为“精度、延迟、能耗的 Pareto 最优”。

检索证据：本方向纳入/暂缓记录见 [[docs/LNN_训练论文检索矩阵_2026-05-28]]。

## 2. 代表论文与资料

| 论文或资料 | 硬件/任务 | 关键启发 |
|---|---|---|
| *Exploring Liquid Neural Networks on Loihi-2* | Loihi-2 / CIFAR-10 | 报告 91.3% accuracy 与 213 microJoules/frame，说明 LNN 可面向低功耗硬件 |
| *When Smaller Wins* | Arduino Nano 33 BLE Sense / 电池预测 | Euler 离散化、双阶段蒸馏、int8 部署、Pareto 选择 |
| *Closed-form continuous-time neural networks* | 通用序列任务 | CfC 避免 ODE solver，更适合边缘推理 |
| 本项目 Jetson benchmark | Jetson Orin Nano | 已有 daily smoke benchmark，可扩展为正式边缘评估 |

## 3. 数据集构建方案

边缘方向需要额外记录硬件相关字段：

```text
sample_id
X, y
scenario: normal / noise / drift / OOD
device_target: Jetson / MCU / Loihi
calibration_split: bool
latency_budget_ms
memory_budget_kb
energy_budget_optional
```

分割建议：

- train：正常训练数据。
- val：常规验证。
- calibration：量化校准数据，不能和 test 混用。
- test_id：常规测试。
- test_ood：漂移、噪声、缺失、极端场景。
- hardware_test：真实设备上采集的延迟和能耗。

## 4. 架构搭建方案

### 4.1 CfC 小模型

首选边缘基线：

```text
Input -> small CfC(hidden=8/16/32) -> Linear -> output
```

优点：

- 无 ODE solver。
- 参数少。
- 容易导出和量化。

本项目已有近似入口：

- `scripts/jetson_lnn_benchmark.py`

正式实验建议把 smoke benchmark 中的 `CfCStyle` 替换或并列为官方 `ncps.CfC`。

### 4.2 Euler-LTC

适合保留液态动力学但降低 solver 成本：

```text
LTC ODE -> fixed-step Euler discretization -> quantization-aware student
```

注意：

- 必须固定步长，便于部署。
- 用训练后的 teacher 监督小型 student。
- 记录 Euler step 和误差变化。

### 4.3 蒸馏与 Pareto 筛选

推荐训练链路：

```text
large teacher CfC/LTC
-> student hidden sweep: 8, 16, 32
-> distill output + hidden trajectory
-> prune / quantize int8
-> hardware measure
-> Pareto select
```

Pareto 目标：

```text
minimize: error, model_size, latency, energy
constraint: accuracy_drop <= threshold
```

## 5. 训练方法

多目标 loss：

```text
loss = task_loss
     + alpha * distill_output_loss
     + beta * hidden_trajectory_loss
     + gamma * smoothness_or_stability_loss
```

推荐配置：

```text
teacher_hidden: 64/128
student_hidden: 8/16/32
lr_teacher: 1e-3
lr_student: 3e-4
gradient_clip: 1.0
quantization: int8 post-training 起步，必要时 QAT
```

硬件测量指标：

- 参数量。
- 模型文件大小。
- peak memory。
- batch=1 latency。
- throughput。
- 能耗或功耗，若设备支持。
- OOD 场景延迟和精度。

## 6. 优化与调参

重点调参：

- `hidden_size`：最直接影响参数量和延迟。
- `seq_len`：影响缓存和推理时间。
- `ode_method`：边缘优先 CfC 或 Euler-LTC。
- quantization calibration set：要覆盖常见幅值和 OOD。
- int8 后激活范围：检查是否因动态时间常数导致饱和。

部署建议：

- Jetson：先 PyTorch eager benchmark，再考虑 TorchScript/ONNX/TensorRT。
- MCU：优先固定步长 Euler-LTC 或极小 CfC，避免动态 shape。
- Loihi/神经形态：需要专门映射神经元和突触，不能直接复用 PyTorch 模型。

## 7. 本项目落地建议

短期：

- 使用 `scripts/jetson_lnn_benchmark.py --pareto` 运行 hidden sweep、seq_len sweep 和多 seed。
- 输出 `analysis/jetson/`，记录 latency/throughput、params、MSE、device，并标记 Pareto front。
- 增加 CPU 与 CUDA 分别测量。

中期：

- 加 `scripts/edge_pareto_lnn.py`。
- 加 student distillation。
- 加 post-training quantization 试验。

## 8. 可行结论

该方向可行，并且与项目现有 Jetson 自动化最契合。下一步应把当前 smoke benchmark 扩展为正式 benchmark：固定数据、固定 seed、多配置、多次重复、真实设备延迟统计。

## 9. 参考资料

- *Exploring Liquid Neural Networks on Loihi-2*：https://arxiv.org/abs/2407.20590
- *When Smaller Wins: Dual-Stage Distillation and Pareto-Guided Compression of Liquid Neural Networks for Edge Battery Prognostics*：https://arxiv.org/abs/2601.06227
- *Closed-form continuous-time neural networks*：https://www.nature.com/articles/s42256-022-00556-7
- 本项目 Jetson 说明：[[docs/每日自动化任务与Jetson验证]]
