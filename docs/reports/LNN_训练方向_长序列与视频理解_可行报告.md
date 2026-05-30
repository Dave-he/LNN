---
title: LNN 训练方向：长序列与视频理解可行报告
date: 2026-05-28
tags: [LNN, Liquid-S4, video, long-sequence, temporal-action-detection]
---

# LNN 训练方向：长序列与视频理解可行报告

## 1. 方向定位

长序列与视频理解要求模型处理数千到数万步依赖。直接逐步求解 LTC 不适合该场景，优先路线是 Liquid-S4 或把 liquid 动态蒸馏为可并行 temporal operator。

检索证据：本方向纳入/暂缓记录见 [[docs/LNN_训练论文检索矩阵_2026-05-28]]。

## 2. 代表论文与数据源

| 论文或数据源 | 任务 | 关键启发 |
|---|---|---|
| *Liquid Structural State-Space Models* | 长程序列建模 | 把 LTC 思想并入结构化状态空间模型，适合长依赖 |
| *LiquidTAD* | Temporal Action Detection | 将 liquid exponential relaxation 变成并行算子，避免顺序 ODE |
| Long Range Arena | 长序列 benchmark | 序列长度覆盖 1K 到 16K，适合测试长依赖 |
| THUMOS-14、ActivityNet-1.3、Ego4D | 视频动作检测 | 需要边界定位、长视频背景建模和计算效率 |

## 3. 数据集构建方案

### 3.1 长序列通用格式

```text
tokens_or_features: [T, F]
target: class / sequence label / next token / regression target
mask: [T]
chunk_id, original_sequence_id
```

处理策略：

- 对超长序列做 chunk，但保留原始 sequence_id，避免同一序列泄漏到不同 split。
- 对序列长度分桶，减少 padding。
- 单独报告 extrapolation：训练短序列，测试更长序列。

### 3.2 视频动作检测格式

```text
video_id
features: [T, F]  # 预提取 I3D/SlowFast/VideoMAE 等特征
segments:
  start_time, end_time, class_id
fps_or_stride
duration
```

处理策略：

- 先预提取视频特征，不在第一版端到端训练视觉 backbone。
- 保留时间边界的真实秒数和 feature index 映射。
- 对背景片段做负样本采样。
- 按视频切分 train/val/test，不能按 clip 随机切分。

## 4. 架构搭建方案

### 4.1 Liquid-S4 路线

```text
input sequence
-> projection
-> Liquid-S4 blocks
-> pooling or sequence head
```

适合：

- 文本、音频、医学长序列。
- Long Range Arena 类 benchmark。
- 需要线性或近线性复杂度的长依赖任务。

工程状态：

- 本项目已新增 `lnn/core/long_sequence.py` 的轻量 Liquid-S4-style block，用于本机 smoke run。
- 该实现不是官方完整 Liquid-S4；正式复现仍应对接外部 repo 或论文实现。

### 4.2 LiquidTAD 路线

```text
video features
-> temporal feature pyramid
-> parallel liquid-inspired relaxation blocks
-> classification + boundary regression heads
```

适合：

- Temporal Action Detection。
- 长视频中定位动作边界。
- 需要硬件友好的并行计算。

关键点：

- 不复现完整 LNN ODE，而是保留指数松弛和动态 decay-rate 先验。
- 输出用 mAP、参数量和 FLOPs 共同评估。

### 4.3 简化本项目路线

如果先做轻量验证，可用：

```text
pre-extracted features
-> CfC with truncated windows
-> temporal pooling
-> classifier
```

这不是最终长序列方案，只适合作为快速 sanity check。

## 5. 训练方法

长序列分类：

```text
loss = CrossEntropy(sequence_prediction, label)
```

视频检测：

```text
loss = classification_loss + lambda * boundary_regression_loss
```

推荐配置：

```text
chunk_len: 512, 1024, 2048
batch_size: 依显存调整
lr: 3e-4 或 1e-4
warmup: 5% steps
gradient_clip: 1.0
mixed_precision: 可开启
```

评估：

- Long sequence：Accuracy、F1、longer-than-train generalization。
- TAD：mAP at IoU thresholds、average mAP、FLOPs、参数量、延迟。
- 工程：训练吞吐 tokens/s 或 frames/s。

## 6. 优化与调参

重点：

- chunk 长度和 overlap。
- decay-rate sharing 或跨层时间尺度共享。
- feature stride 与边界精度的权衡。
- 长序列任务必须看内存增长曲线。
- 与 Transformer、S4、Mamba、GRU/LSTM 做同预算对比。

增强策略：

- temporal jitter：随机缩放或偏移动作边界。
- frame dropping：模拟采样率变化。
- long-context curriculum：从短序列逐步增加到长序列。
- class-balanced sampling：动作检测中背景样本通常占比过高。

## 7. 本项目落地建议

短期：

- 不建议从零实现完整 Liquid-S4。
- 已新增 `scripts/experiment_long_sequence.py`，支持长序列分类和 LiquidTAD-style frame-level smoke test。
- 已新增 `SyntheticLongSequenceDataset`，用于验证 chunk/mask、frame label 和 boundary target 的数据格式。
- 每日追踪中把 LiquidTAD、Liquid-S4 相关论文标为中期复现。

中期：

- 克隆或子模块化 Liquid-S4 官方实现。
- 加入 LRA 小任务或预提取视频特征数据。
- 将当前轻量 LiquidTAD-style head 扩展到真实预提取视频特征，并补 mAP/IoU 指标。

## 8. 可行结论

该方向适合中期研究，不适合作为最先落地的训练教程。它的核心不是“普通 LNN 怎么训练”，而是“如何把 liquid 动态改写为可并行、可扩展的长序列算子”。

## 9. 参考资料

- *Liquid Structural State-Space Models*：https://arxiv.org/abs/2209.12951
- *LiquidTAD: Efficient Temporal Action Detection via Parallel Liquid-Inspired Temporal Relaxation*：https://arxiv.org/abs/2604.18274
- Long Range Arena：https://arxiv.org/abs/2011.04006
- Liquid-S4 官方实现：https://github.com/raminmh/liquid-s4
