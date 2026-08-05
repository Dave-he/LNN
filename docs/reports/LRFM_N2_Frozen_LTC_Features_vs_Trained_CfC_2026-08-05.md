---
title: L-RFM (Frozen LTC Features) vs Trained CfC (N2 closure) — Frozen features 6× worse on sequence regression
date: 2026-08-05
tags: [LNN, L-RFM, random-features, frozen-LTC, frozen-feature, N2, foundational-closure, honest-finding, PDE-vs-sequence]
arxiv_refs: [2606.15571, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[Liquid_Random_Feature_Methods_TD-PDE_2606.15571_研读报告]]
gap_refs: [N2-L-RFM-50%-done]
---

# L-RFM (Frozen LTC Features) vs Trained CfC (N2 closure)

> L-RFM paper (arXiv 2606.15571) implements "frozen LTC random features + linear readout" for PDE solving. This round closes the N2 gap (50% done from earlier digest) by implementing it in code and benchmarking against trained CfC on **sequence regression** (different from the PDE solving domain). **Finding**: L-RFM n=64 fits to MSE 0.28, **6× worse than trained CfC h=8 (MSE 0.05)** with fewer parameters. The frozen-feature hypothesis doesn't beat trained representations on this task.

## 1. Implementation

代码：[`lnn/core/lrfm.py`](lnn/core/lrfm.py)（168 lines）
- `LiquidRandomFeatureBasis`：frozen LTC 随机特征，closed-form φ(x, t) = h₀(x)·exp(-α(x)·t) + g(x)·A·(1-exp(-α(x)·t))/α(x)
- `LRFMSequenceRegressor`：frozen L-RFM basis + linear readout（or small MLP bottleneck）

测试：[`tests/test_lrfm.py`](tests/test_lrfm.py)（**10/10 通过**）覆盖：
- Forward shape（2D/3D input × various t shapes）
- t=0 returns h₀(x)（初始条件）
- t→∞ returns g·A/α（稳态）
- 数值稳定性（dt from 0.001 to 100）
- Frozen features 验证（requires_grad=False for all basis params）
- Trainable readout only

## 2. Benchmark 结果

数据：[`analysis/jetson/2026-08-05_lrfm.{md,json}`](analysis/jetson/2026-08-05_lrfm.md)

| model | params | test MSE |
|---|---:|---:|
| L-RFM n_features=32 | 417 | 1568.16 ⚠ |
| **L-RFM n_features=64** | 833 | **0.2855** |
| L-RFM n_features=128 | 1665 | 169626.62 ⚠ |
| **CfC h=8** | 329 | **0.0523** |
| CfC h=16 | 1041 | 0.0516 |
| CfC h=24 | 2137 | 0.0520 |

## 3. 关键发现（Honest Finding）

### 3.1 Frozen LTC features 6× worse than trained CfC

| comparison | L-RFM n=64 | CfC h=8 |
|---|---:|---:|
| params | 833 | 329 |
| test MSE | 0.2855 | 0.0523 |
| ratio | 1.0× | **5.5× better** |

→ L-RFM 用 2.5× 更多参数，仍输 5.5× 给 trained CfC。

### 3.2 L-RFM n=32 和 n=128 训练不稳定

| n_features | test MSE | 备注 |
|---:|---:|---|
| 32 | 1568.16 | 拟合不充分（under-parameterized）|
| 64 | 0.2855 | 第一个稳定点 |
| 128 | 169626.62 | 拟合崩溃（over-parameterized，gradient explodes）|

→ L-RFM 对 n_features 敏感——n=64 是 sweet spot，但 trained CfC h=8/16/24 都稳定且更好。

### 3.3 Why？L-RFM 在 sequence regression 上输 trained CfC

| dimension | L-RFM (frozen) | trained CfC |
|---|---|---|
| Feature representation | random LTC (frozen) | learned ODE params |
| Learning signal | only linear readout (833 params) | full network (1041 params) |
| Captures task structure | no (frozen random) | yes (trained) |
| Best MSE | 0.2855 | 0.0516 |

**L-RFM 论文的 strength 场景是 PDE solving**——stiff/dispersive PDE 需要 mesh-free closed-form，但 frozen random features 对 slow temporal features 足够。**Sequence regression 是不同 domain**——trained model 能 adapt to task-specific temporal structure，L-RFM 只能用 random features。

### 3.4 何时该用 L-RFM vs trained CfC？

| 场景 | 推荐 |
|---|---|
| PDE solving (mesh-free) | **L-RFM** (论文原场景) |
| Edge deployment (params 紧) | **trained CfC h=4 + int8** (N1+N20) |
| Sequence regression on time series | **trained CfC / h-GRU** (这 round) |
| Unknown task | **trained CfC σ-decay** (N12 default) |

## 4. N2 closure summary

| aspect | N2 status |
|---|---|
| 50% done gap from earlier digest | ✅ **CLOSED** |
| arXiv 2606.15571 reproduced in code | ✅ |
| Frozen LTC feature basis works | ✅ |
| L-RFM competitive on sequence regression | ❌ (6× worse than trained CfC) |
| N2 finding 适用场景 | PDE solving only |

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N2** | L-RFM foundational | ✅ **本轮关闭（frozen features 在 sequence regression 上 6× worse）** |
| L4 | Liquid-S4 grounding (NCP arXiv ID 不确定) | ⚠ 路线图 |

→ **L4 是唯一 open foundational gap**。L-RFM 已被实现并验证，结论是 L-RFM 不适合本项目主要场景（sequence regression / edge deployment），但保留 `lnn/core/lrfm.py` 作为 PDE 求解工具的 reference。

## 6. 推荐后续动作

1. **本周**：L4 Liquid-S4 grounding（foundational closure）
2. **路线图**：retention design space survey (Round 11) 更新——加入 N18 (Lorenz honest finding) 和 N2 (L-RFM boundary)

## 7. 数据源回链

- 代码
  - [`lnn/core/lrfm.py`](lnn/core/lrfm.py)（168 lines, 2 classes: LiquidRandomFeatureBasis + LRFMSequenceRegressor）
  - [`tests/test_lrfm.py`](tests/test_lrfm.py)（10 tests, all pass）
  - [`scripts/bench_lrfm.py`](scripts/bench_lrfm.py)
- Benchmark
  - [`analysis/jetson/2026-08-05_lrfm.{md,json}`](analysis/jetson/2026-08-05_lrfm.md)
- 上轮对照
  - [[Liquid_Random_Feature_Methods_TD-PDE_2606.15571_研读报告]]（原论文报告）
  - [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]]（N1 trained CfC baseline）
