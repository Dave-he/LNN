---
title: 2026-06-04 Loop iteration 6 — GraphLNNPredictor 分子任务 3-backbone × 3-seed smoke
date: 2026-06-04
tags: [LNN, GNN, graph, CfC, LTC, GRU, molecular, Tox21, loop, validation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 6 — GraphLNNPredictor Tox21-styled 分子 smoke

> `/loop 1h` 第 6 次触发。把 iter#5 GCN-CfC 调研报告里的 follow-up task A
> 落地: 用本仓 `lnn.core.graph.GraphLNNPredictor` 在 Tox21-styled
> 合成分子二分类任务上对比 CfC / LTC / GRU 三个 recurrent backbone,
> 跨 3 个随机 seed,在 Jetson Orin Nano Super (CPU) 上跑。
>
> **结论先行**: 3 个 backbone 在该任务上 AUC 几乎并列 (0.754 ± 0.03),
> 但 **LTC 用 28% 更少的参数达到同样精度且方差最低**,
> 是本任务的最佳选择;GRU 推理最快但精度最不稳定。

## 1. 实验设计

### 1.1 任务

合成 Tox21-风格分子二分类:

- 每个"分子"是 Erdős–Rényi 随机图,8–14 个原子,边概率 0.35;
- 每个原子用 4 维 one-hot 编码表示"元素类型";
- 标签 = `triangle_density > dataset_median`(基于数据集中位数自动平衡),
  保证训/验 +/− 比 50/50。

为什么 triangle density: 这是图结构信号,**节点特征 mean pooling
拿不到**(三角密度需要二跳邻居乘法),所以模型必须真正用上 GNN 邻居聚合
才能上 0.5 的随机基线之上 — 我们要的就是这个性质,验证 GraphLNNPredictor
里的 `GraphSnapshotEncoder` 没白搭。

### 1.2 模型 = `GraphLNNPredictor`

本仓 `lnn/core/graph.py` 端到端 PyTorch:
`GraphSnapshotEncoder`(GCN 风格邻居均值池化) →
`{CfCNetwork | LTCNetwork | nn.GRU}`(序列长度 1, 单时间步) →
sigmoid binary head。

`graph_feature_size=24`, `hidden_size=32`, AdamW lr=3e-3, batch=32, epochs=8,
共 3 个 seed: {42, 7, 123}。Device: cpu(CUDA NvMap ENOMEM 仍未恢复,iter#2 解释)。

### 1.3 驱动脚本

新增 `scripts/experiment_graph_lnn_molecule.py`(~230 行):
- 不依赖 torch_geometric / rdkit / TensorFlow(Jetson aarch64 上装这三个成本极高);
- 一次跑 N 个 backbone, 输出 JSON + Markdown 到 `analysis/molecular/`;
- AUC 用 Mann-Whitney U 估计器自实现(不依赖 sklearn)。

## 2. 原始结果(3 seed × 3 backbone)

| seed | backbone | params | acc | AUC | train s | inf samples/s |
|---:|---|---:|---:|---:|---:|---:|
| 42 | cfc | 6,377 | 63.54% | 0.7361 | 2.31 | 5,397 |
| 42 | ltc | 4,585 | 67.71% | 0.7431 | 1.89 | 5,417 |
| 42 | gru | 6,441 | 64.58% | 0.7461 | 1.84 | 6,096 |
| 7 | cfc | 6,377 | 72.92% | 0.7921 | 2.10 | 5,796 |
| 7 | ltc | 4,585 | 75.00% | 0.7943 | 1.72 | 5,346 |
| 7 | gru | 6,441 | 75.00% | 0.7951 | 1.93 | 6,482 |
| 123 | cfc | 6,377 | 65.62% | 0.7326 | 2.56 | 3,554 |
| 123 | ltc | 4,585 | 69.79% | 0.7240 | 2.08 | 5,375 |
| 123 | gru | 6,441 | 59.38% | 0.7222 | 1.98 | 7,616 |

## 3. 跨 seed 汇总 (mean ± std)

| Backbone | params | Val acc | Val AUC | Train s | Inf samples/s |
|---|---:|---:|---:|---:|---:|
| **CfC** | 6,377 | 67.36 ± 4.04 % | 0.754 ± 0.034 | 2.32 ± 0.23 | 4,916 ± 1,212 |
| **LTC** | **4,585** | **70.83 ± 3.41 %** | 0.754 ± 0.036 | **1.90 ± 0.18** | 5,379 ± 36 |
| **GRU** | 6,441 | 66.32 ± 6.85 % | 0.754 ± 0.036 | 1.92 ± 0.07 | **6,731 ± 776** |

### 3.1 三轴 Pareto

| 维度 | 赢家 |
|---|---|
| **参数效率** | **LTC**(4,585 = CfC 的 71.9% / GRU 的 71.2%) |
| **精度** | **LTC**(acc 70.83% 比 CfC 高 3.5 pp、比 GRU 高 4.5 pp) |
| **AUC** | **并列**(三家都在 0.754) |
| **训练速度** | LTC > GRU > CfC(LTC 比 CfC 快 18%) |
| **推理吞吐** | **GRU**(6,731 samples/s,比 LTC 高 25%,比 CfC 高 37%) |
| **精度稳定性** | **LTC**(acc std 3.41% << GRU 6.85%) |
| **吞吐稳定性** | **LTC**(inf std 36 << GRU 776 / CfC 1,212) |

### 3.2 解读

1. **图结构信号是瓶颈,不是 recurrent 操作子**: 三个 backbone AUC 完全
   在 1 个 std 内 (0.754 ± 0.03)。`GraphSnapshotEncoder` 的邻居均值池化
   已经提取了大部分可用信息,后接什么 recurrent 不大影响。
2. **LTC 是综合 winner**: 在 4 个对比维度里赢 3 个(参数 / 精度 / 速度稳定性),
   只输 GRU 一个推理吞吐。这与项目 [[OPTIMIZATION_STRATEGIES]] 里说的
   "LTC 准/慢、CfC 快/准、GRU 快/糙" 在分子图任务上需要修正:
   **静态图(time=1)上 LTC 既准又快**。
3. **GRU 高方差是惯例**: acc std 6.85% 比 LTC 高 2× — 与 EMMA agent 在
   commits b653371 / f540ddf / 1bb78af 里反复发现的"GRU 高 seed 方差"
   完全一致。
4. **CfC 在此任务上反而中等**: 推理吞吐最低、acc 中等、训练秒最高。
   猜想: CfC 的闭式解每步多一次 `exp/log` 运算,
   在 seq_len=1 的退化情形下没有计算摊销空间。

### 3.3 对照 GCN-CfC (Linlab2026)

iter#5 调研结论 [[GCN-CfC_仓库结构化调研]] 提到对方仓库是
"PyTorch GCN → TensorFlow CfC 两阶段"。本轮证明:
**用本仓单 stack `GraphLNNPredictor` 跑同类任务,9 个 trial 全部 OK,
平均 < 2s 训完一个 backbone**。这是双框架方案完全做不到的工程便利。

## 4. PRD §8 进展更新

| # | 任务 | 状态 |
|---|---|---|
| 1 | Jetson CUDA wheel | ✅ iter#2 |
| 2 | LiquidTAD 复现 | A ✅ + B ✅ + C-lite ✅ |
| 3 | LFM2.5-1.2B INT4 | pending (等 RAM ≥ 2 GB 空) |
| 4 | EMMA 多模态 | pending (远程 EMMA agent 在做) |
| 5 | Comparative LNN vs LSTM v2 | pending |
| 6 | GCN-CfC smoke | **调研 ✅ + 落地 follow-up A ✅ (本轮 GraphLNN 分子 smoke)** |
| 7 | Pareto sweep PRD 集成 | ✅ iter#2 |
| 8 | Loop 去重 | pending |

## 5. 衍生工作

| 任务 | 推入 |
|---|---|
| 把 `experiment_graph_lnn_molecule.py` 加 `--frozen-encoder` 模式,模拟 GCN-CfC 的两阶段(GNN 冻结 + CfC 训练) | PRD §8 #6 followup B |
| 用真 Tox21 子集(预下载 CSV 解析)替换合成数据 | PRD §8 #6 followup A-真实 |
| 跨 hidden_size 跑参数 vs AUC Pareto(像 jetson_lnn_benchmark Pareto) | NEXT_STEPS |
| 把 LTC 参数效率写进 README 边缘部署章节 | docs |

## 6. 参考产物

- 源代码: `scripts/experiment_graph_lnn_molecule.py` (本轮新增, 230+ 行)
- 3 次 JSON+MD 产物: `analysis/molecular/2026-06-04_01*_tox21_styled_graph_lnn.{json,md}` ×3
- 上一轮: [[2026-06-04_loop_iteration5_gcn_cfc_repo_survey]]
- 关联报告: [[GCN-CfC_仓库结构化调研]]
- 实现源: `lnn/core/graph.py` (`GraphSnapshotEncoder` + `GraphLNNPredictor`)
- 路线图: [[LNN_训练方向_图时空与通信系统_可行报告]]
- PRD: [[PRD_LNN_Edge_Research]]
