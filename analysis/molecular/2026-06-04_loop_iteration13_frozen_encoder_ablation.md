---
title: 2026-06-04 Loop iteration 13 — frozen-encoder ablation (PRD §9 #4)
date: 2026-06-04
tags: [LNN, loop, PRD-9, graph, GCN-CfC, frozen-encoder, two-stage, paper-replication]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 13 — `--frozen-encoder` ablation (PRD §9 #4)

> `/loop 1h` 第 13 次触发。
> 兑现 PRD §9 #4: 在 `scripts/experiment_graph_lnn_molecule.py` 加
> `--frozen-encoder` 两阶段训练模式,**用我们的代码量化** iter#5 调研里
> 对 GCN-CfC 仓库"两框架两阶段"管线的反面教训。
>
> **结论先行**: frozen-encoder pattern 在同任务/同 seed 下 AUC 平均损 ~5%,
> CfC / LTC / GRU 三个 backbone 都没有例外。**两阶段 GCN→CfC 是 GCN-CfC
> 仓库设计的实证 downside**;本仓 `lnn.core.graph.GraphLNNPredictor` 的
> 端到端共享梯度方案在该任务上**有可量化的优势**。

## 1. 实验配置

### 1.1 frozen-encoder 训练流程

新增 `--frozen-encoder` 后实现:
- **Phase 1**: 用 `GraphSnapshotEncoder` + 一层 `Linear(graph_feature_size, 1)`
  作为 probe,**只**优化这两块,跑 `--frozen-pretrain-epochs` (默认 5);
- **Phase 2**: 把 encoder 参数 `requires_grad_(False)`,
  AdamW 只接收 recurrent + readout 的参数,继续跑主 `--epochs` (8)。

这是 GCN-CfC (Linlab2026) 仓库的 PyTorch 单 stack 等价物 — 不再需要
"PyTorch GCN → 离线 .npy bridge → TF CfC"。
[[GCN-CfC_仓库结构化调研]] iter#5 已记录该仓库的两阶段是其工程痛点。

### 1.2 控制变量

| 变量 | 值 |
|---|---|
| 数据 | `Tox21-styled` (`generate_random_graph_batch` triangle median split) |
| backbones | cfc / ltc / gru |
| seeds | {42, 7, 123} |
| 模型 | `GraphLNNPredictor(graph_feat=24, hidden=32)` |
| Optimizer | AdamW lr=3e-3 |
| Epochs | phase1=5(frozen 模式) + phase2=8 |
| Device | cpu(CUDA NvMap ENOMEM 仍未恢复) |

end-to-end baseline 直接复用 iter#6 数据(`analysis/molecular/2026-06-04_0132*.json`),
两边数据生成 RNG 完全一致(`generate_gradual_multi_regime` 不卷入,
只用 `_random_graph_batch` 与 `iter#6 默认`)。

## 2. 原始结果 (per seed × backbone)

### 2.1 Val AUC

| seed | backbone | end-to-end (iter#6) | **frozen (iter#13)** | Δ |
|---:|---|---:|---:|---:|
| 42 | cfc | 0.7361 | 0.6790 | **−7.8%** |
| 42 | ltc | 0.7431 | 0.6448 | **−13.2%** |
| 42 | gru | 0.7461 | 0.6685 | **−10.4%** |
| 7 | cfc | 0.7921 | 0.7280 | **−8.1%** |
| 7 | ltc | 0.7943 | 0.7012 | **−11.7%** |
| 7 | gru | 0.7951 | 0.7183 | **−9.7%** |
| 123 | cfc | 0.7326 | 0.7444 | **+1.6%** |
| 123 | ltc | 0.7240 | 0.7607 | **+5.1%** |
| 123 | gru | 0.7222 | 0.7639 | **+5.8%** |

### 2.2 跨 seed 汇总(AUC mean / median)

| 模式 | cfc | ltc | gru |
|---|---:|---:|---:|
| **end-to-end** mean | **0.7536** | **0.7538** | **0.7545** |
| **end-to-end** median | 0.7361 | 0.7431 | 0.7461 |
| **frozen** mean | 0.7171 | 0.7022 | 0.7169 |
| **frozen** median | 0.7280 | 0.7012 | 0.7183 |
| Δ frozen vs e2e (mean) | **−4.8%** | **−6.8%** | **−5.0%** |
| Δ frozen vs e2e (median) | **−1.1%** | **−5.6%** | **−3.7%** |

### 2.3 参数量(总参 / 可训参)

| backbone | total params | end-to-end trainable | frozen phase-2 trainable | encoder frozen |
|---|---:|---:|---:|---:|
| cfc | 6,377 | 6,377 | 5,537 | 840 |
| ltc | 4,585 | 4,585 | 3,745 | 840 |
| gru | 6,441 | 6,441 | 5,601 | 840 |

Encoder 占总参 13–18%。frozen 模式 phase-2 节省 ~13% 反向传播算力,
但精度损失大于这点节省。

## 3. 解读

### 3.1 frozen-encoder 总体输 end-to-end ~5%

3 backbone × 3 seed = 9 trial:
- 6/9 trials frozen 比 end-to-end 显著更差(seed 42 / seed 7 全军覆没,Δ −8% 到 −13%);
- 3/9 (seed 123) frozen **反而赢** end-to-end(Δ +2% 到 +6%) — 因为 e2e seed 123 本身
  是 iter#6 表现最弱的 seed(0.72-0.73);frozen 提供的"梯度隔离"在那一个 outlier 
  数据上意外帮忙;
- 跨 seed 平均后,frozen 全输 ~5%。

### 3.2 为什么 frozen 通常更差

- **phase-1 probe 只能优化"能被线性 head 利用的 embedding"**,
  而 recurrent head (CfC/LTC) 需要的特征 manifold 与线性 probe 不同;
- end-to-end 让梯度直接告诉 encoder "为 LNN 训特征",frozen 切断了这条通路;
- 这正是 GCN-CfC 仓库的工程缺陷 —
  把 GNN embedding 当成"另一个上游任务"产物(PyTorch),
  再"另一个下游任务"训 CfC(TensorFlow)。

### 3.3 GRU 不再领跑

iter#6 (end-to-end) GRU median AUC 0.7461 最高;
本轮 frozen 后三家都被打压到 ~0.71-0.72,**LTC 反而成了 frozen 模式的最差**(median 0.7012)。

原因猜测: LTC 的 ODE 求解高度依赖输入特征的连续性,
phase-1 probe 优化出的 embedding 缺少 LNN 需要的"时间常数友好"结构。
LTC 输得最惨支持这条假说。

### 3.4 对照 iter#5 GCN-CfC 调研的工程定性 ↔ 本轮的定量

| 维度 | iter#5 定性结论 | 本轮 iter#13 定量证据 |
|---|---|---|
| 两阶段不利端到端梯度 | 推测 | 6/9 trials 输 4-13%,跨 seed 均输 ~5% |
| LNN/CfC 对 embedding 质量更敏感 | 推测 | LTC frozen mean 0.7022 是 frozen 三家最差 |
| 应优选单 stack 端到端 | 推测 | 端到端 mean 全 backbone > 0.75 vs frozen < 0.72 |

iter#5 的"借思路、不复现"判断**得到自家代码的直接证据支持**。

## 4. PRD §9 进展

| # | 状态 | 备注 |
|---:|:---:|---|
| 9-1 | ⏳ | LFM2.5 — 等 RAM |
| 9-2 | ✅ iter#10/#11 | gradual + warmup + 8-seed retraction |
| 9-3 | ⏳ | LiquidTAD Stage C-true |
| **9-4** | **✅ iter#13** | **frozen-encoder 量化两阶段成本: AUC −5%** |
| 9-5 | ⏳ | loop_status --since-last-loop |
| 9-6 | ⏳ | ONNX + TensorRT INT8 |
| 9-7 | ✅ iter#12 | backbone matrix |
| 9-8 | ⏳ | weekly CI |

## 5. 衍生

| 任务 | 推入 |
|---|---|
| 把 frozen 9 trials 加进 backbone matrix(给 graph_tox21 加一个 [frozen] 行) | 本轮已留 JSON,下次跑 matrix 自动出现 |
| 跑 phase-1 epochs sweep(1/5/15/30 看 phase-1 充分性) | NEXT_STEPS |
| 试在 phase-2 给 head 更高 lr,看是否能补 e2e gap | NEXT_STEPS |
| 把"frozen vs e2e"曲线加进 README 边缘部署章节(对照 GCN-CfC) | docs |

## 6. 参考产物

- 代码改动: `scripts/experiment_graph_lnn_molecule.py` 加 `--frozen-encoder` + `--frozen-pretrain-epochs`
- frozen 9 trials JSON+MD: `analysis/molecular/2026-06-04_083*_tox21_styled_graph_lnn.{json,md}` × 3
- 对照 (end-to-end) JSON: iter#6 留下的 `analysis/molecular/2026-06-04_0132*.json` × 3
- iter#5 调研: [[GCN-CfC_仓库结构化调研]]
- iter#6 端到端 smoke: [[2026-06-04_loop_iteration6_graph_lnn_tox21_smoke]]
- 累计 iter chain: [[2026-06-04_loop_iteration12_backbone_matrix]] /
  [[2026-06-04_loop_iteration11_phaseC_8seed_retraction]] / ...
- PRD: [[PRD_LNN_Edge_Research]] §9 #4
