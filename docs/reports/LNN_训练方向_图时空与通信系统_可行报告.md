---
title: LNN 训练方向：图时空与通信系统可行报告
date: 2026-05-28
tags: [LNN, graph, telecom, spatio-temporal, beamforming]
---

# LNN 训练方向：图时空与通信系统可行报告

## 1. 方向定位

图时空与通信系统关注节点关系随时间变化的动态系统，如交通网络、传感器网络、多智能体系统、6G 信道、beamforming 和边缘网络调度。LNN 在这里主要承担“连续时间动态建模器”的角色，通常需要和 GNN、几何表示或优化算法结合。

检索证据：本方向纳入/暂缓记录见 [[docs/LNN_训练论文检索矩阵_2026-05-28]]。

## 2. 代表论文

| 论文 | 任务 | 关键启发 |
|---|---|---|
| *Riemannian Liquid Spatio-Temporal Graph Network* | 动态时空图 | 在黎曼流形上建模 liquid ODE，处理非欧几里得图结构 |
| *Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks* | sub-THz MU-MIMO beamforming | 用 LNN 学习时间信道动态，再结合流形优化压缩搜索空间 |
| *Liquid-Graph Time-Constant Network for Multi-Agent Systems Control* | 多智能体控制 | 把 LTC 与图结构结合，处理多智能体动态 |
| *FUSION* 等边缘网络调度工作 | 需求预测与调度 | 用 LNN 预测时空负载，再接优化或博弈算法 |

## 3. 数据集构建方案

### 3.1 动态图数据

```text
for each time t:
  node_features[t]: [N, F_n]
  edge_index[t]: [2, E]
  edge_features[t]: [E, F_e]
  graph_label[t] or node_label[t]
  dt[t]
metadata:
  city, topology, sensor_type, weather, event
```

切分：

- 时间外推：训练早期时间段，测试未来。
- 空间外推：训练部分区域，测试未见区域。
- topology OOD：训练一种图结构，测试另一种拓扑。

### 3.2 通信信道数据

```text
H[t]: channel matrix
beam_target[t]: analog/digital beamformer
user_state[t]: location, velocity, blockage
snr[t], carrier_freq, antenna_config
scenario_id
```

切分：

- 按城市街区或 ray-tracing 场景切分。
- 按速度、SNR、遮挡强度做 OOD。
- 按天线配置或用户数量做泛化测试。

## 4. 架构搭建方案

### 4.1 GNN + LNN

```text
node/edge features at each t
-> GNN encoder per graph snapshot
-> sequence of graph embeddings
-> CfC/LTC temporal model
-> prediction head
```

适合：

- 图级预测，如拥堵、负载、全局状态。
- 节点级预测时，可对每个节点共享 LNN 或用图卷积后再 LNN。

### 4.2 LNN + manifold optimization

适合 beamforming：

```text
channel history -> LNN -> compressed digital BF initialization
-> manifold optimization refinement
-> spectral efficiency objective
```

训练目标：

- supervised：拟合最优 beamformer。
- self-supervised：最大化 spectral efficiency 或最小化通信损失。
- hybrid：LNN 产生 warm start，优化器做最后约束满足。

### 4.3 Riemannian liquid graph

适合层级、环状或曲率明显的图结构：

```text
Euclidean node features
-> manifold embedding
-> ODE on manifold
-> tangent projection / readout
```

该路线理论价值高，但工程复杂度也高。除非任务明确需要非欧几里得几何，否则先从 GNN + CfC 做 baseline。

## 5. 训练方法

常用损失：

| 任务 | 损失 | 指标 |
|---|---|---|
| 图时序预测 | MSE/MAE | RMSE、MAE、MAPE |
| 节点分类 | CrossEntropy | Accuracy、Macro-F1 |
| beamforming | negative spectral efficiency、MSE to oracle | SE、BER、robustness under imperfect CSI |
| 调度/路由 | supervised loss + objective penalty | social welfare、latency、energy |

推荐配置：

```text
graph_encoder_dim: 32/64/128
lnn_hidden: 32/64
seq_len: 8/16/32
lr: 3e-4
gradient_clip: 1.0
```

## 6. 优化与调参

重点：

- 图拓扑是否随时间变化：静态图可缓存 message passing，动态图需动态 batch。
- `dt` 与采样频率：通信信道 coherence time 很关键。
- 物理约束：beamforming 输出要满足功率、相位、硬件约束。
- OOD 场景：不同 SNR、速度、遮挡和用户数必须单独报告。
- 端到端优化风险：物理目标可能不可微或数值不稳定，可先用 supervised warm start。

## 7. 本项目落地建议

短期：

- 已新增 `lnn/data/graph_timeseries.py` 的轻量合成动态图数据结构。
- 已新增 `lnn/core/graph.py` 与 `scripts/experiment_graph_lnn.py`，可运行 `GNN encoder + CfC/LTC/GRU` 图级预测 smoke 实验。
- 通信方向先用模拟 channel matrix，不直接追求完整 6G pipeline。

中期：

- 引入 PyTorch Geometric 或 DGL。
- 加入 beamforming 约束 head。
- 对比 GRU-GNN、Temporal GCN、Transformer 类模型。

## 8. 可行结论

该方向有研究价值，但不适合作为第一阶段。建议在时间序列和机器人方向稳定后推进。最务实的切入点是 `GNN encoder + CfC`，而不是直接复现完整 RLSTG 或 6G beamforming 系统。

## 9. 参考资料

- *Riemannian Liquid Spatio-Temporal Graph Network*：https://arxiv.org/abs/2601.14115
- *Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks*：https://arxiv.org/abs/2604.07219
- *Liquid-Graph Time-Constant Network for Multi-Agent Systems Control*：https://arxiv.org/abs/2404.13982
- *FUSION: Forecast-Embedded Agent Scheduling...*：https://arxiv.org/abs/2512.14323
