# Tox21-styled GraphLNNPredictor Smoke — 2026-06-05_011136

## 任务
- 模型: `lnn.core.graph.GraphLNNPredictor` (本仓端到端 GNN+LNN, PyTorch only)
- 数据: 合成 Tox21-风格 — Erdős–Rényi 分子图 + (mean_degree>3 ⊻ triangle_ratio>0.05) 二分类
- 训练 / 验证: 512 / 128 | max_nodes=14 | edge_prob=0.35
- 模型: graph_feat=24 hidden=32 | epoch=5 batch=32 lr=0.003 seed=42
- 类平衡: train +50.0% / val +50.0%

## 结果
| Backbone | 参数量 | Val acc | Val AUC | 训练秒 | 推理样本/秒 |
|---|---:|---:|---:|---:|---:|
| `cfc` | 6,377 | 57.03% | 0.6631 | 2.03 | 2923 |
| `ltc` | 4,585 | 60.94% | 0.6570 | 2.03 | 2515 |
| `gru` | 6,441 | 53.12% | 0.6570 | 2.27 | 2533 |
| `liquid_tad` | 16,105 | 50.00% | 0.6670 | 7.57 | 1690 |

## 解读
- AUC ≥ 0.70 → 模型抓到了 degree/triangle 的结构信号;< 0.55 接近随机;
- CfC / LTC vs GRU: 比 acc/AUC + 训练秒 + 推理吞吐三轴;
- 比 `GCN-CfC` (Linlab2026) 的两阶段管线: 端到端单 stack,Jetson 部署友好。

JSON: `analysis/molecular/2026-06-05_011136_tox21_styled_graph_lnn.json`
