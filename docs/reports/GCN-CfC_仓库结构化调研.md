---
title: GCN-CfC (Linlab2026) 仓库结构化调研
date: 2026-06-04
arxiv_id: (unspecified)
github: https://github.com/Linlab2026/GCN-CfC
tags: [LNN, CfC, GCN, molecular, repo-watchlist, paper-replication, drug-discovery]
parent: [[LNN_深度研读报告]]
---

# 仓库调研 — GCN-CfC: Graph continuous molecular screening for noncovalent inhibitor discovery

> 由本仓库 [[PRD_LNN_Edge_Research]] §8 任务 #6 排入,本次 loop 完成代码结构化调研。
> 评级 (本仓库视角): **B(可借鉴 GCN→CfC 解耦思路,不建议直接复现)**。

## 元数据

- **仓库**: [Linlab2026/GCN-CfC](https://github.com/Linlab2026/GCN-CfC)
- **作者**: Linlab2026
- **License**: MIT
- **活跃度**: 11 commits / 0 star / 0 fork / 0 issue (早期研究 artifact)
- **配套论文**: *Graph continuous molecular screening for noncovalent inhibitor discovery*(作者未给 arXiv 链接)

## 1. 解决的问题

化合物-蛋白互作 (CPI) 预测 + PAD4 (蛋白精氨酸脱亚胺酶 4) 非共价抑制剂虚拟筛选。
PAD4 是自身免疫疾病(RA、SLE)关键靶点,
非共价抑制剂相比共价抑制剂特异性更强、毒性更低。

## 2. 数据集

| 数据集 | 用途 | 公开度 |
|---|---|---|
| **BindingDB** | 化合物-蛋白互作记录 | 公开 |
| **PAD4 dataset**(自策) | 来自 PubChem + 高通量筛选记录 | 自策,论文不一定附配 |
| **MoleculeNet** (Tox21, ClinTox, ESOL, FreeSolv, Lipophilicity) | 通用 benchmark | 公开 |

## 3. 技术栈与核心架构

### 3.1 两框架混合(本仓最大痛点)

- **GCN 部分** (`classification/`): **PyTorch + torch_geometric** —
  典型 DenseGCN 风格,`Conv1dReLU + DenseBlock + GraphConvBn`,
  以分子图节点特征为输入,输出图级 embedding。
  `classification/model.py` 251 行,`train.py` 156 行。
- **CfC 部分** (`cfc_part/`): **TensorFlow / Keras** —
  直接 fork 自 raminmh 官方 [tf_cfc](https://github.com/raminmh/CfC)
  (含 `CfcCell`、`MixedCfcCell`、`LTCCell` 三类 Keras layer,
  本仓 `cfc_part/tf_cfc.py` 430 行 与 reference 90%+ 一致)。
- **耦合方式**: GCN 训完后用 `export_plain_gcn_embeddings.py`
  把 embeddings dump 成 .npy,再喂给 TF CfC 训练
  (`train_cfc_on_embeddings.py`)。

这是**离线 pipeline 而不是端到端可微管线**:
GCN 与 CfC 不共享梯度,CfC 只对 frozen embedding 做后处理。

### 3.2 与本仓 `lnn/core/graph.py::GraphLNNPredictor` 的对比

本仓 (`lnn/`) 的 `GraphSnapshotEncoder + GraphLNNPredictor`:
**端到端 PyTorch + 共享梯度** 的 GNN+CfC,
明显比 Linlab2026 的两阶段方案更适合现代部署(TensorRT/ONNX/Jetson)。

| 维度 | GCN-CfC (Linlab2026) | `lnn.core.graph` (本仓) |
|---|---|---|
| 框架 | PyTorch + TensorFlow | 纯 PyTorch |
| GNN 到 LNN 通路 | 离线 .npy 桥接 | 端到端可微 |
| Jetson 部署 | 难(双框架协同) | 易(单 stack) |
| 模型规模 | 中等(DenseGCN depth=3) | 可调 |

## 4. 复现成本(Jetson Orin Nano Super 视角)

| 阶段 | 估时 | 阻塞点 |
|---|---|---|
| 数据准备(PAD4 SMILES 整理) | 1–2 day | 数据策展工作量,作者未附 CSV |
| Python 环境(PyTorch + TF 双栈) | **2–4 hour** | TF on Jetson 较少 wheel,可能要 docker |
| GCN 训练(depth=3, lr=5e-4, batch=512, ep=50) | < 30 min on CUDA | RAM ≥ 4 GB 才稳 |
| GCN→embedding 导出 | < 5 min | — |
| CfC 训练(ep=100, patience=20) | < 15 min | TF GPU memory 与 PyTorch 冲突 |
| MoleculeNet baseline | < 1 hour | — |
| 总计 | ~ 1 work day | 主要在双框架装机 |

复现可行性: 中等。**最大阻塞**是 Jetson 上同时装 PyTorch (cu126) 与 TensorFlow (cu126) 是个坑;
NVIDIA 官方有 `nvcr.io/nvidia/l4t-tensorflow` 镜像但与 `l4t-pytorch` 镜像不能共用。

## 5. 可重复性

- 硬编码 seeds: **7, 17, 37**(只三个!)
- 关键超参文档化: `lr 5e-4, batch_size 512, epochs 50 (CfC), epochs 100 (MoleculeNet), patience 20, GCN depth 3`
- 无 pretrained checkpoint 公开,只有路径模板 `analysis/oversmoothing/PAD4_007_08/checkpoints/plain_gcn_depth3_seed7.pt`
- 无 CI / 无 release / 无 issue,**所有错误你都得自己 debug**。

## 6. 对本仓库的价值

### 6.1 可借鉴的想法

1. **Oversmoothing 分析方向**: `analysis/oversmoothing/` 目录暗示作者做了 GCN
   深层 oversmoothing 与 CfC 校正的对照 — 这是 GCN+CfC 组合特别值得记录的角度。
   可在 [[NEXT_STEPS]] 加 "用 `lnn.core.graph` 复测 oversmoothing in vs out CfC" 实验。
2. **PAD4 真实任务**: 给 `lnn.core.graph.GraphLNNPredictor` 一个真实生物任务用例,
   提升 README 说服力。
3. **`train_cfc_on_embeddings.py` 的两阶段策略**: 当 GNN 已有现成 pretrained
   时,frozen embedding + CfC head 在 Jetson 上**显存预算友好**(GCN forward
   只需 KV cache,梯度只走 CfC),值得在 [[LNN_训练方向_图时空与通信系统_可行报告]] 加一节。

### 6.2 **不建议直接复现**

理由:
- 两框架管线 + 无 pretrained ckpt + 0 star 维护风险 ≈ 复现成本/收益比差;
- 本仓 `lnn.core.graph` 已实现"端到端 GNN+CfC"更现代方案,直接做下游验证更划算;
- 真要测 PAD4 / MoleculeNet,可直接用 PyG 内置 + 本仓 `GraphLNNPredictor` 写 50 行实验脚本。

## 7. 下一步可执行任务(候选)

| 任务 | 出口物 | 估时 |
|---|---|---|
| **A.** 用本仓 `GraphLNNPredictor` 跑 MoleculeNet Tox21 smoke,与 plain GCN 对比 | `analysis/molecular/2026-06-04_lnn_graph_tox21_smoke.md` | 1 loop |
| **B.** 加 `--frozen-gnn` 两阶段训练模式到 `experiment_graph_lnn.py` | code + smoke | 1–2 loop |
| **C.** GCN-CfC 论文找到后做单独 paper-analyzer 研读 | `docs/reports/GCN-CfC_研读报告.md` | 1 loop(待论文链接) |

## 8. 评级

- **学术价值**: B−(GCN+CfC 思路有意义,但实现工程化弱)
- **代码质量**: C+(两框架混合、无 CI、无 release)
- **对本仓优先级**: **B**(借思路、不复现;PAD4 任务可入栈)
- **2026-06-04 watchlist 状态**: 已记录,不投入复现 budget

## 9. 参考

- 仓库: https://github.com/Linlab2026/GCN-CfC
- Daily watchlist (2026-06-03): [[analysis/repo_watchlist/2026-06-03_lnn_open_source_watchlist]] (该仓库首次被追踪)
- 本仓相关代码: `lnn/core/graph.py` (`GraphSnapshotEncoder`, `GraphLNNPredictor`)
- 本仓相关报告: [[LNN_训练方向_图时空与通信系统_可行报告]]
- 上游 CfC 实现: https://github.com/raminmh/CfC (Linlab2026 直接 fork 了 tf_cfc.py)
- PRD: [[PRD_LNN_Edge_Research]] §8 task #6
