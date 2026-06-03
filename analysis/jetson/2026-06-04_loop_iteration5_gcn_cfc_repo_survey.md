---
title: 2026-06-04 Loop iteration 5 — GCN-CfC 调研 + 新一天 daily baseline
date: 2026-06-04
tags: [LNN, loop, GCN, CfC, repo-watchlist, daily]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 5

> `/loop 1h` 第 5 次触发。新的一天,本轮的主要产出是 PRD §8 #6 GCN-CfC
> 仓库结构化调研报告(连接到我们已有的 `lnn.core.graph` 端到端实现),
> 以及今日 daily research / Jetson benchmark 基线。

## 1. 今日基线

- **Jetson benchmark (CPU, samples=192, seq=32, hidden=16, ep=3, batch=16)**:
  CfCStyle `MSE 0.3364` vs GRU `MSE 0.3518` (CfC 优 −4.4%)。
  说明: 该差异小于 iter#1 的 samples=256/epoch=3 配置(那次 CfC 优 21.2%);
  本轮配置稍小、本意是给"每日定时基线"留位置,不追求最大 contrast。
  详见 `analysis/jetson/2026-06-04_lnn_benchmark.{json,md}`。
- **Daily research 抓取**: arXiv API 此时段 429 限流(`papers: 0`),
  GitHub 51 仓库 + HF 21 模型 已抓到。`daily_lnn_research.py` 按设计
  保留昨天的 arXiv 缓存,digest 输出不为空。详见
  `docs/daily/2026-06-04_LNN_research_digest.md`。
- **CUDA 复检**: 仍 NvMap ENOMEM(系统 RAM 138 MB free,并行 agents 占用);
  CUDA init 成功,但 cudaMalloc 持续失败。维持 iter#2 结论:
  CUDA 路径修通,实际跑 GPU 需空载窗口。

## 2. PRD §8 #6 — GCN-CfC 仓库调研

新增 [[GCN-CfC_仓库结构化调研]](`docs/reports/`),按 paper-analyzer SOP 输出:

- **关键发现**: 仓库 CfC 部分是直接 fork 自 raminmh 官方 `tf_cfc.py`(430 行,90%+ 一致),
  **真正新意在 PyTorch GCN → 离线 .npy embedding → TensorFlow CfC 的两阶段管线**;
  这是一个反面教训 — 双框架混合不利于 Jetson 部署。
- **对照**: 本仓 `lnn.core.graph.GraphLNNPredictor` 早已实现"端到端 PyTorch + GNN+CfC 共享梯度"
  的更现代方案。
- **下游可入栈任务**(写进报告 §7):
  - A. 用本仓 `GraphLNNPredictor` 跑 MoleculeNet Tox21 smoke
  - B. 给 `experiment_graph_lnn.py` 加 `--frozen-gnn` 两阶段训练模式
  - C. 找到 GCN-CfC 配套论文后做单独 paper-analyzer 研读

不建议直接复现该仓库;借思路即可。

## 3. PRD §8 进展更新

| # | 任务 | 状态 |
|---|---|---|
| 1 | Jetson CUDA wheel | ✅ iter#2 |
| 2 | LiquidTAD 复现 | A ✅ + B ✅ + C-lite ✅(iter#3/#4) |
| 3 | LFM2.5-1.2B INT4 | pending (等 RAM ≥ 2GB 空) |
| 4 | EMMA 多模态 | pending(远程 EMMA agent 在做) |
| 5 | Comparative LNN vs LSTM v2 | pending |
| 6 | GCN-CfC smoke | **B 级调研报告 ✅(本轮);决定不复现** |
| 7 | Pareto sweep PRD 集成 | ✅ iter#2 |
| 8 | Loop 去重 | pending |

## 4. 衍生工作

| 任务 | 推入 |
|---|---|
| 把 GCN-CfC 调研结论加入 PRD §8 #6 描述 | 本轮已做 |
| 写 `scripts/experiment_graph_lnn_molecule.py` 在 MoleculeNet 上验证 `GraphLNNPredictor` | NEXT_STEPS |
| 给 PRD 加 §9 "已调研但不复现的仓库" 表 | NEXT_STEPS |
| 把 `analysis/molecular/` 目录预留出来 | 下一轮 commit 顺手做 |

## 5. 参考

- 调研报告: [[GCN-CfC_仓库结构化调研]]
- 远程对照: https://github.com/Linlab2026/GCN-CfC (MIT, 11 commits, 0 star)
- 本仓相关代码: `lnn/core/graph.py` (`GraphSnapshotEncoder` + `GraphLNNPredictor`)
- 本仓相关路线图: [[LNN_训练方向_图时空与通信系统_可行报告]]
- 累计 loop iteration 报告: [[2026-06-03_loop_validation_summary]] / [[2026-06-03_loop_iteration2_cuda_fix_pareto]] / [[2026-06-03_loop_iteration3_liquid_tad_stage_ab]] / [[2026-06-03_loop_iteration4_liquid_tad_3way_ablation]]
- PRD: [[PRD_LNN_Edge_Research]]
