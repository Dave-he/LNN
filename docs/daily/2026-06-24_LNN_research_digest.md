---
title: LNN 每日研究追踪 - 2026-06-24
date: 2026-06-24
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-06-24

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：3 篇新增（本地手工筛选）
- GitHub 候选仓库：0 个
- Hugging Face 候选模型：0 个
- 已下载 PDF：0 个

## 当日 cron 状态
- `scripts/cron_lnn_arxiv_daily.sh` 在 2026-06-23 22:33 UTC 触发了 arXiv fetch，
  但收到 `SSL: UNEXPECTED_EOF_WHILE_READING` 错误（出口网络抖动）。
  候选池被保留为空，避免覆盖上一轮成功数据；本轮手工补齐。

## 今日新增论文（手工补齐）

| arXiv ID    | 提交日期       | 标题（节选）                                                   | 关键词命中              | 与本仓关联                |
|-------------|----------------|----------------------------------------------------------------|-------------------------|---------------------------|
| 2606.19109 | 2026-06-17 | Locally Stable Neural ODEs with Characterized Region of Attraction | Lyapunov + Neural ODE   | **高** — 直接对应稳定性证书 |
| 2606.18315 | 2026-06-16 | Ghost Attractor Networks: Basin-Structured Dynamical Decoders | 多模态 + 吸引子          | 中 — 启发幽灵吸引子         |
| 2606.15469 | 2026-06-13 | Learning Context-Aware Neural ODE Dynamics for Adaptive Robotic Control | Neural ODE + 控制        | 中 — 机器人动态适配         |

### 2606.19109 — Locally Stable Neural ODEs with Characterized Region of Attraction
- **核心思想**：把神经 ODE 的动力学约束为联合学习的**最大化 Lyapunov 函数 V(h)** 的梯度场。
  - 形式：dh/dt = -∇V(h)
  - 稳定 ⟺ V(h) > 0 (h ≠ 0), V(0) = 0, dV/dt < 0
  - **吸引域 = V(h) ≤ 1 的 1-子水平集**，精确表征
- **关键定理**：在吸引域内指数稳定动力学可被该结构任意近似；模型吸引域可被 1-子水平集任意逼近真实吸引域
- **本仓适配**：CfC 的离散更新 `h_{t+1} = decay·g + (1-decay)·h_branch` 缺乏稳定性证书。
  本论文提供**离散 Lyapunov 条件**：V(h_{t+1}) - V(h_t) ≤ -α·V(h_t) 即可证明指数收敛
- **落地路径**：在 CfC 上加 (a) 联合学习的 V(h) = h^T P h，P 为半正定矩阵；(b) 收缩损失 `lyap_loss = relu(V(h_next) - (1-α)·V(h) + margin)`；(c) 正定约束 `pd_loss = relu(-λ_min(P) + margin)`
- **PR 候选**：`lnn/core/lyapunov_stable_cfc.py` + `tests/test_lyapunov_stable_cfc.py` + `scripts/bench_lyapunov_stable_cfc.py`
- **预期**：在 toy_sin/structured/random 三 dataset × 100 epoch × 3 seed × {baseline, +lyap α=0.05, +lyap+pd} = 27 cells，
  测任务 loss 是否仍 ±5%（不退化）+ λ_min(P) > 0 是否被满足 + V(h) 沿轨迹是否严格下降

### 2606.18315 — Ghost Attractor Networks
- **核心**：潜在空间由学习势函数 + drift 演化 → 多模态 basin-attractor 结构 → 模式切换通过 saddle-node 分岔（幽灵吸引子逃逸）
- **三要素**：多模态、单遍切换、常数内存
- **本仓适配启发**：可作为 FAME/MR-MoE 的潜在几何正则化（专家空间具备 basin 结构）
- **优先级**：低（实现复杂，且本仓已经做过 ORC/SNNL 等拓扑机制 round 100-101）

### 2606.15469 — Context-Aware Neural ODE for Adaptive Robotic Control
- **核心**：两阶段训练，先离线学基线 dynamics，再在线用 state-action history 微调 context embedding
- **本仓适配**：类似 QuITE（round 102）的两阶段思路；可在 NLTCell（Neuromodulated LTC）上加 context-embedding 微调入口
- **优先级**：低（机器人域，本仓主要测 1D/2D synthetic + PhysioNet gap 已由 round 102 填）

## 落地优先级
1. **2606.19109 Lyapunov-Stable CfC**（本轮首选）：新机制维度（稳定性证书），与现有 91-228 轮的 smoothness/diversity/rank/robustness 维度正交。
2. 2606.18315 Ghost Attractor：作为下一轮 backlog 候选。
3. 2606.15469 Context-Aware：作为更下游 backlog 候选。

## 建议动作
- 对 2606.19109 实现 Lyapunov-Stable CfC，期望：(i) 任务 loss ±5%；(ii) V(h) 沿训练轨迹严格下降；(iii) P 半正定
- 若 H1+H2 满足 → 进 round 240+ 纳入自主栈
- 若 H1 退化（任务 loss >+5%）→ 标注为 target-dependent 机制（类似 round 91-101 的诚实负）