---
title: LNN 每日研究追踪 - 2026-06-25 (round 244)
date: 2026-06-25
tags: [LNN, daily, automation, arxiv, ghost-attractor]
---

# LNN 每日研究追踪 - 2026-06-25 (round 244, session #79, hourly loop #5)

> 自动生成：聚合 arXiv 与本仓 backlog 的 LNN / CfC / LTC 相关更新。

## 本轮新增论文（手工补齐 + backlog）

| arXiv ID    | 提交日期       | 标题                                                              | 关键词命中                    | 与本仓关联             |
|-------------|----------------|-------------------------------------------------------------------|-------------------------------|------------------------|
| 2606.18315 | 2026-06-22 | Ghost Attractor Networks: Basin-Structured Dynamical Decoders | multi-basin + saddle-node    | **高** — 扩展 round 240 Lyapunov |
| 2606.20491 | 2026-06-23 | Fast Human Attention Prediction (GazeLNN) | LNN recurrent engine + RL   | 应用 — 不进入主线     |
| 2606.15807 | 2026-06-22 | MA-GLTC: Cross-Domain Traffic Prediction | Graph + LTC                  | 应用 — 不进入主线     |
| 2606.13571 | 2026-06-21 | Timeflies: Joint Existence + Value Forecasting | observation existence         | 中 — 与 round 102 QuITE 互补 |

## 选定论文 — 2606.18315 Ghost Attractor Networks

### 核心机制
1. **多 basin 势能面** — 隐空间 V(h) 拥有 K 个学习的吸引子盆地（不只是原点）
2. **Saddle-node bifurcation** — 在 basin 之间通过鞍结分岔切换
3. **Ghost attractor escape** — 在 basin 之间"幽灵"转移
4. **Hierarchical phase-space decomposition** — 一阶 basin 收敛 + 二阶微调

### 与本仓的关系
- **Round 240 Lyapunov**：V(h) = h^T P h (单 basin: 原点)
- **Round 241 Controllability**：c_t = ||h_with - h_without|| (无 basin 结构)
- **Round 242 ISS**：V_next ≤ (1-α)V + β·||x||² (单 basin: 原点)
- **Round 244（新）= Multi-Basin Lyapunov**：
  - K 个学习的 basin 中心 {c_k} (K learnable)
  - V(h) = min_k ||h - c_k||² 或更软的 V(h) = -α · log Σ exp(-β · ||h - c_k||²)
  - 收缩损失: V(h_next) ≤ (1-α) · V(h) + margin
  - basin 中心 c_k 可学习的，初始化为 unit-sphere K 个均匀方向

### PR 候选
`lnn/core/multi_basin_lyapunov_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`

### 核心实现
```python
# K learned basin centers in H-space
self.basin_centers = nn.Parameter(torch.randn(K, H) * 0.1)
# Soft min-distance Lyapunov
def multi_basin_V(h, c, beta=2.0):
    # h: (B, H), c: (K, H)
    d = (h.unsqueeze(1) - c.unsqueeze(0)).pow(2).sum(-1)  # (B, K)
    return - (1.0/beta) * torch.logsumexp(-beta * d, dim=-1)  # (B,)
# Multi-basin contraction loss (ISS extension)
def multi_basin_lyap_loss(h, h_next, c, alpha=0.05, beta_v=2.0, margin=0.0):
    V_t = multi_basin_V(h, c, beta_v)
    V_next = multi_basin_V(h_next, c, beta_v)
    return relu(V_next - (1.0 - alpha) * V_t + margin).mean()
```

### 预期
- 18 cells (3 datasets × 3 conditions {baseline, +round-240 lyap, +multi-basin K=3} × 3 seeds, 100 epochs)
- H1 (task safe): multi-basin 不退化任务 loss
- H2 (V_actual < V_max): 实际 V 比最大 basin V 小 — 真实被 basin 吸引
- H3 (basin usage): 不同 sample 落到不同 basin — entropy > log K · 0.5

### 与 ISS (round 242) 的关系
- ISS = 单 basin + β·||x||² input drive
- Round 244 = 多 basin + 可选 β·||x||²
- 多 basin 是对单 basin 的**结构化扩展**：与 ISS 是**正交维度**

## 落地优先级
1. **2606.18315 Multi-Basin Lyapunov CfC**（本轮首选）— 扩展 round 240 单 basin → 多 basin
2. backlog: 2606.13571 Timeflies 存在性建模（与 round 102 QuITE 互补）

## 建议动作
- 实现 `MultiBasinLyapunovStableCfCCell` with `n_basin=3, alpha=0.05, beta_v=2.0`
- 关键测试：固定 h 时 V 与各 basin 距离一致；h 远离所有 basin 时 V 大
- bench 27 cells vs baseline + round-240-Lyap
- 若 H1+H2+H3 全过 → 进 round 245+ 纳入自主栈