---
title: LNN 每日研究追踪 - 2026-06-25 (round 245)
date: 2026-06-25
tags: [LNN, daily, automation, arxiv, flowfake, l-rfm]
---

# LNN 每日研究追踪 - 2026-06-25 (round 245, session #80, hourly loop #6)

> 自动生成：聚合 arXiv 2026-06 LNN / CfC / LTC 相关更新。

## 本轮新增论文

| arXiv ID    | 提交日期       | 标题                                                              | 关键词命中                    | 与本仓关联             |
|-------------|----------------|-------------------------------------------------------------------|-------------------------------|------------------------|
| 2606.19579 | 2026-06-23 | FlowFake: Liquid Networks for Audio Deepfake Detection | 多 timescale (10ms spectral + 2s prosodic) | **高** — 重访 round 76 multi-τ, 非几何 τ |
| 2606.15571 | 2026-06-22 | Liquid Random Feature Methods for Time-Dependent PDEs | frozen trial functions + sampled relaxation scales | **中** — frozen features 范式 |
| 2606.20491 | 2026-06-23 | GazeLNN | LNN recurrent engine + RL | 应用 — 不进主线 |
| 2606.15807 | 2026-06-22 | MA-GLTC | Graph LTC | 应用 — 不进主线 |

## 选定论文 — 2606.19579 FlowFake

### 核心机制
- **Multi-timescale LTC** with task-driven timescale selection:
  - Fast band: ~10ms for spectral features
  - Slow band: ~2s for prosodic features
- **34K parameters, BIBO stable, O(dt⁴) integration error**
- **关键 insight**: 不是几何级数（round 76: 0.1/1.0/10.0），而是 **task-aware 非几何 timescale band**

### 与本仓的关系
- **Round 76**: 多τ CfC (n_tau=3, tau_scales=(0.1, 1.0, 10.0), 几何级数)
- **Round 243**: Adaptive-gated multi-τ (输入条件化 τ, **task regression -96/-96/-178%**)
- **Round 245（新）= Hierarchical Multi-τ CfC**：
  - 两个 timescale **band**: τ_fast (小) + τ_slow (大)
  - **非几何** τ (e.g., 0.1 vs 5.0 vs FlowFake 10ms/2s ratio)
  - Learned mixture weight α ∈ [0, 1] 混合 fast/slow 输出
  - **静态** τ (区别于 round 243 输入条件化)
  - 测试假设: **几何 τ 不是 multi-τ 优势的本质, 关键是 timescale 分层 + 显式 fusion**

### PR 候选
`lnn/core/hierarchical_multitau_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`

### 核心实现
```python
class HierarchicalMultiTauCfCCell(nn.Module):
    def __init__(self, d_in, d_h, tau_fast=0.1, tau_slow=5.0):
        self.fast_cell = CfCCell(d_in, d_h, tau=tau_fast)  # 局部细节
        self.slow_cell = CfCCell(d_in, d_h, tau=tau_slow)  # 全局结构
        # 学习 α 混合 vs 静态 α=0.5 (round 76 等权)
        self.mix = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x_t, h_fast, h_slow):
        h_fast_next = self.fast_cell(x_t, h_fast)
        h_slow_next = self.slow_cell(x_t, h_slow)
        alpha = torch.sigmoid(self.mix)
        h_next = alpha * h_fast_next + (1 - alpha) * h_slow_next
        return h_next, h_fast_next, h_slow_next
```

### 预期
- 27 cells (3 datasets × 3 conditions {baseline, +round-76-multi-τ, +hierarchical} × 3 seeds, 100 epochs)
- H1 (task safe): hierarchical 不退化任务 loss
- H2 (mix variability): mix 系数在不同 dataset 上学到不同值 (toy_sin 学 ~0.3 fast-leaning, random 学 ~0.5)
- H3 (multi-τ effect): hierarchical vs baseline Δ% ≤ round-76 的 2× (不会更差)

### 与 round 76 / 243 的关系
- Round 76: 几何多τ (0.1/1.0/10.0) + 等权 concat
- Round 243: 输入条件化 + softmax gate (**task regression**)
- Round 245: 非几何双 band + 学习 α 混合 (中间地带)

## 落地优先级
1. **2606.19579 Hierarchical Multi-τ CfC**（本轮首选）— FlowFake 启示: non-geometric τ + learned α
2. backlog: 2606.15571 L-RFM (frozen trial features 范式)

## 建议动作
- 实现 `HierarchicalMultiTauCfCCell` with `tau_fast=0.1, tau_slow=5.0, learn_mix=True`
- 关键测试：固定输入时 α 稳定；不同 dataset 上 α 收敛到不同值
- bench 27 cells vs baseline + round-76-multi-τ
- 若 H1+H3 全过 → 进 round 246+ 纳入自主栈