---
title: LNN 每日研究追踪 - 2026-06-25 (round 246)
date: 2026-06-25
tags: [LNN, daily, automation, arxiv, l-rfm, frozen-features]
---

# LNN 每日研究追踪 - 2026-06-25 (round 246, session #81, hourly loop #7)

> 自动生成：聚合 arXiv 2026-06 LNN / CfC / LTC 相关更新。

## 本轮新增论文

| arXiv ID    | 提交日期       | 标题                                                              | 关键词命中                    | 与本仓关联             |
|-------------|----------------|-------------------------------------------------------------------|-------------------------------|------------------------|
| 2606.15571 | 2026-06-22 | Liquid Random Feature Methods for Time-Dependent PDEs | **frozen trial functions + sampled relaxation scales** | **高** — frozen features 范式 |
| 2606.22801 | 2026-06-22 | Multi-τ Liquid-Mamba for All-in-one Image Restoration | adaptive τ + gated fusion | round 243 已响应 |
| 2606.19579 | 2026-06-23 | FlowFake | multi-timescale LTC | round 245 已响应 |
| 2606.20491 | 2026-06-23 | GazeLNN | LNN recurrent + RL | 应用 — 不进主线 |

## 选定论文 — 2606.15571 L-RFM (Liquid Random Feature Methods)

### 核心机制
- **Frozen trial functions** with embedded **sampled relaxation scales**
- 不学习 temporal params — 只学 linear readout
- 核心: temporal structure 来自 frozen basis, 不是 learned weights
- Density theorem: trial spaces dense in continuous space-time
- 在 stiff reaction-diffusion, nonlinear transport, dispersive PDE 上精确度提升

### 与本仓的关系
- **Round 76**: multi-τ (3 τ 静态, learned f/g/h)
- **Round 243**: input-conditioned τ (learned W_tau per branch)
- **Round 245**: 2-band hand-picked τ (0.1, 5.0) + learned α
- **Round 246（新）= FrozenSampledMultiTauCfCCell**:
  - K=4 branches, 每个 branch 有 **frozen 随机 τ** 从 log-uniform [τ_min=0.05, τ_max=20.0] 采样
  - **不学习 τ** — 完整冻结 (L-RFM 核心)
  - 只学 f, g, h, mix weights
  - 测试假设: **frozen random τ coverage 是否能替代 hand-picked τ**

### PR 候选
`lnn/core/frozen_sampled_multitau_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`

### 核心实现
```python
class FrozenSampledMultiTauCfCCell(nn.Module):
    def __init__(self, d_in, d_h, n_branches=4, tau_min=0.05, tau_max=20.0,
                 seed=42, learn_mix=True):
        # 一次性 log-uniform 采样 K 个 τ, 永久冻结
        rng = torch.Generator().manual_seed(seed)
        log_min, log_max = math.log(tau_min), math.log(tau_max)
        self.register_buffer("tau_frozen", 
            torch.exp(torch.rand(n_branches, generator=rng) * (log_max - log_min) + log_min))
        # K 个 CfC cell, 每个 time_scale 强制等于 frozen τ
        self.cells = nn.ModuleList([CfCCell(d_in, d_h) for _ in range(n_branches)])
        for cell, tau in zip(self.cells, self.tau_frozen):
            cell.time_scale.data.fill_(tau.item())
        # 学 α 混合
        if learn_mix:
            self.mix_param = nn.Parameter(torch.zeros(n_branches))
    
    def forward(self, x_t, h):
        # h: list of K hidden states
        outs = [cell(x_t, h_k) for cell, h_k in zip(self.cells, h)]
        if self.learn_mix:
            alpha = torch.softmax(self.mix_param, dim=0)
            return sum(a * o for a, o in zip(alpha, outs))
        return sum(outs) / len(outs)
```

### 预期
- 27 cells (3 datasets × 3 conditions {baseline, +round-245-hierarchical, +frozen-sampled} × 3 seeds, 100 epochs)
- H1 (task safe): frozen-sampled 不退化任务 loss
- H2 (mix entropy high): softmax 权重不是 one-hot — 多 τ 真的被用
- H3 (τ coverage diverse): 采样到的 τ 在 log 空间覆盖 ≥ 2 decades

### 与 round 245 的关系
- Round 245: hand-picked τ (0.1, 5.0), 2 branches, **strict win structured**
- Round 246: random sampled τ, 4 branches, tests L-RFM hypothesis
- 关键对照: **hand-pick vs random sampling** — L-RFM 主张 random sampling 已足够

## 落地优先级
1. **2606.15571 FrozenSampledMultiTauCfCCell**（本轮首选）— 测试 frozen features 范式

## 建议动作
- 实现 `FrozenSampledMultiTauCfCCell` with `n_branches=4, tau_min=0.05, tau_max=20.0, seed=42`
- 关键测试：τ 真的被冻结（requires_grad=False）；log 空间覆盖 ≥ 2 decades
- bench 27 cells vs baseline + round-245-hierarchical
- 若 H1+H2+H3 全过 → 进 round 247+ 纳入自主栈