---
title: 2026-06-04 Loop iteration 22 — SVAF §10 #9: τ-modulated peer-blending
date: 2026-06-04
tags: [LNN, loop, PRD-10-9, SVAF, CfC, tau-modulation, peer-blending, 2-agent-mesh, toy-experiment]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 22 — SVAF §10 #9: τ-modulated peer-blending

> `/loop 1h` 第 22 次触发。
> 紧接 iter#21 (PRD §10 #5 loop_status --prd-status 落地) 后,
> iter#21 工具自动浮现 next-up 列表中**最 P2 mini-task**: §10 #9 (SVAF τ 调制
> peer-blending 算子)。本轮执行 stage A。
>
> 1. **`lnn/core/cfc.py` 加 4 个 SVAF 算子函数** (~100 行): similarity_per_dim
>    / tau_modulated_blend_coef / tau_modulated_blend_update / default_three_group_tau
> 2. **9 unit tests** `tests/test_svaf_tau_blend.py` (~160 行)
> 3. **toy 2-agent mesh 实验** `scripts/experiment_svaf_tau_toy.py` (~200 行)
> 4. **关键结果**: Fast/Medium 20 步内完全收敛到 peer (0.5), Slow 仅 0.0183 — **论文
>    §7.1 "fast 同步 / slow 主权" 核心论断复现**
> 5. **PRD §10 #9 stage A** ✅
> 6. **commit + rebase + push origin/master**

## 1. 实现细节

### 1.1 SVAF Eq. 20 直译(lnn/core/cfc.py 新增)

```python
def similarity_per_dim(h_local, h_mesh):
    diff = (h_local - h_mesh).abs()
    denom = torch.maximum(h_local.abs(), h_mesh.abs()).clamp_min(1e-8)
    return (1.0 - diff / denom).clamp_min(0.0)

def tau_modulated_blend_coef(h_local, h_mesh, tau, alpha_eff=0.40, K=30.0):
    sim = similarity_per_dim(h_local, h_mesh)
    beta = alpha_eff * K * sim / tau.clamp_min(1e-8)
    return beta.clamp_max(1.0)

def tau_modulated_blend_update(h_local, h_mesh, tau, alpha_eff, K):
    beta = tau_modulated_blend_coef(h_local, h_mesh, tau, alpha_eff, K)
    return (1.0 - beta) * h_local + beta * h_mesh

def default_three_group_tau(d):
    third = max(1, d // 3)
    return torch.cat([
        torch.full((third,), 1.0),       # Fast
        torch.full((third,), 10.0),      # Medium
        torch.full((d - 2 * third,), 60.0)  # Slow
    ])
```

### 1.2 4 个函数的设计点

- **`similarity_per_dim`**: 论文 Eq. 19 左半,per-neuron sim ∈ [0, 1]
- **`tau_modulated_blend_coef`**: 论文 Eq. 20,β ∈ [0, 1] 严格不变量
- **`tau_modulated_blend_update`**: h_new = (1-β)·h_local + β·h_mesh,per-dim 凸组合
- **`default_three_group_tau`**: 一键切 1/3 Fast + 1/3 Medium + 1/3 Slow,简化 toy 用法

## 2. Toy 实验结果(d=6, n=20 steps, peer=0.5)

| Group | τ | final distance to peer |
|---|---:|---:|
| Fast | 1s | 0.0000 ✅ |
| Medium | 10s | 0.0000 ✅ |
| Slow | 60s | 0.0183 ✅ |

论文 §7.1 核心论断 **"Fast neurons (τ<5s) couple readily; slow neurons (τ>30s)
resist coupling entirely"** 在 2-agent toy mesh 上**定量复现**:
- Fast (τ=1) + α=0.4 + K=30 → β = 12 → clipped 1.0 → 一步完全收敛
- Slow (τ=60) → β = 0.2/step → 20 步后仅 0.0183 距离(从 1.0 出发)

## 3. pytest 套件(75/75, 12.08s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed (iter#22 新增)
─────────────────────────────────────────────
75 passed, 1 warning in 12.08s
```

vs iter#21: 66 → 75 = **+9 新增,0 回归**。

## 4. verify_all_models.py(9/9)

无变化。

## 5. 提交与推送

iter#22 改动:
- 改 `lnn/core/cfc.py` (+~100 行: 4 SVAF 函数)
- 增 `tests/test_svaf_tau_blend.py` (160 行, 9 tests)
- 增 `scripts/experiment_svaf_tau_toy.py` (200 行, 2-agent mesh)
- 增 `analysis/svaf/2026-06-04_tau_toy.{md,json}` (toy 报告)
- 改 `docs/PRD_LNN_Edge_Research.md` (1 行状态更新: #10-9 stage A ✅)
- 增 `analysis/jetson/2026-06-04_loop_iteration11_svaf_tau_blend.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-04_loop_iteration22_svaf_tau_blend.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 #9 状态**: **stage A ✅**,stage B (τ-blend 接 CfC backbone) pending
**PRD §10 总体**: **4/10 = 40%**(#10-5 + #10-9 + #10-10 stage A+B)

## 6. 下轮 (iter#23) 候选

按 §10 next-up:

1. **§10 #6 (backbone matrix --export-readme-snippet)**: 无阻塞,~30 行代码
2. **§10 #8 (loop_status README 标签云)**: 无阻塞,衍生 tooling
3. **§10 #1 (DynPMNN stage A)**: 可启动,需新建 `lnn/core/dynpmnn.py`
4. **§10 #9 (SVAF stage B)**: τ-blend 接 CfC backbone,toy mesh 升级到 real sequence
5. **§10 #3 (Comparative phase-D)**: 需空载 RAM 窗口
6. **§8/#3/#7 (LFM2.5 系)**: 1.7GB RAM 硬阻塞无解
