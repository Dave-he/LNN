---
title: 2026-06-05 Loop iteration 36 — RLSTG stage A design doc
date: 2026-06-05
tags: [LNN, loop, rlstg-stage-a, design-doc, planning, tangent-space, hyperbolic, geoopt, prd-10-pending, iter36]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-05 Loop iteration 36 — RLSTG stage A design doc

> `/loop 1h` 第 36 次触发。
> 紧接 iter#35 (Retinal LNN paper deep read) 后,本轮执行 **RLSTG §10 复现路线 stage A**:
> 设计文档,决定复现深度 + 列出 dependencies + 风险评估。
> **纯设计文档, 无 lnn/ 代码改动**。
>
> 1. **新建目录** `analysis/riemannian_lnn/`
> 2. **新设计文档** `2026-06-05_rlstg_stage_a_design.md` (~150 行, 10 节)
> 3. **复现深度决策**: §3.2 + §3.3 完整复现, §4 理论跳过, §5 ENRON 部分用 synthetic fallback
> 4. **dependencies**: `geoopt` (PyTorch manifold ops), Hyperboloid 流形
> 5. **8 unit test 计划** 列出 + 集成测试策略
> 6. **零回归** pytest 102/102 + verify 9/9
> 7. **commit + rebase + push origin/master**

## 1. 关键决策

- 公式与本仓 LTC 几乎同构 → 复现门槛低
- 4 块论文中, 跳 §4 理论证明(仓库无 formal proof 传统)
- §5 ENRON 数据无 → 用 synthetic hyperbolic graph fallback
- stage B 启动需先 `pip install geoopt` 验兼容性

## 2. 关键公式(本仓实现)

```python
# Tangent-space LTC (与本仓 ltn.core.ltc.py 几乎同构)
h_{t+Δt} = h_t + Δt · (-α ⊙ h_t + tanh(W_h h_t + u_t))

# Riemannian wrapper (新增)
x_{t+Δt} = exp_map(x_t, Δt · f_tan(x_t, u_t))
```

## 3. pytest 套件(102/102, 76.91s)

```
102 passed, 1 warning in 76.91s
```

vs iter#35: 102 → 102 = **0 变动,0 回归**(纯设计文档)。

## 4. verify_all_models.py(9/9)

无变化。

## 5. 提交与推送

iter#36 改动:
- 增 `analysis/riemannian_lnn/2026-06-05_rlstg_stage_a_design.md` (~150 行, 10 节)
- 增 `analysis/jetson/2026-06-05_loop_iteration22_rlstg_stage_a.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-05_loop_iteration36_rlstg_stage_a.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 总体**: 14/16 = 87.5%(无变化)
**新目录**: `analysis/riemannian_lnn/` 建立

## 6. 下轮 (iter#37) 候选

按 §10 next-up + iter#36 计划:
1. **§10 RLSTG stage B** (iter#37): `lnn/core/riemannian_ltc.py` + 8 unit tests — **首选**
2. **EntroLnn stage A** (iter#38): 调研 + design
3. **Retinal LNN stage A** (iter#39): 调研 + design
4. **§10 #3 (Comparative phase-D)**: 需空载 RAM
5. **§10 #7 (LFM2.5 INT8)**: RAM blocker
6. **paper deep-read** (下一个未覆盖)
