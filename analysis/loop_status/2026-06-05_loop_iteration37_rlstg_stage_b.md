---
title: 2026-06-05 Loop iteration 37 — RLSTG §10 stage B: Riemannian LTC implementation
date: 2026-06-05
tags: [LNN, loop, rlstg-stage-b, tangent-space, hyperbolic, geoopt, prd-10-pending, new-backbone, iter37, smoke-implementation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-05 Loop iteration 37 — RLSTG §10 stage B

> `/loop 1h` 第 37 次触发。
> 紧接 iter#36 (RLSTG stage A design doc) 后,本轮执行 **stage B**:
> 实现 `lnn/core/riemannian_ltc.py` (TangentSpaceLTC + RiemannianLTC + RiemannianLTCNetwork),
> 加 9 个 unit test。**完整可工作** smoke 实现。
>
> 1. **新依赖**: `pip install geoopt` (0.5.1)
> 2. **lnn/core/riemannian_ltc.py** (+205 行): 3 个类 (TangentSpaceLTC / RiemannianLTC / RiemannianLTCNetwork)
> 3. **9 unit tests** `tests/test_riemannian_ltc.py` (+175 行)
> 4. **零回归** pytest 111/111 + verify 9/9
> 5. **commit + rebase + push origin/master**

## 1. 关键代码结构

```python
# TangentSpaceLTC — 论文 Eq. 10 直译
h_{t+1} = h_t + dt * (-α ⊙ h_t + tanh(W_h h_t + W_u u_t + b))

# RiemannianLTC — 论文 Eq. 12 直译 (origin-only)
x_{t+1} = expmap0( dt * LTC(logmap0(Linear(u))) )
```

## 2. geoopt 0.5.1 局限 + 4 步缓解

| 局限 | 缓解 |
|---|---|
| 完整 expmap/logmap 需 parallel transport (autograd 不支持) | 退 origin-only expmap0/logmap0 |
| logmap0(v) v[0] != 0 时返回 NaN | 显式 `u_amb[..., 0] = 0.0` |
| expmap0 对大 ||v|| 爆 (cosh/sinh overflow) | tangent norm clip `max_tangent_norm=1.0` |
| 初始随机 weight 太大 → 一步 NaN | `std=0.1` 初始化 + `dt=0.001` 默认 |

## 3. pytest 套件(111/111, 26.67s)

```
102 旧 + 9 新 (test_riemannian_ltc) = 111 passed
```

vs iter#36: 102 → 111 = **+9 新增,0 回归**。

## 4. verify_all_models.py(9/9)

无变化。

## 5. 提交与推送

iter#37 改动:
- 改 `lnn/core/riemannian_ltc.py` (+205 行, 3 classes)
- 增 `tests/test_riemannian_ltc.py` (175 行, 9 tests)
- 增 `analysis/jetson/2026-06-05_loop_iteration23_rlstg_stage_b.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-05_loop_iteration37_rlstg_stage_b.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 总体**: 15/16 = **93.8%**(本轮 +1: §10 RLSTG stage B 落地)

## 6. 下轮 (iter#38) 候选

按 iter#37 计划 + §10 next-up:
1. **RLSTG stage C** (iter#38 首选): `synthetic_hyperbolic_graph.py` + 3-seed × 4-backbone ablation + matrix ingest
2. **EntroLnn stage A**: 调研 + design (iter#34 调研已深, design 是 0.5 loop)
3. **Retinal LNN stage A**: 调研 + design
4. **§10 #3 (Comparative phase-D)**: 需空载 RAM
5. **§10 #7 (LFM2.5 INT8)**: RAM blocker
6. **paper deep-read**
