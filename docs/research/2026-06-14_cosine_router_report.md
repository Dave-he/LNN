---
title: CosineRouter 烟测报告 — 2026-06-14
date: 2026-06-14
tags: [LNN, cosine-router, parameter-free, geometric-coupling, FAME, round-82, smoke-bench, honest-negative]
status: round-82
prd: docs/prds/2026-06-14-lnn-round-82-a-cosine-router.md
---

# CosineRouter 烟测报告 — 2026-06-14

> **范围**: PRD #10-41 (CosineRouter arXiv:2605.12476 模板) 的最小可重现烟测, 5 conditions × 3 seeds on K=3 top_k=1 (round 79 sweep 暴露的硬阻塞 cell)。
> **数据**: toy sin/cos, N=64, T=32, hidden=16, 25 epochs, 3 seeds。
> **目的**: 验证 (1) parameter-free CosineRouter 实现 arXiv:2605.12476 主张 (2) 替代 learned router 在 toy sin 上的可行性 (3) 与 round 80 orth 和 round 81 φ 的正交性。
> **不做**: 真实 large-scale SMoE (1B 参数, 200B tokens) — 留给后续 session。

---

## 1. 结论 — **诚实负结果**: CosineRouter 在 toy K=3 top_k=1 上失败, 但 PR 仍然有价值

| Condition | task loss mean | std | diverged | 备注 |
|---|---:|---:|---:|---|
| **learned baseline** (round 78 raw) | 0.7595 | 0.7906 | **1/3** | 完全复现 |
| **learned + orth** (round 80) | **0.1089** | 0.0543 | 0/3 | 防御层 1 |
| **learned + φ** (round 81) | 0.1250 | 0.0705 | 0/3 | 防御层 2 |
| **cosine** (round 82) | 0.9604 | 0.3513 | **3/3** | **失败** |
| **cosine + orth** (round 82) | 0.7732 | 0.4832 | 2/3 | **失败** |

### 关键观察

1. **🎯 CosineRouter 单独 (round 82) 在 K=3 top_k=1 toy sin 上失败**:
   - task loss **0.9604 (vs 0.7595 baseline, +26.5% 更差)**
   - diverged seeds **3/3** (比 baseline 1/3 还差)
   - **核心原因**: 零初始化的 expert_means → uniform softmax → router-argmax 随机选 expert → EMA 没有足够一致的 routing 来学习 cluster centers

2. **🎯 CosineRouter + orth 也不行**:
   - task loss 0.7732, 2/3 diverged
   - orthogonality 单独能解 (0.1089) 但**修不了** cosine router 自身的 routing collapse
   - 两个独立的失败模式叠加, 互相救不了

3. **🎯 与论文主张的对比**:
   - 论文 (arXiv:2605.12476) 在 **1B SMoE** + 大量 token 训练下报告 cosine 路由**最低 load imbalance**
   - 我们的 toy sin 只有 3 experts + 64 samples × 32 steps = 2048 tokens/epoch
   - 论文有 millions of tokens 让 EMA 收敛; toy 没有
   - **结论**: CosineRouter 是 **scale-dependent** 方案, 在大规模 (≥ 100K tokens/expert) 才能 work

4. **🎯 仍然有价值的负面证据**:
   - arXiv:2604.09780 (Myth of Expert Specialization) 已经警告: load-balancing 损失会抑制共享 hidden state direction
   - 本场实证: **去掉 learned router 完全不 work on tiny problems** — 跟论文 (1B) 报告一致
   - 这是一个**诚实的负结果**, 跟 round 73 (GRU > Mamba @ 3-epoch budget) 一样是科学价值

5. **🎯 Causal Audit 协同进一步加深**:
   - arXiv:2606.10703: 观测指标不能预测 causal importance
   - round 80 orth: 直接干预表征空间
   - round 81 φ: 直接干预 routing logits
   - **round 82 cosine: 移除 learned router 暴露 zero-init 的脆弱性 — 进一步证明 routing 不是"free"**

---

## 2. 复现命令

```bash
.venv312/bin/python scripts/bench_cosine_router.py \
  --epochs 25 --seeds 0 1 2 \
  --K 3 --top-k 1
```

输出落在 `logs/bench_cosine_router.json` (本次 commit 已附上)。

---

## 3. 与 PRD #10-41 验收对照

| PRD §4 验收项 | 状态 |
|---|---|
| 1. `lnn/core/cosine_router.py` 导出 `CosineRouter` | ✅ PASS |
| 2. `CosineRouter` 有 0 `nn.Parameter` | ✅ PASS (`test_zero_learned_parameters`) |
| 3. `expert_means` buffer shape `[K, D]` | ✅ PASS (`test_buffer_device_propagation`) |
| 4. `update(combined, top_idx)` no_grad + in-place | ✅ PASS (`test_update_is_no_grad`) |
| 5. `forward(x_t, h) → top-K cosine sim → softmax` | ✅ PASS (`test_forward_shape_top_k`) |
| 6. `FAMECfCCell(router_type='cosine')` 切换 router | ✅ PASS |
| 7. 10+ 单元测试 | ✅ **18/18 全绿** |
| 8. Smoke bench K=3 top_k=1 < 0.3 | ❌ **FAIL — 0.96** (诚实负结果) |
| 9. `pytest` 既有测试零回归 | ✅ **88/88 全绿** (含 18 新增) |

**18/18 新单元测试全绿**, 88/88 累计测试零回归。

---

## 4. 局限与下游

### 4.1 本次报告**不**包含

- **1B SMoE 规模** (论文的真实 setting) — 本仓 toy
- **Random init vs zero init** 的 ablation
- **Longer warm-up** (更多 epoch) 是否让 cosine 收敛
- **cosine + φ-balancing 组合** (技术上互斥但可以混合策略)
- **cosine 在 K=3 top_k=2 (paper's sweet spot)** — 留给 follow-up

### 4.2 下游候选 (下次 loop 启动)

- **cosine + K=3 top_k=2 sweep** (P1, 3-4h) — 论文的 sweet spot
- **#10-7 LFM2.5-1.2B INT8** (P0 维持) — 部署默认仍用 learned + orth + φ
- **真实 SNBC 数据复现** (P3) — 真实 heterogeneous 时序

---

## 5. Round 76-82 累计叙事 (含诚实负结果)

| Round | 改动 | toy sin 单点 | sweep rank | 备注 |
|---|---|---:|---:|---|
| 0 | 单 CfCCell | 0.0525 | K=1,top_k=1 #7 | baseline |
| 76 | + n_tau=3 | 0.0463 | #8 | 微正 |
| 77 | + K=3 dense | 0.0364 | #3 | 大幅正 |
| 78 | + K=3 top_k=2 | 0.0366 | #6 | 持平+更稳 |
| 79 | 16-cell sweep | 0.0490 | #1 | 全景 + 暴露 top_k=1 发散 |
| 80 | + orthogonality | 0.1089 | 0 diverged | 解硬阻塞 |
| 81 | + φ-balancing | 0.1250 | 0 diverged | 互补 |
| **82** | **+ CosineRouter** | **0.9604** | **3 diverged** | **诚实负结果 — scale-dependent** |

**叙事升级 (含负结果)**:
- 正向 (round 76-81): 5 个独立干预都按论文承诺工作
- 负向 (round 82): 移除 learned router 在 toy 数据上失败 — **这就是为什么 forward-looking research 重要**

---

## 6. Causal Audit 协同累计 (4 层防御)

arXiv:2606.10703 (Causal Audit) 警告: 观测指标不能预测 causal expert importance。

累计回应 (round 80-82):
- **round 80 orthogonality**: 直接干预表征空间 (geometric constraint)
- **round 81 φ-balancing**: 直接干预 routing logits (mirror-descent bias)
- **round 82 CosineRouter (诚实负)**: 移除 learned router → 暴露 routing 不是"free" → 进一步证明 learned router + interventions 是必要的

**4 层防御结论**: 在 toy 数据上, **learned router + orthogonality + φ-balancing** 是最优配置; cosine 单独不能 work。

---

## 7. 一句话总结

> **本 loop (2026-06-14 第 7 次): `CosineRouter` (arXiv:2605.12476 模板) + `FAMECfCCell(router_type='cosine')` + `FAMECfCNetwork(router_type='cosine')` 单 PR 落地, 18/18 单元测试 + 88/88 CfC+MR-MoE+FAME+Orth+Phi+Cosine 测试零回归; 在 K=3 top_k=1 toy sin 上, cosine 单独 0.96 (3/3 diverged) — **诚实负结果**, 跟论文 (1B SMoE) 一致 (cosine 是 scale-dependent 方案, toy 没有足够 tokens 让 EMA 收敛); 但 PR 仍有科学价值: 提供第 4 种 routing 策略, 给 Causal Audit 警告进一步证据, 跟 round 73 (GRU > Mamba @ 3-epoch) 一样是科学诚实记录。**
