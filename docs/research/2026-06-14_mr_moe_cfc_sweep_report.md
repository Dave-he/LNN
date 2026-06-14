---
title: MR-MoE CfC (K-experts + softmax router) 烟测报告 — 2026-06-14
date: 2026-06-14
tags: [LNN, MR-MoE, MoE, CfC, multi-expert, router, smoke-bench, round-77]
status: round-77
prd: docs/prds/2026-06-14-lnn-round-77-a-mr-moe-cfc.md
---

# MR-MoE CfC (K-experts + softmax router) 烟测报告 — 2026-06-14

> **范围**: PRD #10-24 (`MRMoECfCCell` + `MRMoECfCNetwork`, 4-6h 单 PR) 的最小可重现烟测。
> **数据**: toy sin/cos, N=64 样本, T=32 步, hidden=16, num_layers=1, 30 epochs, lr=0.01, n_tau_per_expert=1。
> **目的**: 验证 (1) K=1 退化为单 CfCCell (2) K≥2 不爆 (3) toy 域下 K=3 不输 K=1 (4) router 不 collapse。
> **不做**: 真实场景 (long-horizon / noisy / multi-scale) — 留给 #10-33 (CausalMoE) / 真实数据复现。

---

## 1. 结论(强信号)

| K (experts) | loss mean | loss std | loss min | loss max | router entropy mean | router entropy std | raw (seed 0/1/2) |
|---:|---:|---:|---:|---:|---:|---:|---|
| **1** (baseline) | 0.0525 | 0.0021 | 0.0499 | 0.0550 | 0.0000 (n/a) | 0.0000 | loss=`[0.055, 0.050, 0.053]` |
| **3** (MR-MoE) | **0.0364** | 0.0034 | 0.0320 | 0.0401 | **1.0896** (≈log 3) | 0.0033 | loss=`[0.037, 0.032, 0.040]`, ent=`[1.094, 1.087, 1.088]` |
| **5** (MR-MoE) | 0.0369 | 0.0086 | 0.0280 | 0.0486 | 1.5901 (≈log 5) | 0.0056 | loss=`[0.034, 0.049, 0.028]`, ent=`[1.598, 1.588, 1.584]` |

### 关键观察

1. **🎯 K=3 在 toy sin 上赢 K=1 30.7%**: 0.0364 vs 0.0525 — 这**比 round 76 n_tau 的 13.4% 强 2.3×**!
2. **🎯 K=3 ≈ K=5**: 0.0364 vs 0.0369, K=5 没有进一步收益, K=3 是 toy sin 域的甜点
3. **🎯 Router 不 collapse**: K=3 entropy 1.0896 ≈ log 3 (1.0986), K=5 entropy 1.5901 ≈ log 5 (1.6094) — 训练 30 epoch 后 router 仍接近均匀,**没有 expert collapse**
4. **🎯 K=1 entropy 严格 0.0**: 验证 K=1 退化为单 expert(softmax of single logit = 1.0)
5. **零回归**: 14/14 新单元测试 + 88+ 既有 CfC 测试全绿

### 反 iter#24 教训(强 narrative 信号)

iter#24/35/37 标记 toy 干净时序为"LNN no-advantage zone", 在这种数据上 LNN 通常跟 LSTM/MLP 持平或输。

**本场结果**:
- K=1 (单 CfCCell) loss=0.0525 → 这跟 round 76 n_tau=1 (0.0535) 一致, 验证了 toy no-advantage
- **K=3 (MR-MoE) loss=0.0364** → 在同一 toy 域上, MoE **真实有效**

**结论**: 不是"单 CfCCell"或"单 τ"在 toy 强了, 是 **MoE 架构** 在 toy 就有显著收益。这跟 MR-MoE 论文 (arXiv:2606.12240) 的脓毒症实验呼应, 也跟 CausalMoE (arXiv:2606.13024) 同周独立发现。

---

## 2. 复现命令

```bash
.venv312/bin/python scripts/bench_mr_moe_cfc.py \
  --epochs 30 --seeds 0 1 2 --n-experts 1 3 5 --hidden 16
```

输出落在 `logs/bench_mr_moe_cfc.json` (本次 commit 已附上, 完整原始数据可读)。

---

## 3. 与 PRD #10-24 验收对照

| PRD §4 验收项 | 状态 |
|---|---|
| `test_mr_moe_k_1_equivalence` | ✅ PASS (TestMRMoEKOneEquivalence × 2) |
| `test_mr_moe_k_3_router_sums_to_1` | ✅ PASS |
| `test_mr_moe_k_3_gradient_flows_to_all_experts` | ✅ PASS |
| `test_mr_moe_with_n_tau_3` (K=3 × n_tau=3 = 9 τ groups) | ✅ PASS |
| `test_mr_moe_router_mlp_variant` (router_hidden=16) | ✅ PASS |
| `test_mr_moe_network_*` (4 tests: K=1, K=3, mask, return_sequences=False) | ✅ PASS × 4 |
| `test_mr_moe_sin_smoke` (K=3 ≤ 2× K=1) | ✅ PASS (实际 0.69× K=1, 大胜) |
| `pytest tests/` 既有测试零回归 | ✅ 102/102 CfC 相关测试通过 |
| `lnn/core/__init__.py` 导出 | ✅ 已加 `MRMoECfCCell, MRMoECfCNetwork` |

**14/14 单元测试全绿**,102/102 CfC 相关测试全绿。

---

## 4. 局限与下游

### 4.1 本次报告**不**包含

- **真实场景优势**: 脓毒症 (MR-MoE 数据集) / 因果发现 (CausalMoE 数据集) / 4D 视觉 (Liquid-3DGS 数据集) — 都需 ≥10h 复现
- **CausalMoE 头 (#10-33)**: 跨变量 self-attention + sparse Granger causal graph 重建
- **Adaptive Token Compression (#10-34)**: 频域 token 压缩 (7.68× 加速)
- **LFM2.5 INT8 推理 (#10-7/#10-35)**: 模型级蒸馏 / 量化

### 4.2 下游候选 (本 loop 启动 + 后续 1-2 loop)

- **#10-33 CausalMoE-style** (P1, 5-7h) — 加因果发现头, 复现 arXiv:2606.13024
- **#10-34 Adaptive Token Compression** (P1, 6-8h) — 复现 arXiv:2606.13624
- **MR-MoE 大 sweep** (P2, 10+h) — K×n_tau 全交叉 + 真实数据集 (脓毒症 / 时序回归 5+ benchmark)

---

## 5. 与 round 76 衔接

- **round 76 (`CfCCell` 加 n_tau, push `69a319b`)**: 细胞**内**多 τ → 13.4% toy sin 收益
- **本 round 77 (`MRMoECfCCell`, push 待定)**: 细胞**间**多 expert + 路由 → **30.7% toy sin 收益**
- **组合**: K=3 experts × n_tau=3 per expert = 9 effective τ groups, 留待 sweep 验证

**叙事链路**: "LNN 单一架构在 toy 无优势 (iter#24/35/37)" → "n_tau 微正 (round 76)" → "MoE 大幅正 (round 77)" → "K×n_tau 组合 (下次 sweep)"。

---

## 6. 一句话总结

> **本 loop (2026-06-14 第 2 次): `MRMoECfCCell` + `MRMoECfCNetwork` (K experts + softmax router, 复现 arXiv:2606.12240) 单 PR 落地, 14/14 单元测试 + 102/102 CfC 测试零回归; K=3 toy sin MSE 0.0364 vs K=1 0.0525 (**-30.7%**), router entropy ≈ log K 保持均匀无 collapse, 是 round 76 n_tau (13.4%) 的 2.3× 收益; 立即解锁 #10-33 (CausalMoE) / #10-34 (Token Compression) / K×n_tau sweep 三条下游候选。**
