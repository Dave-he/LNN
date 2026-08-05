---
title: MR-hybrid_gate-CfC — N13 三层综合（含 honest finding：h=24 时 multi-rate 不如 single-expert）
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, multi-rate, MR-MoE, three-layer-synthesis, N13, honest-finding, small-hidden]
parent: [[LNN_深度研读报告]]
companion: [[MR_TFP_CfC_Second_Layer_Synthesis_2026-08-05]], [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]
gap_refs: [N13-three-layer-synthesis]
---

# MR-hybrid_gate-CfC — N13 三层综合（honest finding）

> 把"input-dep α conditional gating"（N11）与"multi-rate EC routing"（round 282 / b8d8879）组合——理论上三层综合应该结合 cfC σ-decay 的 robustness + TFP 的 explicit dt + 多速率 expert specialization。**结果（honest finding）**：degradation 持平 CfC (1.00×) 但 precision 比 single-expert 差 ~11%，原因是 small hidden (24/4=6 per expert) 不够。

## 1. 设计：三层综合

```
                arXiv 2606.12240 (MR-MoE)
                         ↓
        MultiRateTfpCfC expert = MemoryFusionCfCCell(retention_kind="hybrid_gate")
                         ↓
                arXiv 2607.08283 (TFP)
                         ↓
                Lechner 2022 (CfC) [arXiv 2106.13898]
                         ↓
                N11 (input-dep α via per-expert MLP)
```

**架构**：
- `MultiRateTfpCfCNetwork(expert_retention_kind="hybrid_gate")` —— 每个 expert 是 hybrid_gate
- EC Router 选 top-K experts per step
- 每个 expert 有自己的 input-dep α MLP，per-expert 独立 conditional gating
- 单 expert hidden = 24 / n_tau=4 = **6 维**

## 2. 实现

代码修改：[`lnn/core/multirate_tfp_cfc.py`](lnn/core/multirate_tfp_cfc.py) — 重构 `expert_retention_kind` 参数支持 `"tfp" | "cfc" | "nsfd" | "hybrid" | "hybrid_gate"`。向后兼容（默认 `"tfp"` 不变，13 个原有测试仍通过）。

测试：[`tests/test_mr_hybrid_gate.py`](tests/test_mr_hybrid_gate.py) — **14/14 通过**：
- init 接受合法 expert_retention_kind，拒绝非法值
- hybrid_gate expert 内部确实有 `gate_mlps`，`alpha=None`（区别于 hybrid 的 static）
- forward shape 正确（含 dt tensor 与 scalar 两种）
- 每个 expert 的 α 真的依赖 x 和 dt
- 端到端训练 loss 下降
- 网络 wrapper 加 `dt` 参数支持 per-step dt

## 3. Benchmark 结果

数据：[`analysis/jetson/2026-08-05_mr_hybrid_gate_benchmark.{md,json}`](analysis/jetson/2026-08-05_mr_hybrid_gate_benchmark.md)

### 3.1 主表（7 模型 × 2 repeats × 4 epochs, irregular dt 训练）

| 模型 | 参数量 | regular MSE | irregular MSE | **degradation** |
|---|---:|---:|---:|---:|
| cfc-baseline (h=24) | 2137 | 0.0564 | 0.0565 | **1.00×** |
| mfc-cfc (h=24) | 2137 | 0.0560 | 0.0560 | **1.00×** |
| mfc-tfp (h=24) | 2113 | 0.0586 | 0.0618 | 1.05× |
| mfc-hybrid (h=24) | 2857 | 0.0556 | 0.0574 | 1.03× |
| **mfc-hybrid_gate (h=24)** | 3577 | **0.0558** | **0.0579** | 1.04× |
| MR-TFP-CfC (n_tau=4, h=24) | 833 | 0.0650 | 0.0649 | **1.00×** |
| **MR-hybrid_gate-CfC (n_tau=4, h=24)** | 1433 | 0.0643 | 0.0643 | **1.00×** |

### 3.2 关键观察

1. **MR-hybrid_gate-CfC degradation 1.00×** —— **与 CfC 持平** ✅
2. **MR-hybrid_gate-CfC 比 MR-TFP-CfC 略优**（irregular 0.0643 vs 0.0649，↓1.0%）—— input-dep α 在 multi-rate 框架内仍贡献
3. **但 MR-hybrid_gate-CfC 比 single-expert mfc-hybrid_gate 差 11%**（0.0643 vs 0.0579）—— multi-rate 在 hidden=24 上**参数利用效率**不如 single-expert
4. **参数差异**：MR-hybrid_gate-CfC 1433 vs single-expert mfc-hybrid_gate 3577 —— multi-rate **更省参数**（节省 60%）但精度受损

## 4. Honest finding 解读

**N13 没达到"best of all worlds"**。三层综合在 degradation ratio 上与 CfC 持平（**这个目标达到了**），但在 precision 上不如 single-expert hybrid_gate。

**原因**：
- n_tau=4 + h=24 → **每个 expert 仅有 6 维 hidden**
- 每个 expert 的 input-dep α MLP 需要 `Linear(input_size+1, 6) → Sigmoid → Linear(6, 6) → Sigmoid` —— 6 dim 太小，α 不能精细 conditional gating
- 与 round 282 (b8d8879) 的 finding **完全一致**：MR-TFP-CfC 在 h=16 时退化 ~29%，需要 h ≥ 64

**结论**：N13 在 **架构设计** 上是三层综合成功（input-dep α + multi-rate + EC routing 都正常工作），但 **规模配置**（h=24 / n_tau=4）太小，没让组合优势发挥。

## 5. 与前序 round 的对照

| Round | commit | 配置 | N13 关联 |
|---|---|---|---|
| 282 | b8d8879 | MR-TFP-CfC, h=16, 退化 29% | 验证 "multi-rate 在 small hidden 下退化" |
| 286 | 55d81dc | mfc-hybrid_gate (single expert, h=24), 1.00× degradation | N11 实现 input-dep α |
| **287** | **本轮** | **MR-hybrid_gate-CfC (n_tau=4, h=24), 1.00× degradation** | **N13 三层综合** |

→ N13 是 round 282 (multi-rate) + round 286 (input-dep α) 的合成，**架构正确**但 **规模受限**。

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N13** | hybrid_gate × MR-TFP-CfC 三层综合 | ✅ **本轮关闭（架构 OK，规模受限）** |
| N12 | hybrid_gate 在 dt distribution shift 下的 transferability | ⏳ 下周 |
| **新增 N14** | MR-hybrid_gate-CfC 在 h=64/128 上重评估（验证"small hidden 限制"）| ⏳ 下周 |

## 7. 推荐后续动作

1. **本周**：N14 跑 MR-hybrid_gate-CfC 在 h=64 / 128 上 —— 验证 small hidden 是否是限制因素
2. **下周**：N12 hybrid_gate 在 dt distribution shift 下的 transferability
3. **路线图**：写 "LNN retention mechanism design space" 综合 survey（合并 N3 / N6 / N8 / N9 / N11 / N13 的所有数据）

## 8. 数据源回链

- 代码
  - [`lnn/core/multirate_tfp_cfc.py`](lnn/core/multirate_tfp_cfc.py)（+ expert_retention_kind 参数 + dt slice 修复）
- 测试
  - [`tests/test_mr_hybrid_gate.py`](tests/test_mr_hybrid_gate.py)（14 tests, all pass）
  - [`tests/test_multirate_tfp_cfc.py`](tests/test_multirate_tfp_cfc.py)（13 tests, all pass，向后兼容验证）
- Benchmark
  - [`analysis/jetson/2026-08-05_mr_hybrid_gate_benchmark.{md,json}`](analysis/jetson/2026-08-05_mr_hybrid_gate_benchmark.md)
- 上轮对照
  - [[MR_TFP_CfC_Second_Layer_Synthesis_2026-08-05]]（round 282, MR-TFP-CfC 二层综合）
  - [[MFC_Hybrid_Gate_N11_Input_Dependent_Alpha_2026-08-05]]（N11 single-expert input-dep α）
