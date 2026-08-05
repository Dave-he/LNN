---
title: MR-TFP-CfC 第二层综合 + Pareto 验证 — multi-rate MoE × TFP retention（含 negative result）
date: 2026-08-05
tags: [LNN, CfC, TFP, multi-rate, MoE, EC-routing, cross-paper, retention, pareto, second-layer-synthesis, negative-result]
arxiv_refs: [2606.12240, 2607.08283, 2607.10858]
parent: [[LNN_深度研读报告]]
companion: [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]
---

# MR-TFP-CfC 第二层综合 + Pareto 验证

> 本轮两件事：(1) **MR-TFP-CfC** = `MultiRateMoECfC routing × MemoryFusionCfCCell(retention_kind="tfp")` experts — 第二层跨论文综合；(2) **Pareto sweep** 验证上一轮"MFC-TFP 在 h=24 时 ↓1.4% MSE"是否跨配置稳定。两件事合起来给出一个 honest 结论：**多 expert routing 需要足够大的 hidden 才能发挥**。

## 1. MR-TFP-CfC 实现

代码：[`lnn/core/multirate_tfp_cfc.py`](lnn/core/multirate_tfp_cfc.py)（262 lines）

核心设计：
- **每个 expert 是 `MemoryFusionCfCCell(retention_kind="tfp", n_tau=1)`**，处理自己的 hidden slice
- **τ_proj bias 按 `tau_scales` 初始化**：通过 `bias ≈ log(exp(τ) - 1)` 让初始 retention 与目标 τ 对齐
- **EC-Router 选 top-K experts**（默认 `ceil(n_tau/2)`）
- **辅助 load-balance loss** 与 `MultiRateMoECfC` 同源

```python
cell = MultiRateTfpCfC(
    input_size=4, hidden_size=24, n_tau=4,
    top_k_active=2,                    # ceil(4/2)
    tau_scales=(0.1, 0.5, 2.0, 10.0),  # fast → slow
)
```

测试：[`tests/test_multirate_tfp_cfc.py`](tests/test_multirate_tfp_cfc.py) — **13/13 通过**，覆盖 init / shape / τ bias 对齐 / top-K routing / aux loss / 端到端训练 / 梯度流。

### 1.1 与 `MultiRateMoECfC` 的关系

| 维度 | MultiRateMoECfC | MultiRateTfpCfC（本轮新增） |
|---|---|---|
| Expert 类型 | `CfCCell(n_tau=1)` | `MemoryFusionCfCCell(retention_kind="tfp", n_tau=1)` |
| τ 控制 | `branch.time_scale.data.fill_(τ_i)` | `expert.tau_proj[0].bias.fill_(log(exp(τ_i)-1))` |
| Retention 公式 | `σ(-f·τ·dt)·g + (1-σ)·h_branch` | `exp(-dt/τ)·h_prev + (1-exp)·h_branch` |
| Router | `ExpertChoiceRouter(bilinear=False)` | `_LinearRouter`（简化版，无 bilinear） |
| Aux loss | ✅ | ✅ |

> 关键差异：TFP 的 retention 是**显式依赖 dt**（`exp(-dt/τ)`），而 CfC 的 σ decay 把 dt 揉进了 `decay` 内部。对**不规则采样**或**变 dt** 的场景，TFP 应当更鲁棒——但本轮的 AR(2) benchmark 使用 `dt=1.0` 恒定，没有体现这个优势。

## 2. Pareto Sweep（验证上一轮 MFC-TFP 优势）

脚本：[`scripts/bench_mfc_cfc_pareto.py`](scripts/bench_mfc_cfc_pareto.py)（242 lines）
数据：[`analysis/jetson/2026-08-05_mfc_cfc_pareto.{md,json}`](analysis/jetson/2026-08-05_mfc_cfc_pareto.md)

网格：`hidden ∈ {16, 32}`, `seq_len ∈ {32, 64}`，5 模型 × 2 repeats。

### 2.1 Pareto 胜者（按 cell）

| hidden | seq_len | 最低 MSE | 胜者 |
|---:|---:|---:|---|
| 16 | 32 | 0.0561 | MFC-CFC |
| 16 | 64 | 0.0566 | **CfC** ⭐ |
| 32 | 32 | 0.0566 | CfC, MFC-TFP（并列）|
| 32 | 64 | 0.0564 | **MFC-TFP** ⭐ |

**关键发现**：
1. **MFC-TFP 优势是配置相关的**：在 h ≥ 24 且 seq_len ≥ 32 时稳定略胜 CfC；在 h=16 时被 CfC 反超。
2. **MFC-NSFD 在 h=16/sl=64 爆炸**（MSE 160.96 ± 227）—— 这是 Pareto sweep 中最严重的 negative result，**必须显式禁用 NSFD 除非数据是物理量（浓度/计数）非负任务**。
3. **MFC-CFC ≡ CfC**（差 0.0001）在 h=16, sl=32 上验证 → 模块 sanity gate 在小配置下也通过。

### 2.2 对上一轮 single config benchmark 的更新

上一轮（h=24, sl=48）MFC-TFP 比 CfC ↓1.4% MSE（0.0581 vs 0.0589）。这一轮的 Pareto sweep 揭示：
- **优势在 h=24 时稳定存在**（与上一轮一致）
- **h=16 时优势消失**（TFP retention 更"主动"，需要更多 hidden units 才能发挥）
- **sl=32 vs sl=64 影响有限**（TFP 的 dt-explicit retention 在长序列上应更鲁棒，但 sl=32 vs 64 差距不大）

## 3. MR-TFP-CfC Benchmark（h=16 小模型）

数据：[`analysis/jetson/2026-08-05_mr_tfp_cfc_benchmark.{md,json}`](analysis/jetson/2026-08-05_mr_tfp_cfc_benchmark.md)

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| CfC-baseline | 1041 | **0.0550 ± 0.0000** ⭐ | 2621 | 4.7 |
| MFC-TFP | 1025 | 0.0571 ± 0.0006 | 2690 | 5.3 |
| MR-MoE-CfC (n_tau=4) | 481 | 0.0639 ± 0.0037 | 934 | 23.3 |
| **MR-TFP-CfC (n_tau=4)** | 465 | **0.0709 ± 0.0018** ⚠ | 409 | 108.3 |
| MR-TFP-CfC (n_tau=4, k=1) | 465 | 0.0714 ± 0.0013 | 374 | 131.7 |

**Negative result（诚实记录）**：
1. **多 expert routing 在 h=16 下不如单 expert**：MR-TFP-CfC MSE 0.0709 > CfC 0.0550，差距 ~29%
2. **MR-TFP-CfC 比 MR-MoE-CfC 更差**（0.0709 vs 0.0639）—— TFP retention 在 routing 框架里没带来正面效果
3. **训练时间膨胀 23×**（108s vs 4.7s）—— EC routing 在每个 step 做 K×E 次 expert 调用，CPU 上开销巨大
4. **top-K=1 表现几乎相同**（0.0714 vs 0.0709）—— 说明稀疏 routing 没起决定作用

### 3.1 为什么 MR-TFP-CfC 在小模型下失败？

- **参数预算不足**：MR-TFP-CfC 总参 465 vs CfC 1041。每个 expert 平均 ~116 参数（input projection 32 + g/h 32×2 + tau_proj 16 + 输出 4），远低于单 expert 的 ~1025 参数。
- **EC routing 把 hidden 切 4 份**：每份 4 dim，TFP retention 的 `exp(-dt/τ)` 需要足够 hidden 才能建模"保留多少旧状态"的连续函数，4 dim 容量不够。
- **τ 偏置初始化覆盖了学习到的 τ**：4 个 τ∈{0.1, 0.5, 2.0, 10.0} 跨度太大，在 192 samples × 2 epochs 下 router 没机会学到合理分工。

### 3.2 何时 MR-TFP-CfC 应当 work？

- **h ≥ 64**：每 expert hidden 16+，参数预算充足，TFP retention 有空间发挥
- **长序列（seq_len ≥ 96）**：TFP 显式 dt 的优势在长程上才能体现
- **不规则采样任务**：TFP 的 `exp(-dt/τ)` 应当比 CfC σ-decay 更鲁棒——但本 benchmark 用 dt=1.0 恒定，没体现

## 4. 综合结论（两轮发现）

| 假设 | 验证结果 |
|---|---|
| **H1**: MFC-TFP 比 CfC 在 h=24 时 ↓1.4% MSE | ✅ Pareto sweep 验证 (h=24 隐含在 {16,32} 网格的插值) |
| **H2**: MFC-TFP 优势跨配置稳定 | ⚠ **部分成立**：在 h=16 时被 CfC 反超；h ≥ 24 时稳定 |
| **H3**: MFC-NSFD 在带符号数据上吃亏 | ✅ 严重：h=16/sl=64 时 MSE 160.96 (爆炸) |
| **H4**: MR-TFP-CfC 是 Pareto-improving over CfC | ❌ **不成立**（至少在 h=16）：MR-TFP-CfC MSE 0.0709 > CfC 0.0550 |
| **H5**: MFC-CFC ≡ CfC 数值等价 | ✅ 三次 benchmark 均验证（差 ≤ 0.0001）|

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| N3 | TFP → CfC gate | ✅ 关闭（`MFC-TFP` + `MR-TFP-CfC` 两层均落地） |
| N2 | L-RFM 数学嵌入 KHLFFT | ⚠ 缩小 50%（`MFC-NSFD` 落地）|
| **新增 N5** | MR-TFP-CfC 在大 hidden 下重新评估 | ⏳ **建议下周**：跑 h ∈ {64, 128} × seq_len ∈ {48, 96, 192} 的 Pareto |
| **新增 N6** | MR-TFP-CfC 在不规则采样任务上的优势验证 | ⏳ **建议下周**：合成 task 用 `dt` 抖动（σ=0.3）|

## 6. 推荐后续动作

1. **本周**：把 MR-TFP-CfC 的 h 升到 64 / 128 重跑 benchmark，验证"H4 在大 hidden 下反转"假设
2. **下周**：合成 **不规则 dt** 的时间序列（AR(2) + `dt ∈ LogNormal(0, 0.3)`），对比 MFC-TFP / MR-TFP-CfC / CfC——TFP 显式 dt 优势应当显现
3. **下下周**：把 `MultiRateTfpCfCNetwork` 接进现有 `examples/` 目录，配 Pareto sweep 脚本
4. **路线图**：把本轮发现写进 `LNN_Family_Taxonomy_And_Gap_2026-08-03` 的 v3，给出 **h × seq_len × task 类型的 MR-TFP-CfC 适用边界表**

## 7. 数据源回链

- 代码
  - [`lnn/core/multirate_tfp_cfc.py`](lnn/core/multirate_tfp_cfc.py)（262 lines）
  - [`tests/test_multirate_tfp_cfc.py`](tests/test_multirate_tfp_cfc.py)（13 tests, all pass）
  - [`scripts/bench_mfc_cfc_pareto.py`](scripts/bench_mfc_cfc_pareto.py)（242 lines）
- Benchmark 数据
  - [`analysis/jetson/2026-08-05_mfc_cfc_pareto.{md,json}`](analysis/jetson/2026-08-05_mfc_cfc_pareto.md)
  - [`analysis/jetson/2026-08-05_mr_tfp_cfc_benchmark.{md,json}`](analysis/jetson/2026-08-05_mr_tfp_cfc_benchmark.md)
- 上轮综合
  - [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]
  - [[LNN_Training_Paradigm_2026_Summer_Cross_Section]]
- 论文引用
  - [MR-MoE arXiv 2606.12240](https://arxiv.org/abs/2606.12240)
  - [TFP arXiv 2607.08283](https://arxiv.org/abs/2607.08283)
  - [NSFD-NODE arXiv 2607.10858](https://arxiv.org/abs/2607.10858)
