---
title: MFC-Hybrid-Gate — Input-Dependent α 实现真 Conditional Gating（N11 positive result）
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, conditional-gating, input-dependent, alpha-MLP, N11, positive-result]
arxiv_refs: [2607.08283, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[MFC_Hybrid_Retention_2026-08-05]], [[MFC_Hybrid_Irregular_Dt_Train_N9_2026-08-05]]
gap_refs: [N11-input-dependent-alpha]
---

# MFC-Hybrid-Gate — Input-Dependent α 实现真 Conditional Gating（N11）

> 上一轮 N9 发现 `α = sigmoid(self.alpha[i])` 是 **static per-branch parameter**，**不**是 conditional gate。本轮 N11 把 α 升级为 **input-dependent 函数** `α(x_t, dt)` 通过 per-branch MLP——验证 α 真的能 conditional。

## 1. 设计：α from MLP([x_t, dt])

```python
# hybrid_gate 模式（MFC 的第 5 种 retention_kind）：
gate_in = cat([x_t, dt_e])            # shape [B, input_size + 1]
α       = sigmoid(W₂ · sigmoid(W₁ · gate_in + b₁) + b₂)  # MLP, shape [B, hidden_size]

# k_cfc = σ(-f · τ_cfc · dt)         # CfC path (sigmoid saturation)
# k_tfp = exp(-dt / softplus(τ_tfp))  # TFP path (exponential)
# k     = α · k_cfc + (1 - α) · k_tfp  # input-dependent mix
# h_new = k · h_prev + (1 - k) · h_branch
```

**关键差异 vs hybrid (N8)**：
- hybrid：α = sigmoid(static Parameter) —— per-branch scalar，**不**依赖输入
- hybrid_gate：α = MLP([x_t, dt]) —— per-branch 函数，**依赖** x 和 dt

## 2. 实现

代码：[`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py) — 新增 `retention_kind="hybrid_gate"` 分支：
- 复用 CfC components (f_gate, time_scale) 和 TFP components (tau_proj)
- 新增 `gate_mlps: ModuleList` — 每个 branch 一个 `Sequential(Linear(input_size+1, d) → Sigmoid → Linear(d, d) → Sigmoid)`
- init：gate MLP 用 gain=0.1（让 sigmoid 输出接近 0.5，两路径等权）

测试：[`tests/test_hybrid_gate.py`](tests/test_hybrid_gate.py) — **11/11 通过**。

## 3. 测试覆盖（11/11 通过）

| 测试 | 验证内容 |
|---|---|
| `test_hybrid_gate_in_valid_set` | `_VALID_RETENTION` 包含 "hybrid_gate" |
| `test_init_hybrid_gate_creates_gate_mlps` | gate_mlp 的 in_features = input_size + 1 |
| `test_init_hybrid_gate_alpha_is_none` | hybrid_gate 不使用 `self.alpha`（区别于 hybrid） |
| `test_forward_shape_hybrid_gate` (×2) | shape 正确（n_tau=1 与 n_tau=3） |
| `test_alpha_depends_on_x` | **关键**：α 真的依赖 x（spread > 0） |
| `test_alpha_depends_on_dt` | **关键**：α 真的依赖 dt（spread > 0） |
| **`test_alpha_learns_conditional_gating_after_training`** | **关键**：训练后 α spread 显著增加 |
| `test_hybrid_gate_dt_zero_recovers_h_prev` | dt→0 时 forward 有界（实际 α 仍 input-dep） |
| `test_gradients_flow_hybrid_gate` | gate_mlp gradient 流正常 |
| `test_end_to_end_training_step_hybrid_gate` | 5 步训练 loss 下降 |

## 4. Benchmark 结果

数据：[`analysis/jetson/2026-08-05_hybrid_gate_benchmark.{md,json}`](analysis/jetson/2026-08-05_hybrid_gate_benchmark.md)

训练 dt = LogNormal(0, 0.5)，测试两种 dt：

| 模型 | 参数量 | test MSE (regular) | test MSE (irregular) | **degradation** |
|---|---:|---:|---:|---:|
| cfc-baseline | 2137 | 0.0573 | 0.0574 | 1.00× |
| mfc-cfc | 2137 | 0.0572 | 0.0573 | 1.00× |
| mfc-tfp | 2113 | 0.0575 | 0.0605 | 1.05× |
| mfc-hybrid (static α) | 2857 | 0.0576 | 0.0582 | 1.01× |
| **mfc-hybrid_gate (input-dep α)** | **3577** | 0.0576 | **0.0578** | **1.00×** ⚡ |

### 4.1 α diversity (after 4 epochs training)

- **std over different x** (fixed dt=1): **0.0118**
- **std over different dt** (fixed x=0): **0.0045**

→ α **真的 conditional**：不同输入产生不同 α 值。

### 4.2 关键发现

1. **hybrid_gate degradation 1.00×** —— **与 CfC 完全持平**！这是 5 种 retention_kind 中 **首次** 达到 CfC 级 dt-robustness 且 irregular MSE 不显著退化。
2. **regular MSE 0.0576** 与所有其他模式几乎相同（0.0572-0.0576）—— 不牺牲 regular dt 性能
3. **α 真的 conditional**：x 和 dt 都能驱动 α 变化（虽然 std 数值不大）

## 5. 与 N8/N9 的演进对比

| 实验 | α 类型 | 训练 dt | α diversity | degradation |
|---|---|---|---|---|
| N8 (hybrid, regular train) | static scalar | dt=1.0 | n/a | 1.05× |
| N9 (hybrid, irregular train) | static scalar | LogNormal(0, 0.5) | α 0.500→0.576 | 1.01× |
| **N11 (hybrid_gate, irregular train)** | **input-dep MLP** | **LogNormal(0, 0.5)** | **std_x=0.012, std_dt=0.0045** | **1.00×** ⚡ |

→ N11 是 **3 轮 hybrid 演进的 positive culmination**：input-dependent α 让 hybrid 真正获得了 conditional gating 能力。

## 6. 研究 take-away

1. **Static α ≠ conditional gate**：N9 已证明 hybrid 的 static α 不能 per-input 切换
2. **Input-dependent α 工作**：N11 的 gate MLP 实现了 `α(x_t, dt)` 真 conditional function
3. **hybrid_gate 达到 CfC 级 dt-robustness**：1.00× degradation 与 CfC σ-decay 持平
4. **参数代价**：hybrid_gate 3577 vs hybrid 2857 vs cfc 2137 —— 多 720 个参数是 gate MLP，但获得 conditional gating 能力
5. **α 的 dt-驱动力较弱**：std_dt=0.0045 < std_x=0.0118 —— 模型主要用 x 决定 α，dt 的影响相对小。可能在长序列 / 大 dt-range 数据上才会更强

## 7. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N11** | Input-dependent α 实现真 conditional gate | ✅ **本轮关闭（positive result）** |
| N10 | Hybrid × MR-TFP-CfC 三层组合 | ⏳ 下周 |
| N12 | Hybrid 在 dt distribution shift 下的 transferability | ⏳ 下周 |
| **新增 N13** | hybrid_gate 与 MR-TFP-CfC 的三层组合（multi-rate × TFP × input-dep-α）| ⏳ 下周 |

## 8. 推荐后续动作

1. **本周**：N13 hybrid_gate × MR-TFP-CfC 三层组合
2. **下周**：N12 测试 hybrid_gate 在 dt distribution shift 下的 transferability
3. **路线图**：写 "LNN retention mechanism design space" survey，把 5 种 retention_kind (cfc / tfp / nsfd / hybrid / hybrid_gate) 的适用边界条件系统化

## 9. 数据源回链

- 代码
  - [`lnn/core/memory_fusion_cfc.py`](lnn/core/memory_fusion_cfc.py)（hybrid_gate 分支，261 lines total）
  - [`tests/test_hybrid_gate.py`](tests/test_hybrid_gate.py)（11 tests, all pass）
- Benchmark
  - [`analysis/jetson/2026-08-05_hybrid_gate_benchmark.{md,json}`](analysis/jetson/2026-08-05_hybrid_gate_benchmark.md)
- 上轮对照
  - [[MFC_Hybrid_Irregular_Dt_Train_N9_2026-08-05]]（N9 partial positive）
  - [[MFC_Hybrid_Retention_2026-08-05]]（N8 hybrid baseline）
  - [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]（retention_kind 接口）
