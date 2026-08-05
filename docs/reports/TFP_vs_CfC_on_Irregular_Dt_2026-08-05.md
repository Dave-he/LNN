---
title: TFP 指数 retention vs CfC σ-decay 在不规则 Δt 下的鲁棒性 — 反直觉的 negative result
date: 2026-08-05
tags: [LNN, CfC, TFP, retention, irregular-dt, robustness, dt-explicit, negative-result, sigmoid-vs-exp]
arxiv_refs: [2607.08283]
companion: [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]], [[LNN_Mathematical_Foundations_Comprehensive_2026-08-05]]
---

# TFP 指数 retention vs CfC σ-decay 在不规则 Δt 下的鲁棒性 — 反直觉的 negative result

> 本报告验证 TFP 论文 (arXiv 2607.08283) 的核心 claim "retention 显式依赖 dt → 对 dt 分布变化更鲁棒"。**结论与论文相反**：在合成 AR(2) 任务上，**CfC σ-decay 完全不敏感（ratio 1.00×），而 TFP 指数 retention 退化 14%**。该 negative result 揭示了"sigmoid 压缩"对 dt 分布变化有天然的鲁棒性。

## 1. 假设与设计

### 1.1 假设（H1）

TFP 论文 §IV 的核心论证：
> *"GRU retention $z_t$ 由 recurrent step 索引；不显式依赖 $\Delta t_t$。TFP 在 Eq. (16) 中把 retention 显式参数化为 $k_t = \exp(-\Delta t_t/\tau_t)$，保证 elapsed-time consistency 是 built-in 而不是学到的。"*

由此推论：在训练 dt=1.0、测试 dt~LogNormal(0, 0.5) 的设定下，TFP 应当比 CfC **更鲁棒**（ratio 更接近 1）。

### 1.2 实验设计

- **数据**：合成非平稳 AR(2) + 3-regime（与之前 benchmark 同）
- **训练 dt**：1.0（恒定）
- **测试 dt**：
  - regular：1.0（恒定）
  - irregular：LogNormal(0, 0.5)，有效范围 [0.123, 4.742]，mean 1.007
- **模型**：CfC / MFC-CFC / MFC-TFP（同初始化种子、同 hidden=24、同 seq_len=48）
- **重复**：3 次
- **指标**：test_mse_regular、test_mse_irregular、**degradation_ratio = mse_irregular / mse_regular**

## 2. 结果（反直觉的 negative result）

| 模型 | test_mse_regular | test_mse_irregular | **degradation_ratio** | 训练秒 |
|---|---:|---:|---:|---:|
| **cfc** | 0.0589 ± 0.0001 | 0.0589 ± 0.0001 | **1.00×** | 21.5 |
| **mfc-cfc** | 0.0590 ± 0.0001 | 0.0590 ± 0.0000 | **1.00×** | 37.6 |
| **mfc-tfp** | 0.0586 ± 0.0002 | 0.0671 ± 0.0012 | **1.14×** ⚠ | 37.8 |

完整数据：[`analysis/jetson/2026-08-05_irregular_dt_benchmark.{md,json}`](analysis/jetson/2026-08-05_irregular_dt_benchmark.md)

## 3. 解读：为什么 CfC σ-decay 比 TFP 指数 retention 更鲁棒？

### 3.1 数学对比

```text
CfC  σ-decay:   decay = σ(-f · τ · dt)  ∈ (0, 1)
TFP exp-decay:  k      = exp(-dt / τ)    ∈ (0, 1]
NSFD fraction:  h_new = (h + dt·G) / (1 + dt·L)
```

CfC 和 TFP 都"显式依赖 dt"，但**依赖方式不同**：
- **CfC σ-decay**：把 dt 揉进 sigmoid 的 exponent `(-f·τ·dt)`，sigmoid 的"saturation"特性把 dt 的巨大变化（4.7×）压缩成输出（0.001-0.999）的很小变化
- **TFP exp-decay**：直接把 dt 当作指数的输入 `exp(-dt/τ)`，**指数函数对 dt 极敏感** —— dt=0.12 时 k=0.886，dt=4.74 时 k=0.008

### 3.2 直观解释

想象一个 retention gate 需要在 dt ∈ [0.12, 4.74] 范围内保持合理值：
- **Sigmoid**：`σ(-f·τ·dt)` 通过 sigmoid 的 S 形曲线自动 clamp 输出到 (0, 1)，dt 的极端值只是把 output 推到饱和区
- **指数**：`exp(-dt/τ)` 没有任何 clamp，dt 翻 40 倍直接把 retention 从 0.886 砸到 0.008

### 3.3 对任务的影响

AR(2) 一步回归任务的"目标函数"是 `sum(x_{t+1})`，模型需要捕获状态 `x_t` 的稳定估计。

- **CfC 在 irregular dt 下**：`decay ∈ (0, 1)` 仍然 bounded，hidden state 的更新幅度只是被 sigmoid 压缩但仍可微
- **MFC-TFP 在 irregular dt 下**：`k` 直接由 dt 决定，dt 小 → k ≈ 1（强保留），dt 大 → k ≈ 0（弱保留）；hidden 的 update 量剧烈波动，导致 hidden state 在测试时与训练时分布偏移

### 3.4 TFP 论文的边界条件

TFP 论文声称的"elapsed-time consistency built-in"在 **VLA belief filtering** 场景下成立（因为 belief 是 compact latent state，short-horizon dynamics），但在 **长序列 + 大 dt 分布** 任务上**指数 retention 反而成为劣势**。

→ TFP 的优势是"**显式建模 dt 让模型知道时间过去了多少**"，但**没有保证**模型能"**对任意 dt 分布保持稳定**"。

## 4. 与上一轮 Pareto sweep 的关联

| 实验 | MFC-TFP vs CfC 的对比 |
|---|---|
| 8/5 Pareto sweep（regular dt, h × sl grid） | MFC-TFP 在 h=32/sl=64 略胜 CfC（0.0564 vs 0.0572，↓1.4%）；h=16 时打平或反超 |
| **本轮 irregular dt（h=24, sl=48, dt ~ LogNormal(0, 0.5)）** | **MFC-TFP 退化 14%，CfC 不变** |

→ MFC-TFP 在 **regular dt 下的小优势（↓1.4%）** 在 **irregular dt 下完全反转**（↑14%）。这是关于 retention 机制选择的 important 边界条件。

## 5. 实用建议

| 场景 | 推荐 retention |
|---|---|
| Regular dt (constant) | CfC σ-decay 或 MFC-TFP 均可，MFC-TFP 略优 |
| **Irregular dt (jittered)** | **CfC σ-decay** 显著更鲁棒 |
| Positivity 任务（浓度/计数）| NSFD 闭式（但 MFC-NSFD 在 h=16/sl=64 仍会爆炸，见 8/5 Pareto sweep）|
| VLA belief filtering（短序列）| TFP retention（论文原场景）|

## 6. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N6** | TFP retention 在不规则 dt 任务上的优势验证 | ✅ **本轮验证，但结论与预期相反**（TFP 退化 14%，CfC 不变）|
| **新增 N7** | CfC σ-decay 在大 dt 范围（如 dt ∈ [0.01, 100]）下是否仍鲁棒 | ⏳ 下周 |
| **新增 N8** | TFP retention 与 CfC 的 **hybrid**（gated 选择）能否兼得两边优势 | ⏳ 下周 |

## 7. 研究 take-away

1. **TFP 论文的 claim 有边界条件** — 在 VLA short-horizon 任务成立，但在长序列 + 大 dt 分布下反转
2. **Sigmoid 的"saturation"是天然的 dt-robustness 机制** — 数学上比指数 retention 更适合 irregular sampling
3. **"显式依赖 dt" ≠ "对 dt 分布鲁棒"** — 这是两个不同的 property，TFP 论文混用了
4. **跨论文综合必须用 benchmark 验证理论 claim** — 没有 benchmark，negative result 不会被发现
5. **Negative result 的研究价值** — 这条发现指引 NCP-style retention 设计的明确边界条件

## 8. 数据源回链

- Benchmark 数据
  - [`analysis/jetson/2026-08-05_irregular_dt_benchmark.{md,json}`](analysis/jetson/2026-08-05_irregular_dt_benchmark.md)
- 脚本
  - [`scripts/bench_irregular_dt.py`](scripts/bench_irregular_dt.py) (259 lines)
- 相关综合
  - [[MemoryFusionCfC_Cross_Paper_Synthesis_2026-08-05]]（MFC-TFP 实现）
  - [[MR_TFP_CfC_Second_Layer_Synthesis_2026-08-05]]（上一轮 MR-TFP-CfC benchmark）
  - [[LNN_Mathematical_Foundations_Comprehensive_2026-08-05]]（TFP 与 LTC Eq. (5) 的代数关系）
- 论文引用
  - [TFP arXiv 2607.08283](https://arxiv.org/abs/2607.08283)
  - [Lechner 2022 CfC arXiv 2106.13898](https://arxiv.org/abs/2106.13898)
