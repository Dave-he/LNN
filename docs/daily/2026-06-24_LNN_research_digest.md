---
title: LNN 每日研究追踪 - 2026-06-24 (session #76)
date: 2026-06-24
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-06-24 (session #76, hourly loop #2)

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：1 篇新增（session #76 手工筛选）
- GitHub 候选仓库：0 个
- Hugging Face 候选模型：0 个
- 已下载 PDF：0 个

## 本轮（session #76）新增论文

| arXiv ID    | 提交日期       | 标题                                                              | 关键词命中                    | 与本仓关联             |
|-------------|----------------|-------------------------------------------------------------------|-------------------------------|------------------------|
| 2606.08431 | 2026-06-07 | Control-Theoretic View of Neural ODEs: Empirical Controllability and Observability | Controllability Gramian + Neural ODE | **高** — 新机制维度    |

### 2606.08431 — Control-Theoretic View of Neural ODEs: Empirical Controllability and Observability
- **核心思想**：把 Neural ODE 表示为 **control-affine form** dh/dt = f(h) + g(h)·u，然后：
  - **Controllability** 用 LTV controllability Gramian 衡量：Wc(T) = ∫₀^T Φ(T,τ) B(τ) B(τ)^T Φ(T,τ)^T dτ，trace(Wc) 量化输入对状态的影响
  - **Observability** 用 LTV observability Gramian 衡量：Wo(T) = ∫₀^T Φ(τ,0)^T C(τ)^T C(τ) Φ(τ,0) dτ，trace(Wo) 量化状态对输出的可推断度
- **关键洞察**：低可控 Gramian ⟺ 模型忽略输入（"dead" CfC cell）；高可控 Gramian + 低任务 loss = 良好响应
- **本仓适配**：CfC 的离散更新 `h_{t+1} = decay·g + (1-decay)·h_branch` 可以从 **input sensitivity** 角度评估：
  - `c_t = ||cell(x_t, h) - cell(0, h)||_2 / ||cell(x_t, h)||_2`（每步相对输入灵敏度）
  - **Controllability loss** = relu(margin - c_t.mean())，鼓励输入对隐藏状态的影响
  - 这正好补充 round 240 Lyapunov：**稳定性 vs 可控性** 的 trade-off
- **PR 候选**：`lnn/core/controllability_cfc.py` + `tests/test_controllability_cfc.py` + `scripts/bench_controllability_cfc.py`
- **预期**：在 toy_sin/structured/random × {baseline, +ctrl margin=0.05, +ctrl+lyap} = 27 cells，验证：
  - H1 (task ±5%): control regularizer 不退化任务 loss
  - H2 (input sensitivity ↑): c_t 平均值相比 baseline 上升 ≥20%
  - H3 (与 Lyapunov 兼容): 加 lyap 后可控性不下降（trace 变化 <10%）

### 其他候选（评估后暂不入库）

| arXiv ID    | 标题                                          | 评估            |
|-------------|-----------------------------------------------|-----------------|
| 2606.16693 | Learning Hybrid Biophysical Neuron Models with Neural ODEs | 生物学导向，与 CfC 应用层距离远 |
| 2606.16567 | TNODEV: Toolbox for Neural ODE Verification   | 实现成本高（需要可达性算法）     |
| 2606.10596 | Embedding Hybrid Systems into Continuous Latent Vector Fields | 理论性强，本仓短期不需要         |
| 2606.06351 | Function-Space Priors for Bayesian Neural ODEs | Bayesian 框架，不匹配本仓 deterministic CfC |

## 落地优先级
1. **2606.08431 Controllability-Regularized CfC**（本轮首选）：与 round 240 Lyapunov 正交 — 稳定性 vs 可控性的二维 trade-off。
2. 后续 backlog：2606.18315 Ghost Attractor、2606.15469 Context-Aware。

## 建议动作
- 对 2606.08431 实现 `ControllabilityCfCCell`，期望 H1+H2+H3 全过
- 若 H1 退化 → 标注为 target-dependent（与 round 91-101、240 同模式）
- 若 H3 (与 Lyapunov 兼容) 失败 → 探索 margin/alpha 调优