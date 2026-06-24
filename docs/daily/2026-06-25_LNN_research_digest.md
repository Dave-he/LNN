---
title: LNN 每日研究追踪 - 2026-06-25 (session #77)
date: 2026-06-25
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-06-25 (session #77, hourly loop #3)

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：1 篇新增（手工筛选，session #77）
- GitHub 候选仓库：0 个
- Hugging Face 候选模型：0 个
- 已下载 PDF：0 个

## 本轮新增论文（手工补齐）

| arXiv ID    | 提交日期       | 标题                                                              | 关键词命中                    | 与本仓关联             |
|-------------|----------------|-------------------------------------------------------------------|-------------------------------|------------------------|
| 2606.14136 | 2026-06-12 | Environment-Aware Stable Neural Koopman Dynamics Learning (ESNKD) | ISS + Koopman + Neural ODE  | **高** — 统一 round 240+241 |

### 2606.14136 — ESNKD (Environment-Aware Stable Neural Koopman Dynamics Learning)
- **核心思想**：把 Neural ODE 的 dynamics 分解为：
  - (i) bundle-structured encoder 把环境观测映射到几何结构
  - (ii) Koopman operator 在提升空间中提供线性表示
  - (iii) stability guarantee 通过 Lyapunov-like 函数
  - (iv) **Input-to-State Stability (ISS)** — extension of Lyapunov for input-driven systems
- **ISS 条件**：存在 V(h) > 0, χ ∈ KL 类函数使得
  `V(h_{t+1}) - V(h_t) ≤ -α·V(h_t) + β·||x||²`
  - 输入为零时 → V 指数收缩（稳定性）
  - 输入很大时 → V 增长有界（被 β·||x||² 控制）
- **本仓适配**：这正好是 **round 240 (Lyapunov) + round 241 (Controllability) 的统一框架**
  - round 240 only: V_next ≤ (1-α)V → 可能忽略输入（uncontrollable）
  - round 241 only: 输入驱动 → 可能不稳定
  - **round 242 (新)**: V_next ≤ (1-α)V + β·||x||² → 输入驱动但稳定
- **ISS loss**: `iss_loss = relu(V_next - (1-α)V + β·||x||² + margin)`
- **PR 候选**：`lnn/core/iss_stable_cfc.py` + `tests/test_iss_stable_cfc.py` + `scripts/bench_iss_stable_cfc.py`
- **预期**：在 toy_sin/structured/random × {baseline, +lyap, +ctrl, +iss} = 36 cells
  - H1 (task ±5%): ISS 不退化任务 loss（Lyapunov 退化 +13% on random，ISS 应该缓解）
  - H2 (V bounded by input): V(h) 与 ||x||² 正相关（R² > 0.5）
  - H3 (stability-controllability balance): 同时满足 Lyap-like 收缩（输入=0 时）+ Ctrl-like 响应（输入大时）

## 落地优先级
1. **2606.14136 ISS-Stable CfC**（本轮首选）：统一 Lyap+Ctrl 框架，可能解决 round 240/241 在 random 上退化的问题。
2. 后续 backlog：2606.18315 Ghost Attractor、2606.13571 Timeflies (existence modeling)。

## 建议动作
- 实现 `ISSStableCfCCell` with V(h) = h^T P h, α=0.05, β=0.01
- 关键测试：让 x=0 时 V 收缩，x 很大时 V 有界（不爆炸）
- 若 H1+H2+H3 全过 → 进 round 243+ 纳入自主栈
- 若 H1 退化 → 标注 target-dependent；若有显著改善（Lyap 退化 +13% → ISS 退化 <5%）则是 STRICTLY POSITIVE