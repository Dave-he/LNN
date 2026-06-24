---
title: LNN 每日研究追踪 - 2026-06-25 (session #78)
date: 2026-06-25
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-06-25 (session #78, hourly loop #4)

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：1 篇新增（手工筛选，session #78）
- GitHub 候选仓库：0 个
- Hugging Face 候选模型：0 个
- 已下载 PDF：0 个

## 本轮新增论文（手工补齐）

| arXiv ID    | 提交日期       | 标题                                                              | 关键词命中                    | 与本仓关联             |
|-------------|----------------|-------------------------------------------------------------------|-------------------------------|------------------------|
| 2606.22801 | 2026-06-22 | Multi-τ Liquid-Mamba for All-in-one Image Restoration | 多τ + adaptive gating + Mamba | **高** — 扩展 round 76 多τ CfC |

### 2606.22801 — Multi-τ Liquid-Mamba
- **核心思想**：在 Selective State Space (Mamba) 模型上加入**输入条件化的多时间尺度液体离散化**
- **两个新机制**：
  - **(a) Adaptive τ（自适应时间常数）**：每个 branch 的有效 τ 是输入的函数 — `τ_i = τ_base_i · σ(W_i x + b_i)`
  - **(b) Gated Branch Fusion（门控融合）**：用 softmax(W_gate · x) 给每个 branch 加权，而不是简单的 concat / 等权
- **本仓对应**：本仓已实现 round 76 的 **Multi-τ CfC (n_tau)**，但**所有分支共享输入条件化的 τ 和门控融合尚未实现**：
  - round 76 = 静态 τ（学到的常数）
  - round 218-228 FATC = 频域自适应 τ（基于 FFT 特征）
  - **round 243（新）= 输入条件化的 τ + 门控融合** — 比 round 76 多一个动态维度
- **PR 候选**：`lnn/core/adaptive_gated_multitau_cfc.py` + `tests/test_*.py` + `scripts/bench_*.py`
- **核心实现**：
  ```python
  # Per-branch input-conditioned time scale
  tau_i = tau_base[i] * sigmoid(W_tau[i] @ x + b_tau[i])
  # Per-step gated fusion
  gate = softmax(W_gate @ x_t)         # (B, n_tau)
  output = sum_i gate[i] * branch_i_output
  ```
- **预期**：在 toy_sin/structured/random × {baseline, +n_tau (round 76), +adaptive (new)} = 27 cells
  - H1 (task safe): adaptive gating 不退化任务 loss
  - H2 (gate entropy high): 平均 gate 熵 ≥ log(n_tau)·0.5（gate 不退化为单分支）
  - H3 (τ variability): τ_i(x) 的方差/均值 ≥ 0.1（τ 真的随输入变化）

## 落地优先级
1. **2606.22801 Adaptive-Gated Multi-τ CfC**（本轮首选）：把 round 76 的多τ扩展为输入条件化+门控融合，与 round 240/241/242 (Lyap/Ctrl/ISS) 正交。
2. 后续 backlog：2606.18315 Ghost Attractor、2606.13571 Timeflies (existence modeling)。

## 建议动作
- 实现 `AdaptiveGatedMultiTauCfCCell` with `n_tau=3, tau_base=(0.1, 1.0, 10.0)`
- 关键测试：固定 x 时 τ 稳定；x 变化时 τ 显著变化
- 若 H1+H2+H3 全过 → 进 round 244+ 纳入自主栈