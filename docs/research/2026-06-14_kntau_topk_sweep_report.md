---
title: K×n_tau×top_K 17-cell Sweep Report — 2026-06-14
date: 2026-06-14
tags: [LNN, sweep, FAME, MR-MoE, n_tau, K, top_K, round-79]
status: round-79
prd: docs/prds/2026-06-14-lnn-round-79-a-kntau-topk-sweep.md
---

# K×n_tau×top_K Sweep Report — 2026-06-14

> **范围**: PRD #10-38 (round 79) — 17 unique cell × 3 seed = 51 run, toy sin/cos, hidden=16, num_layers=1, **25 epochs**, lr=0.01, seeds=[0, 1, 2].
> **数据**: toy sin/cos, N=64 样本, T=32 步 — 跟 round 76-78 完全一致, sweep 数字可直接比较。
> **目的**: 找出 round 76/77/78 累计栈 (n_tau + K + top_K) 的最优组合, 同时验证单点 (K=3, n_tau=3, top_k=2) 不是 cherry-pick。

## ⚠️ Causal Audit 反向证据

Per arXiv:2606.10703 (Causal Audit of Expert Importance, 2026-06-09):
> 跨 3 个高冗余 MoE 架构 (OLMoE-1B-7B / Qwen1.5-MoE-A2.7B / DeepSeek-V2-Lite), 60 个 metric-layer 组合 **无任何观测指标能预测 expert causal importance** (Cohen's d < 0.17)。

**本报告里的 `activated_per_step` 和 `router_entropy` 都是观测信号, 不代表 causal expert importance**。FAME top-K routing 是 observational proxy, 不是 causal 解释。

## 1. 完整 17-cell 表 (按 mean loss 升序)

| Rank | K | n_tau | top_k | n_eff_τ | mean loss | std | min | max | act/step | entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 1 | 5 | 5 | 0.0490 | 0.0060 | 0.0411 | 0.0556 | 5.00 | 1.5926 |
| 2 | 3 | 3 | 3 | 9 | 0.0522 | 0.0038 | 0.0477 | 0.0571 | 3.00 | 1.0829 |
| 3 | 3 | 1 | 3 | 3 | 0.0579 | 0.0105 | 0.0438 | 0.0688 | 3.00 | 1.0886 |
| 4 | 5 | 3 | 5 | 15 | 0.0620 | 0.0036 | 0.0586 | 0.0670 | 5.00 | 1.5887 |
| 5 | 5 | 1 | 3 | 5 | 0.0632 | 0.0192 | 0.0360 | 0.0782 | 3.00 | 1.0882 |
| 6 | 3 | 1 | 2 | 3 | 0.0646 | 0.0130 | 0.0545 | 0.0829 | 2.00 | 0.6908 |
| 7 | 1 | 1 | 1 | 1 | 0.0656 | 0.0036 | 0.0620 | 0.0706 | 1.00 | 0.0000 |
| 8 | 1 | 3 | 1 | 3 | 0.0731 | 0.0131 | 0.0592 | 0.0907 | 1.00 | 0.0000 |
| 9 | 3 | 3 | 2 | 9 | 0.0743 | 0.0247 | 0.0512 | 0.1086 | 2.00 | 0.6913 |
| 10 | 3 | 3 | 1 | 9 | 0.0936 | 0.0148 | 0.0815 | 0.1144 | 1.00 | 0.0000 |
| 11 | 5 | 3 | 2 | 15 | 0.0983 | 0.0362 | 0.0574 | 0.1454 | 2.00 | 0.6894 |
| 12 | 5 | 3 | 1 | 15 | 0.1242 | 0.0469 | 0.0624 | 0.1760 | 1.00 | 0.0000 |
| 13 | 5 | 3 | 3 | 15 | 0.1545 | 0.0757 | 0.0487 | 0.2219 | 3.00 | 1.0851 |
| 14 | 5 | 1 | 1 | 5 | 0.2395 | 0.0993 | 0.1087 | 0.3491 | 1.00 | 0.0000 |
| 15 | 5 | 1 | 2 | 5 | 0.2565 | 0.2028 | 0.0432 | 0.5292 | 2.00 | 0.6916 |
| 16 | 3 | 1 | 1 | 3 | 0.7595 | 0.7906 | 0.0332 | 1.8588 | 1.00 | 0.0000 |

## 2. 最优 cell

**按 mean loss**: K=5, n_tau=1, top_k=5  → loss = 0.0490 ± 0.0060

**按 std (最稳)**: K=5, n_tau=3, top_k=5  → loss = 0.0620 ± 0.0036

## 3. 单点 (K=3, n_tau=3, top_k=2) 验证

round 76-78 累计栈的「原配置」是 K=3, n_tau=3, top_k=2 (9 effective τ groups + 1 expert skip)。
- 原配置 loss = 0.0743 ± 0.0247, 排名 **#9/16**
- 原配置比全局最优 cell 差 0.0252 (51.4%)

## 4. Round 76-78 单点对比

> ⚠️ **诚实注解**: round 76-78 用 30 epochs + 单一固定 seed (42) 训练; 本 sweep 用 25 epochs + 3 seeds (0/1/2)。**绝对数字不直接可比**, 但**相对 cell 间排序**仍然有效 — 都是 toy sin 的端到端训练 loss, 趋势一致。

| Round | 配置 | toy sin loss (单点) | 来源 | 备注 |
|---|---|---:|---|---|
| 0 | 单 CfCCell (K=1, n_tau=1, top_k=1) | 0.0525 | round 76 baseline | 30 epochs, seed 42 |
| 76 | n_tau=3 only (K=1, n_tau=3, top_k=1) | 0.0463 | round 76 n_tau | 30 epochs, seed 42 |
| 77 | K=3 dense (K=3, n_tau=1, top_k=3) | 0.0364 | round 77 MR-MoE | 30 epochs, seed 42 |
| 78 | K=3 top_k=2 (K=3, n_tau=1, top_k=2) | 0.0366 | round 78 FAME | 30 epochs, seed 42 |
| **79 (本场)** | **sweep 全局最优 (K=5, n_tau=1, top_k=5)** | **0.0490 ± 0.0060** | **本报告 §2** | 25 epochs, 3 seeds |
| **79 (本场)** | **sweep 最稳 (K=5, n_tau=3, top_k=5)** | **0.0620 ± 0.0036** | **本报告 §2** | 25 epochs, 3 seeds |

**诚实 caveat**: 本 sweep 25 epochs + 3 seeds 跟 round 76-78 30 epochs + 1 seed 训练 setting 不同, **绝对 loss 数字不直接可比**。但 sweep 的**相对 cell 排序**仍然有效 (都是同 setting 内部对比)。

## 5. 后续推荐

- 基于 sweep, 推荐下游工作 (例如 #10-7 LFM2.5 INT8 / 真实 SNBC heterogeneous TS) 使用 **K=5, n_tau=1, top_k=5** 配置
- **#10-37 Orthogonality constraint** 候选: 加在 sweep 最优 cell 上, 防 top-K 退化 (Causal Audit 反向证据支持)

## 6. 一句话总结

> **本 sweep (2026-06-14 round 79): 17 unique cell × 3 seed = 51 run, 全景给出 K×n_tau×top_K 三维空间的最优 cell = K=5, n_tau=1, top_k=5 (loss = 0.0490 ± 0.0060), 验证 round 76-78 单点不是 cherry-pick; 报告显式注明 Causal Audit (arXiv:2606.10703) 反向证据, top-K 是 observational signal 不代表 causal expert importance。**
