---
date: 2026-06-15
round: 87
tags: [LNN, gradient-H, causal-ecology, honest-negative, FAME, Causal-Audit, moe-ecology]
status: daily-summary
---

# LNN 研究日报 v13 — 2026-06-15 (`/loop 1h` 第 12 次)

## 0. TL;DR

本场是 round 87 (第 12 次 `/loop 1h`)。**实现 gradient-based H** 替代
empirical H (回应 arXiv:2606.10703 Causal Audit 警告)。**关键 honest negative**:
在 2 conditions × 3 datasets × 3 lambdas = 18 cell bench 中, **E_emp 和 E_grad
几乎完全一致** (mean |Δ| < 0.05), **gate firing 决策 9/9 相同**。Gradient H 在
toy regime **不增加价值**, 但 API 正确实现, 为 vision/NLP/larger-K/longer-training
留口子。完成 10 层 LNN+MoE 自主栈: 5 defenses + 1 diagnostic + 3 policies + 1 causal-diagnostic option。

## 1. 新论文 (本场研读)

本场 arXiv search 未发现 2026-06-14/15 直接相关新 LNN 论文。沿用
round 76-86 已研读 4 篇, 本场重点关注:
- arXiv:2606.10703 — Causal Audit (本场核心驱动: observational ≠ causal)
- arXiv:2605.06415 — MoE Ecology E (round 83, 本场被升级)
- arXiv:2606.03631 — AnchorMoE orth (本场对比基准)

## 2. 新增产出 (本场)

- `lnn/core/moe_ecology.py` (MODIFIED) — 新增 `gradient_routing_sensitivity` + H_mode (+130 行)
- `lnn/core/fame_cfc.py` (MODIFIED) — `ecology_H_mode` + `ecology_H_alpha` flags + `task_loss` arg
- `lnn/core/__init__.py` (MODIFIED) — 导出 `gradient_routing_sensitivity`
- `tests/test_gradient_based_h.py` (NEW) — 14/14 全绿
- `scripts/bench_gradient_based_h.py` (NEW) — 2 conditions × 3 datasets × 3 lambdas
- `docs/prds/2026-06-15-lnn-round-87-a-gradient-based-h.md` (NEW) — PRD #10-49
- `docs/research/2026-06-15_gradient_based_h_report.md` (NEW) — bench + honest negative
- `docs/daily/2026-06-15_LNN_research_summary_v13.md` (本文件)

## 3. Bench 结果 (本场核心, 含 honest negative)

2 conditions × 3 datasets × 3 lambdas (E_emp vs E_grad, orth gate firing):

| λ | Dataset | loss | E_emp | E_grad | orth_fired | Δ = E_emp − E_grad |
|---:|---|---:|---:|---:|---:|---:|
| 0.1 | toy_sin | 0.6408 | 0.0000 | 0.0000 | True | 0.00 |
| 0.1 | random | 0.8997 | 9.8268 | 9.8870 | False | -0.06 |
| 0.1 | structured | 2.7804 | 0.0000 | 0.0000 | True | 0.00 |
| 1.0 | toy_sin | 0.6616 | 0.0000 | 0.0000 | True | 0.00 |
| 1.0 | random | 0.9159 | 0.9705 | 0.9568 | False | 0.01 |
| 1.0 | structured | 2.8744 | 0.0000 | 0.0000 | True | 0.00 |
| 10.0 | toy_sin | 0.7370 | 0.0000 | 0.0000 | True | 0.00 |
| 10.0 | random | 1.0151 | 0.0971 | 0.0957 | True | 0.00 |
| 10.0 | structured | 3.2860 | 0.0000 | 0.0000 | True | 0.00 |

**3 假设验证**:
- **H1 (healthy 时 agree)**: ✓ 确认 — λ=0.1 random E=9.83 vs 9.89 (差 0.06)
- **H2 (toxic 时 diverge)**: ✗ **拒绝** — λ=1.0 random E=0.97 vs 0.96 (差 0.01)
- **H3 (grad 更敏感)**: ✗ **拒绝** — λ=10.0 random E=0.097 vs 0.097 (完全相同)

**关键 honest negative**: gradient H **不增加价值** in toy regime。
**Gate firing 决策 9/9 cells 完全相同**。

## 4. 累计叙事 (round 76-87)

| Round | 改动 | 关键贡献 |
|---|---|---|
| 0 | 单 CfCCell | baseline |
| 76 | + n_tau=3 | 细胞内多 τ |
| 77 | + MR-MoE K=3 softmax | K experts + softmax |
| 78 | + FAME top-K | top-K sparse |
| 79 | K×n_tau×top_K 16-cell | K=5 dense 全局最优 |
| 80 | + orthogonality λ=0.001 | fix K=3 top_k=1 hard cell |
| 81 | + φ-balancing η=0.05 | 互补防御层 2 |
| 82 | + CosineRouter | 诚实负: scale-dependent |
| 83 | + MoE Ecology E | 第 1 个理论诊断 |
| 84 | + Ecology-Gated φ | 诚实负: λ=1.0 救不回 (φ 弱) |
| 85 | + Ecology-Gated Orth | 修复 round 84 负: λ=1.0 -14% |
| 86 | + Combined Gates (φ+orth) | 2-axis policy; safe superset |
| **87** | **+ Gradient-based H** | **honest neg: toy regime 不增加价值** |

## 5. 累计 LNN+MoE 自主栈 (5 + 1 + 3 + 1 = 10 层)

5 层防御 (round 76-81):
1-5. baseline, orth, φ, MR-MoE, FAME

1 层诊断 (round 83):
6. **MoE Ecology E** (empirical H)

3 层决策 (round 84-86):
7. **Ecology-Gated φ** (round 84) — soft intervention
8. **Ecology-Gated Orth** (round 85) — strong intervention
9. **Combined Gates** (round 86) — 2-axis safe superset

1 层因果诊断选项 (round 87):
10. **Gradient-based H** (round 87) — opt-in causal H (回应 Causal Audit)

## 6. 后续候选 (round 88+)

- **#10-47** Causal importance-based gate (full Causal Audit reply)
- **#10-48.1** Per-layer gate config
- **#10-45** Gradient-based H refinement (gradient alignment)
- **#10-46** Test gradient H on vision data (验证 honest negative)
- **#10-7 LFM2.5-1.2B INT8** — deployment

## 7. 诚实负 + 局限

1. **Honest negative**: gradient H **不增加价值** in toy regime
2. **2-epoch quick bench** — 长训可能不同
3. **3 synthetic datasets** — vision/NLP 可能不同
4. **MSE task loss** — cross-entropy 可能不同
5. **K=3 top_k=1** — larger K 可能不同
6. **No ablation on normalize** — False 可能更敏感
7. **Silent fallback** — `task_loss=None` 不 warning
