---
date: 2026-06-15
round: 85
tags: [LNN, ecology-gated-orth, orth-rescaling, root-cause-fix, FAME, ortho-toxicity, round-84-negative-fix]
status: daily-summary
---

# LNN 研究日报 v11 — 2026-06-15 (`/loop 1h` 第 10 次)

## 0. TL;DR

本场是 round 85 (第 10 次 `/loop 1h`)。**修复 round 84 诚实负结果**:
实现 `EcologyGatedOrth` — 当 live E < 0.5 时, **自动 rescale orth λ 到 0.001**
(round 80 默认)。**关键 headline**: 在 λ=1.0, round 84 的 gated φ 给 0.7302 (≈ baseline), 
**本场 gated orth 给 0.6285 (-14%)**; 在 λ=10.0, **-55% loss 恢复**。Orth 干预比 φ
干预**强**, 因为它攻击 root cause (aux loss weight) 而非 symptom (routing distribution)。
完成 **adaptive policy** 闭环: round 84 gate 选 φ (soft), round 85 gate 选 orth (strong)。

## 1. 新论文 (本场研读)

| 论文 | arXiv | 状态 | 关键贡献 |
|---|---|---|---|
| E = T·H/(O+B) | 2605.06415v1 | A (round 83) | MoE Ecology 头部发现 (本场基线) |
| AnchorMoE orth | 2606.03631v1 | A (round 80) | 本场 gate 的干预对象 |
| φ-Balancing | 2605.15403v1 | A (round 81) | round 84 gate, 本场互补 |
| Causal Audit | 2606.10703v1 | A (round 79+) | Observational ≠ causal (gate 局限) |

## 2. 新增产出 (本场)

- `lnn/core/ecology_gated_balancing.py` (MODIFIED) — 新增 `EcologyGatedOrth` 类 (+150 行)
- `lnn/core/fame_cfc.py` (MODIFIED) — `ecology_gated_orth` flag + `compute_orth_loss()` API
- `lnn/core/__init__.py` (MODIFIED) — 导出 `EcologyGatedOrth`
- `tests/test_ecology_gated_orth.py` (NEW) — 15/15 全绿
- `scripts/bench_ecology_gated_orth.py` (NEW) — 2 conditions × 3 datasets × 3 lambdas
- `docs/prds/2026-06-15-lnn-round-85-a-ecology-gated-orth.md` (NEW) — PRD #10-44
- `docs/research/2026-06-15_ecology_gated_orth_report.md` (NEW) — 烟测 + 关键 headline
- `docs/daily/2026-06-15_LNN_research_summary_v11.md` (本文件)

## 3. 烟测结果 (本场核心, 含 round 84 修复)

2 conditions × 3 datasets × 3 lambdas:

| λ | Dataset | A baseline | B gated | Δ | Gate fired |
|---:|---|---:|---:|---:|---|
| 0.1 | toy_sin | 0.6474 | 0.6285 | **-2.9%** | True (λ_scale=0.01) |
| 0.1 | random | 0.9019 | 0.9019 | 0.0% | **False (no false pos)** |
| 0.1 | structured | 2.7821 | 2.7637 | -0.7% | True (λ_scale=0.01) |
| **1.0** | **toy_sin** | **0.7302** | **0.6285** | **-14.0%** ✓ | **True (λ_scale=0.001)** |
| 1.0 | random | 0.9420 | 0.9420 | 0.0% | **False (no false pos)** |
| 1.0 | structured | 2.8953 | 2.7637 | **-4.6%** | True (λ_scale=0.001) |
| 10.0 | toy_sin | 1.3804 | 0.6285 | **-54.5%** | True (λ_scale=0.0001) |
| 10.0 | random | 1.3110 | 0.8931 | **-31.9%** | True (λ_scale=0.0001) |
| 10.0 | structured | 3.6791 | 2.7637 | **-24.9%** | True (λ_scale=0.0001) |

**关键 headline**: λ=1.0 toy_sin **A 0.7302 → B 0.6285 (-14.0%)**, 完全修复
round 84 诚实负 (round 84 gated φ 给 0.7302, **没救回来**)。λ=10.0 toy_sin
**A 1.3804 → B 0.6285 (-54.5%)**, 巨大恢复。

## 4. 累计叙事 (round 76-85 含负)

| Round | 改动 | 单点 | 关键贡献 |
|---|---|---:|---|
| 0 | 单 CfCCell | 0.0525 | baseline |
| 76 | + n_tau=3 | 0.0463 | 细胞内多 τ |
| 77 | + MR-MoE K=3 softmax | 0.0324 | 13.4% → 30.7% |
| 78 | + FAME top-K | (denser) | top-K sparse MoE |
| 79 | K×n_tau×top_K 16-cell | 0.0490 | K=5 dense 全局最优 |
| 80 | + orthogonality λ=0.001 | 0.1089 (vs 0.7595) | **fixes K=3 top_k=1 hard cell** |
| 81 | + φ-balancing η=0.05 | 0.1250 | 互补防御层 2 |
| 82 | + CosineRouter | 0.96 (3/3 div) | 诚实负: parameter-free 不 work on tiny |
| 83 | + MoE Ecology E | diagnostic | 第 1 个理论诊断 |
| 84 | + Ecology-Gated φ | policy | **诚实负: λ=1.0 orth 救不回** (gate 正确但 φ 弱) |
| **85** | **+ Ecology-Gated Orth** | **strong policy** | **修复 round 84 负: λ=1.0 -14%, λ=10.0 -55%** |

## 5. 累计 LNN+MoE 自主栈 (5 + 1 + 2 = 8 层)

5 层防御 (round 76-81):
1-5. baseline, orth, φ, MR-MoE, FAME

1 层诊断 (round 83):
6. **MoE Ecology E** — tells us *when*

2 层决策 (round 84-85):
7. **Ecology-Gated φ** (round 84) — soft intervention (router bias)
8. **Ecology-Gated Orth** (round 85) — **strong intervention** (rescale aux loss)

**完成 adaptive policy 闭环**: 不同 E regime 选不同 intervention

## 6. 后续候选 (round 86+)

- **#10-48** Combine both gates (orth rescale + φ balance) — 2-axis policy
- **#10-45** Gradient-based H (替代 empirical H)
- **#10-46** Test on vision classification
- **#10-47** Causal importance-based gate (回应 Causal Audit)
- **#10-7 LFM2.5-1.2B INT8** — deployment, 需 full stack stable
- **#10-XX Timeflies 2606.13571** — irregular time series

## 7. 诚实负 + 局限

1. **Rescaling 是 destructive** — 一旦触发, 用户原 λ 被静默覆盖 (latched, no hysteresis)
2. **Honest-negative for diversity-seeking users** — 想刻意用高 orth 多样性的人会被 silent downgrade (real risk)
3. **No multi-layer coordination** — 所有 layer 同样 rescale
4. **E 是 observational** — 用 empirical mixture weights, 不是 gradient-based H
5. **2 epochs** — 长训可能不同
6. **Eval mode 不 rescale** — by design (eval 不改变 model)
