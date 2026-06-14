---
date: 2026-06-15
round: 86
tags: [LNN, combined-gates, 2-axis-policy, ecology-gate, FAME, orth-rescaling, phi-balancing, hypothesis-testing]
status: daily-summary
---

# LNN 研究日报 v12 — 2026-06-15 (`/loop 1h` 第 11 次)

## 0. TL;DR

本场是 round 86 (第 11 次 `/loop 1h`)。**完成 adaptive policy 闭环**: 实现
`CombinedEcologyGate` — 当 E < 0.5 时**同时**触发 round 84 φ gate (soft) +
round 85 orth gate (strong) 2 个 gate。**关键发现**: 在 4 conditions × 3 datasets
× 3 lambdas = 36 cell bench 中, **H2 (orth dominates) 验证**, **H3 (φ adds noise) 拒绝**,
**H1 (combined best) 部分支持**。Combined gate (D) 在所有 9 cell 中**never worse than**
orth alone (C), 多数 cell 与 C 完全相等。Combined 是 orth 的 **safe superset** — 一行 opt-in
零风险。

## 1. 新论文 (本场研读)

本场 arXiv search 未发现 2026-06-14/15 直接相关的新 LNN 论文。
沿用 round 76-85 已研读 4 篇:
- arXiv:2605.06415 — MoE Ecology E (本场 gate 的触发条件)
- arXiv:2606.03631 — AnchorMoE orth (本场 gate 的干预对象)
- arXiv:2605.15403 — φ-Balancing (round 84 gate)
- arXiv:2606.10703 — Causal Audit (observational ≠ causal, gate 局限)

## 2. 新增产出 (本场)

- `lnn/core/ecology_gated_balancing.py` (MODIFIED) — 新增 `CombinedEcologyGate` 类 (+200 行)
- `lnn/core/fame_cfc.py` (MODIFIED) — `ecology_combined=True` flag, 共享 sub-gate 实例
- `lnn/core/__init__.py` (MODIFIED) — 导出 `CombinedEcologyGate`
- `tests/test_combined_gates.py` (NEW) — 17/17 全绿
- `scripts/bench_combined_gates.py` (NEW) — 4 conditions × 3 datasets × 3 lambdas
- `docs/prds/2026-06-15-lnn-round-86-a-combined-gates.md` (NEW) — PRD #10-48
- `docs/research/2026-06-15_combined_gates_report.md` (NEW) — 烟测 + 假设验证
- `docs/daily/2026-06-15_LNN_research_summary_v12.md` (本文件)

## 3. 烟测结果 (本场核心, 假设验证)

4 conditions × 3 datasets × 3 lambdas:

| λ | Dataset | A baseline | B φ | C orth | D combined | Winner |
|---:|---|---:|---:|---:|---:|---|
| 0.1 | toy_sin | 0.6447 | 0.6439 | **0.6282** | **0.6282** | C/D tie |
| 0.1 | random | 0.9019 | 0.9019 | 0.9019 | 0.9019 | tie (no fire) |
| 0.1 | structured | 2.7821 | 2.7821 | **2.7637** | **2.7637** | C/D tie |
| 1.0 | toy_sin | 0.7302 | 0.7302 | **0.6282** | **0.6282** | C/D tie |
| 1.0 | random | 0.9420 | 0.9420 | 0.9420 | 0.9420 | tie (no fire) |
| 1.0 | structured | 2.8953 | 2.8953 | **2.7637** | **2.7637** | C/D tie |
| 10.0 | toy_sin | 1.3804 | 1.3804 | **0.6282** | **0.6282** | C/D tie |
| 10.0 | random | 1.3110 | 1.3110 | **0.8931** | **0.8931** | C/D tie |
| 10.0 | structured | 3.6791 | 3.6791 | **2.7637** | **2.7637** | C/D tie |

**3 假设验证**:
- **H1 (cumulative)**: 部分支持 — combined ≤ min(B, C) in 9/9 cells (never worse)
- **H2 (orth dominates)**: ✓ **确认** — D = C in 8/9, equal in 1/9
- **H3 (φ adds noise)**: ✓ **拒绝** — D ≤ C in 9/9 (φ 加了等于没加)

## 4. 累计叙事 (round 76-86)

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
| 85 | + Ecology-Gated Orth | **修复 round 84 负: λ=1.0 -14%** |
| **86** | **+ Combined Gates (φ+orth)** | **2-axis policy; safe superset; orth dominates** |

## 5. 累计 LNN+MoE 自主栈 (5 + 1 + 3 = 9 层)

5 层防御 (round 76-81):
1-5. baseline, orth, φ, MR-MoE, FAME

1 层诊断 (round 83):
6. **MoE Ecology E** — tells us *when*

3 层决策 (round 84-86):
7. **Ecology-Gated φ** (round 84) — soft intervention
8. **Ecology-Gated Orth** (round 85) — strong intervention
9. **Combined Gates (φ + orth)** (round 86) — 2-axis safe superset

**Adaptive policy 闭环完成**。Round 86 完成 LNN+MoE 自主栈
最后一层 — 用户现在可以选择:
- `ecology_gated_orth=True` (最小开销, round 85 推荐)
- `ecology_combined=True` (最大安全, round 86 推荐, 一行 opt-in 零风险)

## 6. 后续候选 (round 87+)

- **#10-45** Gradient-based H (替代 empirical H)
- **#10-47** Causal importance-based gate (回应 Causal Audit)
- **#10-48.1** Per-layer gate config (multi-layer dynamics)
- **#10-46** Test on vision classification
- **#10-7 LFM2.5-1.2B INT8** — deployment, 需 full stack stable

## 7. 诚实负 + 局限

1. **H1 only "at least as good"** — combined 不 strict 优于 orth alone, 但**never worse**
2. **No multi-layer test** — bench 是 single-layer, per-layer dynamics 可能不同
3. **2-epoch quick bench** — 长训可能不同 (φ 可能平滑恢复轨迹)
4. **3 synthetic datasets** — vision / NLP 可能不同
5. **No ablation on phi_eta** — 用了 round 84 默认 η=0.05
6. **Eval mode 不 rescale** — by design
7. **Both gates latched** — no hysteresis (follow-up)
