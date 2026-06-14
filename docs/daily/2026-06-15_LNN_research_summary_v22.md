# LNN Research Daily Digest v22 — 2026-06-15 (Round 96)

**Focus**: test the open H4 from round 95 — does the round 80 orthogonality constraint (PRD #10-37, arXiv:2606.03631 AnchorMoE) actually increase FAME's expert diversity at the safe λ=0.001 setting?

## 1. Paper survey

June 2026 arXiv returned the same papers as rounds 94-95. No new LNN-specific leads. The fresh angle for round 96 came from backlog: H4 from round 95.

## 2. Implementation: bench only (PRD #10-58)

Round 96 is **bench-only** — no new code in `lnn/core/`. We reuse:
- `expert_diversity_summary` (round 95)
- `orthogonality_loss` (round 80)
- `FAMECfCCell.forward_with_aux` (round 78)
- `activation_space_overlap` (round 90)

`scripts/bench_fame_orth_diversity.py` is the only new file. 18 cells (3 datasets × 2 conditions × 3 seeds).

## 3. Test + bench summary

- **No new unit tests** (this is a bench-only round)
- **39/39** in-domain green (27 effective_rank + 12 orthogonality, all pre-existing)
- **18-cell bench** (3 datasets × {baseline, +orth λ=0.001} × 3 seeds, 100 epochs):

| dataset    | cond     | div_ratio   | mean_eff | task_loss | act_cos       |
|------------|----------|-------------|----------|-----------|---------------|
| toy_sin    | baseline | 1.32 ± 0.08 | 5.13     | 0.1254    | 0.7555        |
| toy_sin    | orth     | 1.31 ± 0.06 | 5.12     | 0.1253    | 0.7337        |
| structured | baseline | 1.15 ± 0.04 | 5.31     | 0.5043    | 0.4238        |
| structured | orth     | 1.16 ± 0.03 | 5.45     | 0.4935    | **0.2600**    |
| random     | baseline | 1.31 ± 0.08 | 5.49     | 0.8857    | 0.3319        |
| random     | orth     | 1.24 ± 0.04 | 5.46     | 0.9142    | **0.2472**    |

- **H1 (orth increases weight diversity) REJECTED**: Δ div_ratio = -0.07 to +0.01, all within noise
- **H2 (task loss within ±10%) CONFIRMED**: Δ = 0% / -2% / +3%
- **H3 (orth reduces activation cos_sim) PARTIAL**: toy_sin -0.022, structured -0.164, random -0.085
- **Cumulative suite**: 649/649 in-domain green (no new tests, all pre-existing)

## 4. Honest verdict

Round 80's orthogonality works **at the activation level** (it decorrelates expert hidden states) but does **NOT** increase weight-level diversity at the safe λ=0.001 setting. The orth constraint is too weak to reshape weight singular value spectra.

This is a **clean honest-negative** for H1 and a **partial confirmation** of round 80's design intent. The orth loss can stay in the stack as a defensive measure, but should not be marketed as a weight-diversity booster.

## 5. The activation-vs-weight distinction

The key insight from round 96:
- **Activation diversity** = expert hidden states are decorrelated on the same input
- **Weight diversity** = expert weight matrices have different singular value spectra

`orthogonality_loss` targets activation diversity directly. To target weight diversity, we'd need a different penalty (e.g. `||W_i W_j^T||_F^2`).

## 6. 6-round audit complete (rounds 91-96)

| Round | Property | Verdict |
|-------|----------|---------|
| 91 | CfC smoothness | ✓ CfC 2× lower than MLP |
| 92 | Smoother → robust (target dropout) | ✗ LSTM wins |
| 93 | Smoother → robust (input dropout) | ✗ MLP wins |
| 94 | Smoother → low rank | ✗ CfC highest rank |
| 95 | FAME → diverse weight experts | Δ=0.24 over MR-MoE |
| **96** | **Orth → diverse weight experts** | **Δ=0.00, orth is activation-only** |

**Each property is independent**: smoothness, robustness, rank, weight diversity, activation diversity. The mechanisms do not cross.

## 7. Cumulative state — 16-layer LNN+MoE 自主栈 (rounds 76-96)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| 92-93 | Dropout audit (target-side, input-side) | diagnostic |
| 94 | Effective rank (Williams/Payeur/Lajoie 2026) | diagnostic |
| 95 | Per-expert effective rank (FAME diversity) | diagnostic |
| **96** | **FAME+orth diversity (round 80 mechanism test)** | **diagnostic** |

## 8. Backlog for round 97+

1. **Weight-level orthogonality** (`||W_i W_j^T||_F^2` penalty) — true weight diversity
2. **K=20, hidden=32, full recurrent training** — paper-scale settings
3. **Regime-labeled task** — does FAME router pick the right expert?
4. **PhysioNet-style irregular time-series** — most important untested domain
5. **Paper-style note** combining rounds 91-96 — 6-round audit complete
