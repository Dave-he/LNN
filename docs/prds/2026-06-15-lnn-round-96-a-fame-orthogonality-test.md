# PRD #10-58 — Test FAME with Orthogonality (Round 96, H4 from Round 95)

**Date**: 2026-06-15
**Round**: 96 (direct follow-up to round 95 H4)
**Status**: Drafted.

## 1. Why round 96

Round 95 left H4 untested: does the orthogonality constraint from round 80 (`orthogonality_loss`, PRD #10-37, arXiv:2606.03631 AnchorMoE) actually increase FAME's expert diversity?

Round 80 added the loss. Round 83 showed λ=0.001 is safe (no task-loss regression). Round 85 added auto-rescaling. But the **diversity claim** — "orthogonality_loss encourages diverse expert representations" — has never been measured directly in our stack.

**Question for round 96**: at the safe λ=0.001 setting, does FAME+orth have higher `diversity_ratio` than FAME-baseline?

## 2. Hypotheses

- **H1 (orth increases diversity)**: FAME+orth(λ=0.001) trained 100 epochs has diversity_ratio ≥ 1.10× the FAME-baseline ratio on toy_sin (where round 95 FAME-baseline = 1.32).
- **H2 (orth is not just a stylistic tax at λ=0.001)**: FAME+orth task loss is within ±10% of FAME-baseline (round 83 said λ=0.001 is safe).
- **H3 (orthogonality is weight-level AND activation-level)**: FAME+orth has higher activation-space diversity too, measured by mean pairwise cos_sim < baseline (round 90's `activation_space_overlap`).

## 3. Plan

### 3.1 Bench (`scripts/bench_fame_orth_diversity.py`) — only new code

18 cells:
- 3 datasets: toy_sin, structured, random (same as round 95)
- 2 conditions: FAME-baseline (no orth), FAME+orth(λ=0.001)
- 3 seeds, 100 epochs

For each cell measure:
- `per_expert_eff_rank` (round 95 tool)
- `diversity_ratio` (max/min)
- `task_loss` (final MSE)
- `orth_loss` (auxiliary, FAME+orth only)
- `activation_cos_sim` (mean pairwise cos_sim of expert outputs, lower = more diverse)

The orth loss is computed via `cell.compute_orth_loss(expert_outs, lambda_coeff=0.001)`. This requires calling `forward_with_aux` to get the K expert outputs.

### 3.2 No new code in `lnn/core/`

We reuse:
- `expert_diversity_summary` (round 95)
- `orthogonality_loss` (round 80)
- `FAMECfCCell.forward_with_aux` and `compute_orth_loss` (rounds 78, 85)
- `activation_space_overlap` (round 90) for the activation-level test

This is a **bench-only** round, no new core API. It directly tests a 4-round arc: round 80 (orth), round 83 (λ=0.001 safe), round 85 (auto-rescale), round 95 (diversity diagnostic).

## 4. Expected outcomes

| dataset    | FAME baseline div | FAME+orth div | Δ |
|------------|-------------------|---------------|---|
| toy_sin    | 1.32              | 1.30-1.45?    | ? |
| structured | 1.15              | 1.15-1.30?    | ? |
| random     | 1.31              | 1.30-1.45?    | ? |

If H1 ✓: orthogonality at λ=0.001 measurably increases expert diversity.
If H1 ✗: orth at λ=0.001 is just a stylistic tax with no diversity benefit — round 80's mechanism is too weak.

## 5. Why this matters

- Round 80's orthogonality was added as a **defensive** measure: "even if routing collapses, expert projections remain decorrelated". This claim has never been measured.
- If H1 ✗, the orth constraint is a stylistic tax (round 90 already showed λ=10 is a stylistic tax). At λ=0.001 it might just be too weak to matter.
- If H1 ✓, the orth constraint is the first **measured** mechanism to increase diversity in the FAME stack.

## 6. Files

- `docs/prds/2026-06-15-lnn-round-96-a-fame-orthogonality-test.md` (this file)
- `scripts/bench_fame_orth_diversity.py` (NEW)
- `results/bench_fame_orth_diversity.json`
- `docs/research/2026-06-15_fame_orth_diversity_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v22.md`
- (No new tests — uses existing 27 effective_rank + round 80 orth tests)

## 7. Risk

Low. This is a bench-only round, no new API. The existing round 83 orth bench is at safe λ=0.001; the diversity measurement is a single function call.

## 8. Compatibility

Reuses round 95's `expert_diversity_summary` and round 90's `activation_space_overlap`. No Pyright warnings expected beyond the pre-existing torch-import false-positives.
