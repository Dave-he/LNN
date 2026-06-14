# PRD #10-59 — Weight-Level Orthogonality Penalty (Round 97)

**Date**: 2026-06-15
**Round**: 97 (direct follow-up to round 96)
**Status**: Drafted.

## 1. Why round 97

Round 96 (PRD #10-58) showed that the existing `orthogonality_loss` decorrelates expert **activations** (hidden states) but does NOT increase expert **weight** diversity. The activation-vs-weight distinction is the headline finding.

**Question for round 97**: if we add a penalty on the **Gram matrix of weight matrices** — `||W_i W_j^T||_F^2` for i ≠ j — does THAT boost weight diversity?

This would close the diversity story:
- Round 80 (activation orth): decorrelates hidden states (works)
- Round 97 (weight orth): decorrelates weight matrices (to be tested)

## 2. Hypotheses

- **H1 (weight orth increases weight diversity)**: FAME+weight_orth(λ=0.001) trained 100 epochs has diversity_ratio ≥ 1.10× FAME-baseline (i.e. 1.32 → 1.45+).
- **H2 (weight orth is safe for task loss)**: FAME+weight_orth task loss within ±10% of baseline (round 83's "safe λ" criterion).
- **H3 (weight orth also decorrelates activations)**: FAME+weight_orth activation cos_sim ≤ baseline (because decorrelated weights produce different hidden states).

## 3. Plan

### 3.1 Implementation (`lnn/core/orthogonality.py`)

Add 1 new function:
- `weight_orthogonality_loss(W_list, lambda_coeff, eps=1e-8)` — penalty `λ * Σ_{i<j} ||W_i W_j^T||_F^2 / (||W_i||_F · ||W_j||_F)`.  We normalize by the product of Frobenius norms to make the penalty dimensionless and bounded.

For K=5 experts with hidden=8, the weight matrices are 2D tensors of varying shapes (input→hidden, hidden→hidden, etc.). The penalty operates on a **list of 2D weight matrices**, not a list of experts. The user (or a helper) decides which matrices to compare.

### 3.2 New helper in FAME cell

Add `compute_weight_orth_loss(lambda_coeff=0.001)` to FAMECfCCell:
- Collect 2D weight matrices from each expert (round 95's pattern)
- For each (expert i, expert j) pair with i<j, take the **first** 2D matrix (a stable choice) and compute `||W_i W_j^T||_F^2 / (||W_i||_F · ||W_j||_F)`
- Sum and scale by λ

### 3.3 Tests (`tests/test_orthogonality.py`) — was 12, +5 new = 17

5 new tests:
1. `test_weight_orth_zero_for_fewer_than_two_matrices` — K<2 → 0
2. `test_weight_orth_zero_when_lambda_0` — λ=0 → 0
3. `test_weight_orth_zero_for_orthogonal_matrices` — `W_i ⊥ W_j` → ~0
4. `test_weight_orth_high_for_identical_matrices` — `W_i = W_j` → 1
5. `test_weight_orth_gradient_flows` — autograd check

### 3.4 Bench (`scripts/bench_fame_weight_orth_diversity.py`)

36 cells:
- 3 datasets × 4 conditions × 3 seeds
- Conditions: baseline, +activation_orth, +weight_orth, +both
- 100 epochs

For each cell measure: diversity_ratio, mean_eff_rank, task_loss, activation_cos_sim.

## 4. Why this matters

- The 6-round audit (rounds 91-96) showed weight diversity is a separate property from activation diversity.
- Round 80 added activation orth; it works.
- Round 97 adds weight orth; this completes the orth toolkit.
- If H1 ✓: weight orthogonality is the first **measured** mechanism to boost weight diversity in our stack.
- If H1 ✗: weight orthogonality at λ=0.001 is also a stylistic tax.

## 5. Files

- `lnn/core/orthogonality.py` — add `weight_orthogonality_loss` (1 new function)
- `lnn/core/fame_cfc.py` — add `compute_weight_orth_loss` (1 new method, ~20 lines)
- `lnn/core/__init__.py` — export new function
- `tests/test_orthogonality.py` — add 5 tests (12 → 17)
- `scripts/bench_fame_weight_orth_diversity.py` (NEW) — 36-cell bench
- `results/bench_fame_weight_orth_diversity.json`
- `docs/research/2026-06-15_fame_weight_orth_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v23.md`
- `README.md` — new section

## 6. Risk

Low. The new function is a thin wrapper. The penalty is normalized so it doesn't blow up with random init. The bench reuses the round 95/96 infrastructure.

## 7. Compatibility

- `weight_orthogonality_loss(W_list, lambda)` follows the same signature as `orthogonality_loss(expert_outputs, lambda)`.
- Returns a 0-d tensor so it composes via simple addition.
- FAME cell's `compute_weight_orth_loss` mirrors the existing `compute_orth_loss` API.
- No Pyright warnings expected beyond pre-existing torch-import false-positives.
