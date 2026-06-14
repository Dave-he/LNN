# PRD #10-60 — Backward Coherence Regularization (Round 98)

**Date**: 2026-06-15
**Round**: 98 (response to arXiv:2606.08934)
**Status**: Drafted.

## 1. Why round 98

arXiv:2606.08934 (Yuan-chin Ivan Chang, June 2026) — *Backward Coherence and Hidden-State Stability in Recurrent Neural Networks: A Quasi-Reverse-Martingale Theory* — proposes a **backward-coherence regularization** for RNN hidden states.

The core idea: an RNN's hidden state `h_t` should be **backward-coherent**, i.e. `h_t ≈ h_{t+1}` in expectation (a quasi-reverse-martingale). The paper validates this on PhysioNet 2012 ICU, FRED-MD, and UCI HAR benchmarks with reportedly better stability and generalization.

**Question for round 98**: does backward coherence regularization improve our CfC stack?

This is a **new mechanism** in our stack, complementary to:
- Round 80 (activation orth): decorrelates expert hidden states
- Round 91 (smoothness): max_grad of hidden states
- Round 95 (per-expert diversity): hidden state distribution
- Round 97 (weight orth): weight-level regularization

## 2. Hypotheses

- **H1 (backward coherence reduces hidden state variance)**: with `λ=0.01`, the std of `(h_t - h_{t+1})` across the trajectory is < baseline.
- **H2 (backward coherence is safe for task loss)**: with `λ=0.001`, task loss within ±10% of baseline (the safe band from round 83).
- **H3 (backward coherence complements smoothness)**: backward-coherent models also have lower max_grad (round 91) — coherence is related to but distinct from smoothness.

## 3. Plan

### 3.1 Implementation (`lnn/core/smoothness_metrics.py`)

Add 1 new function:
- `backward_coherence_loss(states, lambda_coeff=0.001)` — penalty on the mean squared backward difference `λ * mean((h_t - h_{t+1})^2)`.

This is simpler than smoothness: instead of penalizing the magnitude of `dh/dt`, we penalize the **discreteness** of `h` (jumps between consecutive steps).

### 3.2 New helper in CfCCell (or just use directly)

The function takes a `(T, d)` trajectory and computes the loss. No new cell method needed — users can call it directly with the trajectory they collect.

### 3.3 Tests (`tests/test_smoothness_metrics.py`) — was 14, +5 new = 19

5 new tests:
1. `test_backward_coherence_zero_for_constant_trajectory` — h_t = constant → loss = 0
2. `test_backward_coherence_zero_when_lambda_0` — fast path
3. `test_backward_coherence_high_for_changing_trajectory` — h_t jumps → loss > 0
4. `test_backward_coherence_gradient_flows` — autograd check
5. `test_backward_coherence_exported` — module export

### 3.4 Bench (`scripts/bench_cfc_backward_coherence.py`)

12 cells:
- 3 datasets: toy_sin, structured, random
- 4 models: MLP, CfC, LSTM, GRU
- 1 condition: with backward coherence (λ=0.001) on the hidden state trajectory
- 3 seeds, 100 epochs

For each cell measure:
- `task_loss` (final MSE)
- `backward_diff_std` (mean std of `h_t - h_{t+1}` across the trajectory)
- `max_grad` (round 91 metric — for H3)
- `hidden_eff_rank` (round 94 metric — for completeness)

Compare to the round 91/94 baselines (no coherence penalty).

## 4. Expected outcomes

| model | baseline task_loss | +backward_coherence task_loss | baseline max_grad | +coherence max_grad |
|-------|--------------------|--------------------------------|--------------------|----------------------|
| MLP   | 0.17               | 0.18-0.22                      | 3.66              | 3.40-3.60            |
| CfC   | 0.26               | 0.20-0.25 (improvement?)       | 2.03              | 1.85-2.00           |
| LSTM  | 0.34               | 0.34                           | 52.79             | 50-55               |
| GRU   | 0.30               | 0.30                           | 37.98             | 35-40               |

H1 ✓ if std drops. H2 ✓ if task loss ±10%. H3 ✓ if max_grad drops.

## 5. Why this matters

- Backward coherence is a **new property** of the CfC stack not yet measured.
- It connects the round 91 smoothness story (which had 3 rejections in rounds 92-94) with hidden-state stability.
- If it works, it could be a way to **actually** improve task loss via regularization (unlike the orth penalties which were stylistic taxes).

## 6. Files

- `docs/prds/2026-06-15-lnn-round-98-a-backward-coherence.md` (this file)
- `lnn/core/smoothness_metrics.py` — add `backward_coherence_loss` (1 new function)
- `lnn/core/__init__.py` — export
- `tests/test_smoothness_metrics.py` — add 5 tests (14 → 19)
- `scripts/bench_cfc_backward_coherence.py` (NEW) — 12-cell bench
- `results/bench_cfc_backward_coherence.json`
- `docs/research/2026-06-15_cfc_backward_coherence_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v24.md`
- `README.md` — new section

## 7. Risk

Low. The new function is a simple penalty. The bench reuses the round 91/94 infrastructure.

## 8. Compatibility

- `backward_coherence_loss(states, lambda)` follows the same signature pattern as `orthogonality_loss`, `weight_orthogonality_loss`, `total_variation`, etc.
- Returns a 0-d tensor so it composes via simple addition.
- No Pyright warnings expected beyond pre-existing torch-import false-positives.
