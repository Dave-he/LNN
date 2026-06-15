# PRD #10-133 — LearnedPerScaleBeta-CfC (Round 171, 2026-06-15)

## Goal
Test if data-driven per-scale learnable β (gradient-trained scalar
β per scale) beats round 169's hand-tuned β ∈ {0.75, 0.85, 0.95}.

## Mechanism
Each layer has K scalar β values (one per scale, not per-feature).
Initialized at chosen init (0.5/0.75/0.9), trained via gradient
descent with same Adam lr as rest of model.

## Result
**STRICTLY POSITIVE — TWO new bests**:
1. **lb_ps_h2_75** (Kh=2, init β=0.75): sin 0.0064 (-76%, 1pp over
   round 169's -72%), 16,934 params (13% smaller)
2. **lb_ps_h5_75** (Kh=5, init β=0.75): structured 0.0095 (-92%,
   1pp over round 165's -91%)

## Audit context (91-170)
42 strictly positive + 17 target-dep + 35 negatives = 94
mechanism classes. Round 171 adds 1 strictly positive = **95
mechanism classes total**.

## Files
- `lnn/core/learned_beta_ps_cfc.py` (~280 lines, new core class)
- `tests/test_learned_beta_ps_cfc.py` (16 tests, all pass)
- `scripts/bench_learned_beta_ps_cfc.py` (36-cell bench)
- `results/bench_learned_beta_ps_cfc.json`
- `docs/research/2026-06-15_learned_beta_ps_cfc_report.md`
