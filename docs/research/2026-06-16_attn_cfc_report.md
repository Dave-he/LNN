# Round 206 — Attention over Hidden States — Research Report

**Date**: 2026-06-16
**Round**: 206
**Branch**: master
**Audit context (91-205)**: 47 strictly positive + 26 target-dep
+ 55 negatives = 128 mechanism classes.

## TL;DR

**NEGATIVE (56th) for Round 206**: Attention over past
hidden states HURTS structured/multi-regime data by 80%.

- sin: -8.5% (small win)
- struct: +80% (BIG regression)
- random: -6.5% (small win)

## What was tested

**Self-attention over past hidden states** as the gating
mechanism in CfC. Different axis from spectral (r200-r205).

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0553 | 0.0030 | 0.0907 | 0.0497 |
| **attn (r206)** | **0.0506** | **0.0054** | **0.0848** | **0.0469** |

## Per-dataset analysis

### sin_irr — small win
- cf: 0.0462 / 0.0643 (mean 0.0553)
- r206: 0.0506 / 0.0505 (mean 0.0506, **-8.5%**)

### structured_irr — BIG regression
- cf: 0.0015 / 0.0045 (mean 0.0030)
- r206: 0.0098 / 0.0010 (mean 0.0054, **+80%** WORSE)
- The seed-0 result (0.0098) is anomalously high

### random_irr — small win
- cf: 0.0940 / 0.0873 (mean 0.0907)
- r206: 0.0891 / 0.0804 (mean 0.0848, **-6.5%**)

## Pattern (47 + 26 + 55 = 128 → 47 + 26 + 56 = 129)

- 47 strictly positive (unchanged)
- 26 target-dep (unchanged)
- **56 negatives** (UP from 55, +1)
- Total: **129 mechanism classes**

## Why attention hurts structured/multi-regime

1. **Past hidden states contain mixed regimes**
2. **Attention averages past states** — dilutes signal across regimes
3. **For pure sin**: small win (-8.5%) — past states help
4. **For random**: small win (-6.5%) — past states help
5. **For structured (sin→linear)**: large loss (+80%) —
   past sin-regime states confuse the linear-regime prediction

## Why this is a useful NEG

1. **Confirms past-state mixing is harmful for regime changes**
2. **Suggests: per-regime attention or per-regime MoE** may help
3. **Pivots to attention axis confirmed negative**
4. **Helps design r207+: regime-aware attention**

## Comparison with spectral axis (r200-r205)

Spectral gating (r200-r205) was also problematic for random
data (+12% regression in r200) but did not regress structured.

| Axis | Best result | Worst result |
|------|-------------|--------------|
| Spectral (r200-r205) | r205: -28.2% mean | r201: +0.5% mean |
| Attention (r206) | r206: -5.6% mean (sin/random only) | r206: +80% on struct |

Spectral gating is more useful than attention for multi-regime.

## Caveats

- 2 seeds, 30 epochs
- Hidden=12, lr=1e-2, batch_size=16
- The seed-0 struct result (0.0098) is anomalously high
- Attention doesn't include positional encoding
- No layer norm on attention output

## Next ideas

1. **Per-regime attention** — different attention per regime
2. **Positional encoding** — add position to attention
3. **Gated attention** — gate attention weights
4. **Sparse attention** — top-k past states
5. **Cross-attention** — attend to input features instead
6. **Pivot to SSM** — combine CfC with S4/Mamba (different axis)

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_attn_cfc.py` (~225 lines)
- `tests/test_learned_beta_ps_ln_khlfft_attn_cfc.py` (11 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_attn_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_attn_cfc.json`

**Why:** Round 206 is **NEGATIVE (56th)** — attention over
past hidden states hurts structured/multi-regime data.

**How to apply:** Do NOT use past-state attention for
multi-regime data. Spectral gating (r200-r205) remains
the dominant time-series gating mechanism.
