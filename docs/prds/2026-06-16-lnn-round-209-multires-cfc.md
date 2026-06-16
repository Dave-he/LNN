# PRD #10-171 — Round 209 — Multi-Resolution Spectral Gating on CfC

**Date**: 2026-06-16
**Round**: 209
**Branch**: master
**Audit context (91-208)**: 47 strictly positive + 26 target-dep
+ 58 negatives = 131 mechanism classes.

## Background

r200-r205 explored single-resolution spectral gating. After
attention (r206) and SSM (r207-r208) failed, return to
spectral with **multi-resolution** extension (Sonnet 2026
inspiration).

Hypothesis: applying spectral gating at multiple FFT scales
captures both fine and coarse structure.

## Goal

Test if multi-resolution spectral gating (2 scales) provides
better or different results than r200's single-resolution.

## Mechanism

```python
# Res 1: full FFT
H1 = FFT(h_t)
mask1 = sigmoid(linear(|H1|))
g1 = IFFT(H1 * mask1, n=hidden_size)

# Res 2: half FFT (truncate)
H2 = H1[:, :hidden_size//4+1]  # half of rfft size
mask2 = sigmoid(linear(|H2|))
H2_full = pad(H2 * mask2, (0, hidden_size//2))
g2 = IFFT(H2_full, n=hidden_size)

g_combined = (g1 + g2) / 2
h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (2 conds)

1. `cf`: r187 baseline
2. `multires`: r209 (2-scale spectral)

## Result (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0577 | 0.0041 | 0.0886 | 0.0501 |
| **multires (r209)** | **0.0390** | **0.0049** | **0.0836** | **0.0425** |

Per-dataset (r209 vs cf):
- sin: -32.4% ✓
- structured: +19.5% ✗
- random: -5.6% ✓
- mean: -6.2%

## Verdict

**TARGET-DEPENDENT (27th)** — sin and random improve,
struct regresses.

## Pattern (47 + 26 + 58 = 131 → 47 + 27 + 58 = 132)

- 47 strictly positive (unchanged)
- **27 target-dep** (UP from 26, +1)
- 58 negatives (unchanged)
- Total: **132 mechanism classes**

## Why multi-resolution is TD, not SP

1. **sin wins big** (-32.4%) — multi-scale captures fine +
   coarse spectral structure
2. **random wins small** (-5.6%) — coarse scale helps denoise
3. **struct regresses** (+19.5%) — coarse scale interferes
   with regime-switching

## Comparison r200 vs r209

| Round | Mechanism | sin | struct | random | mean |
|-------|-----------|-----|--------|--------|------|
| 200 | spec single | -34.6% | 0% | +12.2% | -2.5% |
| **209** | **multires** | **-32.4%** | **+19.5%** | **-5.6%** | **-6.2%** |

Multi-resolution has **much better random** result.

## Caveats

- 2 seeds, 30 epochs
- Hidden=12, lr=1e-2, batch_size=16
- Struct result has high variance (seed 0 0.0089, seed 1 0.0008)

## Next ideas

1. **Three or more scales** — try 3-scale spectral
2. **Adaptive scale weighting** — learn per-scale weights
3. **Per-regime spectral** — different gating per regime
4. **Move away from spectral** — 7+ rounds on spectral done

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_multiresgated_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_khlfft_multiresgated_cfc.py` (10 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_multiresgated_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_multiresgated_cfc.json`

**Why:** Round 209 is **TARGET-DEPENDENT (27th)** — sin/random
improve, struct regresses unevenly.

**How to apply:** Use multi-resolution spectral gating for
sin-like or random data, NOT for structured/multi-regime.
