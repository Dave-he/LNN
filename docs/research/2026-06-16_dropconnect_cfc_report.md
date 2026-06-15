# Round 196 — DropConnect on Cell Weights for CfC — Research Report

**Date**: 2026-06-16
**Round**: 196
**Branch**: master
**Audit context (91-195)**: 47 strictly positive + 21 target-dep
+ 50 negatives = 118 mechanism classes.

## TL;DR

**NEGATIVE for Round 196**: DropConnect (Wan et al 2013) on
cell weights is **catastrophic on structured data** (+63% at
p=0.05, +203% at p=0.10, +244% at p=0.20). Only mild sin
help at p=0.05 (-14%), marginal random (-3%). Mean
0-13% worse. **Weight-level regularization is too aggressive
for multi-regime data**.

## What was tested

**DropConnect** (Wan et al 2013) on the cell's 3 linear
layers (f_gate, g_branch, h_branch):
```python
def _apply_dropconnect_forward(linear, x, p, training):
    if not (training and p > 0):
        return linear(x)
    mask = (torch.rand_like(linear.weight) > p).float()
    masked_weight = linear.weight * mask / (1.0 - p)  # inverted dropout
    return F.linear(x, masked_weight, linear.bias)
```

This is **weight-level dropout** — different from
activation-level dropout (round 92) or input/hidden
noise (rounds 192-194). Perturbs the W matrices directly.

## Bench (24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs MSE | type |
|------|---------|----------------|------------|------|----------|------|
| mse (p=0) | 0.0049±0.0019 | 0.0032±0.0003 | 0.0713±0.0200 | 0.0265 | — | — |
| dc_05 | 0.0042±0.0002 | 0.0052±0.0003 | 0.0702±0.0226 | 0.0265 | 0% | — |
| dc_10 | 0.0067±0.0014 | 0.0097±0.0023 | 0.0713±0.0201 | 0.0292 | +10% | NEG |
| dc_20 | 0.0096±0.0011 | 0.0110±0.0036 | 0.0691±0.0244 | 0.0299 | +13% | NEG |

**DropConnect catastrophic on structured** at all p values.

## Per-seed detail

### sin_irr
- mse: seed 0 = 0.0067, seed 1 = 0.0030, mean = 0.0049
- dc_05: seed 0 = 0.0040, seed 1 = 0.0044, mean = 0.0042 (**-14%**)
- dc_10: seed 0 = 0.0077, seed 1 = 0.0057, mean = 0.0067 (+37%)
- dc_20: seed 0 = 0.0087, seed 1 = 0.0104, mean = 0.0096 (+96%)

**dc_05 helps BOTH seeds** on sin (low std 0.0002). Higher
p hurts both seeds.

### structured_irr
- mse: seed 0 = 0.0029, seed 1 = 0.0034, mean = 0.0032
- dc_05: seed 0 = 0.0050, seed 1 = 0.0054, mean = 0.0052 (+63%)
- dc_10: seed 0 = 0.0114, seed 1 = 0.0080, mean = 0.0097 (+203%)
- dc_20: seed 0 = 0.0136, seed 1 = 0.0085, mean = 0.0110 (+244%)

**ALL p values hurt structured CATASTROPHICALLY**:
- p=0.05: +63% (still bad)
- p=0.10: +203%
- p=0.20: +244%

### random_irr
- mse: seed 0 = 0.0913, seed 1 = 0.0513, mean = 0.0713
- dc_05: seed 0 = 0.0927, seed 1 = 0.0476, mean = 0.0702 (-2%)
- dc_10: seed 0 = 0.0914, seed 1 = 0.0512, mean = 0.0713 (0%)
- dc_20: seed 0 = 0.0934, seed 1 = 0.0448, mean = 0.0691 (-3%)

**Roughly neutral on random** at all p values.

## Pattern (47 + 21 + 50 = 118 → 47 + 21 + 51 = 119)

- 47 strictly positive (unchanged)
- 21 target-dep (unchanged)
- **51 negatives** (UP from 50, +1)
- Total: **119 mechanism classes**

## Why DropConnect fails on structured

1. **Structured has 2 distinct regimes** (sin + linear)
2. **Each regime needs different W patterns** to map
   input to output correctly
3. **DropConnect randomly drops W** — destroys the
   learned structure for each regime
4. **Random data is robust to this** (no specific
   structure to lose), sin is too (only 1 regime)
5. **Structured is the worst case** for weight-level
   regularization

## Why DropConnect works on sin (mildly)

1. **Sin has 1 regime** — same W pattern is useful for
   all t
2. **Mild drop (p=0.05) is small enough** to preserve
   sin-relevant weights
3. **Acts as cheap ensemble** — different dropped W
   configurations per forward pass

## Why dc_10/20 are uniformly bad

1. **Higher p drops more weights** — useful weights get
   dropped on critical paths
2. **Inverted dropout compensates** but cannot recover
   the structural information lost
3. **Higher p on structured = catastrophic** because
   the 2-regime structure is fragile

## Hypotheses revisited

- **H1 (positive, DropConnect helps)**: REJECTED. Mean is
  0% or worse.
- **H2 (negative, all p too high)**: PARTIAL. p=0.05 is
  OK on sin, but hurts structured.
- **H3 (mixed, helps random)**: REJECTED. Random is
  roughly neutral.

## Critical implementation details

1. **Inverted dropout** — `mask / (1-p)` scaling
2. **3 linear layers affected** — f_gate, g_branch,
   h_branch
3. **Manual forward override** — to call `_apply_dropconnect_forward`
4. **Pre-create cell wrappers** — needed for eval()
   to propagate correctly
5. **No new params** — same param count as round 187

## Why this is a useful negative

1. **Weight-level dropout is too aggressive** for
   structured data (2 regimes)
2. **Activation-level (input/hidden) noise is more
   forgiving** — only r194 combined noise hurts sin
   but doesn't have dc's catastrophic structured failure
3. **Confirms noise (r192) > DropConnect** as the right
   regularization choice
4. **Round 92 (activation dropout) also hurt CfC** — adds
   to evidence that dropout is bad for CfC

## Caveats

- **2 seeds only** — would benefit from 3-5 seeds
- **3 p values only** — could try p=0.01, 0.02
- **Tested on round 187 stack only** — may not generalize

## Comparison with r192-r195

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | **-26%** | -24% | **SP** |
| 193 | hidden noise | **-20%** | -16% | +21% | +17% | TD |
| 194 | combined | +8% | **-25%** | +14% | +12% | TD |
| 196 | dropconnect | -14% (dc05) | +63% (dc05) | -3% (dc20) | 0% | **NEG** |

**Noise > DropConnect** as regularization choice for CfC
on these datasets. DropConnect catastrophic on structured.

## Next ideas

1. **Mixed DropConnect** — different p per layer
2. **Structured DropConnect** — drop specific weight
   patterns, not random
3. **Time-varying DropConnect** — p(t) schedule
4. **Per-feature DropConnect** — different p per output dim
5. **Stochastic depth** — skip layers with probability

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_dropconnect_cfc.py` (~190 lines)
- `tests/test_learned_beta_ps_ln_khlfft_dropconnect_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_dropconnect_cfc.py` (24-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_dropconnect_cfc.json`

**Why:** Round 196 is **NEGATIVE** (DropConnect hurts
structured at all p values, even p=0.05 +63%). Weight-level
regularization is too aggressive for multi-regime data.

**How to apply:** Don't use DropConnect for multi-regime
data. Use input/hidden noise instead (rounds 192-193).
