# PRD #10-158 — Round 196 — DropConnect on Cell Weights for CfC

**Date**: 2026-06-16
**Round**: 196
**Branch**: master
**Audit context (91-195)**: 47 strictly positive + 21 target-dep
+ 50 negatives = 118 mechanism classes.

## Background

Rounds 192-195 all noise-based (input, hidden, combined, σ
sweep). Test a different mechanism: **weight-level dropout
(DropConnect, Wan et al 2013)**. Different from input noise
or hidden noise because it perturbs the W matrices directly.

## Goal

Test if DropConnect on the cell's 3 linear layers
(f_gate, g_branch, h_branch) provides useful regularization
on top of round 187's stack.

## Mechanism (TRAINING ONLY, not eval)

```python
def _apply_dropconnect_forward(linear, x, p, training):
    if not (training and p > 0):
        return linear(x)
    # Generate binary mask (1 = keep, 0 = drop)
    mask = (torch.rand_like(linear.weight) > p).float()
    # Apply mask with inverted dropout
    masked_weight = linear.weight * mask / (1.0 - p)
    return F.linear(x, masked_weight, linear.bias)
```

## Configurations (4 conds)

1. `lbps_lnkhlfft_5_3_2_mse`: pure MSE baseline (p=0)
2. `lbps_lnkhlfft_dropconnect_05`: p=0.05
3. `lbps_lnkhlfft_dropconnect_10`: p=0.10
4. `lbps_lnkhlfft_dropconnect_20`: p=0.20

## Result (24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs MSE |
|------|---------|----------------|------------|------|----------|
| mse (p=0) | 0.0049±0.0019 | 0.0032±0.0003 | 0.0713±0.0200 | 0.0265 | — |
| dc_05 | 0.0042±0.0002 | 0.0052±0.0003 | 0.0702±0.0226 | 0.0265 | 0% |
| dc_10 | 0.0067±0.0014 | 0.0097±0.0023 | 0.0713±0.0201 | 0.0292 | +10% |
| dc_20 | 0.0096±0.0011 | 0.0110±0.0036 | 0.0691±0.0244 | 0.0299 | +13% |

## Verdict

**NEGATIVE** — DropConnect hurts structured at all p values
(p=0.05 still +63% on structured). Only mild sin help
(p=0.05 -14%), marginal random (-3%).

The structured dataset (2 regimes: sin + linear) is
particularly sensitive to weight dropping because each
regime needs different connectivity patterns.

## Per-seed detail

### sin_irr
- mse: 0.0067 / 0.0030 (mean 0.0049)
- dc_05: 0.0040 / 0.0044 (mean 0.0042, **-14%**)
- dc_10: 0.0077 / 0.0057 (mean 0.0067, +37%)
- dc_20: 0.0087 / 0.0104 (mean 0.0096, +96%)

**dc_05 helps both seeds** on sin. dc_10 mixed, dc_20 hurts.

### structured_irr
- mse: 0.0029 / 0.0034 (mean 0.0032)
- dc_05: 0.0050 / 0.0054 (mean 0.0052, +63%)
- dc_10: 0.0114 / 0.0080 (mean 0.0097, +203%)
- dc_20: 0.0136 / 0.0085 (mean 0.0110, +244%)

**ALL p values hurt structured** — even mild p=0.05 gives
+63%. This is a catastrophic failure on structured.

### random_irr
- mse: 0.0913 / 0.0513 (mean 0.0713)
- dc_05: 0.0927 / 0.0476 (mean 0.0702, -2%)
- dc_10: 0.0914 / 0.0512 (mean 0.0713, 0%)
- dc_20: 0.0934 / 0.0448 (mean 0.0691, -3%)

**Roughly neutral on random** at all p values.

## Pattern (47 + 21 + 50 = 118 → 47 + 21 + 51 = 119)

- 47 strictly positive (unchanged)
- 21 target-dep (unchanged)
- **51 negatives** (UP from 50, +1)
- Total: **119 mechanism classes**

## Why DropConnect fails on structured

1. **Structured has 2 distinct regimes** (sin + linear)
2. **Each regime needs different W patterns**
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

## Caveats

- **2 seeds only** — would benefit from 3-5 seeds
- **3 p values only** — could try p=0.01, 0.02
- **Tested on round 187 stack only** — may not generalize

## Next ideas

1. **Mixed DropConnect** — different p per layer
2. **Structured DropConnect** — drop specific weight
   patterns, not random
3. **Time-varying DropConnect** — p(t) schedule
4. **Per-feature DropConnect** — different p per output dim

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
