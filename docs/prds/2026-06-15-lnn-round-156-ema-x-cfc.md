# PRD #10-118 — Round 156: EMA-X-CfC (Input EMA Augmentation)

**Date**: 2026-06-15
**Round**: 156
**Audit context**: 91-155 audit, 16 strictly positive + 11
target-dep + 24 negatives = 51 mechanism classes.

## Hypothesis

The CfC cell receives x_t at each step, but does NOT explicitly
maintain a smooth / low-pass version of x. In real time-series
data, raw x often contains high-frequency noise that CfC has to
filter implicitly.

By maintaining an **Exponential Moving Average (EMA) of x** as
a separate state and providing it to CfC, we give the model
explicit access to a smoothed version of the input::

    ema_t = β · ema_{t-1} + (1 - β) · x_t    (β ∈ [0, 1])
    aug_x_t = concat([x_t, ema_t])             # H + D input

The EMA is a low-pass filter on x that:
- Smooths out high-frequency noise
- Provides long-range context (β ≈ 1 → very smooth)
- Decouples "current observation" from "smooth trend"

This is **structurally different from**:
- **TCC 149 / MSDC 151**: parallel conv context (one-shot,
  not stateful).
- **QuITE 102**: N learnable queries + masked attention
  (different attention mechanism).
- **DiffCfC 145**: input deltas (high-pass, not low-pass).
- **DELTA-CfC 155**: hidden state deltas (different signal).
- **FiLM 153**: γ, β modulation (no temporal state).

## Mechanism

```
Initialize: ema_0 = x_0
At step t:
  ema_t   = β · ema_{t-1} + (1 - β) · x_t
  aug_x_t = f_concat(x_t, ema_t)  # 4 variants
```

Variants:
1. **ema_concat**: aug_x = [x_t, ema_t], input dim = 2D.
2. **ema_gate**: aug_x = α · x_t + (1-α) · ema_t, input dim = D.
3. **ema_diff**: aug_x = [x_t, ema_t - x_t] (residual).
4. **ema_ema_only**: aug_x = ema_t only (control, replace x).

## Bench plan

30 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs:

| cond            | mechanism              | input dim |
|-----------------|------------------------|-----------|
| cfc              | baseline               | D         |
| ema_concat       | [x, ema]               | 2D        |
| ema_gate         | α·x + (1-α)·ema        | D         |
| ema_diff         | [x, ema - x]           | 2D        |
| ema_ema_only     | ema only (control)     | D         |

3 datasets: sin_irr, structured_irr, random_irr.

β is a fixed hyperparameter (start with β=0.9, ablate later).

## Expected outcomes

- **If EMA helps**: STRICTLY POSITIVE (17th winner).
- **If EMA hurts**: NEGATIVE (25th).
- **If mixed**: TARGET-DEPENDENT (12th).

## Risk

EMA has an inherent lag — it's "behind" x_t. On data with fast
regime changes (structured), this lag could hurt performance.
On smooth periodic data (sin), the lag is irrelevant.

## Files to create

- `lnn/core/ema_x_cfc.py` (~220 lines)
- `tests/test_ema_x_cfc.py` (~25 tests)
- `scripts/bench_ema_x_cfc.py` (~250 lines, 30-cell bench)
- `results/bench_ema_x_cfc.json`
- `docs/research/2026-06-15_ema_x_cfc_report.md`
