---
title: "Round 293 — Decorrelation as Default in BlendGated Cell (HONEST NEGATIVE — reverted to opt-in)"
date: 2026-07-12
round: 293
prd: "docs/prds/2026-07-12-lnn-round-293-decorr-default-a.md"
paper: "arXiv:2607.01986 + arXiv:2604.24788"
status: "FAIL — in-cell default at λ=1e-4 regresses Henry Hub +5% / +5%; reverted to opt-in"
parent: "r291 toy SP + r292 Henry Hub opt-in SP"
---

# Round 293 — Decorrelation as Default in BlendGated Cell

## TL;DR

Tried to promote the r291+r292 finding (`state_decorrelation_loss(λ=1e-4)` is SP) from opt-in to default in `BlendGatedLiquidTauCfCCell.extra_loss()`. **Result: HONEST NEGATIVE — the in-cell default at λ=1e-4 regresses Henry Hub by +4.9% / +5.2%, the opposite of the r292 opt-in result (-0.3% / -1.0%).**

**The default-promotion was reverted**: `BlendGatedLiquidTauCfCCell.decorr_lambda` defaults to **0.0 (opt-in)**. Users who want the r292 benefit must explicitly pass `decorr_lambda=1e-4` and use the opt-in bench pattern (a fresh forward inside `extra_loss(x)`).

**Why the discrepancy:** r292's bench wrapper called `extra_loss(x)` which did a *second forward* through the cell to compute decorrelation on a *separate* computation graph. The in-cell default uses the *cached output* from the *first* (task-loss) forward, so the decorrelation gradient is *added to* the task-loss gradient. The two paths have different optimizer dynamics.

## Results (Henry Hub, 30 epochs, 2 seeds)

| mode                | overall MSE | hi_vol MSE |
|---------------------|------------:|-----------:|
| static_tau          | 3.145       | 312.7      |
| blend_old (r280)    | 2.632       | 266.8      |
| blend_new_off (λ=0) | 2.632       | 266.8      |
| **blend_new_default (λ=1e-4 in-cell)** | **2.761 (+4.9%)** | **280.7 (+5.2%)** |

Δ% vs blend_old:
- blend_new_off: 0.0% / 0.0% (clean superset — opt-out works)
- blend_new_default: +4.9% / +5.2% (H8 FAIL)

## Why the in-cell default fails

The r292 bench's `extra_loss(x)` pattern:
```python
def extra_loss(self, x):
    out, _ = self.cell(x)           # FRESH forward → separate graph
    dec = state_decorrelation_loss(out, lambda_coeff=self.decorr_lambda)
    return dec
```

This creates a *second forward* through the cell, so the decorrelation
gradient flows through a separate copy of the cell's parameters.
Combined with the task loss:
```
total_loss = task_loss(out_task, target) + decor_loss(out_decor)
```
Both gradients contribute to the optimizer, but on *independent graphs*.

The r293 in-cell default:
```python
def extra_loss(self):
    dec = state_decorrelation_loss(self._last_outputs, ...)
    return dec
```

This uses the cached output from the task-loss forward. So:
```
total_loss = task_loss(out, target) + decor_loss(out)    # SAME `out`
```
The decorrelation gradient is added to the task-loss gradient on the
*same graph*. The combined gradient may push parameters in a slightly
different direction than the opt-in's separate-graph path.

The empirical effect on Henry Hub: +4.9% / +5.2% regression vs blend_old.

## Files (Round 293)

- `lnn/core/blend_gated_liquid_tau_cfc.py` (EDITED):
  - Added `decorr_lambda` constructor arg (default **0.0**, opt-in).
  - Added `_last_outputs` cache and `extra_loss()` override that
    returns entropy + decorrelation when `decorr_lambda > 0`.
  - Cache is NOT detached (so gradient flows back to cell params when
    user opts in).
- `tests/test_r293_decorr_default.py` (NEW, 5 tests, all green).
- `scripts/bench_henry_hub_default_decorr.py` (NEW, ~270 LOC): 4 modes ×
  2 seeds × 30 epochs, 8 cells total.

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   72   |   72  | 0 |
| Target-dep    |   36   |   36  | 0 |
| Negatives     |   63   |   64  | **+1** |
| **Total**     |  170   |  171 | +1 |

r293 adds **+1 NEGATIVE** — the default-promotion attempt failed.
The opt-in path (r291+r292) is still SP; the in-cell default is not.

## Recommendation

**Keep `BlendGatedLiquidTauCfCCell.decorr_lambda=0.0` as default.**
Users wanting the r292 benefit should:
1. Pass `decorr_lambda=1e-4` to the cell.
2. Use the **opt-in bench pattern** that does a fresh forward inside
   `extra_loss(x)`, not the in-cell default:

```python
def extra_loss(self, x):
    out, _ = self.cell(x)  # FRESH forward (separate graph)
    ent = self.cell.extra_loss() if self.entropy_lambda > 0 else torch.tensor(0.0)
    dec = state_decorrelation_loss(out, lambda_coeff=self.decorr_lambda)
    return ent + dec
```

This is exactly the r292 pattern and produces the SP result.

## Decision for r294

Two options:
1. **Document and move on.** r293 is a clear finding; r292 is the
   working SP path. Document the discrepancy and pivot.
2. **Investigate further.** Try the in-cell default at λ=1e-5 (matching
   r291's toy SP) on Henry Hub to see if the smaller λ avoids the
   regression.

Top recommendation: **r294 = option 1** — commit r293 as a clear
finding and pivot to a different mechanism (e.g. arXiv:2606.21295
neuron-wise topological dynamics, or r99 segment reliability gate on
irregular TS).

## Citation

- Nie, W., Wang, W., Su, Y. (2026-07). *Liquid Latent State Dynamics*.
  arXiv:2607.01986.
- Liu, Y., et al. (2026-04). *LNN for Natural Gas*. arXiv:2604.24788.
- r291 decorrelation toy bench: `docs/research/2026-07-12_round291_noisy_structured_bench_report.md`
- r292 decorrelation Henry Hub: `docs/research/2026-07-12_round292_henry_hub_decorr_report.md`