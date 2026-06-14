# PRD #10-55 — Input-Side Temporal Dropout (Round 93)

**Date**: 2026-06-15 (round 93)
**Response to**: arXiv:2605.27467 (Thu/Oo/Supnithi, May 2026) — *Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility*
**Direct follow-up to**: PRD #10-54 (round 92, target-side dropout), backlog item #1

## 1. The gap in round 92

Round 92 used **target-side temporal dropout**: the model sees the full t grid, but the y target is zeroed out at random positions. The loss becomes:
```
L = mean over kept-i: (y_pred(t_i) - y_i)^2
```

This is "missing labels" semantics. But the **paper's claim** is about **missing input observations** (irregular sampling): the model *doesn't see* y at t_i at all. That's **input-side temporal dropout**:
```
L = mean over kept-i: (y_pred(t_i) - 0)^2  (no information at dropped positions)
```

For stateless models (MLP, CfC stateless), these are equivalent. **But for stateful models (LSTM, GRU), they are NOT equivalent**: input-side dropout changes the model's running state, target-side dropout does not.

## 2. Why this matters

Round 92's HONEST-NEGATIVE was that LSTM was more robust than CfC under target-side dropout (1.29x vs 2.06x degradation). The plausible explanation was "LSTM's gating+state provide a robustness mechanism separate from smoothness." But there's a simpler possibility we cannot rule out:

> **LSTM is just better at recovering from a zeroed target because it has a state to interpolate from, while CfC stateless has no information to interpolate.**

If input-side dropout tells a different story, then:
- LSTM might fail (state corrupted by zero inputs)
- CfC stateless might be unchanged (statelessness is no longer a disadvantage)
- The smoothness-prior hypothesis might be **rescued** for input-side dropout

## 3. Test design

### 3.1 Input-side vs target-side dropout (the key change)

For each training run, **drop the inputs** at random t positions:
- Input `(t_i, y_i)` is replaced with `(t_i, 0)` (zero-filled) at dropped positions
- Loss is computed only at kept positions
- Eval is on the original dense grid (no dropout)

Compared to round 92:
| | round 92 (target-side) | round 93 (input-side) |
|---|---|---|
| What gets masked | y in the loss | x in the input to the model |
| Stateless impact | same | same |
| Stateful impact | state unaffected | state **corrupted by zero inputs** |

### 3.2 Models (4 same as round 92)

- **MLP**: stateless
- **CfC stateless**: stateless (h=0 each t)
- **LSTM**: stateful, full unroll
- **GRU**: stateful, full unroll

### 3.3 Input-side dropout implementation

Two strategies:
1. **Zero-fill** (simple): replace (t_i, y_i) with (t_i, 0) at dropped positions
2. **Skip both input and target** (clean): drop (t_i, y_i) pair entirely, only train on kept positions

Strategy 1 is closer to the paper's clinical scenario. Use strategy 1.

### 3.4 Dropout p: 0%, 10%, 20%, 40%, 60%, 80% (6 levels, same as round 92)
### 3.5 Seeds: 3 per cell
### 3.6 Total cells: 4 × 6 × 3 = 72 (matches round 92)

## 4. Hypotheses

- **H1 (paper claim, rescuer)**: Under input-side dropout, CfC is more robust than LSTM (reverses round 92's verdict)
- **H2 (stateless recovery)**: CfC's degradation@0.8 is similar to round 92's (2.06x), since stateless models treat input/target dropout identically
- **H3 (LSTM collapse)**: LSTM's degradation@0.8 is much worse than round 92 (1.29x), since zeroed inputs corrupt its state
- **H4 (regularization)**: Under input-side dropout at small p (10-20%), models still IMPROVE due to regularization, like round 92

If **H1+H2+H3** ✓: round 92's negative result was an artifact of target-side dropout, paper's claim is rescued for input-side
If **H2 only** ✓: stateless models are indifferent to dropout direction, but LSTM/GRU behave similarly to round 92 → smoothness prior stays rejected
If **H2+H3** ✓: LSTM is significantly worse under input-side, but CfC doesn't beat it → mixed result

## 5. Implementation

### 5.1 New helper: `input_dropout`

Add to `lnn/core/temporal_dropout.py`:
```python
def input_dropout(t, y, p, seed=None):
    """Mask p fraction of (t, y) pairs by zeroing y BEFORE feeding to model.

    Same return signature as temporal_dropout: (t, y_masked).
    The difference is conceptual: caller passes y_masked as INPUT to the
    model, not as the loss target.
    """
    if p == 0: return t, y
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand(y.shape, generator=gen) > p
    else:
        mask = torch.rand_like(y) > p
    return t, y * mask.to(y.dtype)
```

This is functionally identical to `temporal_dropout` — the **distinction is in how the caller uses the result**. The key value is the *semantic naming*.

### 5.2 Bench: `scripts/bench_cfc_input_dropout.py`

Same 4 models, same 6 dropout p, same 3 seeds, same target function.
Difference: the loss is computed against `y_train` (unmasked) but the model input is `y_train_masked`.

## 6. Success criteria

- **STRONG POSITIVE** (H1+H2+H3 ✓): input-side dropout reverses round 92's verdict. CfC ≤ LSTM on degradation, smoothness prior is rescued.
- **PARTIAL** (H2 only ✓): stateless models don't care, stateful models behave like round 92. Smoothness prior stays rejected.
- **HONEST NEGATIVE** (H1 ✗): input-side dropout doesn't change the round 92 ranking. LSTM still wins. The paper's claim is firmly rejected in 1D.
- **MIXED** (H3 ✓ only): LSTM collapses, CfC doesn't beat MLP, but MLP wins. Different mechanism entirely.

## 7. Out of scope

- Real clinical data
- Variable dropout rates per region of the input (e.g., contiguous masks)
- Continuous-time ODE solvers

## 8. Deliverables

- `docs/prds/2026-06-15-lnn-round-93-a-input-side-temporal-dropout.md` (this file)
- `lnn/core/temporal_dropout.py` — add `input_dropout` helper
- `lnn/core/__init__.py` — export new helper
- `tests/test_temporal_dropout.py` — add 5 tests for `input_dropout` (18 total)
- `scripts/bench_cfc_input_dropout.py` — 72-cell bench
- `results/bench_cfc_input_dropout.json`
- `docs/research/2026-06-15_cfc_input_dropout_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v19.md`
- `README.md` — add "Input-Side Dropout" section
- Memory: `lnn-round-93-input-side-temporal-dropout.md`

## 9. Why this is round 93

1. **Backlog item #1** from round 92 (top of the list)
2. **Rescues the smoothness-prior hypothesis** if H1+H3 ✓ — round 92's negative may have been an artifact
3. **Small scope** — 1 helper function + 1 bench script + 5 tests, ~200 LOC
4. **Direct comparison to round 92** — same setup, just swaps which tensor is masked
5. **Honest-negative friendly** — even if H1 ✗, we close the loop on the dropout story

## 10. Backlog updates

After this round:
- ~~Input-side temporal dropout~~ ← round 93 closes this
- Real irregular time-series (PhysioNet-style) — still open, much larger
- Combined smoothness + state — still open, harder
