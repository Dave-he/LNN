# PRD #10-116 — Round 154: MONO-CfC (Monotonic Activation CfC)

**Date**: 2026-06-15
**Round**: 154
**Audit context**: 91-153 audit, 14 strictly positive + 10
target-dep + 22 negatives = 46 mechanism classes.

## Hypothesis

The CfC closed-form solution::

    h_t = σ(-f · τ) · g + (1 - σ(-f · τ)) · h_branch

where both `g` and `h_branch` use **Tanh** activations.

Tanh is **NOT monotonic in the output direction** — negative input
gives negative output, positive gives positive. For ODE solutions
that need to preserve input-direction information, monotonic
activations like **Softplus** (monotonic increasing) are
mathematically preferable.

**Hypothesis**: replacing Tanh with Softplus in the g_branch
and/or h_branch may help CfC preserve the temporal ordering
of inputs. Inspired by monotonic networks
(Chilinski & Silva 2020, "Neural Likelihoods for Continuous-Time
Markov Chains").

## Mechanism

Four variants:

1. **mono_g**: replace Tanh in g_branch with Softplus.
2. **mono_h**: replace Tanh in h_branch with Softplus.
3. **mono_both**: replace Tanh in BOTH g_branch and h_branch.
4. **mono_sig**: replace Tanh with Sigmoid (control — bounded
   non-monotonic in [0, 1]).

Key code::

    class MonoCfCCell(nn.Module):
        def __init__(self, input_size, hidden_size,
                     mono_mode="g_only"):
            # mono_mode: g_only, h_only, both, sigmoid
            base = CfCCell-equivalent with custom activations
            self.f_gate = nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.Sigmoid(),  # unchanged
            )
            if mono_mode == "g_only":
                g_act = nn.Softplus()
                h_act = nn.Tanh()
            elif mono_mode == "h_only":
                g_act = nn.Tanh()
                h_act = nn.Softplus()
            elif mono_mode == "both":
                g_act = nn.Softplus()
                h_act = nn.Softplus()
            elif mono_mode == "sigmoid":
                g_act = nn.Sigmoid()
                h_act = nn.Sigmoid()
            ...

## Bench plan

24 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs:

| cond       | activation | notes                       |
|------------|-----------|----------------------------|
| cfc         | Tanh+Tanh | baseline                    |
| mono_g      | Softplus+Tanh | replace g_branch only       |
| mono_h      | Tanh+Softplus | replace h_branch only       |
| mono_both   | Softplus+Softplus | both branches        |
| mono_sig    | Sigmoid+Sigmoid | control (bounded)          |

3 datasets: sin_irr, structured_irr, random_irr.

## Expected outcomes

- **If Softplus helps**: would be a new STRICTLY POSITIVE
  mechanism (15th winner).
- **If Softplus hurts**: NEGATIVE (23rd) — Tanh's non-monotonicity
  is a feature not a bug (allows negative output for negative
  input direction).
- **If mixed**: TARGET-DEPENDENT (11th).

## Risk

Softplus outputs are positive, breaking the bidirectional
information flow that Tanh provides. May cause mode collapse on
sin data which oscillates positive/negative.

## Files to create

- `lnn/core/mono_cfc.py` (~180 lines)
- `tests/test_mono_cfc.py` (~22 tests)
- `scripts/bench_mono_cfc.py` (~250 lines, 30-cell bench)
- `results/bench_mono_cfc.json`
- `docs/research/2026-06-15_mono_cfc_report.md`
