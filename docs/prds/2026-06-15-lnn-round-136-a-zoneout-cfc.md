# PRD #10-98 — Zoneout for CfC (Round 136)

**Date**: 2026-06-15
**Round**: 136 (response to Zoneout, Krueger et al. 2016, arXiv:1606.01305, ICLR 2017)
**Status**: Drafted.

## 1. Why round 136

The **Zoneout** paper (Krueger, Maharaj, Kratz, Ramalho, Ballas,
2016/2017) introduced a regularizer for RNNs that randomly preserves
the previous hidden state with probability p::

    h_t = h_{t-1}                with prob p (zone out)
    h_t = cf_c_step(x, h_{t-1})  with prob 1 - p

Unlike dropout which DROPS units (replaces with 0), Zoneout PRESERVES
units (replaces with the previous state). This is a form of stochastic
depth applied to recurrent cells.

Key results from Krueger et al.:
- Outperforms dropout and norm-clipping on language modeling and
  music prediction tasks.
- Reduced L2 norm of activations (more stable training).
- Effective at preventing vanishing gradients.
- Per-cell p works as well as per-timestep p.

### 1.1 Mechanism for CfC

Apply Zoneout to the **output of the CfC step** (the new hidden
state)::

    h_new_cfc = cf_c_step(x, h)    # standard 3-branch CfC
    if self.training:
        mask = bernoulli(p_zoneout).expand_as(h_new_cfc)
        h_final = mask * h + (1 - mask) * h_new_cfc
    else:
        h_final = h_new_cfc

The mask is per-cell (same mask across timesteps for a given cell),
which is the standard "recurrent dropout" pattern. With prob p_zoneout
the cell keeps the previous state (zone out the new computation);
with prob (1 - p_zoneout) the cell uses the new computation.

### 1.2 Why this should win per the 91-135 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (MoE experts, input-side processing, additive
  shortcuts).
- 1 TARGET-DEPENDENT (LN) — structural preprocessing.
- 8 negatives propose alternatives to the recurrent step (HGRN,
  Antisymm, etc.) or add unsupervised terms (FastWeights).

Zoneout:
- **Preserves the recurrent step architecture** (the cell output is
  still computed the same way).
- **Adds a useful regularizer** — random preservation of previous
  state prevents overfitting and stabilizes gradients.
- **Is structural** — modifies the recurrent step's output, not the
  input.

The risk: Zoneout might not be useful for short 1D sequences
(T=32) where overfitting is less of a problem. The 30-epoch training
on small batches may not overfit much, so Zoneout's regularizing
effect might be wasted.

## 2. Hypotheses

- **H1 (Zoneout helps generalization)**: with Zoneout, the variance
  of test_mse across seeds is < baseline (because Zoneout reduces
  overfitting variance).
- **H2 (Zoneout helps on noisy data)**: with Zoneout, test_mse on
  `random_irr` is < baseline (because Zoneout prevents the model
  from overfitting to noise).
- **H3 (no regression on smooth data)**: with Zoneout, test_mse on
  `sin_irr` is not worse than baseline by >10%.

## 3. Plan

### 3.1 Implementation (`lnn/core/zoneout_cfc.py`)

Two classes:
- `ZoneoutCfCCell(nn.Module)`: standard 3-branch CfC cell with
  Zoneout applied to the new hidden state during training.
- `ZoneoutCfCStackedNetwork(nn.Module)`: 2-layer stack with
  per-cell Zoneout.

Key design choices:
- Zoneout mask is per-cell (per-neuron), not per-batch.
- The same mask is used across timesteps for a given cell (recurrent
  dropout pattern).
- At eval time, Zoneout is disabled (full computation).
- p_zoneout default = 0.1 (modest Zoneout, similar to dropout 0.1).

### 3.2 Tests (`tests/test_zoneout_cfc.py`)

20+ unit tests covering:
- Init: p_zoneout stored correctly.
- Forward at train mode: some cells zone out, some don't.
- Forward at eval mode: no Zoneout, all cells compute.
- Mask is per-cell (not per-batch).
- Gradient flows through Zoneout (when not zoned out).
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: Zoneout reduces activation L2 norm at training time.

### 3.3 Bench (`scripts/bench_zoneout_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `zoneout_low` (p=0.1)
- `zoneout_med` (p=0.3)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

- **Best case (probability ~25%)**: H1 + H2 + H3 all confirmed.
  Zoneout is the **15th STRICTLY POSITIVE** winner.
- **Likely case (probability ~50%)**: H3 confirmed, H1/H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE**.
- **Worst case (probability ~25%)**: All 3 hypotheses rejected.
  Zoneout is unnecessary for short 1D sequences. 9th negative.

## 5. Why this is worth testing

The 91-135 audit strongly suggests "additive + useful" mechanisms
win. Zoneout is a clean "additive regularizer" that hasn't been
tested on CfC. If it wins, it would be a high-confidence production
candidate (no extra parameters, just stochastic preservation).

## 6. Files to create

- `lnn/core/zoneout_cfc.py` (~200 lines)
- `tests/test_zoneout_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_zoneout_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_zoneout_cfc_report.md`
