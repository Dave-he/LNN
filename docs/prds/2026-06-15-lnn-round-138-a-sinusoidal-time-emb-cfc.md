# PRD #10-100 — Sinusoidal Time Embedding for CfC (Round 138)

**Date**: 2026-06-15
**Round**: 138 (response to positional encoding literature, Vaswani 2017)
**Status**: Drafted.

## 1. Why round 138

Transformer positional encoding (Vaswani et al. 2017) is the canonical
"add explicit time information to the input" mechanism. The standard
sinusoidal positional encoding is:

```
PE(t, 2i)   = sin(t / 10000^(2i/D))
PE(t, 2i+1) = cos(t / 10000^(2i/D))
```

For CfC, the input `x_t` is just the current observation. The cell has
no explicit knowledge of WHICH timestep it's at. This is fine for
uniformly-sampled sequences, but problematic when:

1. The sequence has REGIME SWITCHES (e.g., sin_irr/structured_irr in
   our bench) — the cell would benefit from knowing the time elapsed
   to detect the switch.
2. The cell has different optimal dynamics at different times (start
   vs middle vs end).
3. We want to use a learned time encoding that's data-adaptive.

Currently CfC has a single `time_constant` parameter per neuron, which
is the same at every timestep. Adding an explicit time embedding would
let the cell have a different effective time constant at each timestep.

## 2. Mechanism

```
t_emb = sinusoidal_encoding(t)        # [B, D_te] or [D_te]
x_aug = concat([x_t, t_emb])           # or add
h_t = cf_c_step(x_aug, h_{t-1})
```

Three variants:
- `concat_d4`: concatenate 4-dim sinusoidal embedding to x_t
- `add_d2`: add 2-dim sinusoidal embedding to x_t
- `learned_d8`: 8-dim learned time embedding (nn.Embedding)

## 3. Hypotheses

- **H1 (time emb helps on smooth data)**: with time embedding, test_mse
  on `sin_irr` is < baseline (cell benefits from knowing elapsed time).
- **H2 (time emb helps on structured data)**: with time embedding,
  test_mse on `structured_irr` is < baseline (regime switch is more
  detectable).
- **H3 (no regression on noisy data)**: with time embedding, test_mse
  on `random_irr` is not worse than baseline by >10%.

## 4. Why this should win per the 91-137 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (MoE experts, input-side processing, additive
  shortcuts).
- 1 TARGET-DEPENDENT (LN 135) + 1 NEW TARGET-DEPENDENT (conv_k3 137).
- 9 negatives propose alternatives to the recurrent step or add
  unsupervised/regularizer terms.

Sinusoidal time embedding:
- **Preserves the recurrent step** entirely.
- **Adds useful input-side structure** — explicit time information
  that the cell can use.
- **Is structural** — modifies the input, not the recurrent step.
- **Similar to QuITE+MoE (round 103)** and **GIS (round 134)** in
  spirit — adds useful input-side information that the recurrent step
  can use.

The risk: the f-gate might already implicitly learn the time
information if the input has time-correlated structure. But for
sequences with explicit time-varying patterns (regime switches), the
cell should benefit from explicit time info.

## 5. Plan

### 5.1 Implementation (`lnn/core/sinusoidal_time_emb_cfc.py`)

Two classes:
- `SinusoidalTimeEmbCfCCell(nn.Module)`: standard 3-branch CfC cell
  with sinusoidal time embedding applied to the input.
- `SinusoidalTimeEmbCfCStackedNetwork(nn.Module)`: 2-layer stack with
  per-timestep time embedding.

Key design choices:
- 4-dim sinusoidal embedding (concat) by default.
- Frequencies follow Vaswani et al. 2017 (1, 1/100, 1/10000, ...).
- For variable-length sequences, normalize t/T to [0, 1].
- Time embedding is computed per-timestep and added to the input.

### 5.2 Tests (`tests/test_sinusoidal_time_emb_cfc.py`)

20+ unit tests covering:
- Init: time embedding parameters.
- Forward: shape preservation.
- Time: time embedding at different t values is different.
- Gradient: flows to time embedding (if learnable).
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: time emb is periodic and bounded.

### 5.3 Bench (`scripts/bench_sinusoidal_time_emb_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `te_concat` (concat 4-dim time embedding)
- `te_learned` (concat 8-dim learned time embedding)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 6. Expected outcomes

- **Best case (~40%)**: H1 + H2 + H3 all confirmed. Time embedding
  is the **14th STRICTLY POSITIVE** winner.
- **Likely case (~40%)**: H1 + H3 confirmed, H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE** (helps smooth/structured, neutral
  noisy).
- **Worst case (~20%)**: All 3 hypotheses rejected. The f-gate
  already implicitly learns time information. 10th negative.

## 7. Why this is worth testing

The 91-137 audit strongly suggests "input-side processing + add to
recurrent step" mechanisms win. QuITE+MoE (round 103), GIS (round 134)
were winners. Sinusoidal time embedding is a 2-line addition that
could be a 14th winner. If it wins, it would be a high-confidence
production candidate (very simple, no extra recurrent parameters).

## 8. Files to create

- `lnn/core/sinusoidal_time_emb_cfc.py` (~200 lines)
- `tests/test_sinusoidal_time_emb_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_sinusoidal_time_emb_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_sinusoidal_time_emb_cfc_report.md`
