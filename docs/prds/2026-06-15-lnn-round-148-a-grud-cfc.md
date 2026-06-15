# PRD #10-110 — GRU-D Time-Decay CfC (Round 148)

**Date**: 2026-06-15
**Round**: 148
**Verdict target**: TARGET-DEPENDENT (7th) or STRICTLY POSITIVE (14th) or NEGATIVE (21st)

## 1. Motivation

The 91-147 audit shows a strong pattern: **multi-timescale mechanisms
need either parallel slow streams (SCRN α=0.5, target-dep 146) or
multi-τ within a single cell (n_tau 76, strictly positive)**.

Round 147's Clockwork partition (NEGATIVE 20th) failed because
slow modules get only 4 gradient updates per T=32 sequence (K=4,
period 8) and the carry-forward h creates discontinuities.

Round 148 tests a structurally different multi-timescale design —
**GRU-D / CT-RNN style time decay** (Che et al. 2018 "GRU-D", Jia
& Benson 2019 "ODE-RNN", Lechner & Hasani 2020 "CT-RNN"):

- Expose the actual time delta Δt to the cell
- Apply a learnable per-feature decay to h between observations:
  `h_t = exp(-γ * Δt) * h_{t-1}`
- Then CfC updates h as usual using the new observation

This is fundamentally different from Clockwork (binary carry) and
SCRN (fixed α):

1. **Clockwork 147 (NEGATIVE)**: binary carry-forward, h stays the
   same for K-1 steps then jumps. Discontinuous.
2. **SCRN 146 (target-dep α=0.5)**: parallel slow context with
   FIXED α. The slow context evolves every step, just with EMA
   decay. Decay rate is FIXED.
3. **GRU-D / CT-RNN 148 (this round)**: continuous time-aware
   decay. The decay rate is **LEARNABLE per-feature** and **DEPENDS
   ON Δt**. The hidden state smoothly decays between observations
   based on how much time has passed.

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

GRU-D CfC adds time decay BEFORE the CfC update:

```
# dt = time difference between observations (B, T, 1)
# gamma = learnable per-feature decay rate (1, hidden_size)
# x_t and h_{t-1} are inputs to CfC
for t in range(T):
    if t == 0:
        h = zeros(B, hidden_size)
    else:
        # Time-aware decay (GRU-D style)
        # decay_factor in (0, 1) — closer to 0 means more decay
        # gamma=0 → decay=1 (no decay), gamma=1 → decay=exp(-dt)
        decay = exp(-gamma * dt[:, t, :])
        h = h * decay
    h = CfCCell(x_t, h)
    outputs.append(h)
```

Where `gamma ∈ R^{hidden_size}` is learnable, initialized near 0
(decay=1, no decay initially). The `dt[:, t, :]` is the time
difference from t-1 to t.

For the benchmark datasets, we set `dt = 1.0` for all steps (regular
time series). The decay then becomes a CONSTANT per step, similar
to SCRN. The difference from SCRN is:

- **SCRN**: parallel context stream, fixed α, separate from main h
- **GRU-D CfC**: decay applied to MAIN h, learnable per-feature γ

## 3. Hypotheses

- **H1** (Sin data): per-feature decay should help smooth data
  (decay = 1, no effect, but it could LEARN to decay slow features).
  **EXPECTED: neutral to slightly positive.**
- **H2** (Structured data): per-feature decay should help on
  regime-change data. **EXPECTED: positive on regime change.**
- **H3** (Random data): per-feature decay should hurt noisy data
  (decay=exp(-γ) wipes memory faster). **EXPECTED: negative.**
- **H4** (Different γ init): init γ=0 (no decay) vs γ=0.1
  (some decay) vs γ=1.0 (heavy decay). **EXPECTED: γ=0 = baseline,
  γ=0.1 = positive on structured, γ=1.0 = negative.**

## 4. Implementation

`lnn/core/grud_cfc.py` (~140 lines) — `TimeDecayCfCCell` +
`TimeDecayCfCStackedNetwork`.

Key design choices:

1. **γ parameterization**: `gamma = softplus(gamma_param)` so γ ≥ 0.
   Init at gamma_param = -3.0 (γ ≈ 0.05).
2. **Per-feature**: one γ per hidden unit. So different features
   can decay at different rates.
3. **Time-aware**: dt is an input. We use dt[:, t, :] as the
   per-step time delta.
4. **NaN handling**: zero-fill input AND zero-fill dt (use dt=0 =
   decay=1 = no decay on missing data).
5. **Preserves CfC**: h goes through the standard CfC update after
   decay. This is a STRUCTURAL addition (preserves the recurrent
   step), not a per-step modification.

## 5. Bench

24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs:
- cfc (baseline)
- grud_gamma_0 (γ ≈ 0, no decay) — control
- grud_gamma_005 (γ ≈ 0.05) — light decay
- grud_gamma_05 (γ ≈ 0.5) — heavy decay

## 6. Why this might win (mechanism reasoning)

The audit pattern: input-side processing that PRESERVES x wins.
The time decay is applied to h, not x. So this is a different
mechanism axis (h-modification, not x-modification).

GRU-D has been validated for irregular TS (Che et al. 2018). The
key insight is that **h is time-dependent** and the network should
know how much time has passed. Even for regular TS, the learnable
γ can adapt to the data structure.

The risk: decay applied to h is closer to Clockwork (h
modification) than to SCRN (parallel context). But the decay is
CONTINUOUS (not binary), so it should be less harmful.

## 7. Critical implementation details

1. **softplus γ**: `γ = softplus(gamma_param) ≥ 0`.
2. **dt = 1.0 for regular TS**: matches standard time step.
3. **NaN handling**: zero-fill dt, then decay=1 (no decay) on NaN.
4. **Per-feature γ**: each hidden unit has its own decay rate.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.
