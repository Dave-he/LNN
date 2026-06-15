# PRD #10-115 — FiLM-CfC (Feature-wise Linear Modulation) (Round 153)

**Date**: 2026-06-15
**Round**: 153
**Verdict target**: TARGET-DEPENDENT (10th), STRICTLY POSITIVE (15th), or NEGATIVE (22nd+)

## 1. Motivation

The 91-152 audit shows 14 strictly positive + 9 target-dep = 23
mechanisms that follow a clear pattern: **preserve x, add parallel
context**. Recent winners:
- **MSDC 151 (strictly positive 14th)**: multi-scale conv
- **TCC 149 (target-dep 8th)**: single-K conv (concat)
- **LiNo 150 (target-dep 9th)**: linear projection (sum)
- **TDSA 152 (negative 21st)**: self-attention (concat, undertrained)

What's NOT been tested: **FiLM (Feature-wise Linear Modulation)**
— a context-driven multiplicative + additive modulation of the
cell's hidden state::

    out = γ ⊙ h + β

where γ and β are computed from a context (either the input itself
or a global summary).

FiLM was introduced by Perez et al. 2018 ("FiLM: Visual Reasoning
with a General Conditioning Layer") for visual reasoning. The
key insight: **a small conditioning network can produce powerful
modulation parameters** (γ, β) that change the behavior of the
main network.

Round 153 tests **FiLM-CfC** with three variants:
- `film_self`: γ, β from x_t at each timestep (self-modulation)
- `film_global`: γ, β from global mean of x (sequence-level)
- `film_concat`: aug_x = concat([x, global_mean]) (control, no
  modulation)

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

FiLM-CfC::

    # Context (sequence-level summary)
    ctx = x.mean(dim=1, keepdim=True)  # [B, 1, D]
    # Or per-step (self)
    # ctx = x  # [B, T, D]

    # Modulation parameters
    gamma = Linear_gamma(ctx)  # [B, T, hidden_size]
    beta = Linear_beta(ctx)    # [B, T, hidden_size]

    # Standard CfC
    h_t = CfCCell(x_t, h_{t-1})  # [B, T, hidden_size]

    # Modulated output
    out = gamma * h + beta  # [B, T, hidden_size]

This is a **structural modulation** (γ, β come from a context,
not the cell state).

## 3. Hypotheses

- **H1** (Sin data): FiLM helps periodic data — γ, β from
  sequence-level context can encode the phase. **EXPECTED: positive.**
- **H2** (Structured data): FiLM helps regime-change data — γ, β
  can shift behavior around the boundary. **EXPECTED: positive.**
- **H3** (Random data): FiLM may hurt noise — modulation on noise
  is noise. **EXPECTED: neutral or negative.**
- **H4** (Self vs global): global is more stable (same γ, β for
  all timesteps). **EXPECTED: global wins.**

## 4. Implementation

`lnn/core/film_cfc.py` (~150 lines) — `FiLMCfCCell` +
`FiLMCfCStackedNetwork`.

Key design choices:

1. **Two modulation projections**: Linear_gamma, Linear_beta.
2. **Three context modes**: self (per-step), global (sequence-level),
   concat (control, no modulation).
3. **Multiplicative + additive**: γ * h + β. Standard FiLM.
4. **NaN handling**: zero-fill input.
5. **Preserves CfC**: h goes through the standard CfC update, then
   is modulated by γ, β.

## 5. Bench

24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs:
- cfc (baseline)
- film_self (γ, β from x_t per step)
- film_global (γ, β from global mean of x)
- film_concat (concat x + global mean, no modulation — control)

## 6. Why this might win (mechanism reasoning)

The audit pattern: input-side processing that PRESERVES x wins.
FiLM preserves x (CfC still takes raw x), preserves h (CfC's
recurrent step is unchanged), and ADDS modulation parameters
from a context.

FiLM is structurally different from all prior mechanisms:
- TCC 149, MSDC 151, TDSA 152: ADD context to input via concat
- LiNo 150: ADD context via sum
- **FiLM 153: MODULATE hidden state via multiplicative + additive**

This is the first mechanism in the audit to use **multiplicative
interaction** with the cell's hidden state (vs additive concat/sum).

Risks:
- Multiplicative modulation is powerful but can be unstable
- γ, β from x_t (self) is essentially a per-step modification
  (loses pattern in audit)
- γ, β from global mean is constant across timesteps (may not
  have enough expressivity)

## 7. Critical implementation details

1. **Two Linear projections**: γ = Linear(D, H), β = Linear(D, H).
2. **Context expansion**: if ctx is [B, 1, D], expand to [B, T, D]
   via `.expand(-1, T, -1)`.
3. **Multiplicative + additive**: γ * h + β. Standard FiLM formula.
4. **NaN handling**: zero-fill input.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.
