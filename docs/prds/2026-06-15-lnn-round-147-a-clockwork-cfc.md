# PRD #10-109 — Clockwork CfC (Round 147)

**Date**: 2026-06-15
**Round**: 147
**Verdict target**: TARGET-DEPENDENT (7th) or STRICTLY POSITIVE (14th) or NEGATIVE (20th)

## 1. Motivation

The 91-146 audit shows a clear pattern: **6 of 6 target-dep winners
are input-side processing that PRESERVES x** (LN 135, conv 137,
GLU+skip 139, decoupled/IndRNN 143, bidi_concat 144, scrn_alpha_05
146). 19 negatives include input-side REPLACEMENTS (diff_only 145,
long-α SCRN 146) and per-step modifications.

Round 147 tests the **Clockwork RNN (CW-RNN)** idea from Koutnik
et al. 2014 ("A Clockwork RNN"). CW-RNN partitions the hidden
state into K groups, each group updating only at its assigned
period (2^k). This is a **structural partition** (not a per-step
modification, not a replacement):

- Module 0: period 1 (updates every step) — "fast"
- Module 1: period 2 (updates every 2 steps) — "medium"
- Module 2: period 4 (updates every 4 steps) — "slow"
- Module 3: period 8 (updates every 8 steps) — "very slow"

When a module doesn't update on step t, it **carries forward** its
previous h (preserves h). All modules' outputs are concatenated
for the final hidden state.

This is **different from**:
- **SCRN 146**: parallel slow context stream, not partition. SCRN
  uses a single slow context; CW-RNN uses K independent modules.
- **ELM 129 (NEGATIVE)**: per-step multi-timescale mixing, not
  partition. ELM mixed all timescales at every step.
- **n_tau 76 (POSITIVE)**: multi-τ within a single cell, not
  partition. n_tau uses K branches that all run at every step.

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

CW-CfC: hidden is partitioned into K groups::

    # At step t, only module k updates if t mod 2^k == 0
    for k in range(K):
        if t mod (2 ** k) == 0:
            h_k_new = CfCCell_k(x_t, h_k_old)
        else:
            h_k_new = h_k_old   # carry forward
    h_combined_t = concat([h_0, h_1, ..., h_{K-1}])   # [B, K * module_size]

This is a STRUCTURAL PARTITION (not a per-step modification) and
PRESERVES both x and h (carrying forward).

## 3. Hypothesis

**H1** (Smooth data): slow modules capture long-term trends, help sin_irr.
- sin_irr test_mse should drop from 0.0094 → <0.007.

**H2** (Structured data): slow modules capture regime-level patterns.
- structured_irr test_mse should drop from 0.0053 → <0.003.

**H3** (Random data): slow modules "smooth out" local noise patterns.
- random_irr test_mse should not regress by >50% (some loss expected).

**H4** (Different K): K=2 minimal, K=4 maximum (periods 1,2,4,8).
- Test K ∈ {2, 3, 4}.

## 4. Implementation plan

`lnn/core/clockwork_cfc.py` — `ClockworkCfCCell` + `ClockworkCfCStackedNetwork`.

```python
class ClockworkCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_modules=3, module_sizes=None):
        # Partition hidden into num_modules
        # Each module has its own CfCCell
        # Period for module k is 2^k
        # On each forward, only some modules update

    def forward(self, x):
        # For each timestep t:
        #   For each module k:
        #     if t mod (2 ** k) == 0: h_k = cell_k(x_t, h_k)
        #     else: h_k stays the same
        # Return concat of all h_k
```

Default config: `num_modules=3`, `module_sizes=None` (auto: hidden_size // num_modules each).

## 5. Variants to test

| Cond | Description | K | Periods | Params |
|------|-------------|---|---------|--------|
| cfc (baseline) | Vanilla CfC | - | - | 2545 |
| cw_k2 | 2 modules, periods 1,2 | 2 | 1, 2 | ~2700 |
| cw_k3 | 3 modules, periods 1,2,4 | 3 | 1, 2, 4 | ~2800 |
| cw_k4 | 4 modules, periods 1,2,4,8 | 4 | 1, 2, 4, 8 | ~2900 |

## 6. Bench plan

- 4 conditions × 3 datasets × 2 seeds = 24 cells
- 30 epochs, B=8, T=32, hidden=16 (matching round 144 setup)
- 3 datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 7. Risk

- **Training instability for slow modules**: a module that updates
  every 8 steps gets only 4 gradient updates per T=32 sequence.
  Could underfit.
- **Carry-forward h for slow modules**: h stays the same for 7
  steps in a row, then jumps. This is a discrete "phase change" that
  may not align with smooth data.
- **NaN handling**: zero-fill input per step.
- **Module size allocation**: with hidden=16 and K=4 modules, each
  module has 4 units. CfC's f_gate, g_branch, h_branch are
  input_size+hidden_size+1 each, so smaller modules are cheaper.

## 8. Why this is likely TARGET-DEPENDENT (7th) or NEGATIVE (20th)

**Per the audit pattern**:
- 6 of 6 target-dep winners are input-side processing that
  PRESERVES x (LN/conv/GLU+skip/decoupled/bidi_concat/scrn_05).
- CW-RNN preserves x (modules still receive x_t at every step).
- It's a structural partition, not a per-step modification.
- Different from ELM 129 (negative) which was per-step multi-timescale.

**Risks**:
- **CW-RNN's slow modules don't get enough gradient updates** for
  T=32 sequences. K=4 with period 8 means 4 updates per sequence.
  Likely underfits the slow modules.
- **Carry-forward h is a per-step "decision"** — even though the
  cell call is per-step, the structure is partition. Per the audit,
  per-step decisions have been negative (e.g., adaptive time constant).

**Prior**:
- TARGET-DEPENDENT 50% (similar to SCRN 146)
- STRICTLY POSITIVE 20% (CW-RNN is well-established, but 1D bench
  may not show wins)
- NEGATIVE 30% (training instability + per-step carry-forward
  decisions)

## 9. Mechanism notes

- **CW-RNN is NOT new** (Koutnik 2014), but its application to CfC
  is novel. The original CW-RNN was tested on LSTM/RNN baselines.
- **Multi-timescale structure** is a classical idea (also seen in
  phased LSTM, multi-τ ELM, hierarchical RNN). CW-RNN is the
  cleanest "partition" formulation.
- **Key invariant**: modules PRESERVE their h when not updating
  (carry-forward), which means the recurrent step is the same CfC
  call but only on the "active" modules.
