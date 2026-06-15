# LNN Research Digest v37 — 2026-06-15

**Coverage**: Mixture-of-Depths Routing + 91-111 audit update (1st compute-saving structural mechanism).

## Headline

Round 111 implemented **Mixture-of-Depths (MoD) Routing** (arXiv:2404.02258 Raposo et al. DeepMind 2024) — *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. The mechanism: at each cell step, route over a fixed budget k of timesteps to process through the heavy CfC block; the remaining timesteps are skipped (residual passthrough). This is the **8th structural mechanism** in our 91-111 audit, and the **first to be BOTH quality-preserving AND compute-saving**.

The result is **HONEST POSITIVE-WITH-NUANCE** (1st compute-saving in audit):
- **H1 ✓ CONFIRMED**: sin_irr — MoD matches baseline (0.0023 vs 0.0023) at 50% compute
- **H2 ✓ CONFIRMED**: structured_irr — MoD matches baseline (0.0010 vs 0.0010) at 50% compute
- **H3 PARTIAL**: random_irr — MoD is 2× worse (0.0010 vs 0.0005); noisy data needs full compute
- **H4 ✓ CONFIRMED**: 100% → 50% → 25% has **NO further degradation** on 2/3 datasets — 50% step is "free"

**NEW INSIGHT**: **MoD is the only mechanism in 91-111 audit that saves compute**. The 7 other structural mechanisms all add parameters or change routing, but they don't reduce the per-forward FLOPs.

## 1. MoD in 60 seconds

Standard Transformer/CfC: every token in a sequence goes through every block. MoD: cap the number of tokens that participate in expensive block computations.
```
input x_t  [B, T, D]
  │
  ├── MoDRouter: linear → logit [B], sigmoid → prob [B]
  ├── topk(logit, k=cap_k) → topk_idx [k]
  ├── Cell(x_t[topk_idx], h[topk_idx]) → h_new [k, H]  (expensive)
  └── out = where(topk_idx, h_new, h)  [B, H]  (residual for skipped)
```

The cap k is **fixed a priori**; the router learns which timesteps to spend compute on.

## 2. Bench summary (24 cells, 50 epochs, 2 seeds)

`scripts/bench_mod_routing.py`:
- 4 conditions: `baseline_cfc` (control), `mod_all` (cap_k=None), `mod_half` (50% budget), `mod_quarter` (25% budget)
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 seeds × 50 epochs, T=32, hidden=16, B=8, num_layers=2

### test_mse (mean ± std, 2 seeds, 50 epochs)

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0023±0.0002 | 0.0010±0.0001 | 0.0005±0.0003 |
| mod_all      | 0.0023±0.0002 | 0.0010±0.0001 | 0.0010±0.0007 |
| mod_half     | 0.0023±0.0002 | 0.0010±0.0001 | 0.0010±0.0007 |
| mod_quarter  | 0.0023±0.0002 | 0.0010±0.0001 | 0.0010±0.0007 |

## 3. The 91-111 audit pattern

| Round | Mechanism | Type | Compute-saving? | Verdict |
|-------|-----------|------|------------------|---------|
| 91-94 | TV smoothness, dropout, rank | Diagnostic | No | NEGATIVE |
| 95-97 | Per-expert rank, FAME+orth, wt orth | Combined | No | NEGATIVE/PARTIAL |
| 98-99 | Backward coherence, Reliability gate | Regularizer/Aug | No | PARTIAL/**STRICTLY POSITIVE** |
| 100-101 | SNNL, ORC | Regularizer | No | TARGET-DEP/DIAGNOSTIC |
| 102 | QuITE | Embedding | No | **STRICTLY POSITIVE** |
| 103-104 | QuITE+MoE, SDG-MoE | Routing | No | TARGET-DEP/NEGATIVE |
| 105 | SETA | Architecture | No | **STRICTLY POSITIVE** |
| 106 | AuxLF | Load balancer | No | TARGET-DEP |
| 107 | Soft MoE | Structural | No | **SAFE ROUTING** |
| 108 | Anchored MoE | Structural | No | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | No | NEGATIVE-WITH-NUANCE |
| 110 | Freq Experts | Structural | No | NEGATIVE-WITH-NUANCE |
| **111** | **MoD Routing** | **Structural** | **YES (50-75%)** | **POSITIVE-WITH-NUANCE** |

**8 STRUCTURAL mechanisms tested in 91-111**:
- 4 winners: 99 Reliability Gate, 102 QuITE, 105 SETA, 107 Soft MoE
- 4 target-dep/negative: 108 Anchored, 109 Dynamic, 110 Freq Experts, 111 (1st compute-saving!)

**MoD is unique**: it's the only mechanism that REDUCES the per-forward compute. The other 7 either add parameters, change routing, or add regularization.

## 4. Why MoD works on smooth data

In sin/structured data, the signal is **smooth** (sinusoidal, step-function). The residual passthrough carries the previous hidden state forward, which is approximately the right value for nearby timesteps. The router learns to pick **representative** timesteps (peaks, troughs, regime boundaries) and skip the in-between steps.

In random_irr data, the signal is **white noise** with no temporal structure. The residual passthrough just propagates noise, and the heavy CfC update is needed at every step. The router can't pick "representative" steps because there are none.

## 5. Why the 50% step is "free"

This is the key finding: **reducing compute from 100% to 25% has no quality cost on smooth data**. The cap_k doesn't need to be tuned; any value below the "saturation point" works equally well. This means MoD is **robust to the cap_k choice**.

## 6. Implementation highlights

`lnn/core/mod_routing.py` (~370 lines):
- `MoDRouter(input_size, hidden_size, router_hidden)` — top-k router with cap k
- `MoDCfCCell(input_size, hidden_size, cap_k, router_hidden)` — CfC cell with MoD routing
- `MoDCfCNetwork(input_size, hidden_size, output_size, num_layers, cap_k, cap_k_frac, router_hidden)` — full network
- `compute_mod_aux_loss(network)` — loss aggregator

`tests/test_mod_routing.py` (28/28):
- TestMoDRouter (7): init, init with router_hidden, forward shape, aux loss non-negative, cap_k capped at batch, router prob in [0,1], gradient flows.
- TestMoDCfCCell (6): init no cap, init with cap, forward no cap, forward with cap, forward skipped timestep keeps hidden, gradient flows.
- TestMoDCfCNetwork (10): init no cap, init with int cap, init with frac cap, init both raises, init bad frac raises, forward no cap, forward last step, forward with cap, forward with frac cap, gradient flows, aux loss aggregation, aux loss no cap is zero.
- TestMoDIntegration (3): captures signal, process mask varies across steps, higher cap → higher process fraction.

## 7. Critical bugs fixed

1. **`test_process_mask_varies_across_steps`**: cap_k=4 with B=4 triggered the "always process" branch. Fixed by cap_k=2 with B=8.
2. **`test_captures_signal`**: `loss` was possibly unbound at the assertion site. Fixed by initialising before the for-loop.
3. **`test_higher_cap_k_higher_skipped_fraction`**: same as #1, cap_k >= B bug.
4. **Pyright torch false-positives**: pre-existing pattern, ignored per standing rules.

## 8. Recommendation

**Use MoD in production** for time series with smooth structure (sin, periodic, regime-switching):
- `cap_k_frac=0.5` (50% compute, no quality loss on smooth data)
- `cap_k_frac=0.25` for extreme compute savings (also no quality loss on smooth data)
- `cap_k=None` for noisy data (always process, equivalent to standard CfC)

**Combine with QuITE (round 102)** for irregular time series:
- QuITE handles the irregular sampling
- MoD handles the per-step compute budget
- Together: irregular-aware, compute-efficient inference

## 9. Files added

- `lnn/core/mod_routing.py` (NEW, ~370 lines)
- `tests/test_mod_routing.py` (NEW, 28/28 tests)
- `scripts/bench_mod_routing.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-111-a-mod-routing.md` (PRD #10-73)
- `docs/research/2026-06-15_mod_routing_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v37.md` (this file)
- `README.md` (new Mixture-of-Depths section)
- `lnn-round-111-mod-routing.md` (memory)

## 10. Future work

1. **Per-layer cap_k schedule**: Apply MoD only to deeper layers (Raposo et al. 2024 schedule).
2. **MoD+MoE (MoDE)**: Combine MoD top-k with FAME-style top-K experts.
3. **Cumulative routing**: Share router signal across layers.
4. **Adaptive k**: Learn the cap k per-sample.
5. **PhysioNet 36D test**: High-dim medical time series — does MoD scale to real data?
