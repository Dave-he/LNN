# PRD #10-65 — QuITE+MoE: Irregularity-Context-Aware Expert Routing (Round 103)

**Date**: 2026-06-15
**Round**: 103
**Status**: Drafted.

## 1. Why round 103

Round 102 (PRD #10-64) implemented **QuITE** (arXiv:2605.28166, Lim ICML 2026) — a plug-and-play query-based embedding for irregular multivariate time series. QuITE was confirmed **STRICTLY POSITIVE, target-agnostic, and first non-target-dependent positive mechanism in our 91-102 audit** on three datasets (sin_irr / structured_irr / random_irr) with synthetic PhysioNet-style missingness.

But QuITE was only wired as a **backbone input embedding**. It has not been connected to the **routing** in our existing MoE stack. The natural next question is:

> *Can QuITE's "irregularity context" be used to **route** experts?*

This is a non-trivial extension. In standard FAME/MR-MoE routing (rounds 77-78), the router uses `[x_t, h_prev]` — a per-step local signal. But:
- `x_t` may be **NaN** for irregular time series (the most common case in PhysioNet)
- The model has no awareness of the **global irregularity pattern** of the sequence
- Routing decisions are made on noisy, partial information

**QuITE+MoE** solves this by pre-computing a global "irregularity context" vector (the pooled QuITE tokens) and using it as an **additional router input**. This makes routing decisions:
- **Irregularity-aware** (the context knows which timesteps are missing)
- **Noise-robust** (queries aggregate over the full sequence)
- **Distinct per-sequence** (different sequences with different missingness patterns get different context)

This is the **first principled combination of two distinct mechanisms from our 27-layer stack** (QuITE embedding + FAME top-K routing).

## 2. Architecture

```
Input: irregular observations (B, T, D), times (B, T), mask (B, T)
                       ↓
        QuITE module (pre-compute once, on full sequence)
        → context: (B, n_queries, d_model) → mean pool → (B, d_model)
                       ↓
        At each step t:
        ┌──────────────────────────────────┐
        │ x_t:        (B, D)               │
        │ h_prev:     (B, H)               │
        │ context:    (B, d_model)         │  ← QuITE-augmented
        │ router_in:  cat([x_t, h_prev, context])  → (B, D+H+d_model)
        │ logits:     Linear → (B, K)      │
        │ top-K:      → sparse g          │
        └──────────────────────────────────┘
                       ↓
        Mixture: h_new = Σ_k g_k · expert_k(x_t, h_prev)
```

Key design choices:
- **QuITE pre-computed once per sequence** (not per-step): T=1 attention call per sequence, not T.
- **Mean pool over query tokens**: collapses (B, n_queries, d_model) → (B, d_model) for compact routing signal.
- **Concatenation with [x_t, h_prev]**: preserves local information; context augments (not replaces) it.

## 3. Hypotheses

- **H1 (QuITE+MoE has lower test MSE than FAME baseline on irregular TS)**: QuITE+MoE = use pooled QuITE context as extra router input. Baseline = FAME with [x_t, h_prev] only. Expect 5-20% lower test MSE on synthetic PhysioNet-style data.
- **H2 (QuITE+MoE expert utilization is more uniform than FAME baseline)**: with QuITE context, experts can specialize by irregularity REGIME (sparse vs dense missingness) rather than by per-step value, leading to more balanced expert loads.
- **H3 (QuITE+MoE is target-agnostic)**: works equally on smooth / structured / random irregular data.
- **H4 (QuITE+MoE training is stable)**: no NaN losses, no expert collapse, gradient norms bounded.

## 4. Plan

### 4.1 Implementation (`lnn/core/quite_moe.py` — NEW file)

Add 3 new components:
- `QuiteRouter(input_size, hidden_size, n_experts, top_k, d_context, router_hidden=0)` — router that concatenates QuITE context to [x_t, h_prev]
- `QuiteMoECfCCell(input_size, hidden_size, n_experts, top_k, n_queries, d_context, n_heads, n_tau_per_expert, tau_scales)` — K CfCCell experts + QuiteRouter
- `QuiteMoECfCNetwork(input_size, hidden_size, n_experts, top_k, n_queries, d_context, n_heads, output_size, n_tau_per_expert, tau_scales)` — full network wrapper that pre-computes QuITE context at sequence start and routes per step

Key implementation details:
- **Pre-compute QuITE context once**: call `self.quite(observations, times, mask) → (B, n_queries, d_context)`, mean pool → (B, d_context), cache for the sequence.
- **Network.forward(observations, times, mask) → output per step**: takes the full irregular sequence and returns the per-step output.
- **Cell.forward(x_t, h, context) → h_new**: takes the cached context, mixes with [x_t, h] for routing.

### 4.2 Tests (`tests/test_quite_moe.py` — NEW file)

12 new tests in 3 classes:
1. `TestQuiteRouter` (4 tests):
   - initialization
   - forward shape with and without context
   - context vs no-context produces different routing
2. `TestQuiteMoECfCCell` (4 tests):
   - initialization with default args
   - forward shape
   - cell accepts pre-computed context
   - different contexts produce different expert choices
3. `TestQuiteMoECfCNetwork` (4 tests):
   - initialization
   - full forward on irregular input
   - expert utilization is non-degenerate
   - NaN handling (mask propagates correctly)

### 4.3 Bench (`scripts/bench_quite_moe.py` — NEW)

24 cells:
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 conditions: FAMECfC baseline, QuiteMoECfC
- 2 K settings: K=2, K=3
- 2 seeds, 100 epochs

For each cell measure:
- `test_mse` (held-out with HIGHER missing rate than training)
- `expert_utilization` (entropy of expert assignments)
- `dead_experts` (experts never used)
- `training_stable` (no NaN losses, gradient norm < 1.0)

H1: QuITE+MoE lower test_mse than baseline. H2: QuITE+MoE expert_utilization > baseline. H3: QuITE+MoE wins on all 3 datasets. H4: QuITE+MoE training stable.

### 4.4 Decision rule

QuITE+MoE is a **STRICTLY POSITIVE** round if H1, H2, H3 all pass. Otherwise it is an **HONEST TARGET-DEPENDENT** or **HONEST NEGATIVE** round.

## 5. Why this matters

- **First principled combination of QuITE and FAME** in the 27-layer stack.
- **Fills the irregular-TS MoE gap**: our 78-100 rounds of MoE work assumed uniform-time input.
- **Enables deployment in PhysioNet-style clinical settings** where missingness is the norm.
- **Routes by global irregularity pattern**, not per-step noise.

## 6. Files

- `docs/prds/2026-06-15-lnn-round-103-a-quite-moe-routing.md` (this file)
- `lnn/core/quite_moe.py` (NEW) — 3 new components
- `lnn/core/__init__.py` — export
- `tests/test_quite_moe.py` (NEW) — 12 tests
- `scripts/bench_quite_moe.py` (NEW) — 24-cell bench
- `results/bench_quite_moe.json`
- `docs/research/2026-06-15_quite_moe_routing_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v29.md`
- `README.md` — new section

## 7. Risk

Medium. The QuITE module from round 102 is well-tested. The new code only adds:
1. A router that concatenates context (linear layer, well-known pattern)
2. A cell wrapper (K experts + router, mirrors FAMECfCCell structure)
3. A network wrapper (pre-compute QuITE once, call cell per step)

The bench reuses round 102's irregular-TS data generators.

## 8. Backlog for round 104+

1. **QuITE++ hierarchical** — combine with round 102 hierarchical variant
2. **Real PhysioNet dataset** — wire to actual data loader
3. **Per-step QuITE** — re-compute context at every step (more expensive, but time-aware)
4. **Compose 4-axis gates** in single QuiteMoECfC stack (round 99)
5. **arXiv:2606.07500 SETA** — subspace-to-expert sharing for continual learning
6. **K=20, hidden=32, paper-scale settings**
