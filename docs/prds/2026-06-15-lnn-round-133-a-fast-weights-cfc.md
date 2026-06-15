# PRD #10-95 — Hebbian Fast Weights for CfC (Round 133)

**Date**: 2026-06-15
**Round**: 133 (response to "Using Fast Weights to Attend to the Recent Past", Ba, Hinton, Mnih, Romoff, Veness, NIPS 2016, arXiv:1610.06258)
**Status**: Drafted.

## 1. Why round 133

The **Fast Weights** paper introduces a **Hebbian-style fast weight
matrix** `F_t` that augments the standard recurrent weight `W_h`. The
key idea:

- **Slow weights** `W_h`: standard recurrent weights that change slowly
  (one update per training step).
- **Fast weights** `F_t`: a matrix that evolves at every recurrent step
  via `F_t = λ * F_{t-1} + η * (h_t ⊗ h_{t-1})` — a **Hebbian outer
  product** (with decay).
- The recurrent step uses BOTH: `h_t = σ(W_h h_{t-1} + F_t h_{t-1} + W_x x_t + b)`.

The fast weights provide a **short-term memory** that captures
**pairwise interactions between recent hidden states**. This is a
**structural** change that PRESERVES `W_h` and ADDS to the recurrent
step. Per the 91-132 audit, mechanisms that ADD a useful inductive bias
to the recurrent step (rather than REPLACE it) are STRICTLY POSITIVE
(12 winners). Rounds 128-132 (oscillator, ELM, MR-MoE+dual attn,
HGRN, Antisymm) all proposed alternatives to the recurrent step and
LOSE in 1D.

### 1.1 Why this is different from rounds 128-132 (the 7 negatives)

HGRN (131) removed W·h, used gating only → 16th negative.
Antisymmetric (132) constrained W_h to be antisymmetric, fixed dt → 17th negative.
Oscillator (128) replaced CfC step with 2nd-order ODE → negative.
ELM (129) replaced CfC with extreme learning machine → negative.
MR-MoE+dual (130) added attention + multi-timescale but on top of W·h-free experts → negative.

**Fast Weights** ADDS a `F_t @ h_{t-1}` term to the standard
recurrent step. `W_h` and CfC's f-gate are PRESERVED. This is the
"additive" pattern that the 11/12 MoE winners followed.

### 1.2 Mechanism

Standard CfC (per Hasani):
- `f = sigmoid(W_f [x, h])`  # per-step interpolation
- `g = tanh(W_g [x, h])`     # candidate
- `h_t = (1 - f) * h + f * g`

Fast Weights CfC (this round):
- `F_t = λ * F_{t-1} + η * h_{t-1} ⊗ h_{t-2}`  # Hebbian update
- `f = sigmoid(W_f [x, h, F_t @ h])`  # gate sees fast-weight interaction
- `g = tanh(W_g [x, h, F_t @ h])`     # candidate sees fast-weight interaction
- `h_t = (1 - f) * h + f * g`

The fast-weight term `F_t @ h` is a learned, time-varying projection
that captures "which past hidden states are most similar to the current
input" — a kind of soft attention over the recent past.

## 2. Hypotheses

- **H1 (fast weights help on noisy data)**: with λ=0.9, η=0.1, test_mse
  on `random_irr` is < unconstrained CfC baseline.
- **H2 (fast weights help on regime switching)**: with λ=0.9, η=0.1,
  test_mse on `structured_irr` (regime switch at T/2) is < baseline
  (because fast weights capture pairwise interactions between adjacent
  states that help at the switch).
- **H3 (no regression on smooth data)**: with λ=0.9, η=0.1, test_mse on
  `sin_irr` is not worse than baseline by >10%.

## 3. Plan

### 3.1 Implementation (`lnn/core/fast_weights_cfc.py`)

Two classes:
- `FastWeightsCfCCell(nn.Module)`: single recurrent step with fast weights
  `F_t` maintained across the forward pass
- `FastWeightsCfCStackedNetwork(nn.Module)`: 2-layer stack

Key implementation details:
- `F` is a buffer (not a parameter) — it changes per forward pass
- `λ` (decay) and `η` (learning rate) are learnable scalars
- `F` is reset to 0 at the start of each forward pass (or use a
  per-call reset)
- Use an upper-triangle approximation for F to reduce parameters
  (F is symmetric: F_t = F_t^T, so we only need the upper triangle)

### 3.2 Tests (`tests/test_fast_weights_cfc.py`)

20+ unit tests covering:
- Init: F is initialized to 0
- Forward: F evolves across steps (not constant)
- Forward: F is reset between sequences
- Hebbian update: F_t has trace proportional to ||h||^2
- Gradient: flows to W_h, W_f, W_g, λ, η
- Stability: no NaN/Inf over 100 steps
- Smoke: learns toy sin in 30 epochs

### 3.3 Bench (`scripts/bench_fast_weights_cfc.py`)

18-24 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `fw_weak` (λ=0.95, η=0.01) — slow fast weights
- `fw_strong` (λ=0.9, η=0.1) — strong fast weights
- `fw_long` (λ=0.99, η=0.05) — long-term fast weights

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

- **Best case (probability ~30%)**: H1 + H2 + H3 all confirmed. Fast
  weights is the **13th STRICTLY POSITIVE** winner. The Hebbian
  short-term memory helps on structured/noisy data without hurting
  smooth data.
- **Likely case (probability ~50%)**: H3 confirmed (no regression on
  sin), H1/H2 partial (modest gains on some datasets, neutral on
  others). **TARGET-DEPENDENT-WITH-NUANCE**.
- **Worst case (probability ~20%)**: All 3 hypotheses rejected. The
  fast weight term F_t @ h is noise on smooth data, doesn't add
  information. 18th negative.

## 5. Why this is worth testing

The 91-132 audit shows that mechanisms that ADD to the recurrent step
(MoE = 11/12 winners) tend to win, while mechanisms that REPLACE the
recurrent step (rounds 128-132) tend to lose. Fast Weights is the
cleanest "additive" mechanism that:
- Preserves W_h (the primary recurrent weight)
- Preserves CfC's f-gate (the per-step interpolation)
- ADDS a fast-weight term that provides short-term Hebbian memory
- Has a learnable decay λ and learning rate η

The risk: fast weights add a H*H matrix per step (potentially
expensive), and the Hebbian update is unsupervised (no learning
signal) which may not match the supervised task.

## 6. Files to create

- `lnn/core/fast_weights_cfc.py` (~200 lines)
- `tests/test_fast_weights_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_fast_weights_cfc.py` (~250 lines, 18-24 cells)
- `docs/research/2026-06-15_fast_weights_cfc_report.md`
