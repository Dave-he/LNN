# PRD #10-94 — AntisymmetricRNN for CfC (Round 132)

**Date**: 2026-06-15
**Round**: 132 (response to AntisymmetricRNN, Chang, Chen, Haber, Chu, \"AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks\", ICLR 2019)
**Status**: Drafted.

## 1. Why round 132

The **AntisymmetricRNN** paper proposes a simple structural change to the
recurrent step: constrain the hidden-to-hidden weight matrix `M` to be
**antisymmetric** (`M = -M^T`). The key insight is dynamical:

- A general W_h has eigenvalues in the complex plane.
- If the maximum real part of any eigenvalue > 0, the system DIVERGES.
- If all real parts < 0, the system is **stable** but converges monotonically
  (not expressive enough for time series with reversals).
- If `M` is ANTISYMMETRIC, all eigenvalues are PURE IMAGINARY (real part = 0) →
  the system is at the **margin** of stability and behaves like an oscillator
  with bounded amplitude.

The paper shows this is a *structural* way to control stability of the
recurrent dynamics without resorting to gating (LSTM/GRU) or saturation
(nonlinearities).

### 1.1 Why this is different from HGRN (round 131)

HGRN **REMOVES** the W·h term and replaces it with gating. It lost because
gating alone lacks CfC's W·h nonlinearity (1.99-10.75× worse in 1D).

AntisymmetricRNN **PRESERVES** the W·h term. It constrains W_h to be
antisymmetric, which is half the parameters (n*(n-1)/2) but keeps the
nonlinearity. This is the right kind of structural change per the 91-131
audit (mechanisms that ADD to the recurrent step win; mechanisms that
REPLACE it lose).

### 1.2 Mechanism

Standard CfC has W_h in the candidate. For AntisymmetricCfC we replace W_h
with an antisymmetric matrix M:

```
h_t = h_{t-1} + dt * (tanh(M @ h_{t-1} + W_x @ x_t + b) - h_{t-1})  # AntisRNN update
M = -M^T  (constrained, half the params of full W_h)
```

With M antisymmetric, the linearization has eigenvalues `±iω` (pure
imaginary), giving the system a "rotation" tendency in state space —
neither diverging nor collapsing to zero.

## 2. Hypotheses

- **H1 (stability constraint helps on noisy data)**: with antisymmetric M,
  test_mse on `random_irr` is < unconstrained CfC baseline (because the
  system can't diverge on noisy inputs that would otherwise blow up W_h
  eigenvalues).
- **H2 (oscillatory behavior helps regime switching)**: with antisymmetric
  M, test_mse on `structured_irr` (regime switch at T/2) is < unconstrained
  baseline (because the rotation tendency matches the regime flip).
- **H3 (no regression on smooth data)**: with antisymmetric M, test_mse on
  `sin_irr` is not worse than unconstrained baseline by >10%.

## 3. Plan

### 3.1 Implementation (`lnn/core/antisymmetric_cfc.py`)

Two classes:
- `AntisymmetricMatrix(nn.Module)` — parameterized antisymmetric matrix using
  upper-triangle storage (n*(n-1)/2 params, reconstructed as M - M^T)
- `AntisymmetricCfCCell(nn.Module)` — single step with antisymmetric W_h
- `AntisymmetricCfCStackedNetwork(nn.Module)` — multi-layer

### 3.2 Tests (`tests/test_antisymmetric_cfc.py`)

20+ unit tests covering:
- Init: M is antisymmetric (M + M^T = 0) by construction
- Forward: shape preservation
- Forward: h stays bounded over 100 steps (no divergence)
- Gradient: flows to upper-triangle storage, reconstructed M, W_x, b
- Stacked: monotonic layer init
- Smoke: learns toy sin in 30 epochs

### 3.3 Bench (`scripts/bench_antisymmetric_cfc.py`)

24 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline, vanilla CfC)
- `antisym_cfc` (AntisymmetricCfC with antisymmetric M)
- `cf_compare_lipschitz` (use Lipschitz-style weight normalization?)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

- **Best case (probability ~30%)**: H1 + H2 + H3 all confirmed. Antisymmetric
  is the **13th STRICTLY POSITIVE** winner. The stability constraint is
  a real regularizer for noisy data, the oscillation tendency helps on
  regime-switching data.
- **Likely case (probability ~50%)**: H3 confirmed (no regression on sin),
  H1 partial (modest gain on random), H2 partial (some gain on structured
  but with caveats). This is **TARGET-DEPENDENT-WITH-NUANCE**.
- **Worst case (probability ~20%)**: All 3 hypotheses rejected. Half the
  parameters mean less capacity, and the oscillation tendency is a
  mismatch for non-oscillatory 1D data. 17th negative.

## 5. Why this is worth testing

The 91-131 audit shows that **structural mechanisms which preserve the W·h
nonlinearity AND add a useful inductive bias win**. AntisymmetricRNN fits
this profile — it constrains the recurrent dynamics to be marginally
stable, which is a stronger inductive bias than CfC's unconstrained W_h.

The risk: antisymmetric M has half the parameters, so the per-layer
capacity is reduced. The reward: stability is built in, not learned.

## 6. Files to create

- `lnn/core/antisymmetric_cfc.py` (~200 lines)
- `tests/test_antisymmetric_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_antisymmetric_cfc.py` (~250 lines, 24 cells)
- `docs/research/2026-06-15_antisymmetric_cfc_report.md`
