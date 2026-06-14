# PRD #10-53 — CfC Temporal Smoothness (Round 91)

**Date**: 2026-06-15 (round 91)
**Response to**: arXiv:2606.07670 (Li, Pal, Tan, June 2026) — *Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting*
**Stack target**: validate CfC's smoothness claim directly on our 1D time-series forecasting bench

## 1. The claim being tested

arXiv:2606.07670 makes a specific empirical claim about CfC (Closed-form Continuous-depth):

> "We redesign the deformation field as a stack of Closed-form
> Continuous-time (CfC) cells, a Liquid Neural Network (LNN), that is
> the closed-form solution of the Liquid Time-constant ODE ... Each
> cell exposes a 'sigmoidal time gate that interpolates between two
> candidate hidden states,' embedding a learned smooth response to t
> directly into the loss landscape. ... Temporal smoothness is now a
> built-in property rather than an emergent artifact."

**Hypothesis**: CfC's closed-form time-constant provides **temporal smoothness** as a built-in property, whereas an equivalent MLP has no such inductive bias and produces outputs that are functionally discontinuous in t even if smooth in weights.

This is a **drop-in replacement** claim — same parameter count, same architecture, but different inductive bias. We can test it directly on our 1D time-series forecasting bench.

## 2. Why this matters for our stack

Our CfC-based networks (`CfCCell`, `CfCNetwork`, `FAMECfCCell`, etc.) are used for:
- Time-series forecasting (round 73 CfC vs SSM)
- Control imitation (`LNNImitationPolicy`)
- Long-sequence classification (`LongSequenceLiquidClassifier`)

If the smoothness claim is **task-independent** (i.e., CfC produces smoother outputs than MLP for the same parameter count, regardless of task), we can use this as a **prior** for tasks where smoothness matters (control, physics-informed models).

If the claim is **task-dependent** (only helps for the 3DGS deformation field), we shouldn't generalize.

## 3. Test design

### 3.1 Setup

Generate a smooth 1D function with high-frequency content:

```python
f(t) = sin(2π t) + 0.5 * sin(10π t) + 0.1 * noise
t ∈ [0, 1] sampled at 64 points (training), 256 points (eval)
```

The training set is a sparse subset; the eval set is dense to test interpolation smoothness.

### 3.2 Models (matched parameter count ≈ 200 params)

- **MLP**: `MLP(1 → 16 → 16 → 1)` (small, ~200 params, ReLU activations)
- **CfC**: `CfCCell(1, 16) → head(16, 1)`, single-step unroll with t as input

Both are trained with the same optimizer (Adam, lr=1e-2) for the same number of epochs (50-100).

### 3.3 Metrics

For each model on the dense eval set:
- **MSE**: prediction error
- **Smoothness (TV)**: total variation = Σ|f'(t_i)| ≈ Σ|f(t_{i+1}) - f(t_i)| (lower = smoother)
- **Smoothness (L2 derivative)**: ∫|f'(t)|² dt (lower = smoother)
- **Max gradient**: max|f'(t)| (lower = smoother)
- **OOD generalization**: MSE on t ∈ [1.0, 1.2] (extrapolation)

### 3.4 Hypotheses

- **H1** (paper claim): CfC produces **smoother** outputs than MLP for the same parameter count (lower TV / lower max gradient)
- **H2** (interpolation): CfC interpolates **better** on dense eval set between sparse training points
- **H3** (extrapolation): CfC extrapolates **better** outside training range
- **H4** (task-equivalent): CfC has comparable or lower MSE than MLP

If H1 ✓, the smoothness claim is **architectural, not task-specific** → we have a strong prior for control/physics applications.
If H1 ✗, the claim is **task-specific** to 3DGS deformation → we should not generalize.

## 4. Implementation

### 4.1 Step 1: define models (1 file, ~100 LOC)

Use the existing `lnn/core/cfc.py: CfCCell` for CfC. For MLP, use a simple `nn.Sequential`.

### 4.2 Step 2: smoothness metrics (1 file, ~50 LOC)

```python
def total_variation(y: torch.Tensor) -> float:
    """TV = mean |y[i+1] - y[i]|. Lower = smoother."""
    return float((y[1:] - y[:-1]).abs().mean().item())

def l2_derivative(y: torch.Tensor, dt: float = 1.0) -> float:
    """L2 norm of finite-difference derivative."""
    d = (y[1:] - y[:-1]) / dt
    return float((d ** 2).mean().sqrt().item())

def max_gradient(y: torch.Tensor, dt: float = 1.0) -> float:
    """Max |f'(t)| via finite differences."""
    d = (y[1:] - y[:-1]) / dt
    return float(d.abs().max().item())
```

### 4.3 Step 3: bench (1 file, ~100 LOC)

For each model × 5 random seeds × 100 epochs:
- Train on 64-point sparse grid
- Evaluate on 256-point dense grid (interpolation) + 64-point OOD grid (extrapolation)
- Report MSE, TV, L2-deriv, max-gradient, OOD-MSE

Pretty-print 2 (model) × 5 (seeds) → mean ± std per metric.

## 5. Success criteria

- **STRONG POSITIVE** (H1+H2 ✓): CfC has measurably lower TV and max-gradient than MLP, with comparable MSE → write up the smoothness claim as a stack property, cite arXiv:2606.07670 as the 3DGS counterpart
- **PARTIAL** (H1 ✓, H2 ✗): CfC is smoother but doesn't interpolate better → smoothness is real but doesn't directly help interpolation
- **NEGATIVE** (H1 ✗): MLP is smoother or equivalent → 2606.07670's claim is task-specific to 3DGS, not a general property of CfC
- **HONEST NEGATIVE** (H1 ✗ + H4 ✗): CfC is worse on smoothness AND MSE → the claim doesn't hold in 1D, possibly because 1D doesn't have the structure the paper's domain has

## 6. Out of scope

- 3D Gaussian Splatting (would need a major pipeline)
- Real-time control (we just test 1D function fitting)
- Comparison to SSM/Mamba (round 73 already did this; we focus on MLP-vs-CfC)
- FAME/MoE variants (smoothness is a base-CfC property)
- Round 85-89 gates (we test raw MLP vs raw CfC)

## 7. Deliverables

- `docs/prds/2026-06-15-lnn-round-91-a-cfc-temporal-smoothness.md` (this file)
- `lnn/core/smoothness_metrics.py` — TV, L2-deriv, max-gradient
- `tests/test_smoothness_metrics.py` — unit tests
- `scripts/bench_cfc_temporal_smoothness.py` — MLP vs CfC bench
- `results/bench_cfc_temporal_smoothness.json` — bench output
- `docs/research/2026-06-15_cfc_temporal_smoothness_report.md` — findings
- `docs/daily/2026-06-15_LNN_research_summary_v17.md` — digest
- `README.md` — new "CfC Temporal Smoothness" section

## 8. Why this is a worthwhile round 91

1. **Direct response** to a fresh June 2026 paper (2606.07670)
2. **Completes the audit story** from round 90 — we now have audit + property-test
3. **Small scope** (~250 LOC, 5-10 min wall time)
4. **Stack-level finding**: if H1 ✓, the smoothness property becomes a documented prior for our CfC-based cells
5. **Honest-negative friendly**: if H1 ✗, we still learn that 2606.07670's claim doesn't generalize to 1D

The audit cost is low and the upside (validated smoothness prior, or honest negative limiting the claim) is high.
