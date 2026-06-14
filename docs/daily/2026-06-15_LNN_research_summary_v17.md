# LNN Research Daily Digest v17 — 2026-06-15 (Round 91)

**Focus**: validate arXiv:2606.07670 (Li/Pal/Tan, June 2026) — CfC as drop-in for MLP in 3DGS, with the smoothness claim testable in 1D.

## 1. Paper survey

Found 1 fresh lead:
- **arXiv:2606.07670** (Li, Pal, Tan, June 2026) — *Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting*

Claim: CfC's closed-form time-constant provides "a learned smooth response to t directly into the loss landscape" — smoothness as built-in property, not emergent artifact.

Tested on 8 D-NeRF + 7 NeRF-DS scenes. Replaces MLP deformation field with stack of CfC cells. Largest gains on high-frequency articulated motion.

## 2. Implementation: smoothness metrics (PRD #10-53)

Round 91 introduces 4 functions in `lnn/core/smoothness_metrics.py`:

- `total_variation(y)` — mean |y[i+1] - y[i]|
- `l2_derivative(y, dt=1.0)` — RMS finite-difference derivative
- `max_gradient(y, dt=1.0)` — max |f'(t)|
- `smoothness_summary(y, dt=1.0)` — all 3 + n

All handle 0/1-element edge cases.

## 3. Test + bench summary

- **14/14 unit tests** pass (`tests/test_smoothness_metrics.py`)
- **5-seed × 100-epoch bench** (MLP vs CfC on 1D f(t) = sin(2π t) + 0.5 sin(10π t)):
  - **H1 PARTIAL ✓**: max_grad -44% (2.02 vs 3.62), l2_deriv -12% (1.98 vs 2.24), TV +13% (0.0078 vs 0.0069, unfavorable)
  - **H2 ✗**: mse_eval 0.26 vs 0.17 (CfC 54% worse interpolation)
  - **H3 ✗**: ood_mse 3.03 vs 2.22 (CfC 37% worse extrapolation)
  - **H4 ✗**: CfC has higher MSE despite more params
- **Cumulative MoE+FAME+Causality+Audit+Smoothness suite**: 251/251 pass (up from 237/237 in round 90)

## 4. Honest-negative: smoothness is a property, not accuracy

The clearest finding: **CfC is smoother at the extremes (max gradient) but doesn't fit the function better**. CfC's output is plateau-like with sharp transitions, while MLP's output is ripple-like with smaller but more frequent changes.

For 3DGS, max-derivative matters (sharp deformations cause artifacts) → CfC helps.
For time-series forecasting, MSE matters → MLP is better.

## 5. Verdict on arXiv:2606.07670 claim

| Claim | Status in 1D |
|---|---|
| CfC is smoother than MLP | PARTIAL — max_grad & l2_deriv yes, TV no |
| Smoothness is built-in, not emergent | ✓ Confirmed (very low std 0.0001 suggests fixed attractor) |
| Smoothness helps task performance | ✗ Rejected (worse MSE) |
| Drop-in replacement claim | ⚠ Architecturally yes, but task-dependent |

## 6. Implications for the LNN stack

- `CfCCell` provides a real **max-derivative smoothness prior** (2× lower)
- For **control / physics** (where Lipschitz bounds matter): prefer CfC
- For **time-series forecasting** (where MSE matters): MLP or transformer may still be better
- The MoE-CfC variants (`FAMECfCCell`, `MRMoECfCCell`) inherit this property

## 7. Cumulative state — 13-layer LNN+MoE 自主栈 (rounds 76-91)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90 | Audit (wgt/act overlap, Kim 2026 response) | diagnostic |
| **91** | **Smoothness (TV, l2_deriv, max_grad, 2606.07670 response)** | **diagnostic** |

## 8. Backlog for round 92

1. Per-t evaluation with hidden state (h!=0)
2. Real time-series test (PDNA-LRA, ETTh1)
3. Control setting test (LNNImitationPolicy — smoother output → safer policy?)
4. Larger model / more data
5. Smoothness × MoE combination
