# LNN Research Daily Digest v19 — 2026-06-15 (Round 93)

**Focus**: input-side temporal dropout (follow-up to round 92's target-side) — does the paper's "CfC > LSTM" claim survive when the dropout is on the input (not the loss target)?

## 1. Paper survey

Two fresh signals from this round's arXiv sweep:

- **arXiv:2606.00243** (Williams/Payeur/Lajoie, ICML 2026) — *Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks*. Theoretical analysis of how locality constraints on learning (RFLO, tBPTT) shape solutions in linear RNNs. Key finding: RFLO solutions are confined to low-rank perturbations of initial parameters. Suggests a representational bottleneck in biologically-plausible learning rules.
- **arXiv:2606.02623** (Chandra/Kapoor, June 2026) — *Oscillatory State-Space Models as Inductive Biases for Physics-Informed Neural PDE Solvers*. Embeds linear-oscillator dynamics as a structural prior for PINNs. Improves accuracy and memory vs general SSMs. Sits at the intersection of structured SSMs, continuous-time modeling, and physics-informed learning.

Also surfaced (not pursued this round, on the watchlist):
- **"Rederived Closed-Form Continuous-Time Neural Networks"** (Xia Min, IEEE TNNLS 2026) — needs the actual arXiv preprint, not just the faculty page mention
- **LFM2 Technical Report** (Liquid AI, arXiv:2511.23404) — practical on-device model, abstract-only; gated short conv + grouped query attention; relationship to LNN/CfC is implicit

## 2. Implementation: input-side temporal dropout helper (PRD #10-55)

Round 93 adds 2 functions to `lnn/core/temporal_dropout.py`:
- `input_dropout(t, y, p, seed=None)` — semantic wrapper for input-side masking
- `apply_input_dropout_to_input(t, y, p, seed=None)` — convenience that returns only the masked y

The key conceptual addition: 2D model input (t, y_masked) instead of 1D t. This matches the paper's clinical scenario where the model sees a sparse, gap-filled input stream.

19/19 unit tests pass (up from 13 in round 92; +6 new for input_dropout).

## 3. Test + bench summary

- **19/19 unit tests** pass
- **72-cell bench** (4 models × 6 dropout p × 3 seeds, 100 epochs, 2D input):
  - **H1 ✗ (paper claim NOT rescued)**: CfC degradation@0.8 = 0.41x vs MLP = 0.23x. MLP wins.
  - **H2 ✗ (stateless recovery)**: CfC improves 5x from round 92 to round 93. Two dropout types are NOT equivalent.
  - **H3 PARTIAL (LSTM collapse)**: LSTM 1.39x at p=0.4 then 0.61x at p=0.8. Non-monotonic.
  - **H4 ✓ (regularization)**: ALL models have degradation < 1.0x at p=0.8. Input-side dropout is a strong regularizer.
- **Cumulative suite**: 621/621 in-domain green (up from 602 prior; +19 new)

## 4. Honest-negative: 3-round chain firmly broken

| Round | Hypothesis | Verdict |
|-------|-----------|---------|
| 91 (smoothness) | CfC has lower max_grad than MLP | ✓ (2.03 vs 3.66) |
| 92 (target-side dropout) | Smoother → more robust | ✗ (LSTM wins) |
| 93 (input-side dropout) | Smoother → more robust | ✗ (MLP wins) |

The smoothness prior is a *property* of the model but not a *predictor* of robustness. Robustness hierarchy depends on the dropout regime.

## 5. Verdict on arXiv:2605.27467

| Claim | Status across rounds 92 + 93 |
|---|---|
| CfC is more robust than LSTM | **REJECTED** in both target-side and input-side |
| CfC is more parameter-efficient | Not measured |
| Robustness under missing data | Confirmed only for stateless MLP in input-side regime |

## 6. Implications for the LNN stack

- For 1D function fitting: **MLP is the most robust** to input-side dropout
- For stateful tasks: LSTM/GRU's state-corruption is real but non-monotonic
- For clinical irregular time-series: untested; the paper's domain is plausible

## 7. Cumulative state — 15-layer LNN+MoE 自主栈 (rounds 76-93)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| 92-93 | Dropout audit (target-side, input-side) | diagnostic |

## 8. Backlog for round 94+

1. Real irregular time-series (PhysioNet-style) — test the paper's actual domain
2. Combined smoothness + state — gating to CfC for seq-model-style robustness
3. Audit other layer types (FAME top-K under dropout? ecology gate under dropout?)
4. Multi-axis robustness profile — combine smoothness + 2 dropout regimes
5. Paper-style note combining rounds 91-93 — 3 rejections of the smoothness chain
6. Pivot to a new problem domain (control imitation, long-sequence forecasting)
