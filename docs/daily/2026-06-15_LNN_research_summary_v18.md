# LNN Research Daily Digest v18 — 2026-06-15 (Round 92)

**Focus**: test the prediction from round 91 that CfC's smoothness (max_grad -44%) implies robustness to temporal dropout (arXiv:2605.27467, Thu/Oo/Supnithi May 2026).

## 1. Paper survey

Found 1 fresh lead this round:
- **arXiv:2605.27467** (Thu, Oo, Supnithi, May 2026) — *Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility*

Claim: CfC provides "superior parameter efficiency and significantly higher robustness" compared to LSTM under temporal dropout (missing observations).

Tested on N-MNIST, QuickDraw, IAM, PhysioNet Sepsis-3. Stress-test with temporal dropout injection.

## 2. Implementation: temporal dropout helper (PRD #10-54)

Round 92 introduces 2 functions in `lnn/core/temporal_dropout.py`:

- `temporal_dropout(t, y, p, seed=None)` — mask p fraction of y values
- `dropout_mask(n, p, seed=None)` — return boolean keep-mask

13/13 unit tests pass.

## 3. Test + bench summary

- **13/13 unit tests** pass
- **72-cell bench** (4 models × 6 dropout p × 3 seeds, 100 epochs, 1D f(t) fitting):
  - **H1 ✓**: CfC 30% more robust than MLP at p=0.8 (2.06x vs 2.96x)
  - **H2 ✗**: smoothness does NOT predict robustness across architectures
  - **Paper claim ✗ REJECTED in 1D**: LSTM is 60% more robust than CfC at p=0.8 (1.29x vs 2.06x)
  - **Regularization bonus**: LSTM/GRU IMPROVE under small dropout (0.87x-0.89x at p=0.1)
- **Cumulative suite**: 265/265 in-domain green (up from 251/251; +13 new dropout tests)

## 4. Honest-negative: smoothness doesn't generalize as a robustness predictor

The clearest finding: **within stateless models (MLP, CfC), smoothness predicts robustness. But across architectures (LSTM, GRU), it's a different mechanism entirely**.

| model | max_grad@0 | degradation@0.8 | mechanism |
|---|---|---|---|
| MLP | 3.66 | 2.96x | (no defense) |
| CfC | 2.03 | 2.06x | smoothness prior |
| GRU | 37.98 | 1.68x | gating + state |
| LSTM | 52.79 | 1.29x | gating + state |

LSTM has **26× higher max_grad** than CfC but is **60% more robust**. The smoothness prior from round 91 only matters within stateless models.

## 5. Verdict on arXiv:2605.27467

| Claim | Status in 1D |
|---|---|
| CfC is more robust than LSTM | REJECTED — opposite is true |
| CfC is more parameter-efficient | Not measured (would need matched-accuracy comparison) |
| Robustness under missing data | Confirmed for stateless models only |

The paper's claim may still hold in their domain (clinical irregular sampling) but doesn't generalize to clean 1D function fitting with target-side dropout.

## 6. The 2-round hypothesis chain: smoothness → robustness

**Round 91 (smoothness)**: CfC has 2× lower max_grad than MLP
**Round 92 (robustness)**: CfC is 30% more robust than MLP, but LSTM is 60% MORE robust than CfC

**Verdict on chain**: works for stateless models, breaks across architectures. The smoothness prior is **one** robustness mechanism, not THE mechanism.

## 7. Implications for the LNN stack

- For 1D function fitting: prefer LSTM/GRU (better robustness via state)
- For 3DGS-style tasks (arXiv:2606.07670): prefer CfC (smoother matters for artifacts)
- For clinical irregular time-series: untested — paper's claim may hold
- For real production: pick the right model for the right task, don't generalize

## 8. Cumulative state — 14-layer LNN+MoE 自主栈 (rounds 76-92)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| **92** | **Dropout robustness (4-model, 6-p sweep)** | **diagnostic** |

## 9. Backlog for round 93+

1. Input-side temporal dropout (mask t values, not y) — closer to real clinical scenario
2. Real irregular time-series (PhysioNet-style)
3. Combined smoothness + state — can we add gating to CfC for seq-model-style robustness?
4. Audit other layers (FAME top-K under dropout? ecology gate under dropout?)
5. Write paper-style note combining rounds 91+92
6. Pivot to a new problem domain (control imitation, long-sequence forecasting)
