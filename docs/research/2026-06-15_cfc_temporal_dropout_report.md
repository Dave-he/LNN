# Round 92 — CfC Temporal Dropout Robustness (PRD #10-54)

**Date**: 2026-06-15 (round 92)
**Response to**: arXiv:2605.27467 (Thu, Oo, Supnithi, May 2026) — *Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility*
**Verdict**: **H1 ✓ (CfC more robust than MLP), H2 ✗ (smoothness doesn't predict robustness across architectures), arXiv:2605.27467 specific claim "CfC > LSTM" REJECTED in 1D**

## 1. The claim being tested

arXiv:2605.27467 reports that **LNNs (CfC) provide "superior parameter efficiency and significantly higher robustness"** compared to LSTM under **temporal dropout** (randomly missing input observations). The paper benchmarks across 4 datasets (N-MNIST, QuickDraw, IAM, PhysioNet Sepsis-3) and stress-tests with temporal dropout.

**The mechanism hypothesis**: CfC's closed-form time-constant provides a smooth interpolation between known data points, naturally handling missing observations.

## 2. Connection to round 91

The round 91 audit established that **CfC has 2× lower max_grad** than MLP (smoother). The natural prediction:

> **If f is smooth, then small perturbations to f's input cause small perturbations to f's output. So max_grad predicts robustness.**

This is the **inverse-Lipschitz argument**: lower max_grad → more robust to input perturbations.

Round 92 tests this **directly** by adding 4 models and a 6-point dropout sweep.

## 3. New helper

Added to `lnn/core/temporal_dropout.py`:
- `temporal_dropout(t, y, p, seed=None)` — mask p fraction of y by setting to 0
- `dropout_mask(n, p, seed=None)` — return boolean keep-mask of shape (n,)

13/13 unit tests pass (range checks, reproducibility, edge cases).

## 4. Bench design

- **Target**: f(t) = sin(2π t) + 0.5 sin(10π t) on t ∈ [0, 1] (round 91 setup)
- **Train**: 64 sparse points, 100 epochs
- **Eval**: 256 dense points (no dropout at eval)
- **Models** (4):
  - MLP: 1 → 16 → 16 → 1, ReLU, 321 params
  - CfC: CfCCell(1, 16) + head, 897 params (stateless, h=0 each t)
  - LSTM: nn.LSTM(1, 16) + head, 1233 params (seq2seq)
  - GRU: nn.GRU(1, 16) + head, 929 params (seq2seq)
- **Dropout p**: 0%, 10%, 20%, 40%, 60%, 80% (6 levels)
- **Seeds**: 3 per cell
- **Total cells**: 4 × 6 × 3 = 72

## 5. Full bench results (100 epochs, 3 seeds)

| model | params | max_grad@0 | p=0.0       | p=0.4       | p=0.8       | degradation@0.8 |
|-------|--------|------------|-------------|-------------|-------------|------------------|
| MLP   | 321    | 3.66       | **0.169**   | 0.240       | 0.501       | **2.96x**        |
| **CfC** | 897  | **2.03**   | 0.259       | 0.314       | 0.533       | 2.06x            |
| **LSTM** | 1233 | 52.79     | 0.337       | **0.287**   | **0.436**   | **1.29x**        |
| **GRU**  | 929  | 37.98     | 0.298       | **0.218**   | 0.500       | 1.68x            |

### 5.1 H1 (CfC degrades less than MLP) — **CONFIRMED**

- MLP degradation@0.8 = 2.96x
- CfC degradation@0.8 = 2.06x
- **CfC is 30% more robust than MLP** (lower degradation at p=0.8)

This **partially** supports the round 91 → round 92 hypothesis chain: smoother model (CfC) IS more robust than less smooth model (MLP).

### 5.2 H2 (smoothness predicts robustness across all models) — **REJECTED**

The ranking by max_grad@0:
- CfC 2.03 (smoothest)
- MLP 3.66
- GRU 37.98
- LSTM 52.79 (least smooth)

The ranking by degradation@0.8:
- LSTM 1.29x (most robust)
- GRU 1.68x
- CfC 2.06x
- MLP 2.96x (least robust)

**The orderings are nearly inverted.** LSTM has 26× higher max_grad than CfC but is dramatically more robust. The smoothness-prior hypothesis from round 91 is **rejected** as a general predictor.

### 5.3 arXiv:2605.27467 specific claim "CfC > LSTM for robustness" — **REJECTED in 1D**

- LSTM degradation@0.8 = **1.29x**
- CfC degradation@0.8 = **2.06x**
- LSTM is **60% more robust** than CfC at p=0.8

The paper's central claim is the **opposite** of what we observe. LSTM is the most robust model in our 1D bench, not CfC.

### 5.4 H3 (no degradation at p=0) — **REJECTED**

At p=0 (no dropout), all models have different mse:
- MLP 0.169 (best)
- CfC 0.259
- GRU 0.298
- LSTM 0.337 (worst)

The models are not equivalent at baseline. MLP is the most accurate at p=0, LSTM is the least.

### 5.5 Crucial observation: dropout acts as REGULARIZATION for seq models

Looking at the degradation ratios for small dropout (p=0.1, 0.2):
- MLP: 1.08x, 1.13x (slight degradation)
- **CfC: 1.02x, 1.05x** (almost no degradation)
- **LSTM: 0.87x, 0.91x** (IMPROVES — mse drops below baseline)
- **GRU: 0.89x, 0.98x** (IMPROVES at p=0.1)

**LSTM and GRU IMPROVE under small-to-moderate dropout** (p=0.1-0.4). The dropout acts as **regularization** for the seq models. This is a separate mechanism from the smoothness prior.

## 6. Honest interpretation

### 6.1 What we learned

1. **CfC IS more robust than MLP** in 1D (H1 ✓) — round 91's smoothness finding has some predictive power within the stateless models
2. **LSTM IS more robust than CfC** in 1D (paper claim ✗) — the paper's claim is task-specific (likely the clinical data has its own structure that benefits from CfC's smoothness)
3. **Smoothness does NOT generalize as a robustness predictor** (H2 ✗) — seq models' gating + state provide a separate robustness mechanism
4. **Dropout can be a regularizer** for seq models — small dropout reduces generalization error
5. **The 2-round hypothesis chain (smoothness → robustness) is partially broken** — it works for stateless models, fails for stateful models

### 6.2 Why the paper might still be right in their domain

The paper tests on 4 datasets including PhysioNet Sepsis-3 (irregular clinical time-series with truly missing data). The 1D bench here uses **target-side dropout** (mask y values), not input-side dropout (mask t values from the model). For real clinical data with irregular sampling, the smoothness prior may matter more.

### 6.3 Verdict: **paper claim REJECTED in 1D, smoothness hypothesis PARTIALLY confirmed**

- **Within stateless models** (MLP vs CfC): smoothness predicts robustness ✓
- **Across architectures** (MLP, CfC, LSTM, GRU): smoothness does NOT predict robustness ✗
- **Paper's "CfC > LSTM" claim**: REJECTED in 1D — LSTM is more robust
- **Mechanism**: seq models have their own robustness via gating + state, which dominates the smoothness prior

## 7. Files

- `docs/prds/2026-06-15-lnn-round-92-a-cfc-temporal-dropout.md` — PRD #10-54
- `lnn/core/temporal_dropout.py` — `temporal_dropout`, `dropout_mask`
- `lnn/core/__init__.py` — export both
- `tests/test_temporal_dropout.py` — 13/13 unit tests
- `scripts/bench_cfc_temporal_dropout.py` — 72-cell bench
- `results/bench_cfc_temporal_dropout.json` — bench output
- `docs/research/2026-06-15_cfc_temporal_dropout_report.md` — this report
- `docs/daily/2026-06-15_LNN_research_summary_v18.md` — digest
- `README.md` — new "CfC Temporal Dropout" section

## 8. Cumulative state — 14-layer LNN+MoE 自主栈 (rounds 76-92)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| **92** | **Dropout robustness (4-model, 6-p sweep, Thu/Oo/Supnithi 2026)** | **diagnostic** |

**Cumulative suite**: 265/265 in MoE+FAME+Causality+Audit+Smoothness+Dropout domains (up from 251/251 in round 91; +13 new dropout tests).

## 9. Backlog for round 93+

1. **Input-side temporal dropout** (mask t values, not y) — closer to real clinical scenario
2. **Real irregular time-series** (PhysioNet-style) — test the paper's actual domain
3. **Combined smoothness + state** — is there a way to add gating to CfC for seq-model-style robustness?
4. **Audit other layer types** — e.g., is FAME's top-K robust to dropout?
5. **Write a paper-style note** combining rounds 91+92 — the 2-round chain is more nuanced than expected
6. **Pivot to a new problem domain** — the stack is well-audited; consider control imitation or long-sequence forecasting
