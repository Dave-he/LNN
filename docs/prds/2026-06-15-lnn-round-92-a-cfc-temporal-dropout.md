# PRD #10-54 — CfC Temporal Dropout Robustness (Round 92)

**Date**: 2026-06-15 (round 92)
**Response to**: arXiv:2605.27467 (Thu, Oo, Supnithi, May 2026) — *Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility*
**Connection to round 91**: smoothness prior predicts robustness to perturbations

## 1. The claim being tested

arXiv:2605.27467 reports that **LNNs (CfC) provide "superior parameter efficiency and significantly higher robustness"** compared to LSTM under **temporal dropout** (randomly missing observations in the input sequence). The paper benchmarks across 4 datasets (N-MNIST, QuickDraw, IAM, PhysioNet Sepsis-3) and finds:

> "LNNs consistently provide superior parameter efficiency and significantly higher robustness" in native temporal domains and clinical settings with data sparsity.

The **mechanism hypothesis**: CfC's closed-form time-constant solution provides a smooth interpolation between known data points, naturally handling missing observations. This connects directly to round 91's finding (max_grad -44% for CfC vs MLP) — **smoother functions should be more robust to perturbations**.

## 2. Why this matters for our stack

The round 91 audit (PRD #10-53) established that CfC is **dramatically smoother at the max-derivative level** (2× lower max_grad than MLP). The natural prediction is:

> **If f is smooth, then small perturbations to f's input cause small perturbations to f's output. So max_grad predicts robustness.**

This is the **inverse-Lipschitz argument**: Lipschitz constant L = max |f'(t)| bounds the degradation rate. Lower L → more robust to input perturbations.

If round 91's smoothness finding **predicts** round 92's robustness finding, the two combine into a coherent property: **CfC is a Lipschitz-regularized function approximator** — smoothness is not just a property, it's a robustness mechanism.

## 3. Test design

### 3.1 Setup

Same as round 91: f(t) = sin(2π t) + 0.5 sin(10π t), 64 training points, 256 eval points.

### 3.2 Models (4 total)

- **MLP**: stateless, ~321 params (round 91 baseline)
- **CfC**: stateless (h=0 each t), ~897 params (round 91 baseline)
- **LSTM**: nn.LSTM(input=1, hidden=16), full sequence unroll
- **GRU**: nn.GRU(input=1, hidden=16), full sequence unroll

LSTM/GRU added for the paper's direct comparison (the paper compares CfC vs LSTM).

### 3.3 Temporal dropout (the new axis)

For each training run, randomly **mask a fraction p of input points**:
- Replace masked (t_i, y_i) with (t_i, 0) or skip entirely
- Eval on the original dense grid (no dropout at eval time)
- p ∈ {0%, 10%, 20%, 40%, 60%, 80%}

### 3.4 Metrics

- **mse_eval**: prediction error on dense grid
- **mse_degradation_ratio**: mse(p) / mse(p=0) — how much worse than no-dropout
- **max_grad** (from round 91): for the p=0 model
- **smoothness_degradation**: how much smoothness changes under dropout (should be small for CfC)

### 3.5 Hypotheses

- **H1 (paper claim, robust degradation)**: CfC's `mse_degradation_ratio` is **lower** than MLP/LSTM/GRU at p=40%, 60%, 80%
- **H2 (round 91 prediction)**: The ordering of degradation rates matches the ordering of max_grad: smoother model → more robust
- **H3 (no degradation at p=0)**: All models have comparable mse at p=0 (sanity check)
- **H4 (catastrophic threshold)**: At some dropout p*, the model collapses (mse → baseline). The p* should be HIGHER for CfC

If H1+H2 ✓: round 91's smoothness is a real Lipschitz bound, and CfC is robust to missing data.
If H1 ✗: paper's claim is task-specific (clinical data has its own structure), doesn't generalize to 1D.

## 4. Implementation

### 4.1 Step 1: temporal dropout helper (1 file, ~30 LOC)

```python
def temporal_dropout(t: torch.Tensor, y: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask p fraction of (t, y) pairs by setting y to 0."""
    if p == 0:
        return t, y
    mask = torch.rand_like(y) > p
    return t, y * mask.float()
```

### 4.2 Step 2: LSTM/GRU baselines (~50 LOC)

```python
class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)
    def forward(self, t, x_masked):
        # t: (B, T), x_masked: (B, T) — use x as input
        _, (h, _) = self.lstm(x_masked.unsqueeze(-1))
        return self.head(h.squeeze(0)).squeeze(-1)
```

### 4.3 Step 3: bench (~100 LOC)

For each (model × dropout p × 3 seeds):
- Train 100 epochs on dropout-masked data
- Eval on dense grid
- Record mse_eval, max_grad, smoothness_summary

Pretty-print 4 × 6 degradation table.

## 5. Success criteria

- **STRONG POSITIVE** (H1+H2 ✓): CfC's degradation curve is consistently below MLP/LSTM/GRU; ordering matches max_grad ranking
- **PARTIAL** (H1 ✓, H2 ✗): CfC is more robust but ordering doesn't perfectly match max_grad
- **HONEST NEGATIVE** (H1 ✗): All models degrade similarly — paper's claim is task-specific, doesn't generalize to 1D
- **UNEXPECTED** (LSTM wins): LSTM/GRU are more robust than CfC (would be a strong honest negative on the smoothness-prior hypothesis)

## 6. Out of scope

- Real clinical data (the paper's domain)
- Multi-modal sequential data
- Continuous-time ODE solvers (we use the closed-form directly)
- Round 85-89 gates (we test raw models)

## 7. Deliverables

- `docs/prds/2026-06-15-lnn-round-92-a-cfc-temporal-smoothness.md` (this file)
- `lnn/core/temporal_dropout.py` — `temporal_dropout` helper
- `tests/test_temporal_dropout.py` — unit tests
- `scripts/bench_cfc_temporal_dropout.py` — bench
- `results/bench_cfc_temporal_dropout.json` — bench output
- `docs/research/2026-06-15_cfc_temporal_dropout_report.md` — findings
- `docs/daily/2026-06-15_LNN_research_summary_v18.md` — digest
- `README.md` — new "CfC Temporal Dropout" section

## 8. Why this is a worthwhile round 92

1. **Direct response** to a fresh May 2026 paper (2605.27467)
2. **Tests a specific prediction from round 91** — does smoothness prior → robustness?
3. **Completes the smoothness story** — round 91 measured the smoothness, round 92 measures the consequence
4. **4-model comparison** — MLP, CfC, LSTM, GRU (the paper's two, plus two from our stack)
5. **Small scope** (~250 LOC, 5-10 min wall time)
6. **Honest-negative friendly** — if H1 ✗, we learn that the paper's robustness claim is task-specific

The audit cost is low and the upside is high: validates or refutes a 2-round hypothesis chain (smoothness → robustness).
