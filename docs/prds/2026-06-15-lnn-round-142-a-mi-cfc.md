# PRD #10-104 — Round 142: Multiplicative Integration CfC (MI-CfC)

**Date**: 2026-06-15
**Round**: 142
**Session**: /loop 1h #67
**Author**: heyongxian + Claude (MiniMax-M3)
**Status**: PROPOSAL

## 1. Paper

**On Multiplicative Integration with Recurrent Neural Networks**
Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, Ruslan Salakhutdinov
NIPS 2016
Paper: http://papers.nips.cc/paper/6215 (Wu et al. 2016)

## 2. Problem

CfC uses the standard additive integration pattern:
```
combined = W_x x_t + W_h h_{t-1}   # [B, hidden]
f = σ(W_f combined + b_f)
g = tanh(W_g combined + b_g)
h_out = tanh(W_h combined + b_h)
h_t = σ(-f·τ) * g + (1 - σ(-f·τ)) * h_out
```

Wu et al. (NIPS 2016) show that replacing the additive integration
with **multiplicative integration** improves language modeling and
speech recognition:

```
x_proj = W_x x_t            # [B, hidden]
h_proj = W_h h_{t-1}        # [B, hidden]
inter = x_proj ⊙ h_proj     # [B, hidden]   <-- element-wise
f = σ(W_f inter + b_f)
g = tanh(W_g inter + b_g)
h_out = tanh(W_h inter + b_h)
h_t = σ(-f·τ) * g + (1 - σ(-f·τ)) * h_out
```

The key insight: **multiplicative interaction allows the input to
modulate the hidden state per-dimension**, rather than being summed.
This is similar in spirit to gating (LSTM input gate, GRU update
gate) but applied at the integration level.

## 3. Hypothesis

**H1**: Multiplicative integration helps on smooth data
(`sin_irr`). The per-dimension product `x_proj ⊙ h_proj` can
capture phase relationships that additive `+` cannot.
**PASS criterion**: test_mse ≤ baseline × 0.95.

**H2**: Multiplicative integration helps on structured data
(`structured_irr`). The product can capture regime-specific
feature interactions (different `x_proj` patterns for different
regimes).
**PASS criterion**: test_mse ≤ baseline × 0.95.

**H3**: Multiplicative integration does not regress on noisy data
(`random_irr`) by more than 10%.
**PASS criterion**: test_mse ≤ baseline × 1.10.

## 4. Design

### 4.1 Cells

`MultiplicativeIntegrationCfCCell` — standard 3-branch CfC with
multiplicative integration:

```python
class MultiplicativeIntegrationCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, tau_init=1.0):
        # Separate projections for x and h
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)
        # 3-branch CfC on the multiplicative interaction
        self.f_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.g_branch = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.h_branch = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
```

### 4.2 Stacked network

`MultiplicativeIntegrationCfCStackedNetwork` — 2-layer stack with
per-layer MI.

### 4.3 Variants

- `mi_cfc` — pure multiplicative (no additive)
- `mi_add_cfc` — additive + multiplicative (inter + W_h h as residual)
  - Variant to test if MI alone is good or MI + additive is better

Wait — let me keep it simple: only `mi_cfc` (pure multiplicative).
This is the paper's recommendation. Adding additive would mix two
priors and confuse the ablation.

## 5. Risks

1. **Multiplicative may overfit on noisy data** — the product
   `x_proj ⊙ h_proj` is more sensitive to noise than `+` because
   noise multiplies. We need H3 to hold for the verdict.
2. **MI loses information** — additive preserves both x and h
   contributions even when one is zero. Multiplicative zeros out
   when either is zero. Mitigated by the gating structure.
3. **Param count** — `x_proj` and `h_proj` are the same size as
   in the baseline, but we don't need `combined = W [x, h]`. Net
   effect: similar param count.

## 6. Validation

- **Unit tests** (≥15): init, forward, gradient flow, stability
  (100 steps), NaN handling, stacked smoke, smoke learns sin.
- **Bench**: 18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs).
- **Datasets**: `sin_irr`, `structured_irr`, `random_irr`
  (same as rounds 134-141).

## 7. Critical Implementation Details

1. **Initialize `h_proj` to small** (e.g. `std=0.01`) to prevent
   the product from blowing up at init. Or use orthogonal init.
2. **Tau preserved** — same CfC tau mechanism, only the
   integration is multiplicative.
3. **No bias in the multiplicative product** — biases are inside
   the gates (`W_f inter + b_f`), not in the inter itself.

## 8. Expected Verdict

**Speculative POSITIVE** or **HONEST NEGATIVE**:
- If MI is a strong inductive bias for time-series with
  structured patterns, → POSITIVE on structured_irr.
- If MI is too sensitive to noise, → NEGATIVE on random_irr.

The 91-141 audit pattern says structural changes are more
likely to be POSITIVE than per-step modifications.

## 9. Files

- `lnn/core/multiplicative_integration_cfc.py` (new)
- `tests/test_multiplicative_integration_cfc.py` (new)
- `scripts/bench_multiplicative_integration_cfc.py` (new)
- `results/bench_multiplicative_integration_cfc.json` (new)
- `docs/research/2026-06-15_mi_cfc_report.md` (new)
- `lnn-round-142-mi-cfc.md` (memory)
