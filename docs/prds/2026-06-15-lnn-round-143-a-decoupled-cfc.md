# PRD #10-105 — Round 143: Decoupled CfC + IndRNN-CfC

**Date**: 2026-06-15
**Round**: 143
**Session**: /loop 1h #68
**Author**: heyongxian + Claude (MiniMax-M3)
**Status**: PROPOSAL

## 1. Background

Round 142 (Multiplicative Integration CfC, Wu et al. 2016) was
**CATASTROPHICALLY NEGATIVE** (14th negative in 91-142 audit):

- mi_pure: sin 3.6×, structured 5.2×, **random 19.4× worse**
- mi_x_residual: sin 1.4×, structured 1.8×, random 6.2× worse

The key failure mode: replacing the standard additive integration
`W_x x + W_h h` with the element-wise product `W_x x ⊙ W_h h`
amplifies noise and loses information.

**The natural control experiment**: replace the product with
**additive** combination. This isolates whether the failure was
due to:
- (a) the **element-wise product** (multiplicative amplifies noise)
- (b) the **decoupling** itself (separate W_x, W_h projections)

This round tests both:

1. **Decoupled CfC**: `inter = W_x x + W_h h` (additive, both are
   d×d matrices)
2. **IndRNN-CfC**: `inter = W_x x + u ⊙ h` (additive, h is
   element-wise vector per Li et al. 2018)

## 2. Paper

**Independently Recurrent Neural Network (IndRNN): Building A
Longer and Deeper RNN**
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
CVPR 2018
arXiv:1803.04831

Key idea: replace the d×d recurrent weight matrix U with a
d-vector u (element-wise multiplication `u ⊙ h_{t-1}`). Each
neuron has its own recurrent weight (scalar), not a shared
matrix.

Benefits per the paper:
- Solves gradient vanishing/exploding
- Allows deeper stacking (with ReLU activations)
- Each neuron is independent → interpretable

## 3. Hypothesis

**H1 (Decoupled / IndRNN helps on smooth data)**: with
decoupled/element-wise recurrent weights, test_mse on `sin_irr`
is < baseline.
**PASS criterion**: test_mse ≤ baseline × 0.95.

**H2 (Decoupled / IndRNN helps on structured data)**: with
decoupled/element-wise recurrent weights, test_mse on
`structured_irr` is < baseline.
**PASS criterion**: test_mse ≤ baseline × 0.95.

**H3 (no regression on noisy data)**: with decoupled/element-wise
recurrent weights, test_mse on `random_irr` is not worse than
baseline by >10%.
**PASS criterion**: test_mse ≤ baseline × 1.10.

## 4. Design

### 4.1 Decoupled CfC

`DecoupledCfCCell` — separate W_x and W_h projections, additive
combination:

```python
class DecoupledCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, ...):
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)
        # 3-branch CfC on (x_proj + h_proj)
        ...
    def forward(self, x_t, h, dt=1.0):
        x_proj = self.x_proj(x_t)
        h_proj = self.h_proj(h)
        inter = x_proj + h_proj  # ADDITIVE
        f = self.f_gate(inter)
        ...
```

### 4.2 IndRNN-CfC

`IndRNNCfCCell` — element-wise recurrent weights, additive
combination:

```python
class IndRNNCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, ...):
        self.x_proj = nn.Linear(input_size, hidden_size)
        # u is a vector of size hidden_size (element-wise)
        self.u = nn.Parameter(torch.ones(hidden_size) * 0.5)
        # 3-branch CfC on (x_proj + u * h)
        ...
    def forward(self, x_t, h, dt=1.0):
        x_proj = self.x_proj(x_t)
        h_proj = self.u * h  # element-wise
        inter = x_proj + h_proj
        f = self.f_gate(inter)
        ...
```

### 4.3 Stacked networks

`DecoupledCfCStackedNetwork` and `IndRNNCfCStackedNetwork` — 2-layer
stacks with per-layer decoupled/IndRNN.

## 5. Critical comparison vs round 142 (MI-CfC)

| Variant | Combination | Param count | Round |
|---------|-------------|-------------|-------|
| cfc (baseline) | concat([x,h]) → linear | 2545 | (baseline) |
| mi_pure | x_proj ⊙ h_proj | 2545 | 142 (NEG) |
| mi_x_residual | x_proj ⊙ h_proj + x_proj | 2545 | 142 (NEG) |
| **decoupled** | x_proj + h_proj | 2545 | 143 |
| **indrnn** | x_proj + u ⊙ h | **<2545** | 143 |

IndRNN has fewer parameters (u is d-vector, not d×d matrix).

## 6. Risks

1. **Decoupled CfC might be equivalent to standard CfC** — the
   linear layer in `W[x, h]` can learn the same function as
   `W_x x + W_h h`. This is a degenerate case.
2. **IndRNN may have limited expressiveness** — element-wise
   recurrent weights mean neurons don't interact, which is
   structurally limited.
3. **Fewer parameters in IndRNN might underfit** on small data.

## 7. Validation

- **Unit tests** (≥18): init, forward, gradient flow, stability
  (100 steps), NaN handling, stacked smoke, smoke learns sin,
  IndRNN param count test.
- **Bench**: 18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs).
- **Datasets**: `sin_irr`, `structured_irr`, `random_irr`.

## 8. Expected Verdict

**Speculative TARGET-DEPENDENT** or **HONEST NEGATIVE**:

- If decoupled helps smooth but not noisy → TARGET-DEPENDENT
  (similar to glu_residual 139)
- If IndRNN's element-wise is too restrictive → NEGATIVE
- If both are equivalent to standard CfC → NEUTRAL

The 91-142 audit pattern says additive structures can be
target-dependent (LN 135, conv 137, glu_residual 139) and pure
input-side modifications can be positive (GIS 134, QuITE 102).
So this round has a chance to be TARGET-DEPENDENT or POSITIVE.

## 9. Files

- `lnn/core/decoupled_cfc.py` (new)
- `tests/test_decoupled_cfc.py` (new)
- `scripts/bench_decoupled_cfc.py` (new)
- `results/bench_decoupled_cfc.json` (new)
- `docs/research/2026-06-15_decoupled_cfc_report.md` (new)
- `lnn-round-143-decoupled-cfc.md` (memory)
