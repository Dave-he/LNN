# PRD #10-56 — Effective Rank of CfC Trained Solutions (Round 94)

**Date**: 2026-06-15 (round 94)
**Response to**: arXiv:2606.00243 (Williams/Payeur/Lajoie, ICML 2026) — *Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks*
**Connection to round 91**: smoothness prior → low effective rank of trained solutions

## 1. The claim being tested

arXiv:2606.00243 proves that **RFLO** (random feedback local online, a biologically-plausible learning rule) finds solutions that are **low-rank perturbations of the initial parameters**, even in non-data-aligned settings. The result holds for linear RNNs and depends on the *locality* of the learning rule.

**Hypothesis for round 94**: CfC's smoothness prior (round 91: max_grad -44% vs MLP) is functionally a *locality constraint* on the function space. If so, CfC's trained solutions should have **lower effective rank** than MLP, LSTM, or GRU when trained on the same task. This would explain why CfC's robustness is task-dependent (smoothness is a kind of low-rank bias) and would validate the smoothness-prior story from a different angle.

## 2. Effective rank — definition

For a matrix W (e.g., a hidden-to-output weight matrix), the **effective rank** is:
```
eff_rank(W) = (sum_i σ_i)^2 / (sum_i σ_i^2)
```
where σ_i are the singular values of W. eff_rank ∈ [1, min(m, n)] and equals 1 for rank-1, equals min(m, n) for full-rank. It's a continuous, differentiable proxy for algebraic rank.

For an architecture with multiple weight matrices, we can compute:
- `weight_eff_rank` — average over the hidden-state transition matrices
- `hidden_eff_rank` — effective rank of the running hidden states on a held-out input (measures the "manifold dimension" the network actually uses)
- `output_eff_rank` — effective rank of the output layer

## 3. Test design

### 3.1 Setup (same as rounds 91, 92, 93)

- Target: f(t) = sin(2π t) + 0.5 sin(10π t) on t ∈ [0, 1]
- 64 training points, 256 eval points
- 100 epochs, 3 seeds
- Models: MLP, CfC stateless, LSTM, GRU (4, same as rounds 92, 93)

### 3.2 The 3 effective-rank axes (one per model component)

1. **weight_eff_rank**: per-matrix, mean across the 4 models
2. **hidden_eff_rank**: per-time-step, on the dense eval trajectory
3. **output_eff_rank**: per-batch, on the eval predictions

### 3.3 Hypotheses

- **H1 (paper prediction)**: CfC has **lower effective rank** than MLP/LSTM/GRU in at least one of the 3 axes (predicted by 2606.00243's locality argument)
- **H2 (correlation with smoothness)**: Effective rank correlates negatively with max_grad (smoother → lower rank) across the 4 models
- **H3 (CfC is genuinely low-rank)**: CfC's hidden_eff_rank < 4 (the hidden size is 16, so eff_rank < 4 would mean it uses <25% of its representational capacity)
- **H4 (no rank collapse at p=0)**: All models have eff_rank > 2 at baseline (no degenerate solutions)

If H1+H2 ✓: the smoothness prior IS a low-rank bias, and the LNN stack has a principled connection to locality-constrained learning theory.

If H1 ✗: CfC's smoothness is NOT a low-rank bias — it's just a smoothness bias, and the 2606.00243 theory doesn't apply to our stack.

## 4. Implementation

### 4.1 Step 1: effective rank helper (~30 LOC)

```python
def effective_rank(W: torch.Tensor) -> float:
    """eff_rank(W) = (sum σ_i)^2 / (sum σ_i^2) using SVD."""
    s = torch.linalg.svdvals(W.float())
    if s.sum() < 1e-12: return 0.0
    return float((s.sum() ** 2 / (s ** 2).sum()).item())
```

### 4.2 Step 2: per-architecture weight extraction

- MLP: net.0.weight, net.2.weight (Linear layers)
- CfC: cell.f_gate.weight, cell.W_tau.weight, head.weight
- LSTM: lstm.weight_ih_l0, lstm.weight_hh_l0, head.weight
- GRU: gru.weight_ih_l0, gru.weight_hh_l0, head.weight

### 4.3 Step 3: hidden_eff_rank collection

- Train each model for 100 epochs on f(t) (round 91 setup)
- On the dense eval grid, capture the hidden state at every step
  - For LSTM/GRU: capture h_n at every step (output, batch, hidden)
  - For CfC stateless: capture h_new at every step (already a tensor)
  - For MLP: this is N/A (no hidden state) — use the activations of the 2 hidden layers
- Stack the 256 hidden states into a (256, hidden_dim) matrix
- Compute eff_rank on this matrix

### 4.4 Step 4: bench (~150 LOC)

For each (model × seed):
1. Train 100 epochs
2. Compute weight_eff_rank for each trainable weight matrix
3. Compute hidden_eff_rank on the dense eval trajectory
4. Compute output_eff_rank on the dense eval predictions
5. Store all 3 axes

Pretty-print a 4×3 table (model × axis).

## 5. Success criteria

- **STRONG POSITIVE** (H1+H2+H3 ✓): CfC has the lowest eff_rank in all 3 axes, eff_rank correlates with max_grad, CfC's hidden_eff_rank < 4
- **PARTIAL** (H1 ✓ only): CfC is lowest in 1-2 axes
- **HONEST NEGATIVE** (H1 ✗): CfC is not lower-eff_rank than other models, the 2606.00243 theory doesn't apply to our stack
- **UNEXPECTED**: LSTM is lowest-eff_rank (would suggest CfC's smoothness is NOT locality, while LSTM's gating IS)

## 6. Out of scope

- Computing algebraic rank exactly (eff_rank is sufficient)
- Hidden state of MLP — we'll use intermediate activations
- Per-expert rank for MoE (not in this round)

## 7. Deliverables

- `docs/prds/2026-06-15-lnn-round-94-a-effective-rank-cfc.md` (this file)
- `lnn/core/effective_rank.py` — `effective_rank`, `mean_effective_rank`
- `lnn/core/__init__.py` — export
- `tests/test_effective_rank.py` — unit tests
- `scripts/bench_cfc_effective_rank.py` — bench
- `results/bench_cfc_effective_rank.json` — bench output
- `docs/research/2026-06-15_cfc_effective_rank_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v20.md`
- `README.md` — new "Effective Rank" section
- Memory: `lnn-round-94-effective-rank-cfc.md`

## 8. Why this is round 94

1. **Direct response** to a fresh ICML 2026 paper (2606.00243) on locality-constrained learning
2. **Rescues the smoothness story** from a different angle: not "smoothness → robustness" (broken in rounds 92, 93) but "smoothness → low-rank" (theoretically motivated)
3. **Small scope** — 1 helper, 1 bench, 6-8 tests
4. **Connects to 4 papers** in the audit chain: round 91 smoothness + round 92 target-side + round 93 input-side + 2606.00243 locality
5. **Honest-negative friendly** — if H1 ✗, we learn that CfC's smoothness is NOT a low-rank bias

## 9. Backlog updates

After this round:
- ~~Multi-axis robustness profile~~ ← deferred (now split into 4 audit rounds)
- **NEW** Combined smoothness + state (gating for CfC) — backlog #2 stays open
- Real irregular time-series (PhysioNet-style) — backlog #1 stays open
