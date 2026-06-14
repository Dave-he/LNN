# PRD #10-52 — Orthogonality Audit: Weights vs Activations (Round 90)

**Date**: 2026-06-15 (round 90)
**Response to**: arXiv:2601.00457 (Hyunjun Kim, 2026) — "Geometric Regularization in Mixture-of-Experts: The Disconnect Between Weights and Activations"
**Stack target**: stress-test rounds 80-89 orthogonality stack

## 1. The claim we are auditing

Kim 2026 (arXiv:2601.00457) reports that **weight-space geometric regularization in MoE fails**:
- Mean weight-space overlap (MSO) **increases** by up to 114% under the loss
- Activation-space overlap stays at ~0.6 regardless of regularization strength
- Across 7 regularization strengths, correlation between weight and activation orthogonality: r=−0.293, p=0.523 (no significance)
- PTB results have std > 1.0 (inconsistent signal)

**The "disconnect"**: minimizing weight overlap does NOT translate to reduced activation overlap.

## 2. Why this matters for our stack

Rounds 80-89 built a sophisticated orthogonality stack:
- **Round 80**: `orthogonality_loss(outs, lambda_coeff)` — acts on **per-expert outputs** (activations)
- **Round 85**: `EcologyGatedOrth` — auto-rescale λ when E < 0.5
- **Round 89**: `CausalityGatedOrth` — auto-rescale λ when per-expert gradient imbalance > threshold

Critically, our round 80 orthogonality loss operates on the **activations** (`outs` from each expert), NOT on the expert weight matrices directly. This is structurally different from Kim 2026's setup (which regularizes W_i matrices).

**Hypothesis**: our activation-space orthogonality may NOT exhibit the Kim 2026 disconnect, because it operates on the right side of the disconnect (activations, not weights). If confirmed, this validates the stack; if disconfirmed, the stack has a hidden flaw.

## 3. Audit design

### 3.1 New metrics

Add two functions to `lnn/core/moe_ecology.py` (or a new `lnn/core/orthogonality_metrics.py`):

```python
def weight_space_overlap(expert_weights: list[torch.Tensor]) -> float:
    """Mean pairwise |W_i W_j^T| / (||W_i|| ||W_j||) for i != j.
    Expert weights are (out, in) or (out, in) shaped tensors."""
    ...

def activation_space_overlap(expert_outs: list[torch.Tensor]) -> float:
    """Mean pairwise |h_i · h_j| / (||h_i|| ||h_j||) for i != j,
    averaged over batch and time. expert_outs are (B, T, D) per expert."""
    ...
```

### 3.2 Bench

Reuse round 89's bench harness. 2 conditions × 3 datasets × 4 orth λ ∈ {0, 0.1, 1.0, 10.0}:

- (A) `orth_only=False` (baseline, no orth loss)
- (B) `orth_only=True` with `causality_gated_orth=False` (round 80 raw orth)

For each cell, measure:
- `activation_overlap` (mean of last-epoch)
- `weight_overlap` (mean of last-epoch)
- `loss_final`
- `E_emp_last` (round 83)
- `max_min_ratio_grad` (round 88)

Datasets: `toy_sin`, `random`, `structured` (round 83's set).

### 3.3 Hypotheses

- **H1** (Kim 2026 disconnect): weight_overlap **increases** when applying orth loss (the "blowup")
- **H2** (our target): activation_overlap **decreases** when applying orth loss (we hit our objective)
- **H3** (no disconnect in our stack): weight_overlap and activation_overlap are **negatively correlated** under our orth (correlation r < 0)
- **H4** (clean signal): performance (loss_final) **monotonically improves** with λ (no PTB-style noise)

If H2 + ~H3 hold: stack is valid, our activation-space orth works as designed.
If H1 also holds: our stack has a hidden disconnect despite operating on activations.

## 4. Implementation

### 4.1 Step 1: add metrics (1 file, ~30 LOC)
- `weight_space_overlap(expert_weights) -> float`
- `activation_space_overlap(expert_outs) -> float`
- Unit tests verifying:
  - identical matrices → overlap = 1.0
  - orthogonal matrices → overlap = 0.0
  - mean over all pairs (not self)

### 4.2 Step 2: bench script (1 file, ~150 LOC)
- Reuse round 89's train_one harness
- Add weight/activation snapshot at end of training
- Pretty print 2 conditions × 3 datasets × 4 λ matrix
- JSON output

### 4.3 Step 3: 9-cell bench
- 2 conditions × 3 datasets × ~3 λ (λ=0 baseline, λ=0.1, λ=1.0) = 9 cells
- 5 epochs each
- Wall time ~3-5 min

## 5. Success criteria

- **STRONG POSITIVE**: H2 + H3 confirmed → write a paper-style note "Our activation-space orth does NOT exhibit Kim 2026 disconnect, by construction"
- **PARTIAL**: H2 confirmed, H1 not → note the activation-side orthogonality works but with the warning that weight overlap can still grow (relevant for future regularization design)
- **HONEST NEGATIVE**: H1 + H2 both fail → orth loss doesn't even reduce activation overlap; investigate whether `orthogonality_loss` has a bug or is misconfigured
- **KIM 2026 REPRODUCED**: H1 confirmed (weight blowup) but H2 confirmed (activation reduction) → original Kim 2026 finding holds in this codebase AND orth loss is doing the right thing anyway (a nuanced "no, but actually yes" verdict)

## 6. Out of scope

- Scaling to large K (K=3 only — toy regime)
- Long training (>5 epochs)
- Real datasets (PTB, WikiText) — toy only
- Modifying the orthogonality_loss itself (we audit, not change)
- Causality/Ecology gates (round 85-89) — we test raw orth only (λ=0 vs λ>0)

## 7. Deliverables

- `docs/prds/2026-06-15-lnn-round-90-a-orth-weights-vs-activations.md` (this file)
- `lnn/core/moe_ecology.py` (or new metrics file) — add `weight_space_overlap`, `activation_space_overlap`
- `tests/test_orth_audit_metrics.py` — unit tests
- `scripts/bench_orth_weights_vs_activations.py` — bench
- `results/bench_orth_weights_vs_activations.json` — bench output
- `docs/research/2026-06-15_orth_weights_vs_activations_report.md` — findings
- `docs/daily/2026-06-15_LNN_research_summary_v16.md` — digest
- `README.md` — new "Orthogonality Audit" section

## 8. Why this is a worthwhile round 90

The LNN+MoE 自主栈 is closed (rounds 76-89). Kim 2026 (arXiv:2601.00457) is a direct challenge to the foundational assumption of round 80. Stress-testing the stack against an external audit is the right kind of "trust but verify" activity. Findings feed into:
- Whether the stack needs revising (if H1+H2 both fail)
- Whether the stack can be defended in a paper (if H2+H3 hold)
- Future round 91+ directions (audit other layers? extend to real datasets?)

The audit cost is low (3-5 min wall time, ~250 LOC total) and the upside (validated or rejected stack foundation) is high.
