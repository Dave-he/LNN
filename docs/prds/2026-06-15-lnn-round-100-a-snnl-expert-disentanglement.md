# PRD #10-62 — Soft Nearest Neighbor Loss for Expert Disentanglement (Round 100)

**Date**: 2026-06-15
**Round**: 100
**Status**: Drafted.

## 1. Why round 100

arXiv:2603.26734 (Agarap & Azcarraga, March 2026) — *Mixture of Experts with Soft Nearest Neighbor Loss: Resolving Expert Collapse via Representation Disentanglement* — applies the **Soft Nearest Neighbor Loss (SNNL)** to pre-condition the latent space of a Mixture-of-Experts (MoE) model. The paper claims SNNL produces "structurally diverse experts" and "significantly improves classification accuracy on FashionMNIST, CIFAR10, and CIFAR100."

The mechanism is **fundamentally different from** the regularizers we've tested in rounds 80-99:

| Regularizer | Mechanism |
|-------------|-----------|
| Round 80 (orth) | Decorrelate expert hidden states |
| Round 81 (φ-balancing) | Equalize expert routing probability |
| Round 97 (weight orth) | Decorrelate weight matrices |
| Round 98 (backward coherence) | Penalize hidden state step size |
| Round 99 (reliability) | Dampen output on noisy inputs |
| **Round 100 (SNNL)** | **Cluster same-class inputs in feature space** |

SNNL is a **clustering loss** — it makes same-class examples have similar features in the latent space. For an MoE, "class" can be interpreted as the expert that handles the example, so SNNL encourages each expert to handle a **distinct cluster** of inputs.

This is the **most distinct mechanism** in our 91-100 audit — it's about **feature-space organization**, not weight organization, decorrelation, or smoothness.

## 2. The SNNL formula

For a batch of `B` feature vectors `f_1, ..., f_B` with labels `y_1, ..., y_B`:

```
L_SNNL = -1/B * Σ_i log( Σ_{j: y_i = y_j, j≠i} exp(-||f_i - f_j||²/T) / Σ_{k≠i} exp(-||f_i - f_k||²/T) )
```

where:
- `T` is the temperature (lower T = sharper, higher T = softer)
- The numerator is over same-class examples (excluding i)
- The denominator is over all examples (excluding i)
- The "log" is taken after the ratio

Intuitively: maximize the probability that same-class examples are closer than different-class examples, in a soft k-NN sense.

## 3. Hypotheses

- **H1 (SNNL reduces pairwise embedding similarity)**: with SNNL enabled, the pairwise cosine similarity between expert weights (round 90 metric) is lower than baseline.
- **H2 (SNNL is safe for task loss)**: with SNNL at λ=0.001, task loss within ±10% of baseline.
- **H3 (SNNL complements orthogonality)**: SNNL + weight orthogonality together produce experts that are MORE diverse than either alone.

## 4. Plan

### 4.1 Implementation (`lnn/core/snnl.py` — NEW file)

Add 1 new function:
- `soft_nearest_neighbor_loss(features, labels, temperature=1.0)` — implements the SNNL formula.

Edge cases:
- `temperature <= 0`: raise ValueError
- Single-class batch: degenerate (no negative class) — return 0
- Empty same-class denominator: return 0 (no positive pairs)
- `features` shape: `(B, d)` or `(B,)` — flatten if needed

### 4.2 Apply SNNL to per-expert features (`lnn/core/snnl.py`)

Add 1 new function:
- `expert_snnl_loss(expert_features, routing_decisions, temperature=1.0)` — collects features per expert and applies SNNL with `routing_decisions` as labels.

The expert features can be:
- Per-expert weight matrices (flat, with routing decisions as labels)
- Per-expert output trajectories (sum-pooled, with routing decisions as labels)
- Per-expert hidden state means (over a batch of inputs, with routing decisions as labels)

For round 100 we use the simplest: **per-expert hidden state mean over the trajectory**, with the routing decision as the label.

### 4.3 Tests (`tests/test_snnl.py` — NEW file)

10 new tests:
1. `test_zero_for_perfectly_clustered` — features cluster perfectly by label → loss = 0
2. `test_high_for_randomly_mixed` — features don't cluster → loss > 0
3. `test_temperature_scaling` — lower T → lower loss for clustered data, higher loss for unclustered
4. `test_single_class_returns_zero` — degenerate case
5. `test_gradient_flows` — autograd check
6. `test_rejects_zero_temperature` — validation
7. `test_expert_snnl_basic` — wrapper works
8. `test_expert_snnl_exported` — module export
9. `test_handles_1d_features` — flatten
10. `test_handles_empty_batch` — no positives

### 4.4 Bench (`scripts/bench_snnl_expert_disentanglement.py` — NEW)

24 cells:
- 3 datasets: toy_sin, structured, random
- 4 conditions: baseline, +SNNL λ=0.001, +orth λ=0.001 (round 80), +SNNL+orth combined
- 2 seeds, 100 epochs
- 1 model: FAMECfC with K=4 experts (small MoE)

For each cell measure:
- `task_loss`
- `pairwise_weight_similarity` (round 90 metric, lower = more diverse)
- `expert_diversity_ratio` (round 95 metric, higher = more diverse)
- `mean_eff_rank` (round 94 metric, lower = lower-rank)

H1: +SNNL has lower weight_similarity than baseline.
H2: +SNNL task loss within ±10%.
H3: combined is the most diverse.

## 5. Expected outcomes

| condition | task_loss | weight_sim | diversity_ratio | mean_eff_rank |
|-----------|-----------|------------|------------------|----------------|
| baseline  | 0.13      | 0.65       | 1.30             | 5.0            |
| +SNNL     | 0.13      | **0.50**   | **1.45**         | 4.8            |
| +orth     | 0.13      | 0.55       | 1.40             | 4.5            |
| +SNNL+orth | 0.13     | **0.40**   | **1.55**         | 4.2            |

H1 ✓ if weight_sim drops. H2 ✓ if task_loss preserved. H3 ✓ if combined best.

## 6. Why this matters

- **New mechanism dimension**: SNNL is the only regularizer in our stack that targets **feature-space organization** rather than weight organization, decorrelation, or smoothness.
- **Direct follow-up to round 95**: per-expert diversity was 1.32 (FAME) vs 1.15 (MR-MoE). SNNL could close that gap or push both higher.
- **Composes with rounds 80, 97**: SNNL is orthogonal to orthogonality, so combining gives the best of both.

## 7. Files

- `docs/prds/2026-06-15-lnn-round-100-a-snnl-expert-disentanglement.md` (this file)
- `lnn/core/snnl.py` (NEW) — 2 new functions
- `lnn/core/__init__.py` — export
- `tests/test_snnl.py` (NEW) — 10 tests
- `scripts/bench_snnl_expert_disentanglement.py` (NEW) — 24-cell bench
- `results/bench_snnl_expert_disentanglement.json`
- `docs/research/2026-06-15_snnl_expert_disentanglement_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v26.md`
- `README.md` — new section

## 8. Risk

Low. The SNNL function is well-known and the math is straightforward. The bench reuses the round 95/96/97 infrastructure.

## 9. Compatibility

- `soft_nearest_neighbor_loss(features, labels, temperature)` follows the same signature pattern as `orthogonality_loss`, `weight_orthogonality_loss`, etc.
- Returns 0-d tensor for composability.
- Edge cases handled (empty batch, single class, etc.).
- No Pyright warnings expected beyond pre-existing torch-import false-positives.

## 10. Backlog for round 100+

1. **Compose 4-axis gates** in single FAMECfC stack
2. **Per-expert reliability** — extend round 99 to per-expert
3. **Adaptive σ_min** — make round 99's σ_min learnable
4. **arXiv:2606.07500 SETA** — subspace-to-expert sharing
5. **arXiv:2603.22317 GeoMoE** — Ollivier-Ricci Curvature
6. **K=20, hidden=32, full recurrent training** — paper-scale
7. **PhysioNet-style irregular time-series** — most important untested domain
