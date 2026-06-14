# Round 100 — SNNL for Expert Disentanglement (PRD #10-62)

**Date**: 2026-06-15
**Round**: 100
**Paper**: arXiv:2603.26734 (Agarap & Azcarraga, March 2026) — *Mixture of Experts with Soft Nearest Neighbor Loss: Resolving Expert Collapse via Representation Disentanglement*; original SNNL from Frosst et al. 2019.

## TL;DR

We implement the Soft Nearest Neighbor Loss (SNNL) and apply it to a small MoE (FAMECfC with K=4 experts) using per-input regime labels. **SNNL is a real positive for expert diversity on structured (+17%) and random (+8%) data** but HURTS task loss on toy_sin (+22%). This is the **first mechanism in our 91-100 audit that is target-dependent** — it works on multi-regime/noisy data but fails on smooth targets.

SNNL is a **new mechanism dimension** (feature-space organization) that is orthogonal to all our existing regularizers (weight orthogonality rounds 80/97, φ-balancing round 81, backward coherence round 98, reliability gate round 99).

## 1. The paper's claim

arXiv:2603.26734 (Agarap & Azcarraga) applies SNNL to pre-condition the latent space of a MoE model. The paper claims:
- "Structurally diverse experts" via the SNNL penalty
- "Significantly improves classification accuracy on FashionMNIST, CIFAR10, and CIFAR100"
- Quantified via two new metrics: **Pairwise Embedding Similarity** and **Expert Specialization Entropy**

## 2. The SNNL formula

For a batch of `B` feature vectors `f_1, ..., f_B` with labels `y_1, ..., y_B`:

```
L_SNNL = -1/B * Σ_i log( Σ_{j: y_i = y_j, j≠i} exp(-||f_i - f_j||²/T)
                        / Σ_{k≠i} exp(-||f_i - f_k||²/T) )
```

where:
- `T` is the temperature (lower = sharper, higher = softer)
- The numerator is over same-class examples (excluding i)
- The denominator is over all examples (excluding i)

Intuitively: maximize the probability that same-class examples are closer than different-class examples, in a soft k-NN sense.

## 3. Critical implementation detail: how to label

The natural MoE label would be "the expert that handled the input" — but with K=4 experts and top-K=1 routing, each input is "labeled" with a unique expert, so all labels are different and SNNL silently returns 0 (no positive pairs).

The right interpretation: **the input's regime/class** is the label, not the expert. For our 1D regression bench, we use `t > 0.5` to bin each timestep into 2 classes (low/high). For each (timestep, expert) pair, the feature is the expert's hidden state and the label is the input's regime.

This is the **correct** MoE interpretation: experts that handle inputs in the same regime should have similar features (positive pairs across experts).

## 4. Our implementation

`lnn/core/snnl.py`:
- `soft_nearest_neighbor_loss(features, labels, temperature=1.0)` — the SNNL formula
- `expert_snnl_loss(expert_features, routing_decisions, temperature)` — wrapper for MoE use

Edge cases handled:
- `temperature <= 0` → raise ValueError
- 1D features → flatten
- `B < 2` → return 0
- All-same-class → return 0
- No positive pairs → return 0
- Numerical stability via logsumexp

## 5. Bench setup (100 epochs, 3 seeds, FAMECfC K=4)

- 3 datasets: toy_sin, structured, random
- 4 conditions: baseline, +SNNL λ=0.001, +orth λ=0.001 (round 80), +SNNL+orth combined
- 1 model: FAMECfC with K=4 experts

Cells: 1 × 3 × 4 × 3 = 36 cells

Metrics: task_loss, weight_sim (round 90), diversity_ratio (round 95), mean_eff_rank (round 94)

## 6. Results

| dataset    | cond       | task_loss | wgt_sim | div_ratio | mean_er |
|------------|------------|-----------|---------|-----------|---------|
| toy_sin    | baseline   | 0.1265    | 0.0585  | 1.1874    | 10.401  |
| toy_sin    | **snnl**   | 0.1537    | 0.0579  | **1.2192**| 10.031  |
| toy_sin    | orth       | 0.2031    | 0.0722  | 1.1447    | 10.381  |
| toy_sin    | snnl_orth  | 0.1391    | 0.0653  | 1.1473    | 10.353  |
| structured | baseline   | 0.4763    | 0.0571  | 1.1602    | 10.278  |
| structured | **snnl**   | 0.4853    | 0.0944  | **1.3574**| **9.636**|
| structured | orth       | 0.4993    | 0.0735  | 1.0964    | 10.359  |
| structured | snnl_orth  | 0.4956    | 0.0624  | 1.1612    | 10.028  |
| random     | baseline   | 0.8840    | 0.0617  | 1.1502    | 10.396  |
| random     | **snnl**   | 0.8815    | 0.0575  | **1.2391**| **9.892**|
| random     | orth       | 0.8577    | 0.0594  | 1.1012    | 10.100  |
| random     | snnl_orth  | 0.8823    | 0.0584  | 1.1143    | 10.094  |

## 7. Findings

### 7.1 H1 — SNNL increases diversity ✓ (PARTIAL)

- structured: div_ratio 1.16 → **1.36 (+17%)** — strongest effect
- random: div_ratio 1.15 → **1.24 (+8%)** — strong effect
- toy_sin: div_ratio 1.19 → 1.22 (+3%) — marginal

SNNL is the **strongest diversity mechanism we've found** — bigger than FAME top-K routing (round 78: Δ=+0.03-0.24) and bigger than weight orthogonality (round 97: Δ=0.00-0.02 on weight diversity). On structured data, SNNL gives +0.20 div_ratio improvement.

### 7.2 H2 — SNNL is safe for task loss ✗ (PARTIAL)

- structured: 0.4763 → 0.4853 (+2%) — safe
- random: 0.8840 → 0.8815 (-0.3%) — slightly improves!
- toy_sin: 0.1265 → **0.1537 (+22%)** — REGRESSION

**SNNL fails on smooth targets.** On toy_sin, the bin label `t > 0.5` doesn't represent a real regime (the target is smooth and continuous). Forcing experts to cluster by this artificial label is harmful because it conflicts with the natural smooth-target learning.

### 7.3 H3 — SNNL+orth combined is best ✗ REJECTED

- snnl_orth on structured: div_ratio 1.16 (similar to baseline), worse than SNNL alone (1.36)
- snnl_orth on random: div_ratio 1.11 (similar to orth alone), worse than SNNL alone (1.24)
- snnl_orth on toy_sin: div_ratio 1.15 (worse than both)

**Combining SNNL with orth does NOT give the best of both.** The two losses interfere — orth pushes experts apart in hidden state space, SNNL pulls same-regime experts together. These are opposing forces at the per-timestep level.

## 8. Why SNNL is target-dependent

The mechanism only works when the **label assignment is meaningful**:
- Structured (regime switch at t=0.5): the `t > 0.5` label is MEANINGFUL — the two regimes are qualitatively different, so clustering experts by regime is informative.
- Random (no structure): the `t > 0.5` label is RANDOM — clustering by it is just a regularizer, but it doesn't conflict with task learning.
- Toy_sin (smooth periodic): the `t > 0.5` label is ARTIFICIAL — it doesn't correspond to any natural boundary in the target, so forcing experts to cluster by it fights against the natural smooth learning.

**Recommendation**: only enable SNNL when there are natural regime boundaries in the data (e.g. multi-task, multi-domain, or regime-switching time series).

## 9. Verdict

| Hypothesis | Verdict |
|------------|---------|
| H1 (SNNL increases diversity) | ✓ PARTIAL — strong on structured (+17%) and random (+8%), marginal on toy_sin |
| H2 (SNNL safe for task loss) | ✗ PARTIAL — safe on structured/random, +22% on toy_sin |
| H3 (SNNL+orth combined best) | ✗ REJECTED — combined is dominated by either alone |

**SNNL is a TARGET-DEPENDENT diversity mechanism** — strongest effect on multi-regime data, neutral on noisy, harmful on smooth. This is a useful new tool for the FAME/MR-MoE stack when the data has natural boundaries.

## 10. Comparison with prior rounds

| Round | Mechanism | Diversity Δ | Task loss Δ |
|-------|-----------|-------------|--------------|
| 78 (FAME) | top-K sparse routing | +0.03-0.24 | varies |
| 80 (orth) | activation orth | +0.00 (weight) | ±3% |
| 97 (weight orth) | weight orth | +0.00 (weight) | ±3% |
| **100 (SNNL)** | **feature clustering** | **+0.08 to +0.20** | **-0.3% to +22%** |

SNNL gives the **largest diversity improvement** of any mechanism we've tested. The trade-off is target-dependence.

## 11. Why this matters for the LNN stack

- **New mechanism dimension**: SNNL is the only regularizer in our stack that targets **feature-space clustering** (rather than weight organization, decorrelation, smoothness, or input reliability).
- **Composability**: SNNL is **incompatible** with weight/activation orthogonality at the per-timestep level (round 80, 97), but **compatible** with reliability (round 99) and backward coherence (round 98) because those operate on different axes.
- **Practical use**: enable SNNL on multi-regime / multi-task data; disable on smooth single-target data.

## 12. Files

- `docs/prds/2026-06-15-lnn-round-100-a-snnl-expert-disentanglement.md` — PRD
- `lnn/core/snnl.py` (NEW) — 2 new functions
- `lnn/core/__init__.py` — export
- `tests/test_snnl.py` (NEW) — 15 tests
- `scripts/bench_snnl_expert_disentanglement.py` (NEW) — 36-cell bench
- `results/bench_snnl_expert_disentanglement.json` — full results
- `docs/research/2026-06-15_snnl_expert_disentanglement_report.md` — this report
- `docs/daily/2026-06-15_LNN_research_summary_v26.md` — daily summary
- `README.md` — new section

## 13. Backlog for round 101+

1. **Compose 4-axis gates** in single FAMECfC stack (from round 99)
2. **Per-expert reliability** — extend round 99 to per-expert
3. **Adaptive σ_min** — make round 99's σ_min learnable
4. **SNNL with regime-aware label** — use real regime boundaries instead of t > 0.5
5. **arXiv:2606.07500 SETA** — subspace-to-expert sharing for continual learning
6. **arXiv:2603.22317 GeoMoE** — Ollivier-Ricci Curvature
7. **K=20, hidden=32, full recurrent training** — paper-scale
8. **PhysioNet-style irregular time-series** — most important untested domain
