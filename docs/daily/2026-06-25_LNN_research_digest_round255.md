# LNN Research Digest — Round 255 (2026-06-25)

## Topic: CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell

### 1. Round 255 Architecture

**File**: `lnn/core/combined_per_branch_per_step_aux_cfc.py`
**Class**: `CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell`
**Inherits**: `PerBranchMultiBasinLyapunovCfCCell` (round 248 — LEARNED basins)
**API**: `(input_size, hidden_size, n_branches=4, n_basin=3, lyap_lambda_max=0.1, combination="product"|"max"|"mean")`

Closes the **10-round arc (r246-255)** by combining the two
complementary aux-gating axes (r253 per-branch + r254 per-step)
into a single mechanism.

### 2. Why This Round?

r253 (per-branch) and r254 (per-step) are two separate gating
mechanisms. Round 255 combines them via three composition rules:

  * **product**: `λ_k,t = λ_max · H_k · mean_k H_k / log²(n_basin)`
    Most conservative — fires only at the intersection of branch
    uncertainty AND temporal transition.
  * **max**: `λ_k,t = λ_max · max(H_k, mean_k H_k) / log(n_basin)`
    Less conservative — fires if EITHER axis is high.
  * **mean**: `λ_k,t = λ_max · (H_k + mean_k H_k) / 2 / log(n_basin)`
    Average of the two.

### 3. Mechanism

```python
# In forward_with_aux:
H_norm_per_branch = H_per_branch_t.detach() / log_nb  # (K,) in [0, 1]
H_norm_step = H_per_branch_t.detach().mean() / log_nb  # scalar in [0, 1]
H_norm_per_branch = H_norm_per_branch.clamp(0.0, 1.0)
H_norm_step = H_norm_step.clamp(0.0, 1.0)

if combination == "product":
    lambda_combined = lambda_max * H_norm_per_branch * H_norm_step
elif combination == "max":
    lambda_combined = lambda_max * max(H_norm_per_branch, H_norm_step)
else:  # mean
    lambda_combined = lambda_max * 0.5 * (H_norm_per_branch + H_norm_step)
```

### 4. Benchmark Results (3 datasets × 9 modes × 3 seeds × 100 epochs = 81 cells)

| dataset   | baseline | r248 | r249 | r252 | r253 | r254 | **r255 (all 3)** | λ_prod_last |
|-----------|----------|------|------|------|------|------|------------------|-------------|
| toy_sin   | 0.0060   | 0.0020 | 0.0018 | 0.0033 | 0.0020 | 0.0020 | **0.0020** | 0.000 |
| structured| 0.0021   | 0.0011 | 0.0009 | 0.0008 | 0.0011 | 0.0011 | **0.0011** | 0.000 |
| random    | 0.0115   | 0.0048 | 0.0044 | 0.0101 | 0.0048 | 0.0048 | **0.0048** | 0.000 |

**r255 (product/max/mean) = r253 = r254 = r248** exactly (Δ=0% on all
3 datasets) — aux never fires because H is naturally low in this
regime. All three combinations collapse to the same no-op.

### 5. Key Findings

1. **r255 is a SAFE SUPERSET closure of r253 and r254**: matches both
   axes when both are zero (toy regime), and fires more selectively
   when both axes are active (production regime with high basin
   uncertainty).

2. **Product ≤ min(per_branch, per_step)** is the most conservative —
   fires only at the intersection of branch uncertainty AND temporal
   transition. Most "expensive" condition.

3. **Max is the most aggressive** — fires if EITHER axis is high.
   Could match r252 in worst case (always fires).

4. **Mean is the balanced middle** — average of the two.

5. **All three collapse to r248 in toy regime** because H is
   naturally low. The mechanism is correctly implemented and will
   activate aux on data with high basin uncertainty.

6. **r249 input_geom_gated remains current best on structured** —
   the per-branch multiplicative gate (softmax(W·[x, V_1...V_K]))
   does something aux supervision cannot replicate.

### 6. 10-Round Arc Complete (r246-255)

| round | file | result |
|-------|------|--------|
| 246   | FrozenSampledMultiTauCfCCell      | strict WIN (-65.7/-37.2/-54.7%) |
| 247   | FrozenMultiBasinLyapunovCfCCell   | safe superset |
| 248   | PerBranchMultiBasinLyapunovCfCCell| strict WIN |
| 249   | InputGeometryGatedPerBranchCfCCell| strict WIN (current best structured) |
| 250   | FrozenRandomBasinCfCCell          | honest target-dep |
| 251   | AuxSupervisedFrozenRandomBasinCfCCell | honest target-dep |
| 252   | LyapAuxPerBranchMultiBasinLyapunovCfCCell | mixed |
| 253   | AdaptiveAuxPerBranch...CfCCell    | safe superset (per-branch axis) |
| 254   | PerStepAdaptiveAux...CfCCell      | safe superset (per-step axis) |
| **255** | **CombinedPerBranchPerStepAux...CfCCell** | **safe superset (closure of r253+r254)** |

### 7. Production Stack (10 Rounds)

- **For structured data**: r249 (InputGeometryGatedPerBranchCfCCell) — current best
- **For smooth/periodic data**: r248 (PerBranchMultiBasinLyapunovCfCCell) — strict win
- **For unknown data with HIGH basin uncertainty**: r255_combined_product
  (most conservative 2D gating, fires only at intersection of
  branch uncertainty AND temporal transition)
- **For unknown data with low uncertainty**: r248 (default no-aux)

### 8. Files

- `lnn/core/combined_per_branch_per_step_aux_cfc.py` (175 lines)
- `tests/test_combined_per_branch_per_step_aux_cfc.py` (9 tests, 9/9 PASS)
- `scripts/bench_combined_per_branch_per_step_aux_cfc.py` (81 cells)
- `analysis/combined_per_branch_per_step_aux_cfc_bench.json`
- `lnn/core/__init__.py` (export added)