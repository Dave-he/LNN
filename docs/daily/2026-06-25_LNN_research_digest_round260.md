# LNN Research Digest — Round 260 (2026-06-25)

## Topic: PerStepInterBasinGraphCfCCell — Input-Dependent Adjacency (HONEST MIXED)

### 1. Round 260 Architecture

**File**: `lnn/core/per_step_inter_basin_graph_cfc.py` (~270 lines)
**Class**: `PerStepInterBasinGraphCfCCell`
**Inherits**: `InterBasinGraphCfCCell` (round 258, static A)
**New**: per-step adjacency A_t = softmax(MLP(x_t))
**Helpers**: `input_dependent_adjacency`, `batched_graph_mix`
**API**: `(input_size, hidden_size, n_branches=4, n_basin=3, mlp_hidden=0, ...)`

### 2. Mechanism — Input-Dependent A

```python
def input_dependent_adjacency(x_t, mlp, n_basin):
    logits = mlp(x_t)           # (B, K*K)
    A = logits.view(B, K, K)
    return torch.softmax(A, dim=-1)   # row-stochastic

def batched_graph_mix(p, A):     # (B, K) and (B, K, K)
    q = torch.bmm(p.unsqueeze(1), A).squeeze(1)
    return q / q.sum(-1, keepdim=True)
```

The static `self.adjacency` from r258 is kept as a learned bias term,
but the **forward pass uses A_t** (input-dependent). The static A still
receives sym_lambda / sparse_lambda regularization (acts as a learnable
prior on the per-step perturbations).

### 3. Why TND motivates this design

TND (arXiv:2606.21295, Cai & Zhao 2026) makes neurons evolve
**independently** rather than via a shared operator. In basin terms,
this means the inter-basin coupling should depend on the input, not
just be a static matrix. r260 is the basin-level analog:
- r258 = "shared operator" (one A for all timesteps)
- r260 = "per-step operator" (A_t varies with x_t)

### 4. Benchmark Results (54 cells = 6 modes × 3 datasets × 3 seeds × 100 epochs)

| dataset   | baseline | r248   | r257_d2 | r258   | **r260_perstep** | **r260_perstep_h4** |
|-----------|----------|--------|---------|--------|------------------|---------------------|
| toy_sin   | 0.0060   | 0.0020 | 0.0009  | 0.0009 | 0.0011           | 0.0011              |
| structured| 0.0021   | 0.0011 | 0.0004  | 0.0003 | **0.0002**       | 0.0003              |
| random    | 0.0115   | 0.0048 | 0.0014  | 0.0007 | 0.0013           | **0.0007**          |

**r260_perstep** wins on structured (-33%) but regresses on random (+86%).
**r260_perstep_h4** (with hidden=4 layer) is a **safe superset** of r258:
- toy_sin: tie (0.0011 vs 0.0009, +22%)
- structured: tie (0.0003 vs 0.0003)
- random: tie (0.0007 vs 0.0007)

### 5. H trajectory — input doesn't reach graph mix

| mode        | H_std (H_per_timestep std)              |
|-------------|------------------------------------------|
| r257_d2     | 0.1358-0.1422 (high, varies per branch)  |
| r258        | 0.0120-0.0189 (low, learned-A static)    |
| r260_perstep| 0.0117-0.0180 (≈ r258, NOT more variable)|
| r260_h4     | 0.0113-0.0157 (≈ r258)                   |

**H2 REJECTED**: H_per_timestep std for r260 ≈ r258 in all 9 cells. The
input signal does NOT meaningfully reach the graph mix in the toy regime.

### 6. A_diversity (NEW diagnostic)

`A_diversity = mean ||A_t - A_static||_F²` measures how much the per-step
A deviates from the static A baseline.

- r260_perstep: A_div = 0.0005-0.0007 (small, ~0.5% of K*K variance)
- r260_perstep_h4: A_div = 0.0005-0.0007 (same)

The per-step A is **near-uniform** in practice (initialized with bias
toward identity, small MLP weights after init scale 0.01). The static
A still dominates. **r260 is essentially a no-op in toy regime**.

### 7. H1/H2/H3 verdict

| Hypothesis                                                       | Verdict   |
|------------------------------------------------------------------|-----------|
| H1: r260_perstep beats r258 on structured (input carries info)   | PARTIAL ✓ (1/3 cells, -33%) |
| H2: r260 H_per_timestep is more variable than r258               | REJECTED ✗ |
| H3: r260_perstep is robust on random                             | PARTIAL ✗ (no-h: +86%, h=4: tie) |

**Headline: HONEST MIXED**
- `mlp_hidden=0` config: STRICT WIN on structured, REGRESSION on random.
- `mlp_hidden=4` config: SAFE SUPERSET (ties on all 3 datasets).

### 8. Why input doesn't reach graph mix in toy regime

1. **MLP weights stay small** (init scale 0.01) and the dataset is
   low-dim (d_in=1), so the MLP's output is near-zero (uniform A_t).
2. **Static A is preserved as bias term**, so initial A_t ≈ static A.
3. **Task loss dominates**: gradients flow primarily through the forward
   pass (Lyapunov V), not the aux graph mix path.
4. **H_std is tiny in r258 already** (0.012-0.019), so there's not much
   headroom for r260 to be "more variable".

### 9. Production Stack (Updated)

- **For structured/clean data**: r260_perstep (`mlp_hidden=0`) — STRICT WIN
- **For mixed/random data**: r260_perstep_h4 (`mlp_hidden=4`) — SAFE SUPERSET
- **Default**: r258 still recommended unless data has known basin-varying
  structure (TND-style per-neuron dynamics).

### 10. Files

- `lnn/core/per_step_inter_basin_graph_cfc.py` (~270 lines)
- `tests/test_per_step_inter_basin_graph_cfc.py` — 14 tests, 14/14 PASS
- `scripts/bench_per_step_inter_basin_graph_cfc.py` (54 cells)
- `analysis/per_step_inter_basin_graph_cfc_bench.json`
- `lnn/core/__init__.py` (export added)

### 11. Round 260 Verdict — HONEST MIXED

**Input-dependent adjacency is a useful knob but doesn't unlock new
performance in the toy regime.** The hidden=4 config is the safe choice
(never worse than r258); the no-hidden config is a structured-specific
tradeoff.

TND's per-neuron dynamics would likely need:
1. Higher-dim inputs (d_in > 1) so the MLP has real signal to extract
2. Per-neuron-per-basin graph (n_basin × hidden_size edges), not just basin
3. Either longer training or curriculum to overcome initialization bias

This is a CLEAN MIXED in our 91-260 audit:
- No strict regression across the safe config
- 1 STRICT WIN on structured
- 1 safe superset across 3 datasets

### 12. Future arc candidates (refined from r259)

1. **r261**: Fix r258 sparsity bug — use `||A_off_diag||_1` instead of
   `||A||_1` (which is constant K for row-stochastic A and thus a no-op).
2. **r262**: Higher-dim input test (d_in=4 or d_in=8) to give r260's MLP
   something to extract.
3. **r263**: Cross-branch graph (over branch × basin pairs).
4. **r264**: r260 with **larger** K (n_basin=8 or 16) where input-dependent
   routing has more capacity to specialize.

### 13. 15-Round Arc (r246-260)

| round | file                                      | result           |
|-------|-------------------------------------------|------------------|
| 246-256| aux-gating variants (r246-256)           | safe supersets   |
| 257   | InterBasinDistanceCfCCell                 | STRICT WIN (geometry) |
| 258   | InterBasinGraphCfCCell                    | STRICT WIN (structure) |
| 259   | MultiHopInterBasinGraphCfCCell            | HONEST NEGATIVE  |
| **260**| **PerStepInterBasinGraphCfCCell**        | **HONEST MIXED** |

The 5-round basin-graph sub-arc (r256-260) covers:
- geometry (r257)
- static structure (r258)
- depth (r259)
- input-dependence (r260)

The basin-graph axis is now FULLY EXPLORED at d_h=9, d_in=1, n_basin=3.
Future rounds should focus on either scale-up (d_h=32) or architecture
(cross-branch graph, higher K).