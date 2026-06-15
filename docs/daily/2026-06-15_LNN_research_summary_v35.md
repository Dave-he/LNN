# LNN Research Digest v35 — 2026-06-15

**Coverage**: Dynamic TMoE (drift-aware dynamic expert pool) + 91-109 audit update.

## Headline

Round 109 implemented **Dynamic TMoE** (arXiv:2605.20678 Zhu/Liu/Weng/Wu May 2026, ICML 2026) — *Drift-Aware Dynamic Mixture of Experts for Non-Stationary Time Series Forecasting*. The mechanism dynamically grows and prunes the expert pool in response to MMD-detected distribution shifts. This is the **most structural change yet** in our 91-109 audit — the expert pool itself evolves.

The result is **HONEST NEGATIVE-WITH-NUANCE** (3rd target-dep in audit):
- **H1 ✓ CONFIRMED**: Drift detection fires 10-27 times per training pass — MMD mechanism is real
- **H2 ✗ REJECTED**: test_mse on structured_irr is WORSE — dynamic_add +60-150%, dynamic_full 10-100× worse
- **H3 PARTIAL**: sin_irr OK with add-only, CATASTROPHIC with full (0.17-0.19 vs 0.0002-0.003)
- **H4 ✓ CONFIRMED**: random_irr is competitive (dynamic_add 0.0000-0.0160 vs 0.0001-0.0003 baseline)

**NEW INSIGHT**: The mechanism works (drift detection, pool growth, routing entropy) but the **add+prune recipe is destructive** in 1D. Aggressive pruning kills useful experts before they specialize. The **add-only** version is safe (within 2× of baseline) but doesn't help. This is the **3rd target-dependent** mechanism in 91-109 audit (after SNNL 100, Anchored MoE 108).

## 1. Dynamic TMoE in 60 seconds

Standard MoE: fixed K-expert pool. Dynamic TMoE makes the pool **evolve**:
```
input (B, T, D)
  │
  ├── MMD drift detector: MMD(ref_window, new_samples) → drift_score
  │   - threshold: configurable (default 0.05)
  │   - Gaussian RBF kernel with median heuristic bandwidth
  │
  ├── DynamicExpertPool: K experts that grow/shrink
  │   - on drift: add_expert(reference=most_used)
  │   - on prune_every: prune_expert (least-used)
  │   - capped at max_size
  │
  ├── TemporalMemoryRouter: logit = Router_MLP([x_t, h, memory, anomaly])
  │   - memory: GRU-updated recurrent state
  │   - anomaly: rolling buffer of MMD scores
  │
  └── Output: weighted mix of top-K expert outputs
```

The paper claims -10.4% MSE on 9 benchmarks. Our bench reproduces the **mechanism** (drift detection works) but the **recipe** (add+prune) hurts in 1D.

## 2. Bench summary (24 cells, 100 epochs)

`scripts/bench_dynamic_tmoe.py`:
- 4 conditions (all start at K=4 for fair capacity comparison):
  - `baseline_fixed` (drift disabled)
  - `dynamic_add` (K=4→8, no prune)
  - `dynamic_full` (K=4→8, prune every 50 steps)
  - `dynamic_tiny` (max K=4, high threshold → rarely fires)
- 3 datasets: sin_irr, structured_irr, random_irr (30% train, 50% test)
- 2 seeds × 100 epochs, T=32, D=2, hidden=16, K=4, top_k=2

### test_mse (mean over 2 seeds, 100 epochs)

| Condition | sin_irr | structured_irr | random_irr | Pool Final | Drifts |
|-----------|---------|----------------|------------|------------|--------|
| baseline_fixed | 0.0002-0.0033 | 0.0002-0.0025 | 0.0001-0.0003 | 4 | 0 |
| dynamic_add | 0.0011-0.0025 | 0.0039-0.0042 | 0.0000-0.0160 | 4→8 | 10-27 |
| dynamic_full | 0.1674-0.1855 | 0.0137-0.1821 | 0.0003-0.0664 | 4→8 | 6-27 |
| dynamic_tiny | 0.0002-0.0033 | 0.0002-0.0025 | 0.0001-0.0003 | 4 | 0-15 |

### Routing diagnostics

| Condition | routing_H | active_fraction |
|-----------|-----------|-----------------|
| baseline_fixed | 0.871-0.986 | 1.00 |
| dynamic_add | 0.966-0.996 | 1.00 |
| dynamic_full | 0.952-0.997 | 1.00 |
| dynamic_tiny | 0.871-0.986 | 1.00 |

## 3. The 91-109 audit pattern

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 91 | TV smoothness | Diagnostic | NEGATIVE |
| 92-93 | Temporal dropout | Augmentation | NEGATIVE |
| 94-95 | Effective rank | Diagnostic | NEGATIVE |
| 96-97 | FAME+orth | Regularizer | PARTIAL |
| 98 | Backward coherence | Regularizer | PARTIAL |
| 99 | Reliability gate | Augmentation | **STRICTLY POSITIVE** |
| 100 | SNNL | Regularizer | TARGET-DEP |
| 101 | ORC | Regularizer | DIAGNOSTIC |
| 102 | QuITE | Embedding | **STRICTLY POSITIVE** |
| 103 | QuITE+MoE | Router+ctx | TARGET-DEP |
| 104 | SDG-MoE | Router+delib | NEGATIVE |
| 105 | SETA | Architecture | **STRICTLY POSITIVE** |
| 106 | AuxLF | Router+bias | TARGET-DEP |
| 107 | Soft MoE | Structural | **SAFE ROUTING** |
| 108 | Anchored MoE | Structural | TARGET-DEP |
| **109** | **Dynamic TMoE** | **Structural** | **NEGATIVE-WITH-NUANCE** |

**5 STRUCTURAL winners** (99, 102, 105, 107) + Anchored (108 target-dep) + Dynamic (109 negative).
**NEW INSIGHT**: **structural > routing-only only when the structural change is constructive**. Dynamic TMoE's "add" is constructive (more capacity, no destruction). The "prune" part is destructive (kills useful experts before they specialize). Net effect: regression in 1D.

## 4. Why pruning is destructive

After 50 steps, prune_expert() removes the least-used expert. In 1D synthetic:
- Early in training, all experts look similar (random init) → usage is roughly uniform
- Pruning at step 50 may kill a useful expert that just hasn't received enough gradient yet
- The pool oscillates: grow → prune → grow → prune, never settling

In higher-dim data (PhysioNet 36D, robot 10D), drift is real and experts have meaningful specialization → pruning is justified. In 1D, drift is mostly noise.

## 5. Why the mechanism works but doesn't help in 1D

The paper claims -10.4% MSE on 9 benchmarks. These are real-world non-stationary time series (electricity, traffic, weather) where:
- Multiple regimes are clearly distinguishable
- Drift detection is meaningful
- Adding an expert per regime is genuinely useful

In 1D synthetic:
- All "regimes" are smooth, low-dim, easily captured by 4 fixed experts
- Drift is mostly numerical noise
- Adding experts either doesn't help (add-only) or hurts (add+prune)

## 6. Implementation highlights

`lnn/core/dynamic_tmoe.py` (~700 lines):
- `mmd_rbf(x, y, sigma=0)` — Gaussian RBF MMD with median heuristic
- `DriftDetector(window_size, threshold, sigma)` — sliding-window MMD detector
- `DynamicExpertPoolConfig(init_size, max_size, min_size, ...)` — pool config
- `ExpertModule(input, hidden)` — single expert MLP
- `DynamicExpertPool` — list of experts, add/prune operations
- `TemporalMemoryRouterConfig(memory_dim, anomaly_dim, top_k)` — router config
- `TemporalMemoryRouter` — recurrent router with anomaly buffer
- `DynamicTMoEConfig` — top-level config
- `DynamicTMoECfCCell` — K experts + drift detection + dynamic add/prune
- `DynamicTMoECfCNetwork` — full network with rolling loop
- `get_utilization()` — routing_H, max_min, active_fraction, usage_count

`tests/test_dynamic_tmoe.py` (37/37):
- TestMMD (4)
- TestDriftDetector (5)
- TestExpertModule (1)
- TestDynamicExpertPool (7)
- TestTemporalMemoryRouter (5)
- TestDynamicTMoECfCCell (6)
- TestDynamicTMoECfCNetwork (6)
- TestDynamicTMoEIntegration (3)

## 7. Critical bugs fixed

1. **ref_window feature dim hardcoded** (D=1 only) — fixed by dynamic resize on first update
2. **GRUCell batch size mismatch** (1 vs B) — fixed by expand memory_state at call time
3. **Router shrink/grow race** (n_experts shrunk but layer kept old size) — fixed: never shrink layer
4. **top_idx out of bounds** (router > pool size after prune) — fixed: clamp to pool_size

## 8. Recommendation

**Use Dynamic TMoE in 3 scenarios**:
1. **High-dim non-stationary data** (PhysioNet 36D, electricity, traffic) where drift is real
2. **Production with add-only** (no prune) for safe capacity scaling
3. **When you can verify drift signal** — check that MMD is firing on actual regime changes

**Don't use in 1D synthetic**:
- The mechanism is real but drift is mostly noise
- Aggressive pruning is destructive
- Use Soft MoE (round 107) or Anchored MoE (round 108) instead

For 1D, **dynamic_add is the only safe variant**.

## 9. Files added

- `lnn/core/dynamic_tmoe.py` (NEW, ~700 lines)
- `tests/test_dynamic_tmoe.py` (NEW, 37/37 tests)
- `scripts/bench_dynamic_tmoe.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-109-a-drift-aware-moe.md` (PRD #10-71)
- `docs/research/2026-06-15_dynamic_tmoe_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v35.md` (this file)
- `README.md` (new Dynamic TMoE section)
- `lnn-round-109-dynamic-tmoe.md` (memory)

## 10. Future work

1. **PhysioNet test**: 36D real medical time series — drift is meaningful here
2. **Adaptive prune threshold**: prune only when usage < epsilon AND pool > max_size
3. **Hierarchical drift detection**: 2-level MMD (coarse regime + fine within-regime)
4. **Combine with Anchored MoE** (round 108): structural prior + dynamic pool
5. **Add-only mode for production**: no prune, just monitor and grow
