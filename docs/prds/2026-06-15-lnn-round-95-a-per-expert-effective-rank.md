# PRD #10-57 — Per-Expert Effective Rank (Round 95)

**Date**: 2026-06-15
**Round**: 95 (follow-up to rounds 78, 80, 94; closes backlog item #4)
**Response to**: arXiv:2606.08896 (FAME) core claim — "FAME experts are diverse" — and arXiv:2606.12240 (MR-MoE) — multi-rate expert specialization.
**Status**: Drafted.

## 1. Why round 95

The FAME paper (arXiv:2606.08896, Li et al. 2026) claims that its top-K sparse MoE routes to **diverse** experts, with each expert specializing in a particular forecastability regime. The MR-MoE paper (arXiv:2606.12240, Zong et al. 2026) claims its multi-rate experts specialize by time-scale.

Round 78 (FAME) and round 77 (MR-MoE) implemented cell-level versions of these. Round 80 added an orthogonality-constrained AnchorMoE to **enforce** expert diversity at the weight level. But none of these rounds have ever **measured** the resulting diversity at the weight level.

Round 94 added the `effective_rank` tool (PRD #10-56) — eff_rank(W) = (Σσᵢ)²/(Σσᵢ²), a continuous differentiable proxy for algebraic rank. Round 94 found that CfC has the **highest** weight_eff_rank (8.36) among MLP/CfC/LSTM/GRU.

**Question for round 95**: do FAME/MR-MoE experts actually have **distinct weight signatures** — i.e. is the per-expert eff_rank distribution spread enough to support the "diverse experts" claim?

## 2. Hypotheses

- **H1 (diversity)**: The per-expert weight_eff_rank values in a FAME cell with K=5, top_k=2 should span a range with max/min ratio > 1.5 after 100 training epochs (i.e. experts are NOT all the same).
- **H2 (specialization correlates with routing)**: In FAME, the expert with the **highest** weight_eff_rank should receive above-average routing weight (more "active" experts use more rank).
- **H3 (dead experts = collapsed rank)**: FAME experts that are never selected by the router (utilization = 0) should have lower weight_eff_rank than active experts, OR no difference if dead experts stay at initialization (untrained) — test of "do dead experts get trained at all?".
- **H4 (orthogonality boosts diversity)**: FAME with `orthogonality_loss=True` should have **higher** expert diversity (max/min ratio) than FAME without — this is the round 80 mechanism.

## 3. Plan

### 3.1 Implementation (`lnn/core/effective_rank.py`)

Add 3 new functions:
- `per_expert_effective_rank(cell)` — iterate over `cell.experts[i]`, collect 2D params, compute eff_rank per expert, return a list of K floats.
- `expert_diversity_ratio(ranks)` — max(ranks) / min(ranks). 1.0 = uniform, >1.5 = "diverse".
- `expert_diversity_summary(cell)` — combine: dict with `per_expert`, `diversity_ratio`, `mean`, `min`, `max`, `std`.

These work for **any** `nn.Module` with an `experts: nn.ModuleList` attribute — so FAME, MR-MoE, and any future variant.

### 3.2 Tests (`tests/test_effective_rank.py`)

Add 5 new tests (existing 20 + 5 = 25):
1. `test_per_expert_init_uniform` — fresh FAME experts have diversity_ratio ≈ 1.0 (random init).
2. `test_per_expert_fame_trained_diverse` — FAME trained 100 epochs on toy_sin has diversity_ratio > 1.2.
3. `test_per_expert_mr_moe` — same for MR-MoE.
4. `test_diversity_ratio_uniform_returns_one` — `[3.0, 3.0, 3.0] → 1.0`.
5. `test_diversity_ratio_handles_zeros` — `[0.0, 3.0, 0.0] → inf`, gracefully.

### 3.3 Bench (`scripts/bench_per_expert_effective_rank.py`)

12 cells:
- 3 datasets: `toy_sin` (smooth, predictable), `structured` (regime-switching, FAME-friendly), `random` (uniform noise, MoE should struggle)
- 2 models: `FAMECfCCell(K=5, top_k=2)`, `MRMoECfCCell(K=5, dense)`
- 2 conditions: `trained` (100 epochs), `untrained` (random init, control)
- 3 seeds each

For each cell measure:
- `mse` (toy_sin, structured) or `pseudo_loss` (random)
- `per_expert_eff_rank` (list of K floats)
- `diversity_ratio` (max/min)
- `expert_utilization` (fraction of steps each expert is selected, FAME only)

### 3.4 Expected outcomes

| dataset    | FAME diversity_ratio (trained) | MR-MoE diversity_ratio (trained) |
|------------|-------------------------------|----------------------------------|
| toy_sin    | ~1.3-1.6                      | ~1.1-1.3 (dense routing → similar) |
| structured | ~1.5-2.0 (best — regime switches) | ~1.2-1.5 |
| random     | ~1.0-1.2 (nothing to specialize) | ~1.0-1.2 |

Healthy H1: `diversity_ratio > 1.2` for trained FAME on at least 2/3 datasets.

## 4. Why this matters for the LNN stack

- FAME/MR-MoE are core layers in our 15-layer stack. If experts are **not** diverse, the top-K gating adds cost without benefit.
- Round 80 orthogonality was added to **enforce** diversity, but never measured. This round closes that loop.
- Per-expert eff_rank could be added to the live ecology diagnostic (round 83) as a new layer.

## 5. Files

- `lnn/core/effective_rank.py` — add 3 functions
- `lnn/core/__init__.py` — export
- `tests/test_effective_rank.py` — add 5 tests (20 → 25)
- `scripts/bench_per_expert_effective_rank.py` (NEW) — 12-cell bench
- `results/bench_per_expert_effective_rank.json`
- `docs/research/2026-06-15_per_expert_effective_rank_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v21.md`
- `README.md` — add to "Effective Rank" section

## 6. Risk

Low. `per_expert_effective_rank` is a thin wrapper over the existing `effective_rank`. No training-loop changes. If the hypothesis is rejected, that's a clean honest-negative (rounds 80, 84, 85, 86 partially depend on "diverse experts" being real).

## 7. Compatibility

- `per_expert_effective_rank` works for any `nn.Module` with `cell.experts: nn.ModuleList` — covers FAME (round 78), MR-MoE (round 77), and any future per-expert architecture.
- No Pyright warnings expected (same pattern as existing `effective_rank`).
