# LNN Research Daily Digest v15 — 2026-06-15 (Round 89)

**Focus**: turn round 88's per-expert gradient **diagnostic** into a **policy** (Causality-Gated Orth).

## 1. Paper survey (today, 2026-06-15)

Round 89 prioritized closing the LNN+MoE 自主栈 (closed in round 87) with one more policy layer — no new paper survey today. Recent relevant papers held over from rounds 73-87:

- **DLNet 2601.06227** (round 87-b): dual-stage KD, Pareto LNN edge battery
- **Causal Audit 2606.10703** (round 87): motivates per-expert gradient diagnostic
- **MoE Ecology 2605.06415** (round 83): E = T·H/(O+B) basis

## 2. Implementation: Causality-Gated Orth (PRD #10-51)

Round 88 introduced `per_expert_gradient_norms` as a diagnostic. Round 89 closes the loop:

- `CausalityGatedOrth` class: fires when `max_min_ratio_grad > threshold (default 10.0)`, rescales `λ → 0.001` (sticky)
- `FAMECfCCell(causality_gated_orth=True, causality_ratio_threshold=10.0)` wires it
- `compute_orth_loss_causality(outs, user_lambda, task_loss)` method
- Combined with round 85 E-gate: take `min(effective_lambda)` (strict safe superset)

## 3. Test + bench summary

- **12/12 unit tests** pass (`tests/test_causality_gated_orth.py`)
- **226/226 MoE+FAME+Causality** suite pass (round 87 + 88 + 89 cumulative)
- **Bench (5 epochs × 9 cells)**: structured@all-λ has `cau_fired=True` (sticky from early collapse), toy_sin/random ratios 2-7.6 < 10
- **Verdict**: complementary safety net to E-gate; rarely activates in 5-epoch regime

## 4. Honest-negative: threshold calibration

Round 88 found 13-27× ratios in **no-gate** conditions. Round 89 with **E-gate active** shows only 2-7× in 5-epoch bench. Either:
- Lower the threshold to 5-7 to make causality gate more reactive, OR
- Accept that E-gate is doing the heavy lifting and causality gate is a defense-in-depth backstop

For round 90, **threshold sweep (5, 10, 20)** would help calibrate.

## 5. Cumulative state — LNN+MoE adaptive policy stack (10 layers)

| Round | Layer | Status |
|-------|-------|--------|
| 76 | CfC `n_tau` (multi-time-scale) | ✅ 13/13 tests |
| 77 | MR-MoE (K experts + router) | ✅ 14/14 tests |
| 78 | FAME top-K sparse | ✅ 15/15 tests |
| 80 | Orthogonality constraint | ✅ 12/12 tests |
| 81 | φ-Balancing (EMA mirror-descent) | ✅ 16/16 tests |
| 82 | CosineRouter (honest-negative at toy scale) | ✅ 18/18 tests |
| 83 | MoE Ecology `E = T·H/(O+B)` diagnostic | ✅ 14/14 tests |
| 84 | Ecology-Gated φ-balancing (honest-negative at λ=1.0) | ✅ 13/13 tests |
| 85 | Ecology-Gated Orth Rescaling (fixes round 84) | ✅ 15/15 tests |
| 86 | Combined Gates (H2 confirmed, H3 rejected) | ✅ 17/17 tests |
| 87 | Gradient-based H (causal diagnostic option) | ✅ 14/14 tests |
| 88 | Per-Expert Gradient Norms (causal imbalance) | ✅ 16/16 tests |
| **89** | **Causality-Gated Orth (policy from round 88 diag)** | **✅ 12/12 tests** |

**Cumulative suite**: 226/226 in MoE+gate+causality domain (4 pre-existing failures in unrelated domains: multimodal_regime, multimodal_physreg, pdna_lra, transformable_ltc — confirmed not introduced by round 89).

## 6. Backlog for round 90

1. Threshold sweep (5, 10, 20) for `CausalityGatedOrth`
2. Long-horizon bench (10-20 epochs) to see if causality catches collapse before E drops
3. Combined E + causality on a non-toy dataset (PDNA-LRA, ETTh1, etc.)
4. Investigate why `structured` has persistent 4.95× per-expert grad imbalance despite E-gate firing
