# Round 96 — FAME with Orthogonality Test (PRD #10-58)

**Date**: 2026-06-15 (round 96)
**Response to**: open H4 from round 95 — does the round 80 orthogonality constraint (PRD #10-37, arXiv:2606.03631 AnchorMoE) actually increase FAME's expert diversity at the safe λ=0.001 setting?
**Direct follow-up to**: PRD #10-37 (round 80, orthogonality), PRD #10-42 (round 83, λ=0.001 safe), PRD #10-56 (round 94, effective rank), PRD #10-57 (round 95, per-expert diversity).
**Verdict**: **H1 REJECTED** (orth has no effect on weight diversity at λ=0.001, Δ = -0.07 to +0.01), **H2 CONFIRMED** (task loss within ±3% of baseline), **H3 PARTIAL** (orth reduces activation cos_sim on structured/random by 0.08-0.16, no effect on toy_sin). Clean honest-negative: round 80's mechanism is too weak at λ=0.001 to break the weight-level convergence, but it does what it was designed to do (decorrelate activations).

## 1. Why round 96

Round 80 added `orthogonality_loss` (PRD #10-37, arXiv:2606.03631 AnchorMoE) to the FAME stack with the claim that it "encourages diverse expert representations". Round 83 confirmed λ=0.001 is safe (no task-loss regression). Round 85 added auto-rescaling.

Round 94 added the `effective_rank` tool. Round 95 used it to discover that FAME develops modest diversity (1.32 max) but H4 — does adding the orth loss boost this further? — was left untested.

**Question for round 96**: at the safe λ=0.001 setting, does FAME+orth have higher `diversity_ratio` than FAME-baseline?

## 2. The prediction

- **H1**: FAME+orth(λ=0.001) diversity_ratio ≥ 1.10× baseline on toy_sin (1.32 → ≥ 1.45).
- **H2**: FAME+orth task loss within ±10% of baseline (round 83 said λ=0.001 is safe).
- **H3**: FAME+orth mean pairwise |cos_sim| of expert hidden states < baseline (orthogonality_loss targets this directly).

## 3. Setup (round 96)

Same datasets as round 95 (toy_sin, structured, random).
Same model: FAMECfCCell(K=5, top_k=2, hidden=8).
Same training: stateless, h reset each step, 100 epochs, lr=1e-2, 3 seeds.

For FAME+orth, the orth loss is added to the task loss:
```python
h_new, expert_outs = cell.forward_with_aux(x, h, dt=1.0)
# collect per_expert trajectories
orth_aux = orthogonality_loss(per_expert_traj, lambda_coeff=0.001)
total_loss = task_loss + orth_aux
```

The orth loss is computed from the **same forward pass** as the task loss, so gradients flow back to the expert weights.

For each cell we measure:
- `diversity_ratio`: max/min per-expert eff_rank (round 95 tool)
- `mean_eff_rank`: mean per-expert eff_rank
- `task_loss`: final MSE
- `activation_cos_sim`: round 90's `activation_space_overlap`

## 4. Full bench results (100 epochs, 3 seeds, λ=0.001)

| dataset    | cond     | div_ratio       | mean_eff   | task_loss    | act_cos       |
|------------|----------|-----------------|------------|--------------|---------------|
| toy_sin    | baseline | 1.32 ± 0.08     | 5.13       | 0.1254       | 0.7555        |
| toy_sin    | **orth** | 1.31 ± 0.06     | 5.12       | 0.1253       | 0.7337        |
| structured | baseline | 1.15 ± 0.04     | 5.31       | 0.5043       | 0.4238        |
| structured | **orth** | 1.16 ± 0.03     | 5.45       | 0.4935       | **0.2600**    |
| random     | baseline | 1.31 ± 0.08     | 5.49       | 0.8857       | 0.3319        |
| random     | **orth** | 1.24 ± 0.04     | 5.46       | 0.9142       | **0.2472**    |

| dataset    | Δ div_ratio | Δ task_loss | Δ act_cos |
|------------|-------------|-------------|-----------|
| toy_sin    | **-0.01**   | 0%          | -0.022    |
| structured | **+0.01**   | -2%         | **-0.164** |
| random     | -0.07       | +3%         | **-0.085** |

## 5. Hypotheses verdict

### H1 (orth increases weight diversity): **REJECTED**

- diversity_ratio: 1.32 → 1.31 (toy_sin, Δ-0.01)
- diversity_ratio: 1.15 → 1.16 (structured, Δ+0.01)
- diversity_ratio: 1.31 → 1.24 (random, Δ-0.07, slight drop)
- All deltas are within noise. Adding orth at λ=0.001 has **essentially no effect** on weight-level diversity.

The orthogonality loss is computed on the per-step **hidden states** (activations), not on the weight matrices. So the constraint nudges the *direction* of the expert outputs but doesn't reshape the underlying weight singular values. Weight eff_rank is determined by the structure of the weight matrices themselves.

### H2 (orth is safe for task loss at λ=0.001): **CONFIRMED**

- toy_sin: 0.1254 → 0.1253 (0% change)
- structured: 0.5043 → 0.4935 (-2% improvement, actually!)
- random: 0.8857 → 0.9142 (+3% degradation, within noise)

All within the ±10% safe band from round 83. The orth loss is a stylistic tax at most, and can occasionally help (structured data -2%).

### H3 (orth reduces activation cos_sim): **PARTIAL CONFIRMED**

- toy_sin: 0.7555 → 0.7337 (Δ-0.022, marginal)
- structured: 0.4238 → 0.2600 (**Δ-0.164**, strong effect)
- random: 0.3319 → 0.2472 (**Δ-0.085**, strong effect)

The orthogonality loss does what it was designed to do: decorrelate expert **activations**. On structured data (where experts have the most room to specialize) and random data (where there's nothing to fit but the orth gradient dominates), the effect is strong. On toy_sin (where all experts try to fit the same smooth curve), the activation cos_sim is already high and orth has little room to push.

This is consistent with round 90's finding that the **activation-space** orth effect is real at λ=0.01+ (where wgt_ov +44-48%, act_ov -47-54%) — at our safe λ=0.001 setting, the effect is smaller but still measurable on structured data.

## 6. Honest interpretation

### 6.1 What we learned

1. **Round 80's orthogonality loss works as designed**: it decorrelates expert hidden states (activations) when there's room to do so.
2. **It does NOT increase weight-level diversity** at the safe λ=0.001 setting. Weight eff_rank is robust to this small orth nudge.
3. **The "diverse experts" claim in FAME is at the activation level, not the weight level**. If we want diverse weight signatures, the orth constraint at λ=0.001 is too weak; we'd need a different mechanism (e.g. orth on the **weight** matrices themselves, not on activations).
4. **The orth loss is safe** (H2 confirmed), so it can stay in the stack as a defensive measure against the round 80 / 90 "what if routing collapses" scenario.

### 6.2 The activation-vs-weight distinction

This round clarifies a subtle but important distinction:
- **Activation diversity** = expert hidden states are decorrelated on the same input
- **Weight diversity** = expert weight matrices have different singular value spectra

The orthogonality_loss targets activation diversity directly (it penalizes cos_sim of expert hidden states). It does NOT directly target weight diversity. To increase weight diversity, we'd need a penalty on the **Gram matrix of weight matrices** (e.g. `||W_i W_j^T||_F^2` for i ≠ j).

Round 80's mechanism is correct for its stated goal. Round 96 shows the goal was activation diversity, not weight diversity — and the goal was achieved.

### 6.3 The 5-round smoothness + diversity audit (rounds 91-96)

| Round | Property tested | Verdict |
|-------|----------------|---------|
| 91 | CfC smoothness (max_grad) | ✓ CfC 2× lower than MLP |
| 92 | Smoother → more robust (target-side dropout) | ✗ LSTM wins |
| 93 | Smoother → more robust (input-side dropout) | ✗ MLP wins |
| 94 | Smoother → lower rank | ✗ CfC highest rank |
| 95 | FAME → diverse experts (weight) | Δ=0.24 over MR-MoE, modest |
| **96** | **Orth → diverse experts (weight)** | **Δ=0.00, orth is activation-only** |

The audit is now **6 rounds complete**. The headline finding: CfC smoothness, MoE diversity, and orthogonality are **independent properties** of the stack, each measurable at its own level (property, robustness, rank, weight diversity, activation diversity). The mechanisms do not cross.

## 7. Verdict on arXiv:2606.03631 (AnchorMoE)

| Claim | Status in our stack |
|-------|---------------------|
| Orthogonality constraint produces decorrelated expert representations | **CONFIRMED** (H3) — activation cos_sim drops on structured/random |
| Constraint is safe for task loss at small λ | **CONFIRMED** (H2) — within ±3% |
| Constraint increases weight diversity | **REJECTED** (H1) — orthogonality is activation-level only |

## 8. Implication for the LNN stack

- **Round 80 orthogonality is correctly placed in the stack** as a defensive measure. It works.
- **Do not promise weight-level diversity from orth** — the constraint is too weak at λ=0.001.
- **Future orthogonality at the weight level** (e.g. `||W_i W_j^T||_F^2` penalty) would be needed if weight diversity is the goal.
- **The audit chain (rounds 91-96) is now 6 rounds long** and shows that each property is its own thing. Stack composition matters: pick which property you need and use the corresponding mechanism.

## 9. Files

- `docs/prds/2026-06-15-lnn-round-96-a-fame-orthogonality-test.md` — PRD #10-58
- `scripts/bench_fame_orth_diversity.py` (NEW) — 18-cell bench
- `results/bench_fame_orth_diversity.json` — bench output
- `docs/research/2026-06-15_fame_orth_diversity_report.md` — this report
- `docs/daily/2026-06-15_LNN_research_summary_v22.md` — digest
- (No new tests, no new core code — bench-only round)

## 10. Cumulative state — 16-layer LNN+MoE 自主栈 (rounds 76-96)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| 92-93 | Dropout audit (target-side, input-side) | diagnostic |
| 94 | Effective rank (Williams/Payeur/Lajoie 2026) | diagnostic |
| 95 | Per-expert effective rank (FAME diversity) | diagnostic |
| **96** | **FAME+orth diversity (round 80 mechanism test)** | **diagnostic** |

**Cumulative suite**: 649/649 in-domain green (no new tests, all existing green).

## 11. Backlog for round 97+

1. **Weight-level orthogonality** (e.g. `||W_i W_j^T||_F^2` penalty) — if we want true weight diversity
2. **K=20, hidden=32, full recurrent training** — paper-scale settings
3. **Regime-labeled task** — does FAME router pick the right expert?
4. **PhysioNet-style irregular time-series** — most important untested domain
5. **Paper-style note** combining rounds 91-96 — 6-round audit complete
6. **Audit the ecology gate under dropout** (backlog #3)
