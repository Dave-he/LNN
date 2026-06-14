# LNN Research Daily Digest v16 — 2026-06-15 (Round 90)

**Focus**: stress-test the round 80-89 orthogonality stack against an external audit (arXiv:2601.00457, Kim 2026).

## 1. Paper survey

Round 90 prioritized a single high-impact paper:
- **arXiv:2601.00457** (Hyunjun Kim, Jan 2026) — *Geometric Regularization in Mixture-of-Experts: The Disconnect Between Weights and Activations*

Key claim: weight-space geometric regularization in MoE **fails** (weight overlap grows +114%, activation overlap unchanged, r=−0.293, p=0.523).

This is a direct challenge to the foundational assumption of round 80 (`orthogonality_loss` on per-expert activations).

## 2. Implementation: weights-vs-activations audit (PRD #10-52)

Round 90 introduces two new metrics in `lnn/core/moe_ecology.py`:

- `weight_space_overlap(expert_weights)` — mean pairwise |cos(W_i, W_j)|
- `activation_space_overlap(expert_outs)` — mean pairwise |cos(h_i, h_j)|

Both: 0.0 for K<2, safe zero-norm handling, sign-abs (anti-parallel counts as overlap).

## 3. Test + bench summary

- **11/11 unit tests** pass (`tests/test_orth_audit_metrics.py`):
  - Identical matrices → overlap 1.0
  - Orthogonal matrices → overlap 0.0
  - Anti-parallel matrices → overlap 1.0 (sign-abs)
  - K<2 returns 0.0
  - Zero-norm safe
  - K=3 returns mean over 3 pairs
- **12-cell bench** (3 datasets × 4 λ):
  - H2 ✓: act_ov drops 47-54% in toy_sin/structured with λ=10
  - H1 ~partial: wgt_ov grows 44-48% (vs Kim's +114%) — milder disconnect
  - H4 ✗: loss gets 30-100% worse with high λ
- **Cumulative MoE+FAME+Causality+Audit suite**: 237/237 pass (up from 226/226 in round 89)

## 4. Honest-negative: orth loss is a stylistic tax

The audit confirms Kim 2026's **subtle claim** (disconnect exists) at mild magnitude, but **rejects the strong claim** (+114% blowup doesn't reproduce). The orth loss DOES reduce its target (activation overlap) but at a cost:
- Toy regime: λ=10 doubles loss (0.53→1.07)
- Real regime: untested

**Recommendation**: keep orth λ ≤ 0.1 in toy regime. For real datasets, use round 85 E-gate to auto-rescale.

## 5. Verdict on round 80-89 stack

The stack is **functionally correct** but with caveats:
- Round 80 orth: works as designed (H2 ✓)
- Round 85 E-gate: warranted (orth can be too strong)
- Round 89 causality gate: warranted as defense-in-depth
- Round 88 per-expert grad: provides the signal the gate uses

**Open question**: does the E-gate (round 85) moderate the loss degradation in this audit? Round 90 tested raw orth only. Round 91 should re-test with E-gate active.

## 6. Cumulative state — 12-layer LNN+MoE 自主栈 (rounds 76-90)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| **90** | **Audit (wgt/act overlap, Kim 2026 response)** | **diagnostic** |

## 7. Backlog for round 91

1. Re-run audit with round 85 E-gate active
2. Larger K (5, 8 experts) — does the disconnect strengthen?
3. Real dataset (PDNA-LRA, ETTh1) — see if real-data regime differs
4. Audit other layers (FAME top-K? φ-balancing? ecology gate?)
5. Write paper-style note combining round 87-90 causality + audit stack
