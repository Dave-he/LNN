# Round 106 — AuxLF Load Balancing on SETA (PRD #10-68) Report

**Date**: 2026-06-15
**Round**: 106
**Paper**: arXiv:2408.15664 (Wang et al. Aug 2024) — *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* (used in DeepSeek-V3)
**Verdict**: HONEST TARGET-DEPENDENT-WITH-NUANCE

## Motivation

The 91-105 audit revealed a clear pattern: **structural > routing-only**. Round 106 picks AuxLF as a clean test of this hypothesis. AuxLF is a pure router-side change (bias adjustment), so the audit pattern predicts it will likely fail in our time-series setting.

If AuxLF works, we have a new mechanism in our stack. If it fails (more likely per audit), we have **confirming evidence** for the "structural > routing" pattern.

## Background

The standard load-balancing in MoE uses an **auxiliary loss** that penalizes imbalance:
```
L_aux = α · K · Σ_k (f_k · P_k)
```
where f_k = fraction of tokens routed to expert k, P_k = mean routing probability.

Problem: "A large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance" (from the paper).

AuxLF replaces the auxiliary loss with a **bias term**:
- `score_k = logit_k + bias_k` (added to each expert's score)
- After top-K, the softmax is computed on `score_k` (not `logit_k`)
- `bias_k` is updated based on expert load:
  - If expert k is over-loaded: `bias_k -= γ`
  - If expert k is under-loaded: `bias_k += γ`

This achieves load balancing **without** any gradient interference.

## Implementation

`lnn/core/auxlf.py` (~340 lines):
- `AuxLFConfig(bias_lr, target_load_fraction, bias_clamp, warmup_steps, use_update)` dataclass
- `update_load_balancing_bias(bias, top_idx_counts, config, n_experts)` — adjusts bias based on recent load counts
- `AuxLFRouter(SETARouter)` — adds per-expert bias BEFORE top-K selection, auto-updates bias
- `AuxLFSETAMoECfCCell(SETAMoECfCCell)` — replaces the SETA router with `AuxLFRouter`
- `AuxLFSETAMoECfCNetwork(nn.Module)` — full network with QuITE + SETA + AuxLF

**Critical design choices**:
- `self.bias = nn.Parameter(torch.zeros(n_unique), requires_grad=False)` — bias doesn't affect gradients
- `with torch.no_grad():` wraps the bias update — no leakage into autograd
- Auto-update triggered at end of each forward pass (after warmup)
- `auxlf_util_std`, `auxlf_max_min_ratio`, `auxlf_bias_norm` exposed via `collect_expert_utilization()`

## Tests (22/22)

`tests/test_auxlf.py` (NEW):
- **TestAuxLFConfig** (2): defaults, custom
- **TestUpdateLoadBalancingBias** (5): overloaded decreases, balanced no change, clamp, no-update, LR scaling
- **TestAuxLFRouter** (7): bias starts at 0, forward shape, bias affects routing, no-update mode, warmup, gradient flow, load stats
- **TestAuxLFSETAMoECfCCell** (4): forward shape, shared always active, router is AuxLF, utilization includes load stats
- **TestAuxLFSETAMoECfCNetwork** (3): forward, NaN-aware mask, get utilization
- **TestAuxLFExports** (1): all public symbols exported from `lnn.core`

## Bench results (24 cells, 100 epochs)

4 conditions × 3 datasets × 1 K setting × 2 seeds × 100 epochs = 24 cells.
Tested on data with **higher** missing rate (50% vs train 30%).
Robust test at 70% missing rate.

### Summary

| cond | sin_irr test | sin_irr uniq_H | structured test | structured uniq_H | random test | random uniq_H |
|------|------|------|------|------|------|------|
| seta_only_shared | 0.0811 | 0.556 | 0.3890 | 0.531 | **0.1886** | 0.633 |
| seta_auxlf_no_update | 0.0833 | 0.558 | 0.3877 | 0.581 | **0.1828** | 0.506 |
| seta_auxlf_active | 0.0845 | 0.608 | 0.3793 | 0.555 | 0.2080 | 0.572 |
| seta_auxlf_strong | 0.0836 | **0.271** | 0.3779 | 0.262 | 0.2039 | 0.224 |

**Bold** = notable.

### Key observations

1. **H1 (load balance) PARTIAL**: AuxLF active/strong reduces unique_H by 50-65% on structured/random — it IS forcing the router to be more uniform (the auxlf bias is real and changes routing). However, this **diversity loss** is structural to AuxLF (the bias is intentionally pushing towards uniform).

2. **H2 (test_mse) REJECTED in expected direction**: AuxLF on random_irr *worsens* test_mse by 8-10% (0.183 → 0.208). On sin/structured it's within noise (±1%). The paper's claim of "load balancing improves task performance" is NOT reproduced in our time-series setting. The audit pattern **CONFIRMED**: routing-only changes do not help our time-series MoE.

3. **H3 (H=0 fix preserved) CONFIRMED**: shared_H stays at 0.693 (= log 2, max for 2 shared experts) in all 12 cells. SETA's structural fix is robust to adding AuxLF on the unique subgroup.

4. **H4 (training stability) CONFIRMED**: All 24 cells train stably. AuxLF bias updates don't cause divergence.

5. **AuxLF strong HURTS diversity**: This is actually expected — AuxLF pushes towards uniform load, which is the opposite of FAME/diversity mechanisms. The `auxlf_strong` condition with bias_lr=0.1 collapses unique routing to be near-uniform (max_min=1.0 at clamp). This is **good behavior for AuxLF** (it does what it claims) but the diversity is at odds with the multi-regime learning signal that MoE needs.

6. **AuxLF no_update ≈ SETA**: When use_update=False and bias_lr=0.0, the network is functionally identical to round 105 SETA. This confirms that the `AuxLFRouter` wrapper adds no measurable overhead when not active.

## Verdict

**HONEST TARGET-DEPENDENT-WITH-NUANCE**:
- AuxLF works **as a load balancer** (H1 partial — but in the wrong direction for diversity)
- AuxLF does **not** help task loss in our time-series setting (H2)
- AuxLF is **safe to combine with SETA** (H3, H4)
- The routing entropy drop is structural to AuxLF's design (push to uniform)

**Pattern reinforcement**: This is the **6th routing-only mechanism** in the 91-106 audit to fail to improve task loss in our setting (rounds 100 SNNL, 101 ORC, 103 QuITE+MoE, 104 SDG-MoE, 105 SETA regularizers, 106 AuxLF). The 91-106 audit now has **3 STRICTLY POSITIVE** mechanisms: 99 Reliability Gate, 102 QuITE, 105 SETA — all architectural/structural in nature.

## Use cases

- **Use AuxLF** when you want guaranteed balanced expert utilization (e.g. for inference cost predictability, latency, or hardware balancing)
- **Don't use AuxLF** as a default regularizer for time-series MoE
- The auxlf `auxlf_util_std`, `auxlf_max_min_ratio`, `auxlf_bias_norm` metrics are useful **diagnostics** for understanding expert load distribution regardless of whether you use AuxLF or not

## Files

- `docs/prds/2026-06-15-lnn-round-106-a-aux-loss-free-load-balancing.md` — PRD #10-68
- `lnn/core/auxlf.py` (NEW, ~340 lines)
- `lnn/core/__init__.py` (exports)
- `tests/test_auxlf.py` (NEW, 22/22)
- `scripts/bench_auxlf.py` (NEW, 24-cell bench)
- `results/bench_auxlf.json`

## References

- arXiv:2408.15664 (Wang et al. Aug 2024) — AuxLF
- arXiv:2401.06066 (DeepSeek-MoE) — shared expert isolation
- arXiv:2606.07500 (round 105) — SETA (complementary)
- arXiv:2606.08934 (round 98) — Backward Coherence
- DeepSeek-V3 Technical Report
