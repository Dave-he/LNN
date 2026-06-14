# PRD #10-68 — Round 106: Auxiliary-Loss-Free Load Balancing (response to arXiv:2408.15664)

**Date**: 2026-06-15
**Round**: 106
**Paper**: arXiv:2408.15664 (Wang et al. Aug 2024) — *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* (used in DeepSeek-V3)
**Status**: To implement

## Motivation

The 91-105 audit revealed a clear pattern:
- **Architectural fixes** (rounds 102 QuITE, 105 SETA): work
- **Routing-only fixes** (rounds 103 QuITE+MoE, 104 SDG-MoE): often fail or are target-dependent

Round 106 picks **AuxLF** as a **clean test of this hypothesis**. AuxLF is a pure router-side change (bias adjustment), so the audit pattern predicts it will likely fail in our time-series setting.

If AuxLF works, we have a new mechanism in our stack.
If it fails (more likely per audit), we have **confirming evidence** for the "structural > routing" pattern.

## Goal

Implement `AuxLF` router that adds a **per-expert bias term** to the routing scores before top-K selection. The bias is **dynamically adjusted** based on recent expert load (no auxiliary loss, no interference gradients).

## Background

The standard load-balancing in MoE uses an **auxiliary loss** that penalizes imbalance:
```
L_aux = α · K · Σ_k (f_k · P_k)
where f_k = fraction of tokens routed to expert k
      P_k = mean routing probability for expert k
```

Problem: "A large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance" (from the paper).

AuxLF replaces the auxiliary loss with a **bias term**:
- `score_k = logit_k + bias_k` (added to each expert's score)
- After top-K, the softmax is computed on `score_k` (not `logit_k`)
- `bias_k` is updated based on expert load:
  - If expert k is over-loaded: `bias_k -= γ`
  - If expert k is under-loaded: `bias_k += γ`

This achieves load balancing **without** any gradient interference.

## Hypotheses

**H1 — AuxLF balances unique expert load in SETA**: the unique experts' utilization becomes more uniform (lower std, lower max-min ratio).

**H2 — AuxLF doesn't help test_mse in our setting**: the audit pattern predicts routing-only fixes don't translate to time-series.

**H3 — AuxLF works WITHOUT breaking the H=0 fix from SETA**: shared experts remain active, unique experts get balanced.

**H4 — AuxLF training is stable**: bias adjustments don't cause divergence.

## Architecture

```python
class AuxLFRouter(nn.Module):
    def __init__(self, n_experts, top_k, bias_lr=0.01, warmup_steps=100):
        self.router = nn.Linear(...)  # produces logits
        self.bias = nn.Parameter(torch.zeros(n_experts))  # per-expert bias
        self.bias_lr = bias_lr
        self.warmup_steps = warmup_steps
    
    def forward(self, x_t, h, context=None):
        logits = self.router(...)
        adjusted = logits + self.bias  # add bias BEFORE top-K
        # ... standard top-K softmax ...
        return g
    
    def update_bias(self, top_idx_counts):
        # top_idx_counts: (n_experts,) how many times each expert was selected
        # over-loaded → decrease bias, under-loaded → increase bias
        target = self.top_idx_counts.sum() / self.n_experts  # uniform target
        diff = top_idx_counts.float() - target
        with torch.no_grad():
            self.bias -= self.bias_lr * diff  # decrease over-loaded
```

## Test plan

- Test bias starts at 0
- Test biased scores go into top-K (not raw logits)
- Test update_bias decreases bias for over-loaded experts
- Test update_bias increases bias for under-loaded experts
- Test bias update doesn't affect gradient flow
- Test gradient flows through router
- Test SETA + AuxLF combined: shared experts always active, unique balanced

## Bench plan

24 cells:
- 4 conditions: `set_full` (round 105), `seta_auxlf_no_update` (AuxLF bias=0), `seta_auxlf_active` (AuxLF with updates), `seta_auxlf_strong` (AuxLF with strong bias LR)
- 2 datasets: sin_irr, random_irr
- 1 K setting: S=2+U=3
- 2 seeds × 100 epochs
- T=32, D=2, hidden=16, lr=1e-3, Adam
- Measure: test_mse, unique_H, unique_util_std (load balance metric), training stability

## Expected outcomes (per audit pattern)

- **H1 PARTIAL**: AuxLF should reduce load std somewhat
- **H2 NEUTRAL/NEGATIVE**: test_mse unlikely to improve (routing-only fix)
- **H3 CONFIRMED**: SETA's structural fix preserved
- **H4 CONFIRMED**: training stable

## Verdict

If H1 + H4 work but H2 doesn't, this is consistent with the audit pattern (structural > routing). If AuxLF improves test_mse, we update the audit with a new STRICTLY POSITIVE.

## Why this might still help

Unlike FAME/SDG-MoE (which tried to fix H=0 lock-in via routing alone), AuxLF only addresses **load balance** (not H=0). With SETA already providing the H=0 fix, AuxLF's load balancing has a different target. It might successfully **complement** SETA's structural fix.

## Files to create

- `lnn/core/auxlf.py` (NEW, ~300 lines)
- `tests/test_auxlf.py` (NEW, 20+ tests)
- `scripts/bench_auxlf.py` (NEW, 24-cell bench)
- `docs/research/2026-06-15_auxlf_load_balancing_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v32.md`
- `README.md` (new section)

## Risks

1. **Bias update is tricky**: must be done AFTER gradient computation but BEFORE next forward
2. **Bias can drift unboundedly**: needs clamping
3. **Bias interferes with H=0 fix from SETA**: shared experts may not benefit from unique-expert bias

## References

- arXiv:2408.15664 (Wang et al. Aug 2024) — AuxLF
- arXiv:2401.06066 (DeepSeek-MoE) — shared expert isolation
- arXiv:2606.07500 (round 105) — SETA (complementary)
