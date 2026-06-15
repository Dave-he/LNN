# PRD #10-78 — Round 116: Sigmoid Routing for MoE (Qwen MoE style)

**Date**: 2026-06-15
**Round**: 116
**Paper**: arXiv:2407.10671 (Qwen Team, June 2024) — *Qwen Technical Report* / *Qwen2 MoE*
**Related**: Mixtral (arXiv:2401.04088) uses softmax; Qwen2-MoE uses sigmoid.
**Goal**: Add a **purely-sigmoid** router variant to the 91-115 audit.
**Audit pattern**: structural + data-structure-independent + preserves recurrent state mixing.

## The mechanism

Standard FAME/ReMoE routers all use **softmax** (normalized probabilities that sum to 1) or **ReLU** (zero-suppressing gates). Qwen2-MoE (arXiv:2407.10671) replaces softmax with **sigmoid**:

```
g = sigmoid(W x + b)        # [B, K] each entry in [0, 1], NOT normalized
y = sum_i g_i * expert_i(x)  # [B, hidden]
```

The three properties of sigmoid routing:

1. **No normalization** — each expert gets an independent score in [0, 1]. Multiple experts can fire simultaneously with no "softmax budget" competition.
2. **Naturally sparse via small init** — early in training, W ~ 0 so g ~ 0.5 for all experts, but as the network learns, the W magnitudes diverge and naturally only some experts fire strongly.
3. **Per-expert bias optional** — Qwen2-MoE uses a bias term on the routing score (similar to DeepSeek-V3's AuxLF), updated by an EMA of recent load.

## Why this fits the audit pattern

- **Structural**: changes the routing topology (softmax → sigmoid), but experts/cell structure unchanged.
- **Data-structure-independent**: no data-dependent bias, no regime detection, no gating on input structure.
- **Preserves recurrent state mixing**: h_t is updated as `h_new = sum_i g_i * expert_i(x_t, h_t)`, same form as FAME/ReMoE. No modification to the inner CfCCell forward.
- **Fills a real gap**: the 91-115 audit has softmax (rounds 78/103), ReLU (114), cosine (82), and forecastability (78). **Sigmoid is the 4th major router family** and the 1st one without normalization.

## Hypotheses

- **H1 (STRICTLY POSITIVE)**: Sigmoid routing beats softmax+FAME on smooth data (sin_irr, structured_irr) because no normalization budget means experts can co-activate freely when input is rich.
- **H2 (PARTIAL)**: Sigmoid routing matches or beats softmax+FAME on noisy data (random_irr) because small scores are not suppressed — every expert contributes something.
- **H3 (PARTIAL)**: Sigmoid routing is competitive with ReMoE (round 114) and DeepSeek (round 113) on at least 2 of 3 datasets. Sigmoid is between softmax (strict budget) and ReLU (zero-suppressing) in the bias-variance trade-off.
- **H4 (RULE)**: Sigmoid routing preserves recurrent state mixing (h_new shape, gradient flow), so it should not break the 99-115 audit pattern of "winners preserve recurrent state".

## Test plan

- 28+ unit tests covering: sigmoid range, no normalization property, sparse top-K, gradient flow, expert utilization diagnostic, network API.
- 30-cell bench: 3 datasets × 5 conditions × 2 seeds, 50 epochs.
  - baseline_cfc (control)
  - fame_k3_t1 (softmax baseline)
  - sigmoid_k3_dense (new, dense — no top-K)
  - sigmoid_k3_t1 (new, top-K=1)
  - sigmoid_k3_t2 (new, top-K=2)

## What "winning" looks like

- Sigmoid is competitive with FAME on 2/3 datasets (within 1.5× test_mse)
- No divergence in any cell
- Routing entropy is balanced (≥ 1.0 nats for K=3)
- Gradient flows to all K experts
- Sigmoid + DeepSeek is testable (i.e., the architecture composes cleanly)

## Risk

- Sigmoid is **dense by default** (no top-K selection) — this is K=3× more compute per step. We mitigate by also testing top-K variants.
- Sigmoid can be **unstable early in training** (gradient through sigmoid is at most 0.25, leading to vanishing gradients). Mitigate by initializing W ~ N(0, 0.01) instead of N(0, 1).

## Files

- `lnn/core/sigmoid_moe.py` (NEW, ~300 lines) — SigmoidRouter, SigmoidMoECfCCell, SigmoidMoECfCNetwork
- `tests/test_sigmoid_moe.py` (NEW, 28+ tests)
- `scripts/bench_sigmoid_moe.py` (NEW, 30 cells)
- `results/bench_sigmoid_moe.json` (NEW)
- `docs/research/2026-06-15_sigmoid_moe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v42.md`
- `docs/prds/2026-06-15-lnn-round-116-a-sigmoid-moe.md` (this PRD)
- `README.md` (new Sigmoid MoE section)
- `lnn-round-116-sigmoid-moe.md` (memory)

## Out of scope

- Per-expert bias EMA (Qwen2-MoE AuxLF variant) — already tested in round 106.
- Hierarchical routing — overcomplicated for this round.
- Sparse upcycling (start from dense, replicate) — too different from current stack.
