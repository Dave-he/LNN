# PRD #10-79 — Round 117: Gumbel-Softmax Routing (Stochastic MoE)

**Date**: 2026-06-15
**Round**: 117
**Paper**: arXiv:1611.01144 (Jang et al. 2017, ICLR 2017) — *Categorical Reparameterization with Gumbel-Softmax*
**Related**: Switch Transformer (arXiv:2101.03961, Fedus et al. 2021) uses top-1 + Gumbel-style sampling
**Goal**: Add a **stochastic** router variant with annealed temperature to the 91-116 audit.
**Audit pattern**: structural + data-structure-independent + preserves recurrent state mixing.

## The mechanism

All 4 router families tested in 91-116 (softmax, sigmoid, ReLU, cosine) are **deterministic** — the routing decision is a fixed function of the input. This round adds a **5th family: stochastic routing** via Gumbel-Softmax.

```
input x_t
  │
  ├── compute router logits: z = W x
  │
  ├── sample Gumbel noise: g = -log(-log(U)),  U ~ Uniform(0, 1)
  │
  ├── add noise + temperature: z' = (z + g) / T
  │
  ├── Gumbel-Softmax: g_routing = softmax(z')
  │       (with optional hard=True → argmax but soft gradient)
  │
  └── h_new = sum_i g_routing_i * expert_i(x_t, h_t)
```

The temperature `T` is **annealed** during training:
- T=1.0 (start) → near-uniform exploration
- T=0.1 (end) → near-deterministic selection

This gives a natural curriculum: explore all experts early, then commit to specific experts late.

## Why this fits the audit pattern

- **Structural**: changes the routing topology (deterministic → stochastic), but experts/cell structure unchanged.
- **Data-structure-independent**: noise is per-sample, no data-dependent bias.
- **Preserves recurrent state mixing**: h_new has the same form `sum_i g_i * expert_i(x_t, h_t)`.
- **Fills a real gap**: 91-116 has 4 deterministic router families (softmax, sigmoid, ReLU, cosine) and 0 stochastic families.

## Three properties of Gumbel-Softmax routing

1. **Stochastic at training time, deterministic at inference** — `torch.no_grad()` removes the noise
2. **Temperature annealing** — natural curriculum from exploration to exploitation
3. **Gumbel-Softmax is differentiable** — gradient flows through the soft mixture even though the decision is stochastic

## Hypotheses

- **H1 (POSITIVE)**: Gumbel-Softmax with high initial T reduces FAME's H=0 lock-in (round 103 critical finding) by forcing all experts to receive gradient signal early
- **H2 (PARTIAL)**: Annealing T to 0.1 recovers FAME-like behavior (no late-training cost)
- **H3 (PARTIAL)**: Stochastic routing beats deterministic on noisy data (random_irr) because exploration prevents overfitting
- **H4 (RULE)**: Gumbel-Softmax preserves recurrent state mixing (h_new shape, gradient flow), so it should not break the 91-116 audit pattern

## Test plan

- 28+ unit tests covering: gumbel noise sampling, temperature scaling, annealed schedule, gradient flow, expert utilization diagnostic, network API.
- 30-cell bench: 3 datasets × 5 conditions × 2 seeds, 50 epochs.
  - baseline_cfc (control)
  - fame_k3_t1 (softmax baseline)
  - sigmoid_k3_dense (round 116 winner)
  - gumbel_k3_t1_high (T=1.0 constant, top-1)
  - gumbel_k3_t1_anneal (T annealed 1.0 → 0.1, top-1)

## What "winning" looks like

- Gumbel-Softmax beats FAME on random_irr (H3) due to exploration
- Annealing converges to FAME-like performance (H2)
- Gradient flows to all K experts (H1) at high T
- Recurrent state preserved (H4)

## Risk

- Annealing schedule is **brittle in 1D** (round 110 freq experts also annealed and was negative)
- Gumbel-Softmax has higher variance than softmax (per-batch noise)
- May need `hard=True` (straight-through estimator) to be competitive

## Files

- `lnn/core/gumbel_moe.py` (NEW, ~350 lines) — GumbelRouter, GumbelMoECfCCell, GumbelMoECfCNetwork
- `tests/test_gumbel_moe.py` (NEW, 28+ tests)
- `scripts/bench_gumbel_moe.py` (NEW, 30 cells)
- `results/bench_gumbel_moe.json` (NEW)
- `docs/research/2026-06-15_gumbel_moe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v43.md`
- `docs/prds/2026-06-15-lnn-round-117-a-gumbel-moe.md` (this PRD)
- `README.md` (new Gumbel MoE section)
- `lnn-round-117-gumbel-moe.md` (memory)

## Out of scope

- Gumbel-Softmax with hard=True (straight-through estimator) — needs separate test
- Per-expert temperature — too many hyperparameters
- Learnable temperature — defer to future round
