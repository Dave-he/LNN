# LNN Research Digest v38 — 2026-06-15

**Coverage**: Expert Choice Routing + 91-112 audit update (CRITICAL STRUCTURAL NEGATIVE — recurrent dynamics broken).

## Headline

Round 112 implemented **Expert Choice (EC) Routing** (arXiv:2202.09368 Zhou et al. 2022) — *Mixture-of-Experts with Expert Choice Routing*. The mechanism: each **expert** picks its top-k tokens, giving perfect load balance by construction (no aux loss needed). This was the natural complement to MoD (round 111).

**The result is a CRITICAL STRUCTURAL NEGATIVE**: EC routing BREAKS recurrent dynamics in time-series MoE. Bench at 50 epochs (24 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0023±0.0002 | 0.0010±0.0001 | 0.0005±0.0003 |
| ec_dense      | 0.0107±0.0021 | 0.0062±0.0014 | 0.0042±0.0043 |
| ec_half       | 0.0258±0.0054 | 0.0186±0.0040 | 0.0039±0.0020 |
| ec_quarter    | 0.0530±0.0315 | 0.0447±0.0169 | 0.0472±0.0121 |

EC dense (full compute, just averaging expert outputs) is **4-8× worse** than baseline. EC quarter (25% compute) is **23-94× worse**. Worst case: random_irr ec_quarter is **94× worse** than baseline.

## 1. EC in 60 seconds

Standard MoE: each **token** picks its top-k **experts** (token-choice, FAME). EC: each **expert** picks its top-k **tokens** (expert-choice).
```
input x [B, T, D]
  │
  ├── ExpertChoiceRouter: scores = sigmoid(W · [x; h]) [B, T, K]
  │   transpose to [B, K, T]
  │   topk(scores, k=cap_k, dim=T) → topk_idx [B, K, k]
  │   assign_mask = scatter True at topk_idx [B, K, T]
  │
  └── Cell per-step:
      x_t [B, D]
        │
        ├── K experts → K × [B, H]
        │
        └── Mix: out = (Σ g_e(t) / |{e picked t}|) · expert_e(x_t, h)  [B, H]
```

Perfect load balance by construction (each expert processes exactly k tokens), no aux loss needed.

## 2. Why EC fails on CfC

**The EC mixing step (averaging expert contributions at each timestep) washes out the recurrent state.**

In EC-for-Transformers, tokens are processed in parallel — averaging works because each token is independent. In CfC, the hidden state `h_t` is **recurrent** — averaging over experts corrupts `h_t` for the next step, and the corruption compounds through the sequence.

This confirms the FAME H=0 (round 78) and SDG-MoE H=0 (round 104) findings: **time-series MoE is fundamentally different from LLM-MoE** because experts all see correlated inputs and the recurrent state is delicate.

## 3. The 91-112 audit pattern

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 99 | Reliability gate | Augmentation | STRICTLY POSITIVE |
| 102 | QuITE | Embedding | STRICTLY POSITIVE |
| 105 | SETA | Architecture | STRICTLY POSITIVE |
| 107 | Soft MoE | Structural | SAFER ROUTING |
| 108 | Anchored MoE | Structural | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | NEGATIVE-WITH-NUANCE |
| 110 | Freq Experts | Structural | NEGATIVE-WITH-NUANCE |
| 111 | MoD Routing | Structural | POSITIVE-WITH-NUANCE (compute-saving) |
| **112** | **Expert Choice** | **Structural** | **NEGATIVE (recurrent dynamics broken)** |

**9 STRUCTURAL mechanisms tested**:
- 4 winners: 99, 102, 105, 107
- 1 compute-saving: 111
- 4 target-dep/negative: 108, 109, 110, **112**

**New rule**: **mechanisms that modify or constrain the recurrent state mixing are dangerous in time-series MoE**. The original CfC's gate-and-update is delicate; replacing it with averaging (or fixed assignment) destroys the dynamics.

## 4. Implementation highlights

`lnn/core/expert_choice.py` (~430 lines):
- `ExpertChoiceRouter(input_size, hidden_size, n_experts, router_hidden, use_sigmoid)` — per-(expert, token) sigmoid score; each expert picks top-k tokens
- `ExpertChoiceCfCCell(input_size, hidden_size, n_experts, cap_k, router_hidden)` — K independent CfC cells mixed by EC assignment
- `ExpertChoiceCfCNetwork(input_size, hidden_size, output_size, num_layers, n_experts, cap_k, cap_k_frac, router_hidden)` — full network
- `expert_choice_load(cell)` — per-expert token count diagnostic

`tests/test_expert_choice.py` (27/27):
- TestExpertChoiceRouter (7): init, init with router_hidden, init softmax mode, forward shape, perfect load balance, cap_k capped at T, assign_w in range, gradient flows.
- TestExpertChoiceCfCCell (5): init, init with cap, forward with assignment, forward no assignment fallback, gradient flows.
- TestExpertChoiceCfCNetwork (8): init, init with int cap, init with frac cap, init both raises, init bad frac raises, forward dense, forward last step, forward with int cap, forward with frac cap, gradient flows.
- TestExpertChoiceIntegration (4): perfect load balance, expert_choice_load no forward, captures signal, smaller cap_k does not crash.

## 5. Critical bugs fixed

1. **topk dim**: `assign_w.topk(cap_k_eff, dim=-1)` was operating on the K dim (3) instead of T dim (8) because the original shape was `[B, T, K]`. Fixed by transposing to `[B, K, T]` first.
2. **Stash ordering**: `cell.last_assign_mask` was being overwritten by the per-step `[B, K]` slice in the cell's forward, hiding the full `[B, K, T]` assignment stashed by the network. Fixed by stashing the `[B, K, T]` AFTER the per-step loop.

## 6. Recommendation

**DO NOT use EC routing for time-series MoE in production**:
- The averaging mixing step breaks the recurrent dynamics
- Even `ec_dense` (full compute) is 4-8× worse than baseline
- The mechanism is theoretically sound for LLM tokens but fundamentally breaks for recurrent hidden states

**Use EC for**:
- Parallel-input MoE (LLMs, vision) where each token is independent
- **NOT** for recurrent MoE (CfC, LSTM, GRU) where the hidden state is a time-series signal

**Combine with other mechanisms**:
- **MoD (round 111)** works because it skips timesteps, not averages them
- **FAME (round 78)** works because it uses softmax weighting (not averaging)
- **SETA (round 105)** works because it has shared parameters across experts (not full averaging)

## 7. Files added

- `lnn/core/expert_choice.py` (NEW, ~430 lines)
- `tests/test_expert_choice.py` (NEW, 27/27 tests)
- `scripts/bench_expert_choice.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-112-a-expert-choice.md` (PRD #10-74)
- `docs/research/2026-06-15_expert_choice_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v38.md` (this file)
- `README.md` (new Expert Choice section)
- `lnn-round-112-expert-choice.md` (memory)

## 8. Future work

1. **Soft EC**: replace averaging with a learned weighted sum (FAME-style softmax)
2. **EC for embedding only**: use EC routing for QuITE-style embedding but not for recurrent step
3. **Gating on EC**: add a learned gate to suppress EC mixing when harmful
4. **PhysioNet 36D test**: high-dim real medical time series — does EC scale?
5. **Combine with MoD (round 111)**: skip timesteps with MoD, then use EC for remaining (k < T) timesteps
