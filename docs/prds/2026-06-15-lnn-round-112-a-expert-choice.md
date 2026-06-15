# PRD #10-74 — Round 112: Expert Choice Routing (response to arXiv:2202.09368)

**Date**: 2026-06-15
**Round**: 112
**Paper**: arXiv:2202.09368 (Zhou et al., 2022) — *Mixture-of-Experts with Expert Choice Routing*
**Status**: IMPLEMENTED + BENCHED, ready to push
**Audit fit**: 9th structural mechanism in 91-112 audit; **CRITICAL STRUCTURAL NEGATIVE finding**.

## 1. Problem and motivation

Our 91-111 audit established:
- 5 STRUCTURAL winners (99, 102, 105, 107, 111) all **don't modify the recurrent state mixing**
- 3 STRUCTURAL target-dep failures (108, 109, 110) all **depend on data structure** that doesn't exist in 1D

We hypothesised that **Expert Choice (EC) routing** — the natural complement to MoD (round 111) — would be the 9th winner:
- MoD (round 111) = per-timestep compute budget (which timesteps to process)
- EC (round 112) = per-expert compute budget (which expert processes which timesteps)

EC promises **perfect load balance by construction** and removes the need for Switch-Transformer-style aux load-balancing loss. The structural fit:
- Structural: ✓ (changes the routing mechanism)
- Data-independent: ✓ (no data assumptions, just enforces balance)
- Constructive: ✓ (removes aux loss, balances load)

## 2. Solution

Implement EC routing for CfC, adapted to the recurrent setting:

1. **`ExpertChoiceRouter`** — per-(expert, token) sigmoid score; each expert picks its top-k tokens.
2. **`ExpertChoiceCfCCell`** — wraps K independent CfC cells. At each step, the cell receives the pre-computed (B, K) assignment for that timestep and computes the EC-mixed output: `out = (Σ_{e picked t} g_e(t) / |{e picked t}|) · expert_e(x_t, h)`.
3. **`ExpertChoiceCfCNetwork`** — stacked EC layers. The EC assignment is pre-computed once per sequence per layer (using the input + h0), then sliced per-timestep inside the recurrent loop.
4. **`expert_choice_load(cell)`** — diagnostic for per-expert load counts.

The mixing scheme is the **average** of all active expert contributions (not a learned weighted sum), giving a per-token representation that is invariant to how many experts picked it.

## 3. Critical structural finding (the bench result)

**EC routing BREAKS recurrent dynamics in time-series MoE.** Bench results at 50 epochs (24 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0023±0.0002 | 0.0010±0.0001 | 0.0005±0.0003 |
| ec_dense      | 0.0107±0.0021 | 0.0062±0.0014 | 0.0042±0.0043 |
| ec_half       | 0.0258±0.0054 | 0.0186±0.0040 | 0.0039±0.0020 |
| ec_quarter    | 0.0530±0.0315 | 0.0447±0.0169 | 0.0472±0.0121 |

EC dense (all tokens processed, just averaging expert outputs) is **4-8× worse** than baseline. EC quarter (25% compute) is **23-94× worse** on smooth data, **94× worse** on noisy data.

**Mechanism**: The EC mixing step (averaging expert contributions at each timestep) **washes out the recurrent state**. In the original EC-for-Transformers, each token's representation is mixed, but the tokens are processed in parallel. In CfC, the hidden state is **recurrent** — averaging over experts corrupts the temporal dynamics.

This confirms the FAME H=0 (round 78) and SDG-MoE H=0 (round 104) findings: time-series MoE is fundamentally different from LLM-MoE because all experts see correlated inputs and the recurrent state is delicate. EC amplifies this problem by enforcing a fixed assignment per timestep.

## 4. Audit pattern update (91-112)

9 STRUCTURAL mechanisms tested:
- 4 winners: 99 Reliability Gate, 102 QuITE, 105 SETA, 107 Soft MoE
- 1 compute-saving: 111 MoD Routing
- 4 target-dep/negative: 108 Anchored MoE, 109 Dynamic TMoE, 110 Freq Experts, **112 Expert Choice**

**New rule**: **mechanisms that modify or constrain the recurrent state mixing are dangerous in time-series MoE**. The original CfC's gate-and-update is delicate; replacing it with averaging (or fixed assignment) destroys the dynamics.

## 5. Files added

- `lnn/core/expert_choice.py` (NEW, ~430 lines)
  - `ExpertChoiceRouter(input_size, hidden_size, n_experts, router_hidden, use_sigmoid)`
  - `ExpertChoiceCfCCell(input_size, hidden_size, n_experts, cap_k, router_hidden)`
  - `ExpertChoiceCfCNetwork(input_size, hidden_size, output_size, num_layers, n_experts, cap_k, cap_k_frac, router_hidden)`
  - `expert_choice_load(cell)` — diagnostic
- `tests/test_expert_choice.py` (NEW, 27/27 tests)
- `scripts/bench_expert_choice.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-112-a-expert-choice.md` (this PRD)
- `docs/research/2026-06-15_expert_choice_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v38.md` (digest v38)
- `README.md` (new Expert Choice section)
- `lnn-round-112-expert-choice.md` (memory)

## 6. Test coverage (27 tests, all pass)

- `TestExpertChoiceRouter` (7): init, init with router_hidden, init softmax mode, forward shape, perfect load balance, cap_k capped at T, assign_w in range, gradient flows.
- `TestExpertChoiceCfCCell` (5): init, init with cap, forward with assignment, forward no assignment fallback, gradient flows.
- `TestExpertChoiceCfCNetwork` (8): init, init with int cap, init with frac cap, init both raises, init bad frac raises, forward dense, forward last step, forward with int cap, forward with frac cap, gradient flows.
- `TestExpertChoiceIntegration` (4): perfect load balance, expert_choice_load no forward, captures signal, smaller cap_k does not crash.

## 7. Critical bugs fixed

1. **topk dim**: `assign_w.topk(cap_k_eff, dim=-1)` was operating on the K dim (3) instead of T dim (8) because the original shape was `[B, T, K]`. Fixed by transposing to `[B, K, T]` first.
2. **Stash ordering**: `cell.last_assign_mask` was being overwritten by the per-step `[B, K]` slice in the cell's forward, hiding the full `[B, K, T]` assignment stashed by the network. Fixed by stashing the `[B, K, T]` AFTER the per-step loop.

## 8. Recommendation

**DO NOT use EC routing for time-series MoE in production**:
- The averaging mixing step breaks the recurrent dynamics
- Even `ec_dense` (full compute) is 4-8× worse than baseline
- The mechanism is theoretically sound for LLM tokens but fundamentally breaks for recurrent hidden states

**Use EC for**:
- Parallel-input MoE (LLMs, vision) where each token is independent
- **NOT** for recurrent MoE (CfC, LSTM, GRU) where the hidden state is a time-series signal

**Combine with other mechanisms**:
- **MoD (round 111)** works for time-series MoE because it skips timesteps, not averages them
- **FAME (round 78)** works because it uses softmax weighting (not averaging)
- **SETA (round 105)** works because it has shared parameters across experts (not full averaging)

## 9. Comparison with prior structural mechanisms

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

## 10. Future work

1. **Soft EC**: replace the averaging mixing with a learned weighted sum (FAME-style softmax)
2. **EC for embedding only**: use EC routing for the QuITE-style embedding (round 102) but not for the recurrent step
3. **Gating on EC**: add a learned gate to suppress EC mixing when it's harmful
4. **PhysioNet 36D test**: does EC scaling hurt less on high-dim real medical time series?
5. **Combine with MoD (round 111)**: skip timesteps with MoD, then use EC for the remaining (k < T) timesteps

## 11. 33-layer LNN+MoE stack

`rounds 76-112` = 33 layers, extended with Expert Choice routing in round 112.
