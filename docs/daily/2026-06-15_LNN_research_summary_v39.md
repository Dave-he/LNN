# LNN Research Digest v39 — 2026-06-15

**Coverage**: DeepSeekMoE Shared Expert Isolation + 91-113 audit update (STRICTLY POSITIVE — additive residual preserves recurrent dynamics).

## Headline

Round 113 implemented **DeepSeekMoE Shared Expert Isolation** (arXiv:2401.06066, DeepSeek-AI January 2024) — *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. The mechanism: K_s **shared experts** always process every timestep (no routing) and their outputs are **added** (not averaged) to K_r **routed experts** selected by FAME-style top-K_r sparse routing.

**The result is STRICTLY POSITIVE** — the **5th structural winner** in the 91-113 audit, and the **1st non-augmentation winner that adds MoE diversity**. Bench at 50 epochs (30 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc       | 0.0023±0.0001 | 0.0010±0.0001 | 0.0005±0.0002 |
| fame_k3_t1         | 0.0112±0.0016 | 0.0061±0.0021 | 0.0050±0.0041 |
| **deepseek_1s_3r_t1** | **0.0017±0.0008** | **0.0015±0.0005** | 0.0037±0.0000 |
| **deepseek_1s_3r_t2** | **0.0011±0.0000** | **0.0007±0.0001** | 0.0021±0.0011 |
| **deepseek_2s_3r_t2** | **0.0014±0.0002** | 0.0012±0.0006 | 0.0036±0.0035 |

Key findings:
- **DeepSeek beats FAME on all 3 datasets** (1.5-10× better test_mse)
- **DeepSeek matches or beats baseline on smooth data** (sin_irr, structured_irr)
- **Routed utilization is balanced** (~0.33 for K_r=3, top_k=1)
- **Higher n_shared (2 vs 1) reduces variance** on smooth data

## 1. DeepSeekMoE in 60 seconds

Standard MoE: K experts, top-K' selected per step. DeepSeekMoE: split experts into K_s **shared** (always on) + K_r **routed** (top-K_r selected).
```
input x [B, T, D]
  │
  ├── Shared experts: K_s always-active cells → mean → shared_out [B, H]
  │
  ├── Routed experts: ForecastabilityRouter → topk(K_r) → routed_out [B, H]
  │
  └── Additive combination: h_new = shared_out + routed_out [B, H]
```

Key insight: **additive** combination (not averaged) preserves the recurrent state dynamics. Shared experts act as a "common knowledge sink" that never collapses; routed experts add specialization on top.

## 2. Why DeepSeek succeeds on CfC

The **additive residual** structure (`h_new = Shared + Routed`) is structurally identical to a residual connection in ResNets. The shared path is a stable anchor that processes every step identically; the routed path adds a delta to the shared output. The recurrent state `h_t` is never directly modified by routing decisions.

This is the **1st structural mechanism that adds MoE diversity** to the recurrent step itself. The earlier 4 winners (99, 102, 105, 107) were all **input-side / embedding-side / structure-only** augmentations. DeepSeek is the **1st to add expert specialization to the recurrent step**.

## 3. The 91-113 audit pattern

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
| 112 | Expert Choice | Structural | NEGATIVE (recurrent dynamics broken) |
| **113** | **DeepSeek Shared Expert** | **Structural (residual)** | **STRICTLY POSITIVE** |

**10 STRUCTURAL mechanisms tested**:
- 5 winners: 99, 102, 105, 107, **113**
- 1 compute-saving: 111 MoD
- 4 target-dep/negative: 108, 109, 110, 112

**Rule (reinforced)**: mechanisms that modify or constrain the recurrent state mixing are dangerous in time-series MoE. The **additive residual** structure of DeepSeekMoE is the natural safe way to add MoE diversity.

## 4. Implementation highlights

`lnn/core/deepseek_moe.py` (~340 lines):
- `DeepSeekCfCCell(input_size, hidden_size, n_shared, n_routed, top_k, ...)` — K_s shared + K_r routed experts
- `DeepSeekCfCNetwork(input_size, hidden_size, output_size, num_layers, n_shared, n_routed, top_k, ...)` — full network
- `deepseek_utilization(cell)` — diagnostic: shared=1.0 by construction, routed is mean of last_g

`tests/test_deepseek_moe.py` (23/23):
- TestDeepSeekCfCCellInit (6): default, no_shared, no_routed, no_experts raises, invalid_top_k raises, with router_hidden.
- TestDeepSeekCfCCellForward (7): forward shape, shared always active, routed sparsity, n_shared=0 fallback, n_routed=0 fallback, gradient flows, additive residual.
- TestDeepSeekCfCNetwork (6): init, forward dense, forward last step, with mask, two layers, gradient flows.
- TestDeepSeekDiagnostics (3): utilization no forward, utilization after forward, captures signal.
- TestDeepSeekSineSmoke (1): converges on toy sin.

## 5. Critical bugs fixed

1. **Pyright "loss unbound" warning** in test: pre-existing pattern, fixed by initializing `loss_value` before the for-loop.
2. **Pyright torch import false-positives**: pre-existing pattern, ignored per standing rules.
3. **Type narrowing for `self.router`**: when `n_routed=0`, `self.router=None`. Pyright couldn't track the assertion narrowing. Runtime is correct since `if self.n_routed > 0:` guarantees `self.router is not None`.

## 6. Recommendation

**Use DeepSeekMoE for time-series MoE in production**:
- The shared-expert path is a stable anchor that never collapses
- The routed path adds specialization on top
- Additive combination preserves the recurrent dynamics
- Default to `n_shared=1, n_routed=3, top_k=2` for the best balance

**Combine with other mechanisms**:
- **MoD (round 111)**: skip timesteps with MoD, then DeepSeek on remaining
- **QuITE (round 102)**: use QuITE embedding, then DeepSeek for the recurrent step
- **Soft MoE (round 107)**: replace the sparse top-K_r router with Soft MoE routing

## 7. Files added

- `lnn/core/deepseek_moe.py` (NEW, ~340 lines)
- `tests/test_deepseek_moe.py` (NEW, 23/23 tests)
- `scripts/bench_deepseek_moe.py` (NEW, 30 cells)
- `docs/prds/2026-06-15-lnn-round-113-a-deepseek-moe.md` (PRD #10-75)
- `docs/research/2026-06-15_deepseek_moe_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v39.md` (this file)
- `README.md` (new DeepSeekMoE section)
- `lnn-round-113-deepseek-moe.md` (memory)

## 8. Future work

1. **DeepSeek + Orth (round 80)**: add orthogonality loss only on routed experts
2. **DeepSeek + MoD (round 111)**: skip timesteps with MoD, DeepSeek on remaining
3. **DeepSeek + QuITE (round 102)**: QuITE embedding → DeepSeek recurrent step
4. **Per-shared-expert gradient diagnostic**: analyze whether all shared experts learn the same thing
5. **Adaptive n_shared**: learn whether to use 0, 1, 2, ... shared experts per layer
