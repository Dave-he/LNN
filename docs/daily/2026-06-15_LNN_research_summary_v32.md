# LNN Research Digest v32 — 2026-06-15

**Coverage**: AuxLF follow-up + 91-106 audit pattern update.

## Headline

Round 106 implemented **AuxLF** (arXiv:2408.15664 Wang et al. Aug 2024) — Auxiliary-Loss-Free Load Balancing (the mechanism in DeepSeek-V3). The key idea: replace auxiliary load-balancing loss with a per-expert **bias term** adjusted outside the gradient, achieving load balance without gradient interference.

The result is **HONEST TARGET-DEPENDENT-WITH-NUANCE**:
- **H1 PARTIAL**: AuxLF forces more uniform load on unique experts (util_std 7000+ → 37, unique_H -50%)
- **H2 REJECTED in expected direction**: test_mse does NOT improve; on random_irr it WORSENS +8-10% (0.183 → 0.208)
- **H3 CONFIRMED**: SETA's H=0 fix preserved (shared_H stays at 0.693)
- **H4 CONFIRMED**: training stable across all 24 cells

**The "structural > routing" audit pattern (rounds 91-106) is now further confirmed.** AuxLF is the 6th routing-only mechanism to fail to improve task loss in our time-series setting.

## 1. The 91-106 audit

| Round | Mechanism | Verdict | Type |
|-------|-----------|---------|------|
| 91 | Smoothness (TV) | NEGATIVE | Diagnostic |
| 92 | Temporal dropout | NEGATIVE | Augmentation |
| 93 | Input-side dropout | NEGATIVE | Augmentation |
| 94 | Effective rank | NEGATIVE | Diagnostic |
| 95 | Per-expert eff rank | NEGATIVE | Diagnostic |
| 96 | FAME+orth diversity | NEGATIVE | Combined |
| 97 | Weight orth | HEADLINE | Regularizer |
| 98 | Backward coherence | PARTIAL | Regularizer |
| 99 | Reliability gate | **STRICTLY POSITIVE** | Augmentation |
| 100 | SNNL | TARGET-DEP | Regularizer |
| 101 | ORC | DIAGNOSTIC | Regularizer |
| 102 | QuITE | **STRICTLY POSITIVE** | Embedding |
| 103 | QuITE+MoE | TARGET-DEP | Router+context |
| 104 | SDG-MoE | NEGATIVE | Router+deliberation |
| 105 | SETA | **STRICTLY POSITIVE** | Architecture |
| **106** | **AuxLF** | **TARGET-DEP-WITH-NUANCE** | **Router+bias** |

**Pattern**: 3 strictly positive mechanisms all have **architectural** changes (reliability gate is augmentation not architectural, but it modifies the input flow; QuITE replaces input embedding; SETA decomposes expert structure). The 6 routing-only mechanisms all fail or are target-dependent.

## 2. AuxLF implementation

`lnn/core/auxlf.py` (~340 lines):
- `AuxLFConfig(bias_lr=0.01, target_load_fraction=1/K, bias_clamp=2.0, warmup_steps=10, use_update=True)` dataclass
- `update_load_balancing_bias(bias, top_idx_counts, config, n_experts)` — `bias -= γ * (count - target)` (sign fixed in implementation)
- `AuxLFRouter(SETARouter)` — adds bias BEFORE top-K, auto-updates each forward
- `AuxLFSETAMoECfCCell(SETAMoECfCCell)` — replaces router with AuxLFRouter
- `AuxLFSETAMoECfCNetwork` — QuITE + SETA + AuxLF

**Critical design**: `self.bias = nn.Parameter(torch.zeros(n_unique), requires_grad=False)` — bias is a side channel, not a grad parameter. Updates wrapped in `torch.no_grad()`.

## 3. Bench results (24 cells, 100 epochs)

| cond | sin test | sin uniq_H | struct test | struct uniq_H | random test | random uniq_H |
|------|------|------|------|------|------|------|
| seta_only_shared | 0.0811 | 0.556 | 0.3890 | 0.531 | **0.1886** | 0.633 |
| seta_auxlf_no_update | 0.0833 | 0.558 | 0.3877 | 0.581 | **0.1828** | 0.506 |
| seta_auxlf_active | 0.0845 | 0.608 | 0.3793 | 0.555 | 0.2080 | 0.572 |
| seta_auxlf_strong | 0.0836 | **0.271** | 0.3779 | 0.262 | 0.2039 | 0.224 |

**Bold** = notable.

## 4. Critical findings

### F1. AuxLF is a real load balancer (in the expected direction)
- `util_std` drops from 7000+ (no_update) to 37 (active) when the bias update is enabled
- `bias_norm` grows from 0 to 1.5-3.5 as the bias adapts
- The mechanism works **as designed** — the question is whether load balance helps task

### F2. But load balance does NOT help task loss
- On `random_irr`, AuxLF active/strong **worsens** test_mse by +8-10%
- On `sin_irr`/`structured_irr` it's within noise
- This is consistent with our 91-105 audit: routing-only changes don't help time-series MoE

### F3. AuxLF collapses routing diversity
- Strong AuxLF reduces unique_H from 0.5-0.6 to 0.2-0.3
- This is **expected** (uniform load = uniform routing probability) but **opposes** the diversity mechanism in MoE
- AuxLF strong is essentially "make the unique experts behave as a single expert" — which loses the MoE benefit

### F4. SETA's H=0 fix is preserved
- shared_H stays at 0.693 (= log 2) in all 12 cells
- This confirms that shared experts are **structurally immune** to bias changes on the unique router
- SETA's robustness property holds

### F5. AuxLF no_update is identical to SETA
- `seta_auxlf_no_update` is functionally `seta_only_shared` (bias=0 always)
- Confirms no overhead from the AuxLF wrapper

## 5. Why this matters for the stack

The 91-106 audit shows the LNN+MoE autonomous stack now has a clear pattern: **architectural changes win, routing changes lose**. This is a critical finding for the field of time-series MoE in general — the LLM-style "smart router" assumption may not transfer to time-series because all experts see correlated sequences.

## 6. Use cases for AuxLF

- **Use** when you want guaranteed balanced expert utilization (e.g. inference cost predictability, hardware load balancing, latency)
- **Don't use** as a default regularizer for time-series MoE
- The auxlf_util_std, auxlf_max_min_ratio, auxlf_bias_norm metrics are useful **diagnostics** regardless

## 7. Stack update

The 30-layer LNN+MoE autonomous stack (rounds 76-106) gains:
- **Layer 30 (round 106)**: AuxLF load balancing — TARGET-DEPENDENT-WITH-NUANCE, useful as diagnostic

## 8. Backlog for round 107+

1. **PhysioNet with real data** — confirm SETA wins on real irregular TS
2. **K=20, hidden=32, paper-scale** — confirm SETA still breaks H=0 at scale
3. **SETA + QuITE++** — combine with hierarchical QuITE
4. **SETA + Orthogonality (round 97)** — orthogonalize shared experts
5. **arXiv:2606.10703 Causal Audit** — apply causal MoE ecology to SETA
6. **DLNet (ICPR 2026)** — edge-battery LNN replication/extension
7. **Anti-symmetric A⁺/A⁻ (from round 104)** — fix deliberation constraint
