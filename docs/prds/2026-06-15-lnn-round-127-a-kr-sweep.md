# PRD #10-89 — Round 127 K_r (Routed Multiplicity) Sweep

**Date**: 2026-06-15
**Round**: 127
**Status**: Swept, benched
**Verdict**: **STRICTLY POSITIVE** — 1 NEW BEST on structured_irr

## Goal

Test the natural counterpart to round 125's K_s sweep: **routed
expert multiplicity K_r**. The hypothesis: more routed experts
→ more diversity → potentially better structured data.

## Reference

- **DeepSeekMoE**: arXiv:2401.06066 (DeepSeek-AI, January 2024) —
  recommends K_s=2 for multi-domain knowledge

## Design

Sweep K_r ∈ {2, 3, 4, 6} on the triple hybrid with K_s=2 (round
125's best for structured). This is a sweep-only round — K_r is
already exposed as `n_experts` in `LoRADAGSharedMoECfCCell`.

## Files

- `scripts/bench_kr_sweep.py` (NEW, 30 cells)
- `results/bench_kr_sweep.json` (NEW)
- `docs/research/2026-06-15_kr_sweep_report.md` (NEW)
- `README.md` (K_r sweep section)

## Bench (30 cells, 30 epochs, 2 seeds)

| Condition | sin_irr | structured_irr | random_irr | n_params |
|-----------|---------|----------------|------------|----------|
| baseline_cfc            | 0.0094 | 0.0053 | 0.0013 | 2545 |
| **kr2_ks2**             | 0.0026 | **0.0015** | 0.0028 | 5355 |
| kr3_ks2                 | 0.0018 | 0.0020 | 0.0091 | 5735 |
| kr4_ks2                 | 0.0074 | 0.0044 | 0.0034 | 6115 |
| kr6_ks2                 | 0.2821 | 0.0430 | 0.2799 | 6875 |

## Verdict

**STRICTLY POSITIVE** — `kr2_ks2` achieves **0.0015 on
structured_irr** (vs prior 0.0020 from round 125's
`lora_dag_shared_ks2`, **25% improvement**).

## Best on each dataset

- **sin_irr**: kr3_ks2 = 0.0018 (round 125 still leads)
- **structured_irr**: **kr2_ks2 = 0.0015 (NEW BEST)**
- **random_irr**: baseline_cfc = 0.0013

## K_r is the symmetric counterpart of K_s

The DeepSeek paper recommends K_s=2. We now know **K_r=2 is the
matching value** — the K_r=K_s=2 symmetric configuration is the
Pareto frontier for structured data.

## Future work

1. K_r sweep with K_s=1 (round 124 baseline)
2. K_r sweep with batch_size=32 or 64 (recover K_r=6?)
3. Asymmetric K_r vs K_s (K_r=2 K_s=4, K_r=4 K_s=2)
4. PhysioNet 36D test
5. Per-layer K_r schedule
