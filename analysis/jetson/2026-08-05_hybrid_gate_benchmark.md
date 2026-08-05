---
title: MFC-Hybrid-Gate (input-dependent α) vs MFC-Hybrid (static α) vs CfC vs TFP — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, conditional-gating, input-dependent, alpha-MLP]
---

# MFC-Hybrid-Gate (input-dependent α) vs others — 2026-08-05

## Setup
- Training dt: LogNormal(0, 0.5) — IRREGULAR
- Test A: dt = 1.0 (regular)
- Test B: dt ~ LogNormal(0, 0.5) (irregular, same dist as train)
- 4 epochs, 3 repeats

## Results

| model | params | test_mse_regular | test_mse_irregular | degradation ratio |
|---|---:|---:|---:|---:|
| cfc-baseline | - | 0.0573 ± 0.0000 | 0.0574 ± 0.0000 | **1.00×** |
| mfc-cfc | - | 0.0572 ± 0.0001 | 0.0573 ± 0.0002 | **1.00×** |
| mfc-tfp | - | 0.0575 ± 0.0001 | 0.0605 ± 0.0002 | **1.05×** |
| mfc-hybrid (static α) | - | 0.0576 ± 0.0001 | 0.0582 ± 0.0002 | **1.01×** |
| mfc-hybrid_gate (input-dep α) | - | 0.0576 ± 0.0002 | 0.0578 ± 0.0001 | **1.00×** |

## α diversity (after 4 epochs training)

- **std over different x** (fixed dt=1): **0.0118**
- **std over different dt** (fixed x=0): **0.0045**

These non-zero std values confirm that **α is conditional on (x, dt)** in hybrid_gate, not a static parameter like in 'hybrid'.

## Interpretation

1. **Conditional α works**: the gate MLP produces different α values for different inputs and dt, providing true conditional gating that 'hybrid' (static α) cannot.
2. **Compare degradation ratios**:
   - cfc-baseline: 1.00× (untouched by dt — sigmoid saturation)
   - mfc-cfc: 1.00× (same)
   - mfc-tfp: 1.05× (irregular-train improves from regular-train baseline 1.14×)
   - mfc-hybrid (static α): 1.01×
   - **mfc-hybrid_gate (input-dep α): see table** — does conditional gating reduce degradation further?
