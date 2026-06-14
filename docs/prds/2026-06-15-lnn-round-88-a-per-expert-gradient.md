---
PRD: #10-50
date: 2026-06-15
round: 88
title: Per-Expert Gradient Magnitude (per-expert causal ecology E)
builds-on: [round 87 (PRD #10-49), arXiv:2409.12136 (GRIN, Liu et al. 2024)]
status: draft
---

# PRD #10-50 — Per-Expert Gradient Magnitude (per-expert causal ecology E)

## 1. Background

Round 87 (PRD #10-49) added `H_mode="gradient"` to `moe_ecology_number`.
The result was an **honest negative**: in our 9-cell bench, the
**aggregated** E_emp ≈ E_grad (mean |Δ| < 0.05), with identical
gate-firing decisions in 9/9 cells.

This round refines the gradient H from **aggregated** to
**per-expert**.  The motivation is GRIN (arXiv:2409.12136, Liu et al.
2024), which uses **per-expert gradient estimation** to inform MoE
routing.  Their key insight: aggregated signals (entropy, balancing
loss) **average out** per-expert pathologies.  A dead expert can be
masked by healthy experts in the aggregate.

**This round's hypothesis**: per-expert gradient magnitude
`||∂L/∂g_k||` is a **more sensitive** diagnostic than aggregated
H_emp or H_grad for detecting **per-expert collapse**.  It can
catch individual dead experts that aggregated metrics miss.

## 2. Goals

1. **Per-expert H_grad**: function `per_expert_gradient_norms(
   router_logits, task_loss, normalize=True) -> [K]` returns per-expert
   gradient magnitudes.
2. **Per-expert E**: `moe_ecology_number` returns per-expert E when
   `H_mode="per_expert_gradient"`.
3. **Per-expert dead detection**: `MoEEcologyMonitor` exposes
   `per_expert_gradient_diagnostic()` and identifies "dead by
   gradient" experts (per-expert H_grad ≈ 0).
4. **Wire into `FAMECfCCell`** with opt-in flag
   `ecology_per_expert_grad=True`.
5. **Honest bench**: 2 conditions × 3 datasets × 3 orth λ = 9 cells.
   Compare per-expert empirical H vs per-expert gradient H for
   dead-expert detection.

## 3. Design

### 3.1 New function: `per_expert_gradient_norms`

```python
def per_expert_gradient_norms(
    router_logits: torch.Tensor,  # [B, K], requires_grad
    task_loss: torch.Tensor | None,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-expert gradient norms H_grad_k = ||∂L/∂g_k||.

    Returns [K] tensor of non-negative values, one per expert.
    """
    if task_loss is None or not router_logits.requires_grad:
        return torch.zeros(router_logits.shape[-1])
    grads = torch.autograd.grad(
        task_loss, router_logits,
        retain_graph=True, create_graph=False, allow_unused=True,
    )[0]
    if grads is None:
        return torch.zeros(router_logits.shape[-1])
    # grads: [B, K]. Per-expert norm = ||g_b,k|| over batch dim.
    per_expert = grads.norm(dim=0)  # [K]
    if normalize:
        B = router_logits.shape[0]
        per_expert = per_expert / max(B, 1)
    return per_expert
```

### 3.2 New `H_mode="per_expert_gradient"` on `moe_ecology_number`

When `H_mode="per_expert_gradient"` and `task_loss` is provided,
return **per-expert E** as a tensor `[K]` (not scalar).  Else silent
fallback to per-expert empirical (uniform H over experts).

```python
elif H_mode == "per_expert_gradient":
    if task_loss is None:
        # Fall back to per-expert empirical
        return per_expert_empirical_fallback
    h_per_expert = per_expert_gradient_norms(router_logits, task_loss)
    return T * h_per_expert / (denom + eps)  # [K]
```

For `H_mode="empirical"` and `H_mode="gradient"`, behaviour is
**unchanged** (back-compat).  The new mode is purely additive.

### 3.3 New `MoEEcologyMonitor.per_expert_gradient_diagnostic`

```python
@torch.no_grad()
def per_expert_gradient_diagnostic(
    self,
    router_logits: torch.Tensor,
    task_loss: torch.Tensor,
    dead_grad_threshold: float = 1e-6,
) -> dict:
    """Per-expert gradient magnitude diagnostic.

    Returns dict with:
    - per_expert_grad: [K] tensor of gradient norms
    - dead_by_grad: int (count of experts with grad < threshold)
    - max_grad: float
    - min_grad: float
    - alive_experts: list[int] (indices of experts with grad >= threshold)
    """
```

### 3.4 Wire into `FAMECfCCell`

New constructor args (default off, back-compat):

```python
FAMECfCCell(
    ...,
    ecology_per_expert_grad: bool = False,  # opt-in
    ecology_dead_grad_threshold: float = 1e-6,
)
```

`moe_ecology_diagnostic` signature extended:

```python
def moe_ecology_diagnostic(
    self, B=0.0, T=1.0, O=0.0,
    task_loss=None,
    per_expert: bool = False,  # NEW
) -> dict:
    if per_expert or self.ecology_per_expert_grad:
        # Call moe_ecology_number with H_mode="per_expert_gradient"
        ...
```

## 4. Tests (target ≥ 12/12 unit tests)

- `per_expert_gradient_norms` returns [K] tensor
- Returns zero tensor when `task_loss` is None
- Returns zero tensor when `router_logits.requires_grad=False`
- Returns finite non-negative per-expert tensor
- Normalise scales invariantly
- `H_mode="per_expert_gradient"` returns per-expert E (tensor, not scalar)
- Falls back to per-expert empirical when no `task_loss`
- `MoEEcologyMonitor.per_expert_gradient_diagnostic` returns expected dict
- `FAMECfCCell(ecology_per_expert_grad=True)` uses per-expert H
- `FAMECfCCell(ecology_per_expert_grad=False)` is back-compat
- Per-expert H_grad identifies dead experts (synthetic: train K=1 routing)
- Invalid `H_mode="per_expert_gradient"` arguments handled

## 5. Bench

`scripts/bench_per_expert_gradient.py` — 2 conditions × 3 datasets × 3 λ.

Per cell:
- E_agg_emp, E_agg_grad: aggregated (round 87 behavior)
- per_expert_grad: [K] tensor
- dead_by_util: count of dead experts by utilization EMA
- dead_by_grad: count of dead experts by gradient
- max_dead_disagreement: max |dead_by_util - dead_by_grad| over 3 datasets

Hypotheses:
- **H1**: per-expert H_grad detects dead experts that utilization EMA
  misses in some cells
- **H2**: per-expert H_grad and per-expert empirical H (utilization)
  can disagree on **which** experts are dead

## 6. Honesty section

1. **Honest-positive expected**: per-expert H_grad **does** identify
   specific dead experts that aggregated metrics mask.
2. **Honest-negative possible**: per-expert H_grad may agree with
   per-expert empirical H in toy regime (same as round 87).
3. **2-epoch quick bench** — longer training may show more divergence.
4. **K=3 top_k=1** — larger K, larger top_k may show more per-expert
   pathologies.
5. **No normalization ablation** — default normalizes by B; raw norm
   may give different signal magnitude.
6. **Silent fallback** — task_loss=None falls back silently.
7. **Computational cost**: per-expert grad requires extra autograd call
   (slightly slower than aggregated grad which reuses the full
   gradient).  Not a bottleneck at toy scale.

## 7. Files

| File | Action | Lines |
|---|---|---|
| `lnn/core/moe_ecology.py` | MODIFY: add `per_expert_gradient_norms`, `H_mode="per_expert_gradient"` | +60 |
| `lnn/core/fame_cfc.py` | MODIFY: add `ecology_per_expert_grad` flag, `per_expert` arg | +30 |
| `lnn/core/__init__.py` | MODIFY: export `per_expert_gradient_norms` | +1 |
| `tests/test_per_expert_gradient.py` | **NEW** | 12+ tests |
| `scripts/bench_per_expert_gradient.py` | **NEW** | 200 lines |
| `docs/prds/2026-06-15-lnn-round-88-a-per-expert-gradient.md` | **NEW** | this file |
| `docs/research/2026-06-15_per_expert_gradient_report.md` | **NEW** | bench + analysis |
| `docs/daily/2026-06-15_LNN_research_summary_v14.md` | **NEW** | digest v14 |
| `README.md` | MODIFY: add Per-expert Gradient section | +25 |

## 8. Acceptance criteria

- 12+ unit tests pass
- 148+ full MoE+gate+gradient+per-expert regression suite green
- Bench completes 9 cells
- Honest-positive or honest-negative clearly reported
- Push to master via 140.82.112.4
- Memory file `lnn-round-88-per-expert-gradient.md` written
