---
prd: 10-42
title: "MoE Ecology Diagnostic: E = T·H/(O+B) for FAME Cell Health"
date: 2026-06-14
status: draft
round: 83
authors: heyongxian
depends_on:
  - PRD #10-36 (FAME top-K router, round 78)
  - PRD #10-37 (orthogonality constraint, round 80)
  - PRD #10-40 (φ-balancing, round 81)
references:
  - arXiv:2605.06415  # E = T*H/(O+B) — dimensionless MoE ecology parameter (direct template)
  - arXiv:2606.10703  # Causal Audit: observational ≠ causal
  - arXiv:2605.15403  # φ-Balancing
related:
  - arXiv:2606.03631  # AnchorMoE: orthogonality
  - arXiv:2605.12476  # Geometric Coupling: parameter-free K-Means
---

# PRD #10-42 — MoE Ecology Diagnostic: E = T·H/(O+B) for FAME Cell Health

## 0. One-liner

Add a **dimensionless ecology number E = T·H/(O+B)** (Zhang 2026) as a
diagnostic for `FAMECfCCell` health, plus a `MoEEcologyMonitor` that
tracks dead-expert count, expert utilization histogram, and E over
training.  This is the minimum-viable LNN implementation of the
"MoE ecology" framework from arXiv:2605.06415 (Zhang 2026), which
establishes that **E ≥ 0.5 alone is sufficient to guarantee zero dead
experts, removing the necessity for handcrafted load-balancing
auxiliary losses**.  Crucially, the paper also reports **"ortho
toxicity is dataset-dependent, not universal"** — directly challenging
our round 80 orthogonality constraint.

## 1. Problem

Round 76-82 built a 5-layer LNN+MoE stack and empirically showed that
orthogonality (round 80) and φ-balancing (round 81) both fix the
K=3 top_k=1 hard cell.  But we have no **theoretical diagnostic** for
*when* a cell is healthy and *when* it needs an intervention.  The
arXiv:2605.06415 paper proposes such a diagnostic:

- **E = T · H / (O + B)**
  - T = routing temperature
  - H = routing entropy weight
  - O = oracle weight
  - B = balance (load-balancing loss) weight
- **E ≥ 0.5** ⇒ healthy ecology, no dead experts
- **E < 0.5** ⇒ dead experts will emerge, intervention needed

**Key counter-intuitive finding**: the paper's 12 controlled
experiments (8 vision + 4 language, 11K epochs total) show that
**"ortho toxicity is dataset-dependent, not universal"** — meaning
our round 80 orthogonality constraint (λ=0.001) which works on toy
sin/cos may **hurt** on other datasets.

This is a **scientific challenge to our round 80 design** that we
should test, not ignore.

## 2. Goal (Scope)

**Minimum-viable MoE ecology diagnostic for `FAMECfCCell`**:

- New `lnn/core/moe_ecology.py` with:
  - `moe_ecology_number(router_logits, last_g, T=1.0, H=1.0, O=0.0, B=0.0) → E`
  - `MoEEcologyMonitor(n_experts, ema_alpha=0.01)` — tracks per-expert
    utilization EMA, dead-expert count, E trajectory over training
- New `FAMECfCCell.moe_ecology_diagnostic()` method that returns the
  current E, dead-expert count, and per-expert utilization
- New `scripts/bench_moe_ecology.py` — measures E trajectory for the
  round 79 sweep's 16-cell grid, then tests the **ortho toxicity
  hypothesis** by comparing λ ∈ {0, 0.001, 0.01, 0.1, 1.0} ×
  dataset (toy sin, random, structured)
- **Back-compat**: zero changes to existing API; the diagnostic is
  purely additive

## 3. Out of Scope (Non-Goals)

- **Automatic E-based intervention** (e.g., dynamically enabling
  φ-balancing when E drops below 0.5) — that's a follow-up round
- **Replace orthogonality/φ with E-based routing** — the paper's E
  ≥ 0.5 condition is for **preventing dead experts**, not for
  maximizing task accuracy.  Our orth + φ are still needed for the
  K=3 top_k=1 hard cell.
- **Real MoE LLM reproduction** (1B+ params) — out of scope
- **All 12 paper experiments** — we focus on the **ortho toxicity**
  and **E ≥ 0.5** hypotheses that are most relevant to our stack

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `lnn/core/moe_ecology.py` exports `moe_ecology_number` + `MoEEcologyMonitor` | TBD |
| 2 | `E = T·H/(O+B)` matches paper's definition | TBD |
| 3 | `MoEEcologyMonitor` tracks per-expert utilization EMA, dead-expert count | TBD |
| 4 | `FAMECfCCell.moe_ecology_diagnostic()` returns E, dead_count, util | TBD |
| 5 | 10+ unit tests in `tests/test_moe_ecology.py` | TBD |
| 6 | Smoke bench: E measured across λ ∈ {0, 0.001, 0.01, 0.1, 1.0} on toy sin | TBD |
| 7 | Ortho toxicity test: same setup, different synthetic datasets, λ effect | TBD |
| 8 | `pytest tests/test_fame_cfc.py tests/test_orthogonality.py tests/test_moe_ecology.py` all green | TBD |

## 5. Design

### 5.1 Module: `lnn/core/moe_ecology.py`

```python
def moe_ecology_number(
    router_logits: torch.Tensor,  # [B, K] raw router logits (or g)
    last_g: torch.Tensor,         # [B, K] mixture weights (post top-K mask)
    T: float = 1.0,               # routing temperature
    H: float = 1.0,               # routing entropy weight
    O: float = 0.0,               # oracle weight (task-loss specific)
    B: float = 0.0,               # balance (aux load-balancing) weight
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute E = T·H/(O+B) — the MoE ecology diagnostic (Zhang 2026).
    
    Args:
        router_logits: [B, K] raw router logits.
        last_g: [B, K] mixture weights (post top-K mask + softmax).
        T: Routing temperature (paper's notation).
        H: Routing entropy weight (paper's notation).
        O: Oracle weight (task-specific, usually 0 in our setting).
        B: Balance (aux LB loss) weight.  In our stack:
            - 0 for plain learned router
            - 0.05 for φ-balancing (when phi_step_size=0.05)
            - 0.001 for orthogonality (when lambda_coeff=0.001)
    """
    # H_eff = -Σ g log g, normalized by log K.  We approximate T=1
    # (no temperature scaling in our router), so the formula reduces
    # to the empirical health proxy used in the paper.
    g_mean = last_g.mean(dim=0)  # [K]
    entropy = -(g_mean * torch.log(g_mean.clamp_min(eps))).sum()
    return T * entropy / (O + B + eps)


class MoEEcologyMonitor:
    """Track MoE cell health over training.
    
    Records per-expert utilization EMA, dead-expert count, and E
    trajectory.  Use ``step(g)`` to update; ``summary()`` for current
    state.
    """
    def __init__(self, n_experts, dead_threshold=0.01, ema_alpha=0.01):
        ...
    @torch.no_grad()
    def step(self, g: torch.Tensor) -> dict:
        """Update with one batch's mixture weights.  Returns current E, dead_count."""
        ...
    def summary(self) -> dict:
        return {"E": ..., "dead_experts": ..., "utilization": ...}
```

### 5.2 Wire into `FAMECfCCell`

Add a non-invasive diagnostic method:

```python
def moe_ecology_diagnostic(self, B: float = 0.0) -> dict:
    """Return current E, dead-expert count, per-expert utilization.
    
    Args:
        B: Balance weight (pass your lambda_coeff or phi_step_size).
    """
    if not hasattr(self, "last_g") or self.last_g is None:
        return {"E": float("nan"), "dead_experts": -1, "utilization": []}
    E = moe_ecology_number(
        router_logits=self.last_g,  # approximate
        last_g=self.last_g,
        T=1.0, H=1.0, O=0.0, B=B,
    )
    util = self.last_g.mean(dim=0)
    dead = int((util < 0.01).sum().item())
    return {"E": float(E.item()), "dead_experts": dead, "utilization": util.tolist()}
```

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/moe_ecology.py` | **NEW** | ~100 |
| `lnn/core/fame_cfc.py` | MODIFY: add `moe_ecology_diagnostic()` method | ~+20 |
| `lnn/core/__init__.py` | MODIFY: export `moe_ecology_number`, `MoEEcologyMonitor` | ~+1 |
| `tests/test_moe_ecology.py` | **NEW** | ~120 |
| `scripts/bench_moe_ecology.py` | **NEW** | ~150 |
| `docs/research/2026-06-14_moe_ecology_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-14_LNN_research_summary_v9.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add MoE ecology section | ~+25 |

**Net**: 3 new + 3 modified = 6 files, ~620 lines (modest PR).

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| E formula not directly applicable to our FAME stack (different notation) | M | The paper's E is dimensionally consistent: T·H/(O+B).  Our H (entropy) and B (balance weight) map cleanly.  T=1 since we don't use temperature scaling. |
| Dead-expert threshold (0.01) is arbitrary | L | Make it a parameter; sweep 0.001, 0.01, 0.05 |
| Paper's E ≥ 0.5 doesn't hold on toy | H | That's the **point** — we want to see what E looks like on toy.  The ortho toxicity finding is dataset-dependent, and our toy may be in the "ortho helps" regime. |
| Backward compat breaks `FAMECfCCell` API | L | The diagnostic method is purely additive; no existing method changes signature |

## 8. Verification Plan

1. **Unit tests** (`tests/test_moe_ecology.py`):
   - `moe_ecology_number` matches expected formula on synthetic logits
   - Uniform softmax → max entropy → E high
   - Argmax-only → zero entropy → E = 0
   - `MoEEcologyMonitor.step` updates EMA in-place
   - Dead-expert detection triggers correctly
   - `FAMECfCCell.moe_ecology_diagnostic` returns valid dict

2. **Smoke bench** (`scripts/bench_moe_ecology.py`):
   - **E trajectory** on round 79's 16-cell grid (K × top_k × n_tau)
   - **Ortho toxicity**: same K=3 top_k=1 setup, λ ∈ {0, 0.001, 0.01, 0.1, 1.0}
     on **3 synthetic datasets**: (a) toy sin (b) random gaussian (c) structured
     (sin + 2*cos + 0.3*noise)
   - Report: E, dead_count, task loss, per-expert utilization histogram

3. **Regression**: `pytest tests/test_fame_cfc.py tests/test_orthogonality.py` all green

## 9. Rollout

Single PR.  After landing:
- `lnn/core/moe_ecology.py` is the canonical reference impl
- `FAMECfCCell.moe_ecology_diagnostic()` is the API entry point
- README adds a "MoE Ecology" section with the E ≥ 0.5 rule of thumb

## 10. Why this is the right next step (not other candidates)

- **Theoretical depth**: Zhang 2026's E provides a **single number**
  for MoE health — exactly what our 5-layer stack needs for diagnosis
- **Empirical challenge**: "ortho toxicity is dataset-dependent" is a
  **direct counter-claim** to our round 80 — we should test it
- **Empirical challenge**: "E ≥ 0.5 alone sufficient" is a direct
  counter-claim to our round 81 φ-balancing — we should test it
- **Scope**: 1 new file + 2 modifications + 1 bench ≈ 3-4h
- **Honest-negative friendly**: if E ≥ 0.5 doesn't hold on our toy, we
  document that (consistent with round 73, 82 negative findings)

Other candidates rejected:
- **#10-7 LFM2.5-1.2B INT8** (downstream, needs full stack to be deployment default)
- **Timeflies (2606.13571)** — interesting but irregular time series
  modeling, not directly relevant to our LNN+MoE stack
- **MR-MoE 2606.12240** — already cited in round 76; not a new template
- **Real SNBC data** — needs full stack stable

## 11. Open Questions (to resolve in implementation)

- **Q1**: How do we map the paper's H (routing entropy weight) to our
  FAME stack?
  - **A1**: We approximate H as the **empirical entropy of the average
    mixture weights** g_mean.  This is a 0-th order approximation;
    the paper uses gradient-based H.  But for the toy diagnostic
    purpose, the empirical version is enough.
- **Q2**: Should `B` (balance weight) include both orth λ and φ η?
  - **A2**: **No** — the paper's B is one weight.  We pass B as a
    parameter to `moe_ecology_diagnostic` and the user passes the
    **active** balance weight (orth λ, or φ η, or 0).
- **Q3**: What dataset to use for the ortho toxicity test?
  - **A3**: **3 synthetic datasets** (toy sin, random gaussian, structured
    sin+cos) to cover the "easy", "hard", and "structured" regimes.
