---
prd: 10-44
title: "Ecology-Gated Orth Rescaling: auto-reduce λ when E < 0.5"
date: 2026-06-15
status: draft
round: 85
authors: heyongxian
depends_on:
  - PRD #10-37 (orthogonality constraint, round 80)
  - PRD #10-42 (MoE ecology E diagnostic, round 83)
  - PRD #10-43 (Ecology-Gated φ-balancing, round 84)
references:
  - arXiv:2605.06415  # E = T*H/(O+B) — paper that motivates the threshold
  - arXiv:2606.03631  # AnchorMoE orthogonality (the aux loss we rescale)
  - arXiv:2606.10703  # Causal Audit: observational ≠ causal
  - arXiv:2605.15403  # φ-Balancing (complementary intervention)
---

# PRD #10-44 — Ecology-Gated Orth Rescaling: auto-reduce λ when E < 0.5

## 0. One-liner

Round 84's gate picks φ-balancing (a soft intervention on the router)
when E < 0.5.  **Round 84 honest negative**: at λ=1.0 ortho-toxicity
(arXiv:2605.06415 finding #2), gated φ **does not recover** because
the orth loss is too strong for the soft router bias to counteract.
**This round fixes the root cause**: when E < 0.5, the gate **rescales
the orth loss weight λ down to a safe value** (e.g., 0.001 = round 80
default).  This converts the gate from "soft intervention" to **"soft
intervention on the intervention"** — a meta-controller.

## 1. Problem

Round 84 (PRD #10-43) gave us an ecology-gated policy that auto-enables
φ-balancing.  The smoke-bench surfaced a **clean honest negative**:

| Regime | always-φ | gated-φ | baseline |
|---|---:|---:|---:|
| toy_sin (λ=1.0) | 0.7286 | 0.7302 | 0.7347 |
| structured (λ=1.0) | 2.9057 | 2.8953 | 2.8953 |
| random (λ=1.0) | 0.9431 | 0.9420 | 0.9420 |

Gated φ is essentially identical to baseline because **the orth loss
λ=1.0 is so strong that the routing distribution collapses to 1-hot
and stays collapsed** (dead=2 in both toy_sin and structured).  The
gate fires (correctly), the balancer attaches (correctly), but the
intervention is too weak for the orth regime.

**Root cause analysis**: orth loss is **directly penalizing expert
representations** (`orthogonality_loss(outs, λ)` penalizes the
`cos_sim(outs_i, outs_j)` for all i≠j).  At λ=1.0, this gradient
dominates the task gradient, forcing experts to be maximally
orthogonal — but in a tiny K=3 top_k=1 toy, this means the cell
*can't* route to more than one expert without paying the orth cost.

**Solution**: instead of (or in addition to) attaching a soft router
bias, **rescale λ down to a safe value** when E < 0.5.  The gate
becomes a **meta-controller over the aux loss weight**.

## 2. Goal (Scope)

**Minimum-viable ecology-gated orth rescaling for `FAMECfCCell`**:

- New flag `ecology_gated_orth: bool = False` on `FAMECfCCell`.
  When `True`, the cell monitors live E and, if it drops below
  `ecology_E_min`, **scales the orth loss weight λ down** to a safe
  reference value (default `0.001`, the round 80 default).
- The rescaling is **non-destructive** — it doesn't mutate the user's
  `lambda_coeff`; it applies a per-step multiplicative factor
  `lambda_scale` (default 1.0).  When the gate fires, `lambda_scale`
  drops to `lambda_safe / lambda_coeff` so the *effective* λ is
  `lambda_safe`.
- New helper method `FAMECfCCell.set_ecology_orth_target(lambda_safe)`
  — configures the safe λ (default 0.001).
- **Back-compat**: zero changes when `ecology_gated_orth=False`.  The
  existing `orth_lambda` argument continues to work unchanged.
- New `scripts/bench_ecology_gated_orth.py` — 3 conditions × 3
  datasets × orth λ ∈ {0.1, 1.0, 10.0} to show:
  - At λ=0.1: gate doesn't fire, behaviour = baseline
  - At λ=1.0: gate fires, λ scales down, loss recovers
  - At λ=10.0: gate fires immediately, λ scales to 0.001
- **Honest-negative-friendly**: report cases where rescaling HURTS
  (e.g., when the user deliberately wants high orth for
  representation diversity)

## 3. Out of Scope (Non-Goals)

- **Adaptive λ per layer** (per-layer rescaling) — we apply the same
  factor to all layers in a `FAMECfCNetwork`
- **Auto-rescaling φ step size** — round 84's gate handles φ, this
  round handles orth; combined in a follow-up
- **Hard orth disable** (set λ=0 when E<0.5) — we rescale to `lambda_safe`,
  not zero, so we don't lose orth benefits entirely
- **Real MoE LLM reproduction** — out of scope

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `EcologyGatedOrth` class exists with `step(E, lambda_coeff) → scale` | TBD |
| 2 | Default scale = 1.0 (no rescaling) when E ≥ threshold | TBD |
| 3 | Scale = `lambda_safe / lambda_coeff` when E < threshold (latched) | TBD |
| 4 | `FAMECfCCell(ecology_gated_orth=True)` returns `lambda_scale` in diagnostic | TBD |
| 5 | `forward_with_aux(..., orth_lambda=...)` applies `lambda_scale` to orth | TBD |
| 6 | 10+ unit tests in `tests/test_ecology_gated_orth.py` | TBD |
| 7 | Smoke bench: orth λ ∈ {0.1, 1.0, 10.0} × 3 datasets × 2 conditions | TBD |
| 8 | Back-compat: `pytest tests/test_fame_cfc.py tests/test_orthogonality.py` all green | TBD |
| 9 | Honest-negative: report case where rescaling loses to baseline | TBD |

## 5. Design

### 5.1 Module addition: `EcologyGatedOrth` in `lnn/core/ecology_gated_balancing.py`

```python
class EcologyGatedOrth:
    """Ecology-gated orth rescaling: scale λ down to lambda_safe when E<threshold.
    
    Mirrors EcologyGatedBalancer's no-hysteresis semantics: once
    rescaled, stays rescaled (re-enabling high λ mid-training would
    re-collapse routing).
    
    Args:
        E_min: Threshold for intervention.  Default 0.5.
        lambda_safe: Target λ when gate fires.  Default 0.001
            (round 80 default).
        warmup_steps: Don't rescale in the first N steps.
    """
    def __init__(self, E_min: float = 0.5, lambda_safe: float = 0.001,
                 warmup_steps: int = 0): ...
    
    @torch.no_grad()
    def step(self, E: float, lambda_coeff: float, step_idx: int) -> dict:
        """Returns dict with:
            - lambda_scale: float (1.0 normally, lambda_safe/lambda_coeff when fired)
            - intervened: bool
            - triggered_step: int
        """
        ...
```

### 5.2 Wire into `FAMECfCCell`

```python
def __init__(
    self, ..., 
    ecology_gated_orth: bool = False,  # NEW
    ecology_orth_lambda_safe: float = 0.001,  # NEW
):
    ...
    if ecology_gated_orth:
        from lnn.core.ecology_gated_balancing import EcologyGatedOrth
        self.orth_gate = EcologyGatedOrth(
            E_min=self.ecology_E_min,
            lambda_safe=ecology_orth_lambda_safe,
            warmup_steps=self.ecology_warmup_steps,
        )
    else:
        self.orth_gate = None

def forward_with_aux(self, x_t, h, dt=1.0, orth_lambda: float = 0.0):
    """Returns (h_new, outs) with gate-applied λ."""
    # If gate is on and lambda_scale < 1, scale orth_lambda.
    if self.orth_gate is not None and self.training:
        diag = self.moe_ecology_diagnostic(B=orth_lambda)
        gate_info = diag.get("ecology_gate_orth", {})
        scale = gate_info.get("lambda_scale", 1.0)
        effective_lambda = orth_lambda * scale
    else:
        effective_lambda = orth_lambda
    # ... existing forward logic with effective_lambda ...
```

But this is awkward because `forward_with_aux` doesn't currently
take `orth_lambda` as an arg (it's an external loss).  Cleaner
approach: expose `cell.get_effective_orth_lambda(user_lambda)` as
a helper, and document that callers should use it.

### 5.3 Cleaner: `cell.compute_orth_loss(outs, user_lambda)`

```python
def compute_orth_loss(
    self, outs: list[torch.Tensor], user_lambda: float = 0.0
) -> torch.Tensor:
    """Compute orth loss with ecology-gated rescaling applied.
    
    If ecology_gated_orth=True and the gate has fired, scales user_lambda
    down to ecology_orth_lambda_safe.  Otherwise returns the standard
    orthogonality_loss(outs, user_lambda).
    
    Also runs the ecology gate step (uses last_g from prior forward).
    """
    if self.orth_gate is not None and self.training and user_lambda > 0:
        diag = self.moe_ecology_diagnostic(B=user_lambda)
        gate_info = diag.get("ecology_gate_orth", {})
        scale = gate_info.get("lambda_scale", 1.0)
        effective_lambda = user_lambda * scale
    else:
        effective_lambda = user_lambda
    if effective_lambda <= 0:
        return torch.tensor(0.0)
    from lnn.core.orthogonality import orthogonality_loss
    return orthogonality_loss(outs, lambda_coeff=effective_lambda)
```

This is the **recommended API**: callers use `cell.compute_orth_loss()`
instead of `orthogonality_loss()` directly, and the gate is applied
transparently.

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/ecology_gated_balancing.py` | MODIFY: add `EcologyGatedOrth` class | +60 |
| `lnn/core/fame_cfc.py` | MODIFY: add `ecology_gated_orth` flag + `compute_orth_loss()` | +30 |
| `lnn/core/__init__.py` | MODIFY: export `EcologyGatedOrth` | +1 |
| `tests/test_ecology_gated_orth.py` | **NEW** | ~120 |
| `scripts/bench_ecology_gated_orth.py` | **NEW** | ~200 |
| `docs/research/2026-06-15_ecology_gated_orth_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-15_LNN_research_summary_v11.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add Ecology-gated orth section | +25 |

**Net**: 3 new + 3 modified = 6 files, ~640 lines.

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rescaling loses user's deliberate choice (e.g., user wants high orth for diversity) | M | Default `ecology_gated_orth=False`; opt-in only.  Document the trade-off. |
| Gate fires too early, before orth has done its job | M | `warmup_steps` (default 0, configurable).  Recommend ≥ 5 in README. |
| Per-layer rescaling differs from per-network rescaling | L | We apply the same factor across all layers (one gate per cell).  Per-layer is a follow-up. |
| `compute_orth_loss()` requires training mode | L | Document: gate only runs in training mode (consistent with φ gate, round 84). |

## 8. Verification Plan

1. **Unit tests** (`tests/test_ecology_gated_orth.py`):
   - `EcologyGatedOrth` never rescales when E > threshold
   - Rescales to `lambda_safe/lambda_coeff` when E < threshold
   - Latched (no hysteresis)
   - Respects `warmup_steps`
   - `FAMECfCCell.compute_orth_loss(outs, user_lambda)` returns scaled loss
   - `moe_ecology_diagnostic` includes `ecology_gate_orth` key
   - Default `ecology_gated_orth=False` is back-compat

2. **Smoke bench** (`scripts/bench_ecology_gated_orth.py`):
   - 2 conditions × 3 datasets × 3 orth λ ∈ {0.1, 1.0, 10.0}:
     - Condition A: `FAMECfCCell(orth_lambda=λ)` — baseline
     - Condition B: `FAMECfCCell(ecology_gated_orth=True, orth_lambda=λ)` — gated
   - 3 datasets: toy sin, random, structured
   - Report: final loss, λ_scale trajectory, E trajectory, dead_experts

3. **Regression**: all FAME+orth+φ+ecology+gate tests still pass

## 9. Rollout

Single PR.  After landing:
- `EcologyGatedOrth` is the canonical reference impl
- `FAMECfCCell.compute_orth_loss()` is the recommended API
- README adds "Ecology-gated orth rescaling" section
- Back-compat: all existing code unchanged when `ecology_gated_orth=False`

## 10. Why this is the right next step

- **Directly motivated by round 84 honest negative**: gate's
  intervention (φ) is too weak for λ=1.0 orth toxicity.  This round
  fixes the root cause (rescale λ) instead of the symptom (add φ).
- **Closes the orth vs φ loop**: round 84's gate picks φ; this
  round's gate picks orth.  Together they form an **adaptive policy**
  that selects the right intervention per regime.
- **Empirically testable**: 2×3×3 = 18 cells, runs in <5 min
- **Honest-negative-friendly**: at λ=0.1 (healthy regime), the gate
  may *over*-rescale and hurt diversity.  Document it.

Other candidates rejected:
- **#10-45 Gradient-based H** — needs more research
- **#10-46 Test on vision** — out of scope (no vision data)
- **#10-47 Causal importance gate** — needs more design
- **#10-7 LFM2.5-1.2B INT8** — deployment, needs full stack
- **Timeflies 2606.13571** — interesting but not directly relevant

## 11. Open Questions

- **Q1**: Should rescaling be **latched** (no hysteresis) or
  **hysteretic** (re-enable high λ when E recovers)?
  - **A1**: **Latched** for round 85 (consistent with round 84 φ
    gate).  Hysteresis is a follow-up.
- **Q2**: What's the safe λ value?
  - **A2**: 0.001 (round 80 default, validated on 3 synthetic
    datasets per round 83 B).  User-overridable via
    `ecology_orth_lambda_safe` arg.
- **Q3**: Should we rescale to 0 (full disable) or to a small
  positive value (preserving some orth benefit)?
  - **A3**: Small positive value (default 0.001).  Full disable
    is a follow-up.
