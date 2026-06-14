---
prd: 10-49
title: "Gradient-based H: causal MoE ecology E (replaces empirical H)"
date: 2026-06-15
status: draft
round: 87
authors: heyongxian
depends_on:
  - PRD #10-42 (MoE ecology E diagnostic, round 83, empirical H)
  - PRD #10-43 (Ecology-Gated φ-balancing, round 84)
  - PRD #10-44 (Ecology-Gated Orth, round 85)
  - PRD #10-48 (Combined Gates, round 86)
references:
  - arXiv:2605.06415  # E = T*H/(O+B) — paper that motivates the threshold
  - arXiv:2606.10703  # Causal Audit: observational ≠ causal
  - arXiv:2408.15664  # Auxiliary-Loss-Free Load Balancing (gradient-free approach)
---

# PRD #10-49 — Gradient-based H for MoE Ecology E

## 0. One-liner

Replace the **empirical** routing entropy H in `moe_ecology_number`
with a **gradient-based** H that measures how sensitive the loss is
to the routing distribution.  This addresses the **Causal Audit
arXiv:2606.10703** concern: empirical E is **observational** (it
measures what the routing looks like), while gradient-based E is
**causal** (it measures how much the routing matters for the loss).

The new class is `GradientRoutingSensitivity` in
`lnn/core/moe_ecology.py`, and the new opt-in flag is
`moe_ecology_number(..., H_mode="gradient")` (default `"empirical"`
for back-compat).

## 1. Problem

Round 83 (PRD #10-42) introduced `moe_ecology_number` with
**empirical H**:

```python
H = -Σ g_mean log g_mean / log(K)    # in [0, 1]
E = T · H / (O + B)
```

This is **observational**: it measures how uniform the routing
distribution looks.  In practice, this gives:

| Regime | Empirical H | What it implies | What it should imply |
|---|---:|---|---|
| Uniform routing | 1.0 | Healthy | Healthy ✓ |
| 1-hot routing (dead experts) | 0.0 | Unhealthy | Unhealthy ✓ |
| Mixed but loss-flat routing | > 0 | Healthy | **Not necessarily healthy** |
| Sharp but loss-sensitive routing | < 1 | Unhealthy | **Could be healthy** (sharp but working) |

The **Causal Audit (arXiv:2606.10703)** explicitly warns:
**observational ≠ causal**.  An empirical H that looks "healthy"
could mask a regime where the routing distribution is uniform but
the loss is **insensitive** to routing changes — meaning the
experts are functionally identical and the MoE has collapsed into
a single-expert mode in disguise.

**This round's fix**: add a **gradient-based H** that measures
**how much the loss changes when the routing changes**.  This is
the **causal** counterpart: it measures the routing's *impact on
the loss*, not just its *appearance*.

## 2. Goal (Scope)

**Minimum-viable gradient-based H for `moe_ecology_number`**:

- New function `gradient_routing_sensitivity(router_logits, loss)`
  that computes `H_grad = ||∂L/∂router_logits||` (Frobenius norm
  of the gradient of the task loss w.r.t. router logits, averaged
  over the batch).
- New `H_mode` argument on `moe_ecology_number`:
  - `H_mode="empirical"` (default, back-compat): use empirical H
  - `H_mode="gradient"`: use gradient-based H
  - `H_mode="blend"`: H = α·H_emp + (1-α)·H_grad (configurable α)
- New helper `MoEEcologyMonitor.compute_gradient_H(loss)` that
  computes the gradient H on demand (no impact on `step()`).
- New flag `FAMECfCCell.ecology_H_mode` (default `"empirical"`)
  that propagates to the diagnostic and the gates.
- Smoke bench: 4 conditions × 3 datasets × 2 H modes × 3 λ:
  - At λ=0.1, both H modes agree (no aux loss interference)
  - At λ=1.0, gradient H diverges from empirical H
    (the orth toxicity is causally visible in gradients)
  - At λ=10.0, gradient H ≪ empirical H (routing matters less
    when the aux loss dominates everything)
- **Honest-negative-friendly**: report cases where gradient H
  disagrees with empirical H, and which gives better early-warning
  for ecology collapse.

## 3. Out of Scope (Non-Goals)

- **Per-layer gradient H** — we compute the global gradient on
  the last forward step (per-cell, not per-layer)
- **Full Jacobian** — we use a simple Frobenius norm of ∂L/∂g
- **Hessian-based H** — second-order info is a follow-up
- **Real MoE LLM reproduction** — out of scope

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `gradient_routing_sensitivity` returns H_grad ≥ 0 (norm) | TBD |
| 2 | `moe_ecology_number(H_mode="empirical")` matches round 83 behavior | TBD |
| 3 | `moe_ecology_number(H_mode="gradient")` returns E_grad ≥ 0 | TBD |
| 4 | `moe_ecology_number(H_mode="blend", alpha=0.5)` = 0.5·H_emp + 0.5·H_grad | TBD |
| 5 | `FAMECfCCell(ecology_H_mode="gradient")` uses gradient H in gate firing | TBD |
| 6 | 10+ unit tests in `tests/test_gradient_based_h.py` | TBD |
| 7 | Smoke bench: 4 conditions × 3 datasets × 2 H modes × 3 λ | TBD |
| 8 | Back-compat: `pytest tests/test_moe_ecology.py` all green | TBD |
| 9 | Honest-negative: report divergence between empirical and gradient H | TBD |

## 5. Design

### 5.1 New function: `gradient_routing_sensitivity`

```python
def gradient_routing_sensitivity(
    router_logits: torch.Tensor,  # [B, K]
    task_loss: torch.Tensor,      # scalar (must have grad)
    normalize: bool = True,
) -> float:
    """Compute H_grad = ||∂L_task/∂router_logits|| (Frobenius norm).
    
    Args:
        router_logits: [B, K] raw router logits.
        task_loss: Scalar task loss (must require_grad).
        normalize: If True, divide by batch size and log(K) for
            scale-invariance.
    
    Returns:
        Scalar H_grad ≥ 0.  Large value ⇒ loss is sensitive to
        routing (routing matters).  Small value ⇒ loss is
        insensitive to routing (routing doesn't matter — collapse
        imminent).
    """
    if not router_logits.requires_grad:
        return 0.0
    grads = torch.autograd.grad(
        task_loss, router_logits,
        retain_graph=False, create_graph=False,
        allow_unused=True,
    )[0]
    if grads is None:
        return 0.0
    h = grads.norm().item()  # Frobenius norm
    if normalize:
        B, K = router_logits.shape
        h = h / max(B, 1) / max(np.log(max(K, 2)), 1e-8)
    return float(h)
```

### 5.2 New `H_mode` argument on `moe_ecology_number`

```python
def moe_ecology_number(
    router_logits, last_g,
    T=1.0, H=None, O=0.0, B=0.0, eps=1e-8,
    H_mode: str = "empirical",  # NEW
    alpha: float = 0.5,          # NEW (only for H_mode="blend")
    task_loss=None,              # NEW (only for H_mode="gradient" / "blend")
):
    if H is not None:
        H_val = float(H)  # user override
    elif H_mode == "empirical":
        # Same as round 83: empirical routing entropy.
        g_mean = last_g.mean(dim=0).clamp_min(eps)
        H_val = -(g_mean * torch.log(g_mean)).sum() / max(torch.log(torch.tensor(float(K))).item(), eps)
    elif H_mode == "gradient":
        H_val = gradient_routing_sensitivity(router_logits, task_loss, normalize=True)
    elif H_mode == "blend":
        H_emp = ... # empirical (same code as above)
        H_grad = gradient_routing_sensitivity(...)
        H_val = alpha * H_emp + (1 - alpha) * H_grad
    else:
        raise ValueError(H_mode)
    return T * H_val / (O + B + eps)
```

### 5.3 Wire into `FAMECfCCell`

New purely-additive constructor arg:

```python
FAMECfCCell(
    ...,
    ecology_H_mode: str = "empirical",  # NEW (back-compat default)
    ecology_H_alpha: float = 0.5,       # NEW (only for H_mode="blend")
)
```

When `ecology_H_mode != "empirical"`, the diagnostic
(`moe_ecology_diagnostic(B=...)`) needs a `task_loss` argument:

```python
def moe_ecology_diagnostic(
    self, B=0.0, T=1.0, O=0.0,
    task_loss=None,  # NEW
):
    ...
    E = moe_ecology_number(
        router_logits=self.last_g, last_g=self.last_g,
        T=T, H=None, O=O, B=B,
        H_mode=self.ecology_H_mode,
        alpha=self.ecology_H_alpha,
        task_loss=task_loss,
    )
    ...
```

This means **callers must pass `task_loss` explicitly** when
`H_mode != "empirical"`.  The default is `task_loss=None`, which
falls back to empirical H (with a warning).

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/moe_ecology.py` | MODIFY: add `gradient_routing_sensitivity` + H_mode | +100 |
| `lnn/core/fame_cfc.py` | MODIFY: add `ecology_H_mode` + `ecology_H_alpha` flags | +30 |
| `lnn/core/__init__.py` | MODIFY: export `gradient_routing_sensitivity` | +1 |
| `tests/test_gradient_based_h.py` | **NEW** | ~150 |
| `scripts/bench_gradient_based_h.py` | **NEW** | ~250 |
| `docs/research/2026-06-15_gradient_based_h_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-15_LNN_research_summary_v13.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add Gradient-based H section | +30 |

**Net**: 3 new + 3 modified = 6 files, ~750 lines.

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gradient H disagrees with empirical H in surprising ways | M | Bench exposes the divergence honestly; document the trade-off |
| `task_loss` not passed by caller → silent fallback | M | Print warning in fallback; document the API |
| Gradient computation adds memory overhead | L | We use `allow_unused=True` and `retain_graph=False`; gradient is single-pass |
| H_mode="gradient" fires gates too aggressively | M | Backward-compatible default: H_mode="empirical" |
| `torch.autograd.grad` breaks in `torch.no_grad()` context | M | Detect and return 0.0 (no gradient available) |

## 8. Verification Plan

1. **Unit tests** (`tests/test_gradient_based_h.py`):
   - `gradient_routing_sensitivity` returns ≥ 0
   - `gradient_routing_sensitivity` is 0 when `requires_grad=False`
   - `moe_ecology_number(H_mode="empirical")` matches round 83
   - `moe_ecology_number(H_mode="gradient", task_loss=...)` returns finite E
   - `moe_ecology_number(H_mode="blend", alpha=0.5)` = 0.5·H_emp + 0.5·H_grad
   - `FAMECfCCell(ecology_H_mode="gradient")` works in training mode
   - `FAMECfCCell(ecology_H_mode="empirical")` is back-compat (no task_loss needed)
   - Gates fire on gradient H same way as empirical H (when both healthy)

2. **Smoke bench** (`scripts/bench_gradient_based_h.py`):
   - 4 conditions × 3 datasets × 2 H modes × 3 λ
   - Compare E_emp vs E_grad trajectory
   - Compare gate firing decisions (E < 0.5)

3. **Regression**: all FAME+orth+φ+ecology+gate tests still pass

## 9. Rollout

Single PR.  After landing:
- `gradient_routing_sensitivity` is the canonical gradient-based H
- `moe_ecology_number(H_mode="gradient")` is the new opt-in API
- `FAMECfCCell(ecology_H_mode="gradient")` is the recommended opt-in
- README adds "Gradient-based H (causal MoE ecology)" section
- Back-compat: zero changes when `ecology_H_mode="empirical"` (default)

## 10. Why this is the right next step

- **Directly motivated by round 86's Causal Audit note**: the
  empirical H is observational; a gradient-based H is the natural
  causal counterpart.
- **Empirically testable**: 4 conditions × 3 datasets × 2 H modes ×
  3 λ = 72 cells
- **Honest-negative-friendly**: explicitly test where gradient H
  diverges from empirical H, and document which is more reliable
- **Closes the LNN+MoE diagnostic layer**: round 83 added the
  diagnostic; this round makes it **causally sound**
- **Directly testable on round 86's bench**: same 9 cells, swap
  H mode, see if gradient H detects the orth toxicity earlier

Other candidates rejected:
- **#10-47 Causal importance gate** — needs causal inference setup
  (out of scope for a single round)
- **#10-48.1 Per-layer gate config** — orthogonality has been
  consistent across layers in our bench; not yet needed
- **#10-46 Test on vision** — out of scope (no vision data)
- **#10-7 LFM2.5-1.2B INT8** — deployment, needs full stack

## 11. Open Questions

- **Q1**: Is `||∂L/∂g||` the right causal measure?
  - **A1**: It's the simplest.  Alternatives (Hessian trace,
    gradient alignment) are follow-ups.
- **Q2**: Should `task_loss` be the *task* loss or the *total* loss
  (task + aux)?
  - **A2**: We use **task loss** (so we measure the routing's
    impact on the user's objective, not on the aux loss).  The
    aux loss's effect is captured by the B term in the denominator.
- **Q3**: Should we normalise by batch size and K?
  - **A3**: Yes (scale-invariance).  Default `normalize=True`.
- **Q4**: Is gradient H strictly better than empirical H?
  - **A4**: **Unknown — that's what this round tests.**  We
    expect gradient H to be more sensitive to early collapse
    (e.g., when routing becomes a no-op) but less stable in
    high-noise regimes.  Bench will tell.
