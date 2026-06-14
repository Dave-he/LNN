---
prd: 10-40
title: "φ-Balancing: EMA-based Expert Load Balancing for FAMECfC"
date: 2026-06-14
status: draft
round: 81
authors: heyongxian
depends_on:
  - PRD #10-36 (FAME top-K router, round 78)
  - PRD #10-37 (orthogonality constraint, round 80)
references:
  - arXiv:2605.15403  # φ-Balancing: strictly convex symmetric differentiable potential
  - arXiv:2408.15664  # DeepSeek-V3 Loss-Free Balancing (auxiliary-loss-free baseline)
  - arXiv:2605.18498  # DBES diagnostic — expert utilization tracking
related:
  - arXiv:2606.10703  # Causal Audit: observational vs interventional expert importance
---

# PRD #10-40 — φ-Balancing: EMA-based Expert Load Balancing for FAMECfC

## 0. One-liner

Add a **per-expert EMA-tracked routing bias** to the FAME top-K router so that
**frequently-activated experts get demoted and rarely-activated experts get
promoted**, **without introducing any extra gradient** (mirror-descent on a
strictly convex potential). This is the minimum-viable LNN-flavored
implementation of the φ-Balancing framework (arXiv:2605.15403, 2026-05-14).

## 1. Problem

Round 80 showed that the **K=3 top_k=1** FAME cell (round 79 sweep's
hard-blocker) is **unstable without orthogonality**:
- task loss 0.7595 ± 0.7906, 1/3 seeds diverged
- root cause: router-argmax single-expert mode + collapse-prone experts
- **orthogonality fixes it** (λ=0.001 → 0.1089, 0/3 diverged)

But orthogonality is a **defensive auxiliary loss** — it forces experts to be
geometrically decorrelated, not balanced. A **complementary** intervention is
**load balancing**: prevent any single expert from dominating the routing
probabilities (which is the direct cause of routing collapse in top_k=1 mode).

**φ-Balancing** is the principled answer:
- Operates on **population-level expected routing distribution**, not noisy
  mini-batch statistics
- Uses a **strictly convex, symmetric, differentiable potential** φ(·) of the
  expected assignment fractions f_k
- Mirror descent on the dual yields an **EMA-based bias update** — no extra
  gradients, negligible overhead
- Outperforms Switch-style aux losses and DeepSeek-V3's Loss-Free Balancing in
  the paper's large-scale pretraining experiments

## 2. Goal (Scope)

**Minimum-viable φ-Balancing for `FAMECfCNetwork`**, **composable with the
existing orthogonality constraint** (PRD #10-37):
- New `PhiBalancer` module in `lnn/core/phi_balancing.py`:
  - Maintains `f_k = EMA_k(assignment_indicator_k)` per expert
  - Maintains `b_k = mirror_descent_step(f_k)` per expert (no_grad bias)
  - `bias ∈ R^K` is **added to router logits** before the top-K mask
- New `forward_with_aux_balanced` methods on `FAMECfCCell` and
  `FAMECfCNetwork` that apply the bias and **update the EMA in eval-mode-safe
  fashion** (default: update in train mode, frozen in eval)
- New `PHIBalancedFAMECfCNetwork` subclass (or a `phi_balance: bool` flag on
  the base class) that wires the balancer into every layer's router
- **Back-compat**: when `phi_balance=False` (default), the network is
  numerically equivalent to round 80's `FAMECfCNetwork`

## 3. Out of Scope (Non-Goals)

- **Strictly convex potential φ** (the paper has a particular φ choice;
  we use a simple φ(f) = -Σ f_k log f_k, i.e. negative entropy, which is the
  simplest strictly convex symmetric differentiable potential that matches
  the paper's intuition). A future round can swap in the paper's exact φ.
- **Per-token (not per-step) bias** — for the toy sin dataset, per-step is
  sufficient. Real data may want per-token. Hooks for it are noted.
- **Mirror descent step size adaptation** — we use a fixed step size η.
- **φ-Balancing on `MRMoECfCCell`** (dense softmax) — the paper's claims are
  strongest for top-K sparse routing, and the round 79 sweep identified
  K=3 top_k=1 as the failing cell. Dense softmax is less prone to collapse.
- **Integration with SNBC 5000-machine** — the FAME production reproduction
  is downstream work.

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `lnn/core/phi_balancing.py` exports `PhiBalancer` | TBD |
| 2 | `PhiBalancer(forward(...))` returns biased logits: `logits + b_k` | TBD |
| 3 | `PhiBalancer.update(assignments)` updates EMA **in-place** with no grad | TBD |
| 4 | `phi_balance=False` ⇒ back-compat (numerically equivalent to round 80) | TBD |
| 5 | `phi_balance=True` ⇒ bias added + EMA updated each step (train) | TBD |
| 6 | `phi_balance=True` + `model.eval()` ⇒ bias frozen (no update) | TBD |
| 7 | 10+ unit tests in `tests/test_phi_balancing.py` (no regression) | TBD |
| 8 | Smoke bench: K=3 top_k=1 toy sin, **λ=0 + φ=η=0.01** ≤ baseline 0.7595 | TBD |
| 9 | Smoke bench: **λ=0.001 + φ=η=0.01** ≤ orth-only 0.1089 (synergy) | TBD |
| 10 | `pytest tests/test_fame_cfc.py tests/test_orthogonality.py tests/test_phi_balancing.py` all green | TBD |

## 5. Design

### 5.1 Module: `lnn/core/phi_balancing.py`

```python
class PhiBalancer(nn.Module):
    """EMA-based per-expert routing bias (φ-balancing, arXiv:2605.15403).
    
    The bias is a mirror-descent step on the strictly convex potential
    φ(f) = -Σ f_k log f_k (negative entropy), yielding a bias proportional
    to log(1/f_k) = -log f_k.  Frequently-activated experts (high f_k) get
    a large negative bias, rarely-activated experts (low f_k) get a positive
    bias, encouraging uniform utilization.
    
    Args:
        n_experts: K.
        ema_alpha: EMA decay for assignment tracking (default 0.01).
        step_size: Mirror descent step size η (default 0.01).
    """
    def __init__(self, n_experts: int, ema_alpha: float = 0.01, step_size: float = 0.01):
        super().__init__()
        self.n_experts = int(n_experts)
        self.ema_alpha = float(ema_alpha)
        self.step_size = float(step_size)
        # Buffers (non-parameter, but on the right device).
        self.register_buffer("f", torch.full((n_experts,), 1.0 / n_experts))
        self.register_buffer("b", torch.zeros(n_experts))
    
    @torch.no_grad()
    def update(self, assignments: torch.Tensor) -> None:
        """Update EMA from per-batch hard assignment indicators.
        
        Args:
            assignments: [B, K] bool/int float tensor.  1.0 = expert was 
                         selected for this batch element, 0.0 = not.
        """
        # Per-expert fraction in the batch.
        f_batch = assignments.float().mean(dim=0)  # [K]
        # EMA update.
        self.f.mul_(1.0 - self.ema_alpha).add_(self.ema_alpha * f_batch)
        # Mirror descent step on φ(f) = -Σ f_k log f_k.
        # The gradient is -log f_k - 1, mirror step is +log f_k.
        # We add the bias so high-f gets demoted and low-f gets promoted.
        # bias = -step_size * grad_phi = -step_size * (-log f - 1) ≈ step_size * log f
        # Convention: we add b to logits, so positive b promotes.
        # We want high f → negative b → b = -log f (or any monotone-decreasing fn).
        self.b.copy_(-self.step_size * torch.log(self.f.clamp_min(1e-8)))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add the bias to the router logits.
        
        Args:
            logits: [B, K] raw router logits.
        Returns:
            [B, K] biased logits, ready for top-K mask + softmax.
        """
        return logits + self.b.unsqueeze(0)  # broadcast over batch
```

### 5.2 Wire into `FAMECfCCell`

Add an optional `phi_balance` argument to `FAMECfCCell.__init__`:

```python
def __init__(self, ..., phi_balance: bool = False, ema_alpha: float = 0.01, step_size: float = 0.01):
    ...
    self.phi_balance = bool(phi_balance)
    if self.phi_balance:
        self.balancer = PhiBalancer(self.n_experts, ema_alpha=ema_alpha, step_size=step_size)
    else:
        self.balancer = None
```

In `forward_with_aux`, after computing `logits` (or just before the top-K
mask), if `self.balancer is not None` and `self.training`, add the bias and
update the EMA based on the post-top-K `last_top_idx`:

```python
g = self.router(x_t, h)  # [B, K], with K' nonzeros
self.last_g = g.detach()
self.last_top_idx = self.router.last_top_idx.detach()
# NEW: φ-balancing bias and EMA update.
if self.balancer is not None and self.training:
    self.balancer.update(self.last_top_idx)  # [B, top_k] long
    # Bias is already applied inside the router (or we apply here).
```

A cleaner design: **the router holds the bias** (not the cell). Update
`ForecastabilityRouter` to accept an optional `PhiBalancer` and apply the
bias inside `router.forward()`. This keeps the bias update co-located with
the routing decision.

### 5.3 Synergy with orthogonality (PRD #10-37)

The two losses target **different failure modes**:
- **φ-balancing** targets **router collapse** (one expert dominates top-K)
- **orthogonality** targets **expert representation collapse** (experts
  learn similar functions even if they're chosen evenly)

Combined: `total = task + α * orth + β * phi`, where β is the φ-bias
step_size (already inside the balancer, no extra loss term needed because
the bias is a no_grad mirror-descent step — exactly the φ-Balancing paper's
design). The only hyperparameter added is `phi_balance: bool` (and the
optional `ema_alpha`, `step_size`).

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/phi_balancing.py` | **NEW** | ~50 |
| `lnn/core/forecastability_router.py` | MODIFY: optional `balancer` arg, bias in forward, EMA update | ~+20 |
| `lnn/core/fame_cfc.py` | MODIFY: optional `phi_balance` arg wired through to router | ~+5 |
| `lnn/core/__init__.py` | MODIFY: export `PhiBalancer` | ~+1 |
| `tests/test_phi_balancing.py` | **NEW** | ~120 |
| `scripts/bench_phi_balancing.py` | **NEW** | ~150 |
| `docs/research/2026-06-14_phi_balancing_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-14_LNN_research_summary_v7.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add φ-balancing example | ~+25 |

**Net**: 3 new + 3 modified = 6 files, ~520 lines (modest PR).

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| EMA updates during eval pollute state | M | Guard update with `self.training` check; freeze in eval |
| Bias diverges on small data | L | `clamp_min(1e-8)` on f, `clamp` on b |
| Step size η too aggressive | M | Default η=0.01 (small); smoke-bench sweep 0.001, 0.01, 0.1 |
| φ choice wrong (we use -Σf log f, not the paper's exact φ) | M | Neg-entropy is the simplest strictly-convex symmetric potential; matches the paper's intuition. If smoke-bench underperforms, swap. |
| Backward compat breaks `FAMECfCCell` signature | L | `phi_balance=False` default; existing tests must pass unchanged |

## 8. Verification Plan

1. **Unit tests** (`tests/test_phi_balancing.py`):
   - `PhiBalancer.update` is in-place, no grad
   - Bias is zero on init, non-zero after first update
   - `forward(logits) == logits + b` broadcasts correctly
   - `phi_balance=False` ⇒ no balancer, no bias, no EMA update
   - `phi_balance=True` + `model.eval()` ⇒ bias frozen across batches
   - Synergy: bias + orthogonality co-exist in `forward_with_aux`

2. **Smoke bench** (`scripts/bench_phi_balancing.py`):
   - K=3 top_k=1 toy sin, 25 epochs, 3 seeds
   - Conditions: `λ=0/η=0` (round 79 baseline), `λ=0.001/η=0` (orth-only),
     `λ=0/η=0.01` (φ-only), `λ=0.001/η=0.01` (both)
   - Report task loss mean ± std, diverged seed count, expert utilization

3. **Regression**: `pytest tests/test_fame_cfc.py tests/test_orthogonality.py` all green

## 9. Rollout

Single PR.  After landing:
- `lnn/core/phi_balancing.py` is the canonical reference impl
- `phi_balance=True` is the recommended K=3 top_k=1 default
- README adds a "φ-balancing" section analogous to "orthogonality"

## 10. Why this is the right next step (not #10-7 LFM2.5 INT8)

- **Synergy**: φ-balancing + orthogonality is the natural pair (defense in
  depth on routing + representation)
- **Round 80 just shipped orthogonality** — applying φ-balancing on top is
  the immediate complementary move
- **φ-Balancing paper is 30 days old (2026-05-14)** — high novelty, no
  widely-available open-source LNN-flavored implementation
- **Toy sin/cos reproduction is fast** (1 PR cycle, ~3-4h)
- LFM2.5 INT8 deployment (PRD #10-7) is downstream and requires the
  full orthogonality+φ-balancing stack to be the deployment default

## 11. Open Questions (to resolve in implementation)

- **Q1**: Should the bias be applied **before** the top-K mask (so bias can
  demote the argmax) or **after** (so top-K indices are decided by raw
  logits, then bias rescales their softmax weights)?
  - **A1**: **Before** — that's the whole point of φ-balancing (demote the
    over-used expert before deciding who wins).
- **Q2**: Should the EMA be per-expert-per-layer or shared across layers?
  - **A2**: **Per-expert-per-layer** (each layer's router gets its own
    PhiBalancer) — matches DBES findings that expert specialization is
    layer-dependent (arXiv:2605.18498).
- **Q3**: Should we apply φ-balancing to `MRMoECfCCell` (dense softmax)?
  - **A3**: **Out of scope for this PRD** — dense softmax is less prone to
    collapse; future round can add it.
