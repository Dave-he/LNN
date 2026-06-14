---
prd: 10-41
title: "CosineRouter: Parameter-Free Geometric-Coupling Routing for FAMECfC"
date: 2026-06-14
status: draft
round: 82
authors: heyongxian
depends_on:
  - PRD #10-36 (FAME top-K router, round 78)
  - PRD #10-37 (orthogonality constraint, round 80)
  - PRD #10-40 (φ-balancing, round 81)
references:
  - arXiv:2605.12476  # Geometric Coupling — "Routers Learn the Geometry of Their Experts" (direct template)
  - arXiv:2604.09780  # Myth of Expert Specialization — "load-balancing loss suppresses shared hidden state directions" (theoretical motivation)
  - arXiv:2606.10703  # Causal Audit: observational ≠ causal
related:
  - arXiv:2606.03631  # AnchorMoE: orthogonality constraint (round 80 template)
  - arXiv:2605.15403  # φ-Balancing: bias-additive routing (round 81 template)
---

# PRD #10-41 — CosineRouter: Parameter-Free Geometric-Coupling Routing for FAMECfC

## 0. One-liner

Replace the learned linear/MLP router in `FAMECfCCell` with a **parameter-free
cosine-similarity router** that, for each expert, maintains a running
EMA of the hidden states routed to it and assigns tokens based on cosine
similarity.  This is the minimum-viable LNN-flavored implementation of
the "parameter-free online K-Means router" from arXiv:2605.12476
(*Routers Learn the Geometry of Their Experts*, Ahrac et al., 2026-05-12)
which the paper reports achieves the **lowest load imbalance** among
auxiliary-loss, loss-free, and K-Means routers, with only a modest
perplexity cost.

## 1. Problem

Round 76-81 built a 5-layer LNN+MoE stack that converges stably on toy
sin/cos.  But all three routing mechanisms to date have a **learned
linear/MLP router**:

| Round | Router | Training signal | Risk |
|---|---|---|---|
| 77 | `Linear([x;h]) → softmax` | backprop through task | gradient interference with experts |
| 78 | `Linear([x;h]) → top-K sparse` | same | gradient interference + sparsity noise |
| 81 | `Linear + bias from φ-balancer` | backprop + EMA bias | still a learned router, just with a hand-crafted bias |

arXiv:2605.12476 reveals:
- **Geometric coupling** between routers and experts: their gradients
  flow along the same input direction with different scalar coefficients
- **Auxiliary load-balancing losses break this coupling** by spreading
  input-directed gradients across router weights, making distinct router
  directions ~3× more similar
- A **parameter-free K-Means router** (cosine sim to per-expert running
  hidden-state average) achieves **lowest load imbalance** in the paper's
  1B SMoE experiments, with only a modest perplexity cost

This is a **third routing strategy** that's complementary to:
- **φ-balancing** (no_grad bias add, round 81)
- **orthogonality** (aux loss on expert outputs, round 80)

The CosineRouter **removes the learned router entirely** — no router
parameters to train, no auxiliary loss, no bias buffer.  Just cosine
similarity to a per-expert running average of routed hidden states.

## 2. Goal (Scope)

**Minimum-viable CosineRouter for `FAMECfCCell`**:
- New `lnn/core/cosine_router.py` with:
  - `CosineRouter(input_size, hidden_size, n_experts, top_k, ema_alpha=0.01)`
  - `forward(x_t, h) → g ∈ Δ^K` with `K-K'` zeros (top-K mask)
  - `update(hidden_states, top_idx) → @torch.no_grad` EMA update of per-expert mean
- New `CosineFAMECfCCell` and `CosineFAMECfCNetwork` (subclasses or factory)
  that use the parameter-free router
- **No learned router** (`router = nn.Identity()` or removed)
- **Back-compat**: existing `FAMECfCCell` (learned router) untouched

## 3. Out of Scope (Non-Goals)

- **Per-token (not per-step) EMA** — paper does per-token; we do per-step
  (one EMA update per cell.forward call).  Sufficient for toy sin.
- **Multi-step K-Means initialization** — paper uses the first batch's
  mean as init.  We use uniform init (zeros), and let EMA converge.
- **Online K-Means++ warm-up** — K-Means++ warm-up is for the
  first-batch init; we skip and rely on the EMA to find cluster centers
- **Integration with MRMoECfCCell** (dense softmax) — paper's claims
  target sparse top-K; out of scope for this round
- **Sigmoid / temperature scaling** — the paper uses raw cosine; we
  follow that

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `lnn/core/cosine_router.py` exports `CosineRouter` | TBD |
| 2 | `CosineRouter` has NO learned parameters (zero `nn.Parameter`) | TBD |
| 3 | `CosineRouter` has K buffer tensors of shape `[input+hidden]` | TBD |
| 4 | `update(hidden_states, top_idx)` runs in-place, no_grad, EMA | TBD |
| 5 | `forward(x_t, h) → top-K cosine sim → softmax` | TBD |
| 6 | `CosineFAMECfCCell` and `CosineFAMECfCNetwork` use the new router | TBD |
| 7 | 10+ unit tests in `tests/test_cosine_router.py` | TBD |
| 8 | Smoke bench: K=3 top_k=1 toy sin converges to < 0.3 | TBD |
| 9 | `pytest tests/test_fame_cfc.py tests/test_phi_balancing.py tests/test_orthogonality.py tests/test_cosine_router.py` all green | TBD |

## 5. Design

### 5.1 Module: `lnn/core/cosine_router.py`

```python
class CosineRouter(nn.Module):
    """Parameter-free top-K router via cosine similarity to per-expert EMA.
    
    Per the Geometric Coupling paper (arXiv:2605.12476), the best
    parameter-free router maintains a running average of the hidden
    states routed to each expert and assigns tokens based on cosine
    similarity.  This avoids:
    - learned router parameters (no gradient interference)
    - auxiliary load-balancing losses (paper §5: "3x more similar router directions")
    
    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: K (number of experts).
        top_k: K' ∈ [1, K] — number of experts activated per step.
        ema_alpha: EMA decay rate for per-expert mean update.
        eps: Numerical floor on cosine denominator.
    """
    
    def __init__(self, input_size, hidden_size, n_experts, top_k=2,
                 ema_alpha=0.01, eps=1e-8):
        super().__init__()
        # NO learned parameters — `self.parameters()` is empty.
        # Per-expert running mean of the COMBINED [x_t; h] state.
        # Init: zeros (paper allows random init too; we use zeros for
        # determinism).
        self.register_buffer("expert_means", torch.zeros(n_experts, input_size + hidden_size))
        # Diagnostic: last update step.
        self.register_buffer("step", torch.tensor(0))
    
    @property
    def num_parameters(self) -> int:
        return 0  # explicit
    
    @torch.no_grad()
    def update(self, combined: torch.Tensor, top_idx: torch.Tensor) -> None:
        """Update per-expert mean from the routed hidden states.
        
        Args:
            combined: [B, input+hidden] — the [x_t; h] used for routing.
            top_idx: [B, top_k] long — which expert each batch element was routed to.
        """
        K = self.expert_means.shape[0]
        for k in range(K):
            mask = (top_idx == k).any(dim=-1)  # [B] — batch elements routed to k
            if mask.any():
                # Mean of routed states for expert k.
                routed = combined[mask]  # [n_routed, input+hidden]
                m_batch = routed.mean(dim=0)
                self.expert_means[k].mul_(1 - self.ema_alpha).add_(self.ema_alpha * m_batch)
        self.step += 1
    
    def forward(self, x_t, h):
        combined = torch.cat([x_t, h], dim=-1)  # [B, input+hidden]
        # Cosine similarity to each expert's mean.
        # expert_means: [K, D], combined: [B, D]
        # Normalize:
        means_norm = F.normalize(self.expert_means, dim=-1, eps=1e-8)  # [K, D]
        comb_norm = F.normalize(combined, dim=-1, eps=1e-8)            # [B, D]
        sim = comb_norm @ means_norm.t()  # [B, K] — cosine sims
        # Top-K mask (same contract as ForecastabilityRouter).
        if self.top_k == self.n_experts:
            g = F.softmax(sim, dim=-1)
            self.last_top_idx = torch.arange(self.n_experts, device=sim.device).expand(sim.size(0), -1)
        else:
            top_result = sim.topk(self.top_k, dim=-1)
            top_idx = top_result.indices
            del top_result
            mask = torch.full_like(sim, float("-inf"))
            mask.scatter_(-1, top_idx, 0.0)
            g = F.softmax(sim + mask, dim=-1)
            self.last_top_idx = top_idx
        return g
```

### 5.2 Wire into `FAMECfCCell`

Add a `router_type: str = "learned"` argument to `FAMECfCCell.__init__`:
- `"learned"` (default, back-compat): use `ForecastabilityRouter`
- `"cosine"`: use `CosineRouter`, and the cell wires `update` to the
  EMA after `forward_with_aux`

```python
def __init__(self, ..., router_type: str = "learned", ema_alpha: float = 0.01, ...):
    if router_type == "learned":
        self.router = ForecastabilityRouter(...)
    elif router_type == "cosine":
        self.router = CosineRouter(input_size, hidden_size, n_experts, top_k, ema_alpha)
    else:
        raise ValueError(f"unknown router_type: {router_type}")
    self.router_type = router_type
```

In `forward_with_aux`, after computing `g` and `last_top_idx`, if
`router_type == "cosine"` and `self.training`, call
`self.router.update(combined, last_top_idx)`.

### 5.3 Synergy with φ-balancing and orthogonality

The three interventions are **stackable** in principle, but the
parameter-free CosineRouter is **mutually exclusive with the learned
router** (you can't have both compute the routing).  So the design is:

- **`router_type="learned"` (default)** → can stack with orth and φ
- **`router_type="cosine"`** → standalone alternative; orthogonality can
  still be added as aux loss on expert hidden states; φ-balancing
  doesn't apply (no learned logits to bias)

This gives a clean 3-way exploration: learned+orth+φ vs learned+orth
vs cosine (+ optional orth).

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/cosine_router.py` | **NEW** | ~80 |
| `lnn/core/fame_cfc.py` | MODIFY: `router_type` arg wired to either router | ~+15 |
| `lnn/core/__init__.py` | MODIFY: export `CosineRouter` | ~+1 |
| `tests/test_cosine_router.py` | **NEW** | ~120 |
| `scripts/bench_cosine_router.py` | **NEW** | ~120 |
| `docs/research/2026-06-14_cosine_router_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-14_LNN_research_summary_v8.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add CosineRouter example | ~+20 |

**Net**: 3 new + 3 modified = 6 files, ~530 lines (modest PR).

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Zero init of expert_means → all cosine sims = 0 in first step | M | Use small random init (paper: "random init of cluster centers" allowed) — but we keep zeros for determinism; first batch will have all-equal probs which is fine |
| EMA converges slowly with small batches | L | Default ema_alpha=0.01 (paper's value); can sweep |
| K-Means on hidden states needs many tokens to converge | L | For toy sin with T=32 and N=64, EMA gets ~2000 tokens per epoch |
| Three-way comparison (learned / learned+orth / cosine) is hard to interpret | M | Use the same K=3 top_k=1 cell as round 80-81 baseline for direct A/B |
| Backward compat breaks `FAMECfCCell` signature | L | `router_type="learned"` default; existing tests must pass unchanged |

## 8. Verification Plan

1. **Unit tests** (`tests/test_cosine_router.py`):
   - `CosineRouter` has zero `nn.Parameter`
   - Buffers move with `.to(device)`
   - `update` is in-place, no_grad
   - `forward` returns sparse top-K mixture with correct shape
   - Init: `expert_means` is all zeros → all cosine sims = 0 → uniform g
   - After warm-up with K=3 different clusters → top-1 picks the closest cluster

2. **Smoke bench** (`scripts/bench_cosine_router.py`):
   - K=3 top_k=1 toy sin, 25 epochs, 3 seeds
   - Conditions:
     - `learned` (round 78 baseline, no φ, no orth)
     - `learned + orth (λ=0.001)` (round 80)
     - `learned + φ (η=0.05)` (round 81)
     - `cosine` (new, round 82)
     - `cosine + orth (λ=0.001)` (new, round 82)
   - Report task loss mean ± std, diverged seeds

3. **Regression**: `pytest tests/test_fame_cfc.py tests/test_phi_balancing.py tests/test_orthogonality.py` all green

## 9. Rollout

Single PR.  After landing:
- `lnn/core/cosine_router.py` is the canonical reference impl
- `router_type="cosine"` is an alternative to `"learned"`
- README adds a "CosineRouter" section

## 10. Why this is the right next step (not other candidates)

- **Theoretical depth**: arXiv:2605.12476 provides a *mechanistic
  explanation* of why orthogonality (round 80) and φ-balancing (round 81)
  work — and why learned routers with aux losses are *worse*
- **Empirical claim**: paper reports cosine router has **lowest load
  imbalance** of all three methods tested (aux loss, loss-free, cosine)
  with only modest perplexity cost
- **Conceptual variety**: 3rd routing mechanism (learned+orth / learned+φ
  / cosine) gives a more complete picture of the design space
- **Composes with orth**: cosine router can still use orthogonality
  aux loss on expert hidden states (no conflict, since orth is on
  experts, not on the router)
- **Fits the 3-4h PR cycle**: implementation is ~80 lines, tests ~120,
  bench ~120

Other candidates rejected:
- **#10-7 LFM2.5-1.2B INT8** (downstream, needs full stack to be deployment default)
- **Full 16-cell orth sweep** (round 79 already swept; can do later)
- **SNBC production** (downstream, real-data validation)

## 11. Open Questions (to resolve in implementation)

- **Q1**: Should `expert_means` be init to zeros or small random?
  - **A1**: **Zeros** (deterministic, first batch makes them all equal → uniform softmax, then EMA differentiates).  Test will verify init = zeros.
- **Q2**: Should we update per-token (per [B, D] element) or per-batch-mean?
  - **A2**: **Per-batch-mean per expert** (the paper does this for efficiency).  The "mean of routed states for expert k in this batch" is one EMA step.
- **Q3**: Does the cosine router need a learnable temperature?
  - **A3**: **No** — paper uses raw cosine.  Temperature is a future round.
