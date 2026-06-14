> **兄弟项目**:[Dave-he/RoboticsResearch](https://github.com/Dave-he/RoboticsResearch) —— 通用机器人技术的同模式活知识库(本项目的范式扩展)

# LNN

PyTorch implementations, benchmarks, and research logs for Liquid Neural Networks.

`Dave-he/LNN` is a code-first research project. It contains a reusable Python
package under `lnn/` plus an auditable research archive under `docs/` and
`analysis/`. The package includes CfC, LTC, liquid neuron layers, continuous-time
variants, graph models, physics-informed models, multimodal models, and
noise-adaptive CfC backbones. The research archive records paper tracking,
ablation history, Jetson checks, and EMMA rover benchmark results.

Current package status: `0.1.0`. Core sequence models are covered by tests and
are intended for reuse. Timestamped research reports and analysis outputs are
evidence trails, not stable APIs.

## Install

```bash
git clone https://github.com/Dave-he/LNN.git
cd LNN

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
python -m pytest tests -q -m "not large_budget"
```

Run the large-budget EMMA rover tests explicitly when you want the slower
regime checks:

```bash
python -m pytest tests -q -m large_budget
```

## Minimal API Example

```python
import torch
from lnn import CfCNetwork

batch, steps, features = 8, 32, 3
x = torch.randn(batch, steps, features)
dt = torch.full((batch, steps, 1), 0.5)
mask = torch.ones(batch, steps, features)
mask[:, 10:12, :] = 0.0

model = CfCNetwork(
    input_size=features,
    hidden_size=32,
    output_size=1,
    return_sequences=False,
)

y = model(x, dt=dt, mask=mask)
print(y.shape)  # torch.Size([8, 1])
```

## Multi-Time-Scale CfC (n_tau ≥ 2)

`CfCCell` and `CfCNetwork` accept an optional `n_tau: int = 1` argument.  When
`n_tau == 1` (default) the cell is numerically equivalent to the legacy
single-τ path.  Setting `n_tau > 1` splits the hidden state into K independent
time-scale groups, each with its own τ, f_gate, g_branch, h_branch.  This is
the minimum-variance extension that aligns with the multi-τ pattern observed
in arXiv:2606.12240 (MR-MoE), arXiv:2606.11162 (COGENT),
arXiv:2606.07670 (Liquid-3DGS), and arXiv:2604.18274 (LiquidTAD).

```python
from lnn import CfCNetwork

# Three time-scale groups: τ ∈ {0.1, 1.0, 10.0}.
model = CfCNetwork(
    input_size=3,
    hidden_size=24,
    output_size=1,
    n_tau=3,
    tau_scales=(0.1, 1.0, 10.0),  # per-branch initial τ
)
y = model(x)  # hidden dim is split evenly across the 3 branches
```

Smoke-bench on toy sin/cos: `n_tau=3` reaches final MSE 0.0463 vs
`n_tau=1` 0.0535 (-13.4%, std 49% tighter) — see
`docs/research/2026-06-14_cfc_n_tau_sweep_report.md` and the unit
tests in `tests/test_cfc_n_tau.py`.

## Multi-Expert MR-MoE (n_experts ≥ 2)

`MRMoECfCCell` and `MRMoECfCNetwork` (PRD #10-24, round 77) wrap
K independent `CfCCell` experts behind a softmax router that produces
per-step mixture weights `g ∈ Δ^K`.  The cell output is
`Σ_k g_k · expert_k(x_t, h_prev)`.  This is the minimum-viable
implementation of the Multi-Rate Mixture-of-Experts pattern from
arXiv:2606.12240 (Zong et al., 2026) and the pattern-routed
heterogeneous-experts idea from arXiv:2606.13024 (CausalMoE, 2026-06-11).

```python
from lnn import MRMoECfCNetwork

# 3 CfC experts, each with n_tau=1 (combine with round 76 n_tau for
# multi-rate inside each expert: set n_tau_per_expert=3 for 3*3=9 effective τ groups).
model = MRMoECfCNetwork(
    input_size=3,
    hidden_size=24,
    output_size=1,
    n_experts=3,
    n_tau_per_expert=1,             # reuse round 76 n_tau per expert
    router_hidden=0,                # 0=linear router, >0=2-layer MLP
)
y = model(x)                       # [B, T, 1]
# Inspect router weights after a forward pass:
g = model.cells[0].last_g          # [B, K] softmax weights
```

Smoke-bench on toy sin/cos: `K=3` reaches final MSE 0.0364 vs
`K=1` 0.0525 (-30.7%, 2.3× the n_tau gain from round 76).  Router
entropy stays at ≈ log K after 30 epochs (no expert collapse).
See `docs/research/2026-06-14_mr_moe_cfc_sweep_report.md` and the
unit tests in `tests/test_mr_moe_cfc.py`.

## FAME-style Top-K Sparse MoE Routing

`ForecastabilityRouter` + `FAMECfCCell` + `FAMECfCNetwork` (PRD
#10-36, round 78) wrap the round 77 K-expert pool behind a **sparse
top-K router** that activates only `K'` of `K` experts per step.
This is the minimum-viable implementation of the
"cost-aware sparse router" from arXiv:2606.08896 (FAME, 2026-06-08)
— a production-deployed forecastability-aware MoE for retail
vending-machine sales (5000+ machines, 60M+ transactions, Top-2
routing −12.4% MSE vs LightGBM).

```python
from lnn import FAMECfCNetwork

# 3 CfC experts, top-2 sparse routing (FAME paper default).
# top_k=1 → argmax single expert; top_k=K → dense softmax (round 77).
model = FAMECfCNetwork(
    input_size=3,
    hidden_size=24,
    output_size=1,
    n_experts=3,
    top_k=2,                          # K' ∈ [1, K]
    n_tau_per_expert=1,               # round 76 n_tau per expert
)
y = model(x)                         # [B, T, 1]
# Inspect activated experts per step:
g = model.cells[0].last_g            # [B, K], exactly top_k nonzeros per row
idx = model.cells[0].last_top_idx    # [B, top_k] indices of activated experts
```

Smoke-bench on toy sin/cos: `top_k=2` reaches final MSE 0.0366
(≈ equal to `top_k=3` dense at 0.0364) but with **3.7× tighter
std** (0.0012 vs 0.0034) — the FAME paper's core "sparse is
more stable" claim, reproduced on a 1-seed-runner homely toy
setup.  Activated-experts-per-step is exactly `top_k` (sparsity
contract honoured).  See
`docs/research/2026-06-14_fame_cfc_sweep_report.md` and the unit
tests in `tests/test_fame_cfc.py`.

## Orthogonality Constraint (defence against expert collapse)

`orthogonality_loss` (PRD #10-37, round 80) implements the
"geometric orthogonality constraint that penalises representational
redundancy" idea from arXiv:2606.03631 (AnchorMoE, 2026-06-02).
Use it as an auxiliary loss to keep expert hidden states
decorrelated — the round 79 sweep showed that the K=3 top_k=1
(router-argmax single expert) cell is **unstable** without
orthogonality (0.76 ± 0.79 with 1/3 seeds diverging) but
**stable** with `lambda_coeff=0.001` (0.11 ± 0.05, 0/3 seeds
diverging).  This addresses the Causal Audit (arXiv:2606.10703)
warning that observational routing metrics do not predict expert
causal importance — the orthogonality constraint directly
intervenes in the expert representation space.

```python
from lnn import FAMECfCNetwork, orthogonality_loss

net = FAMECfCNetwork(input_size=3, hidden_size=24, output_size=1,
                    n_experts=3, top_k=1)  # unstable without orth
opt = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
for x, y in dataloader:
    opt.zero_grad()
    y_pred, expert_outs = net.forward_with_aux(x)
    task_loss = loss_fn(y_pred, y)
    # Use only the last step's expert outputs from the first layer.
    last_outs = expert_outs[0][-1]            # K × [B, H]
    aux = orthogonality_loss(last_outs, lambda_coeff=0.001)
    (task_loss + aux).backward()
    opt.step()
```

`net.forward(...)` (the back-compat path) still returns only the
prediction tensor; the orthogonality-aware `net.forward_with_aux(...)`
returns the nested `[num_layers][T][K]` expert outputs needed to
compute the auxiliary loss.  See
`docs/research/2026-06-14_orthogonality_report.md` and the unit
tests in `tests/test_orthogonality.py`.

## CosineRouter (parameter-free geometric-coupling routing, optional)

`CosineRouter` (PRD #10-41, round 82) implements the parameter-free
online K-Means router from arXiv:2605.12476 (Routers Learn the
Geometry of Their Experts, 2026-05-12).  It maintains per-expert
running hidden-state means and assigns tokens by cosine similarity.
Unlike `ForecastabilityRouter`, it has **zero learned parameters**.

```python
from lnn import FAMECfCNetwork

# Switch to the parameter-free cosine router.  No learned router,
# no φ-balancing, just cosine similarity to per-expert means.
net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,
    router_type="cosine", ema_alpha=0.05,
)
y = net(x)  # forward and forward_with_aux unchanged
```

**Honest negative result on toy K=3 top_k=1** (round 82 smoke-bench):
the cosine router alone reaches 0.96 ± 0.35 with 3/3 seeds diverging
(worse than the learned-router baseline 0.76 ± 0.79 with 1/3 diverging).
Root cause: zero-init expert means → uniform softmax → EMA needs
many consistent routing events to learn cluster centers, which a tiny
toy sin problem (3 experts, 64 samples × 32 steps) cannot provide.
The paper's claim ("lowest load imbalance") is from a **1B SMoE** with
millions of tokens per expert — the cosine router is **scale-dependent**.
For toy / small-data problems, keep `router_type="learned"` (default)
and combine with `phi_balance=True` + `orthogonality_loss` instead.
See `docs/research/2026-06-14_cosine_router_report.md` and
`tests/test_cosine_router.py`.

## φ-Balancing (EMA-based expert load balancing)

`PhiBalancer` (PRD #10-40, round 81) implements the
EMA-based mirror-descent bias from arXiv:2605.15403 (φ-Balancing,
2026-05-14).  Unlike the orthogonality constraint (which is an
auxiliary loss on hidden states), φ-balancing is a **no-grad bias
added to the router logits** that demotes over-used experts and
promotes under-used ones.  The two are complementary: orthogonality
defends against **representation collapse**; φ-balancing defends
against **routing collapse**.  Together with the round 80
orthogonality constraint, they form the round 81 "LNN+MoE
defence-in-depth" stack.

```python
from lnn import FAMECfCNetwork

# Enable φ-balancing: per-layer balancer, EMA decay 0.05,
# mirror-descent step size 0.05.
net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,                      # the unstable cell
    phi_balance=True, ema_alpha=0.05, phi_step_size=0.05,
)
# Same forward() and forward_with_aux() — no extra args needed.
# The balancer is updated in train mode and frozen in eval mode.
opt = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
for x, y in dataloader:
    net.train()  # balancer updates only in train mode
    opt.zero_grad()
    y_pred, _ = net.forward_with_aux(x)  # or net.forward(x) for back-compat
    task_loss = loss_fn(y_pred, y)
    task_loss.backward()
    opt.step()

# Eval mode: balancer is frozen.
net.eval()
with torch.no_grad():
    y_pred = net(x)  # bias is applied but not updated
```

Smoke-bench on K=3 top_k=1 toy sin: φ-balancing alone reaches
final MSE 0.125 (vs 0.76 baseline, **-83.5%**), comparable to
orthogonality alone (0.11).  Setting `phi_balance=False` (default)
gives the round 80 behaviour — fully back-compatible.  See
`docs/research/2026-06-14_phi_balancing_report.md` and the unit
tests in `tests/test_phi_balancing.py`.

## MoE Ecology Diagnostic (E = T·H/(O+B), optional)

`moe_ecology_number` and `MoEEcologyMonitor` (PRD #10-42, round 83)
implement the **dimensionless control parameter** for MoE health from
arXiv:2605.06415 (Zhang, 2026).  The paper's central claim:

> E = T·H / (O + B) ≥ 0.5  ⇒  zero dead experts, no aux loss needed.

where T = routing temperature, H = routing entropy weight, O = oracle
weight, B = balance (load-balancing aux loss) weight.  This is the
**first theoretical diagnostic** for our LNN+MoE stack — used to
monitor whether a cell is in a healthy ecology regime.

```python
from lnn import FAMECfCCell, MoEEcologyMonitor
import torch

cell = FAMECfCCell(input_size=3, hidden_size=24, n_experts=3, top_k=1)
monitor = MoEEcologyMonitor(n_experts=3, ema_alpha=0.01)

# Run training step.
cell.train()
x_t = torch.randn(8, 3)
h = torch.randn(8, 24)
h_new = cell(x_t, h, dt=1.0)

# Step the monitor with the cell's last-g (mixture weights) and
# the active aux-loss weight (orth λ or φ η or 0).
info = monitor.step(cell.last_g, T=1.0, B=0.001)
# info == {"E": 213.4, "dead_experts": 0, "utilization": [0.27, 0.35, 0.38]}

# Or one-shot diagnostic straight from the cell:
diag = cell.moe_ecology_diagnostic(B=0.001)  # pass lambda_coeff or phi_step_size
# diag == {"E": 213.4, "dead_experts": 0, "utilization": [...]}
```

**Round 83 smoke-bench** (16-cell FAME grid + ortho toxicity test):

- **12/12 toy 16-cell grid configs** have **dead=0** — paper's
  E ≥ 0.5 ⇒ no dead experts claim **reproduced** on our FAME stack.
- **K=3 top_k=2 n_tau=2 wins** (0.538) — round 79 K=5 dense finding
  replicated at K=3 with longer time-scale.
- **Ortho toxicity confirmed at high λ** on all 3 synthetic datasets
  (toy sin / random / structured): λ=1.0 hurts loss by 4.7% to 16.4%
  vs λ=0.  Round 80's default **λ=0.001 is safe** (loss change
  < 0.1% vs λ=0 on all 3 datasets).
- **E scales as 1/λ** (E = 1/(B+eps) in our no-T setting): when
  λ=1.0, E drops to 0.34-0.96, **crossing the paper's 0.5 threshold**
  for toy_sin and structured.

The diagnostic is **purely additive** — `FAMECfCCell.forward(...)` and
`FAMECfCCell.forward_with_aux(...)` are unchanged.  See
`docs/research/2026-06-14_moe_ecology_report.md` and the unit tests in
`tests/test_moe_ecology.py`.

## Ecology-Gated φ-Balancing (auto-enable φ when E < 0.5, optional)

`EcologyGatedBalancer` (PRD #10-43, round 84) **closes the loop on
the MoE ecology diagnostic**: when the live E drops below a
configurable threshold, the cell **automatically enables φ-balancing**
on the affected cell.  This converts the diagnostic from a passive
monitor into an **autonomous cell-health manager** that decides
*when* to intervene, not just *whether* to.

```python
from lnn import FAMECfCNetwork

# Auto-enable φ-balancing when E drops below 0.5 (paper's threshold).
net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,                      # the unstable cell
    ecology_gated_balancing=True,
    ecology_E_min=0.5,                         # paper's threshold
    ecology_warmup_steps=0,                    # no warmup
    phi_step_size=0.05, ema_alpha=0.05,
)
# No need to set phi_balance=True — the gate attaches it on first
# E<0.5 step (in training mode only).  At eval time, the gate is
# silent and the diagnostic just reports E.
y_pred = net(x)
```

**Gate semantics** (no hysteresis for round 84):

- Never fires when E > `ecology_E_min` (no false positives).
- Fires exactly once when E first drops below threshold.
- Stays fired after that (disabling mid-training would re-collapse
  routing, so we err on "intervene early, stay").
- Respects `ecology_warmup_steps` (don't intervene in the first N
  steps even if E < threshold; router needs time to settle).
- In training mode, auto-attaches a `PhiBalancer` on first fire.
- In eval mode, the gate runs but does NOT mutate the cell.

**Round 84 smoke-bench** (3 cells × 3 datasets × orth λ=1.0 to
force E < 0.5):

- **Gate fires correctly** in toy_sin (E=0) and structured (E=0)
  at step 16; auto-attaches a `PhiBalancer`.
- **No false positive** in random (E=0.96): gate correctly silent.
- **Honest negative**: gated φ does not recover from **λ=1.0
  ortho-toxicity** (paper finding #2) — both always-φ and gated-φ
  fail.  The orth loss is too strong for φ to counteract; a stronger
  intervention (e.g., auto-disable orth) is a follow-up.

The gate is **purely additive** — `ecology_gated_balancing=False`
(default) is fully back-compat with rounds 78-83.  See
`docs/research/2026-06-14_ecology_gated_balancing_report.md` and
the unit tests in `tests/test_ecology_gated_balancing.py`.

## Ecology-Gated Orth Rescaling (auto-reduce λ when E<0.5, optional)

`EcologyGatedOrth` (PRD #10-44, round 85) is a **stronger intervention**
than round 84's φ gate.  When E < 0.5, it **rescales the user's
orth loss weight λ down to a safe value** (default `0.001`, the
round 80 default).  This is the **direct fix for round 84's honest
negative**: at λ=1.0, the round 84 gated φ did not recover (gave
0.7302 ≈ baseline), but **the round 85 gated orth recovers to 0.6285**
— a **-14% loss reduction** at λ=1.0 and **-55%** at λ=10.0.

```python
from lnn import FAMECfCNetwork

# Auto-rescale λ → 0.001 when E drops below 0.5.
net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,                      # the unstable cell
    ecology_gated_orth=True,                   # opt in
    ecology_orth_lambda_safe=0.001,            # target effective λ
)
y_pred, expert_outs = net.forward_with_aux(x)

# Use cell.compute_orth_loss() instead of orthogonality_loss()
# to apply the gate transparently:
from lnn.core import FAMECfCCell
aux = cell.compute_orth_loss(expert_outs[-1][-1], user_lambda=1.0)
# If gate has fired, this returns orthogonality_loss(outs, 0.001).
# Otherwise, returns orthogonality_loss(outs, 1.0).
```

**Why orth rescaling is stronger than φ**: at high λ, the orth loss
has gradient magnitude ~λ, which dominates the task loss.  A soft
router bias (round 84 φ, η=0.05) cannot counteract a gradient that's
20× larger.  Rescaling λ down to 0.001 **removes the aux loss
domination**, letting the task loss drive the routing.  This is
**attacking the root cause** rather than the symptom.

**Round 85 smoke-bench** (2 conditions × 3 datasets × 3 λ ∈ {0.1, 1.0, 10.0}):

| λ | Dataset | A baseline | B gated | Δ | Gate fired |
|---:|---|---:|---:|---:|---|
| 1.0 | toy_sin | 0.7302 | **0.6285** | **-14.0%** | True (λ_scale=0.001) |
| 1.0 | structured | 2.8953 | **2.7637** | **-4.6%** | True (λ_scale=0.001) |
| 1.0 | random | 0.9420 | 0.9420 | 0.0% | **False (no false pos)** |
| 10.0 | toy_sin | 1.3804 | **0.6285** | **-54.5%** | True (λ_scale=0.0001) |
| 10.0 | random | 1.3110 | **0.8931** | **-31.9%** | True (λ_scale=0.0001) |
| 10.0 | structured | 3.6791 | **2.7637** | **-24.9%** | True (λ_scale=0.0001) |

**Honest-negative risk**: users who **deliberately** want high λ for
representation diversity will see their λ silently downgraded.  This
is a real risk for downstream-ensembling use cases.  Mitigated by
opt-in (`ecology_gated_orth=False` by default) and a configurable
`ecology_E_min` (set very low to effectively disable).

The gate is **purely additive** — `ecology_gated_orth=False`
(default) makes `compute_orth_loss` identical to
`orthogonality_loss`.  See
`docs/research/2026-06-15_ecology_gated_orth_report.md` and the unit
tests in `tests/test_ecology_gated_orth.py`.

## Combined Ecology Gates (2-axis policy: φ + orth co-active, recommended)

`CombinedEcologyGate` (PRD #10-48, round 86) is the **2-axis adaptive
policy** that closes the LNN+MoE stack.  When `ecology_combined` is
on, **both** the round 84 φ gate (soft) and the round 85 orth gate
(strong) fire co-actively when E < 0.5.

```python
from lnn import FAMECfCNetwork

# One-line opt-in for the full 2-axis adaptive policy.
net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,                      # the unstable cell
    ecology_combined=True,                    # opt in (recommended)
    ecology_orth_lambda_safe=0.001,           # target effective λ
)
```

**Why combined is a safe superset** (verified in 4 conditions × 3
datasets × 3 λ = 36 cells):

| Hypothesis | Verdict |
|---|---|
| H1: combined ≤ min(φ, orth) (combined best) | Partial — never worse, never strictly better in 9/9 cells |
| H2: combined ≈ orth (orth dominates) | **Confirmed** — D = C in 8/9 cells |
| H3: combined > orth (φ adds noise) | **Rejected** — combined never degrades |

The strong intervention (orth rescale) is **dominant** in our toy
bench.  The soft intervention (φ balancer) is **redundant but
harmless** — adding it doesn't hurt, doesn't help either.  The
combined gate is a **safe superset**: opt in for maximum safety,
opt out for minimum overhead.

**Use case recommendation**:
- **Maximum safety (deployment)**: `ecology_combined=True`
- **Minimum overhead (research / quick iteration)**: `ecology_gated_orth=True`
- **Soft intervention only (lightweight)**: `ecology_gated_balancing=True`

The gate is **purely additive** — `ecology_combined=False` (default)
is fully back-compat.  When `ecology_combined=True`, the orchestrator
reuses the **same sub-gate instances** as the cell attributes, so
the diagnostic state stays consistent.  See
`docs/research/2026-06-15_combined_gates_report.md` and the unit
tests in `tests/test_combined_gates.py`.

## Gradient-based H (causal MoE ecology E, opt-in)

`gradient_routing_sensitivity` (PRD #10-49, round 87) is the
**causal counterpart** to the empirical routing entropy H used
since round 83.  Where empirical H asks "how uniform does the
routing look?", gradient H asks "how sensitive is the loss to
changes in the routing?".

```python
from lnn import FAMECfCNetwork
from lnn.core import moe_ecology_number

net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,
    ecology_H_mode="gradient",  # opt in to causal H
)
# ... train loop ...
task_loss = ...  # scalar, with grad
E = moe_ecology_number(
    router_logits=cell.last_g, last_g=cell.last_g,
    B=0.001, H_mode="gradient", task_loss=task_loss,
)
```

**Honest-negative headline (round 87)**: in our 3-dataset × 3-λ
bench, **E_emp ≈ E_grad** in all 9 cells (mean |Δ| < 0.05),
and gate-firing decisions are **identical in 9/9 cells**.  The
empirical H is **sufficient** for gate firing in the toy regime.

**Where gradient H may matter** (out of scope for round 87):
1. **Vision / NLP data** — real-world distributions have
   loss-flat routing pathologies that empirical H can't see
2. **Larger K** (K=8, K=16) — more experts = more opportunities
   for functionally-identical experts to look different
3. **Longer training** — 2 epochs may be too short for the
   routing distribution to fully explore the loss landscape
4. **Self-supervised pre-training** — task loss can be
   trivially low even with collapsed routing

The gate is **purely additive** — `ecology_H_mode="empirical"`
(default) is fully back-compat.  When `ecology_H_mode="gradient"`
is set, callers must pass `task_loss` to
`moe_ecology_diagnostic()` (else silent fallback to empirical).
See `docs/research/2026-06-15_gradient_based_h_report.md` and the
unit tests in `tests/test_gradient_based_h.py`.

## Per-Expert Gradient Magnitude (causal imbalance detection, opt-in)

`per_expert_gradient_norms` (PRD #10-50, round 88) is the
**per-expert refinement** of the round 87 aggregated gradient H.
While aggregated H_grad averages over experts (and can mask
per-expert collapse), per-expert H_grad exposes **which specific
experts** are causally alive or dead.

```python
from lnn import FAMECfCNetwork

net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,
    ecology_per_expert_grad=True,  # opt in
)
# ... train loop ...
task_loss = ...  # scalar, with grad
diag = net.cells[0].moe_ecology_diagnostic(
    B=0.001, task_loss=task_loss,
)
# diag["per_expert_grad_list"]  # [K] list of gradient norms per expert
# diag["dead_by_grad"]          # count of dead experts by gradient
# diag["max_min_ratio"]         # spread of per-expert gradients
```

**Honest-positive headline (round 88)**: in our 9-cell bench,
per-expert H_grad exposes **causal imbalance** that empirical H
cannot see:
- 1-hot collapsed regimes: `max_min_ratio_grad = 13-27×` (one
  expert doing 95% of the causal work)
- Healthy regimes: `max_min_ratio_grad = 2-3×` (balanced experts)
- Even "dead-by-utilization" experts have **non-zero gradient**
  (100-300× smaller, but not zero)

This is the **direct response to the Causal Audit
(arXiv:2606.10703)**: observational E can mask causal collapse,
but per-expert H_grad catches it.

**Bug fix**: `self.last_g` is intentionally detached (so the
routing isn't perturbed by every forward).  Round 88 adds
`self.last_router_logits` (non-detached) for gradient
computation.  `moe_ecology_diagnostic` uses `last_router_logits`
for gradient-based H modes and `last_g` for utilization-based
ones.

The diagnostic is **purely additive** —
`ecology_per_expert_grad=False` (default) is fully back-compat.
When set, callers should pass `task_loss` to
`moe_ecology_diagnostic()`.  See
`docs/research/2026-06-15_per_expert_gradient_report.md` and the
unit tests in `tests/test_per_expert_gradient.py`.

## Causality-Gated Orth (auto-reduce λ when per-expert imbalance > threshold, opt-in)

`CausalityGatedOrth` (PRD #10-51, round 89) is the **policy
counterpart** to round 88's per-expert gradient **diagnostic**.
While the diagnostic reports `max_min_ratio_grad`, the gate
turns it into action: when per-expert causal imbalance exceeds
the threshold (default 10.0), automatically rescale `λ` to a
safe value (default 0.001, sticky).

```python
from lnn import FAMECfCNetwork

net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,
    ecology_gated_orth=True,           # round 85 (observational E)
    causality_gated_orth=True,         # round 89 (causal per-expert)
    causality_ratio_threshold=10.0,    # from round 88 finding
    ecology_orth_lambda_safe=0.001,
)
# ... train loop ...
task_loss = ...  # scalar, with grad
# Use compute_orth_loss_causality instead of compute_orth_loss.
# Combined gate takes min(λ_eff from E, λ_eff from causality).
orth_loss = net.cells[0].compute_orth_loss_causality(
    outs, user_lambda=1.0, task_loss=task_loss,
)
```

**Stack of 2-axis orthogonal gates**:
- Round 85: `EcologyGatedOrth` fires when observational `E < 0.5`
- Round 89: `CausalityGatedOrth` fires when per-expert
  `max_min_ratio_grad > 10.0`
- Combined: `effective_λ = min(λ_E_safe, λ_cau_safe)` if either
  gate fires, else `user_lambda`

**Honest-positive headline (round 89)**: in the 9-cell bench,
with E-gate already active, ratios are bounded at 2-7× in 5
epochs, so causality gate rarely fires (3/9 cells, structured
dataset only). The gate is **defense-in-depth** — correctly
implemented, sticky, and ready to fire if E-gate is disabled
or training runs longer.

**Honest-negative**: in the 5-epoch regime with E-gate, the
causality gate doesn't catch anything E-gate misses. The
13-27× ratios from round 88 only manifest in the no-gate
collapse regime. For round 90: longer training and threshold
sweep (5, 10, 20) to calibrate the sweet spot.

The gate is **opt-in** — `causality_gated_orth=False` (default)
is fully back-compat. When set, callers should pass
`task_loss` to `compute_orth_loss_causality()`.  See
`docs/research/2026-06-15_causality_gated_orth_report.md` and
the unit tests in `tests/test_causality_gated_orth.py`.

## Orthogonality Audit: Weights vs Activations (Kim 2026 response, opt-in)

`weight_space_overlap` and `activation_space_overlap`
(PRD #10-52, round 90) are **audit metrics** for the round 80-89
orthogonality stack, in direct response to arXiv:2601.00457
(Hyunjun Kim, Jan 2026) — *Geometric Regularization in
Mixture-of-Experts: The Disconnect Between Weights and
Activations*.

```python
from lnn import FAMECfCNetwork, weight_space_overlap, activation_space_overlap

net = FAMECfCNetwork(
    input_size=3, hidden_size=24, output_size=1,
    n_experts=3, top_k=1,
)
# ... train loop ...
outs = [o.detach() for o in last_outs]  # K activations (B, T, D)
weights = [p.detach() for p in expert_f_gate_weights]  # K (out, in)
act_ov = activation_space_overlap(outs)    # target metric
wgt_ov = weight_space_overlap(weights)     # disconnect metric
```

**Kim 2026 claim**: weight-space geometric regularization causes
weight overlap to grow (+114%) while leaving activation overlap
unchanged (r=−0.293, p=0.523).

**Our audit (12-cell bench, 5 epochs)**:
- **H2 ✓**: `activation_overlap` drops 47-54% in toy_sin and
  structured under our `orthogonality_loss` (we hit our target)
- **H1 ~partial**: `weight_overlap` grows +44-48% under our orth
  (mild version of Kim's disconnect, but much less than +114%)
- **H4 ✗**: high orth λ (10.0) hurts task loss by 30-100% in toy
  regime — orth loss is a **stylistic tax**

**Verdict**: our round 80 orthogonality loss is functionally
correct (H2) but with a mild version of the Kim 2026 disconnect.
**Recommendation**: keep orth λ ≤ 0.1 in toy regime; use round 85
E-gate to auto-rescale for real datasets. See
`docs/research/2026-06-15_orth_weights_vs_activations_report.md`
and the unit tests in `tests/test_orth_audit_metrics.py`.

## CfC Temporal Smoothness (arXiv:2606.07670 response, opt-in)

`total_variation`, `l2_derivative`, `max_gradient`, and
`smoothness_summary` (PRD #10-53, round 91) are **smoothness
metrics** for testing the claim from arXiv:2606.07670
(Li, Pal, Tan, June 2026) — *Liquid Neural Networks as a Drop-in
Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting*:

> "CfC embeds a learned smooth response to t directly into the loss
> landscape. Temporal smoothness is now a built-in property rather
> than an emergent artifact."

```python
from lnn import total_variation, l2_derivative, max_gradient

y = ...  # 1D tensor of model output over dense t grid
tv = total_variation(y)        # mean |y[i+1] - y[i]|
ld = l2_derivative(y, dt=1/255) # RMS finite-difference derivative
mg = max_gradient(y, dt=1/255)  # max |f'(t)|
```

**Round 91 bench (1D function fitting, MLP vs CfC, 5 seeds × 100
epochs)**:
- **H1 PARTIAL ✓**: max_grad -44% (2.02 vs 3.62), l2_deriv -12%
  (1.98 vs 2.24), TV +13% (0.0078 vs 0.0069, unfavorable)
- **H2/H3/H4 ✗**: CfC has worse interpolation (mse 0.26 vs 0.17)
  and extrapolation (ood_mse 3.03 vs 2.22) despite 2.8× more params

**Verdict**: arXiv:2606.07670's smoothness claim is real at the
max-derivative level (which is what matters for 3DGS artifacts)
but does NOT translate to better task performance in 1D. CfC's
output is plateau-like with sharp transitions, MLP's is
ripple-like with smaller but more frequent changes.

**Implication for our stack**: prefer CfC for control / physics
applications (where Lipschitz bounds matter), prefer MLP /
transformer for time-series forecasting (where MSE matters). The
MoE-CfC variants (`FAMECfCCell`, `MRMoECfCCell`) inherit this
property. See
`docs/research/2026-06-15_cfc_temporal_smoothness_report.md` and
the unit tests in `tests/test_smoothness_metrics.py`.

## CfC Temporal Dropout Robustness (arXiv:2605.27467 response, opt-in)

`temporal_dropout` and `dropout_mask` (PRD #10-54, round 92) are
**dropout helpers** for testing the claim from arXiv:2605.27467
(Thu, Oo, Supnithi, May 2026) — *Comparative Analysis of Liquid
Neural Networks and LSTM for Sequential Pattern Recognition:
Robustness, Efficiency, and Clinical Utility*:

> "LNNs consistently provide superior parameter efficiency and
> significantly higher robustness" compared to LSTM under
> temporal dropout (randomly missing input observations).

```python
from lnn import temporal_dropout

t = torch.linspace(0, 1, 64)
y = torch.sin(2 * 3.14159 * t)
t_out, y_masked = temporal_dropout(t, y, p=0.4, seed=0)
# y_masked has ~40% of its values set to 0
```

**Round 92 bench (4 models × 6 dropout p × 3 seeds, 1D f(t) fitting,
100 epochs)**:

| model | max_grad@0 | degradation@0.8 |
|-------|------------|------------------|
| MLP   | 3.66       | 2.96x            |
| CfC   | 2.03       | 2.06x            |
| LSTM  | 52.79      | **1.29x**        |
| GRU   | 37.98      | 1.68x            |

**Verdict**:
- **H1 ✓ (stateless models)**: CfC is 30% more robust than MLP,
  consistent with round 91's smoothness prior
- **H2 ✗ (across architectures)**: smoothness does NOT predict
  robustness — LSTM has 26× higher max_grad but 60% lower
  degradation than CfC
- **arXiv:2605.27467 claim REJECTED in 1D**: LSTM is significantly
  more robust than CfC, opposite of the paper's claim

**The 2-round chain (smoothness → robustness) is partially broken**:
works for stateless models (MLP, CfC), fails across architectures.
LSTM's robustness comes from gating + state, not smoothness.

**Implication for our stack**: pick the right model for the right
task. CfC's smoothness matters for 3DGS-style tasks; LSTM/GRU's
gating + state matters for 1D function fitting under dropout.
See `docs/research/2026-06-15_cfc_temporal_dropout_report.md` and
the unit tests in `tests/test_temporal_dropout.py`.

## Input-Side Temporal Dropout (arXiv:2605.27467 response, round 93)

`input_dropout` and `apply_input_dropout_to_input` (PRD #10-55,
round 93) are the **input-side counterparts** to round 92's
`temporal_dropout`. The distinction is semantic: the caller passes
the masked y as the **model's input** (not as the loss target).
For stateful models (LSTM, GRU), this corrupts the running state
with zeroed inputs, so the two dropout types have very different
effects on the model.

```python
from lnn import input_dropout

t = torch.linspace(0, 1, 64)
y = torch.sin(2 * 3.14159 * t)
t_out, y_input = input_dropout(t, y, p=0.4, seed=0)
# y_input has ~40% of its values set to 0; pass y_input to the model
```

**Round 93 bench (4 models × 6 dropout p × 3 seeds, 1D f(t) fitting,
100 epochs, 2D input `(t, y_masked)`)**:

| model | max_grad@0 | degradation@0.8 |
|-------|------------|------------------|
| **MLP**   | 0.18       | **0.23x**        |
| CfC   | 0.05       | 0.41x            |
| LSTM  | 19.61      | 0.61x            |
| GRU   | 12.40      | 0.59x            |

**Verdict**:
- **H1 ✗ (paper claim NOT rescued)**: MLP is the most robust, not
  CfC. Paper's "CfC > LSTM" claim is firmly rejected under both
  target-side (round 92) and input-side (round 93) dropout.
- **H2 ✗ (stateless recovery)**: CfC improves 5x from round 92
  to round 93. Statelessness is no longer a disadvantage.
- **H3 PARTIAL (LSTM collapse)**: LSTM 1.39x at p=0.4, then 0.61x
  at p=0.8. Non-monotonic state recovery at high p.
- **H4 ✓ (regularization)**: ALL models improve under input-side
  dropout (degradation < 1.0x at p=0.8).

**The 3-round chain (smoothness → target-side → input-side
robustness) is firmly broken**: smoothness is a *property* of the
model but not a *predictor* of robustness. The robustness hierarchy
depends on the dropout regime.

**Implication for our stack**: **MLP** is the most robust model in
1D for input-side dropout (cheap, simple, stateless). The clinical
irregular-sampling scenario from the paper is untested in our
1D bench but the result suggests CfC's advantage there is
questionable.
See `docs/research/2026-06-15_cfc_input_dropout_report.md` and
the unit tests in `tests/test_temporal_dropout.py` (19/19 pass).

## Effective Rank (arXiv:2606.00243 response, round 94)

`effective_rank`, `mean_effective_rank`, `effective_rank_trajectory`,
and `rank_summary` (PRD #10-56, round 94) test the prediction from
arXiv:2606.00243 (Williams, Payeur, Lajoie, ICML 2026) that
**locality-restricted learning rules find low-rank solutions**.

```python
from lnn.core import effective_rank
import torch

W = torch.randn(16, 32)
er = effective_rank(W)  # eff_rank = (Σσᵢ)² / (Σσᵢ²)
```

**Round 94 bench (4 models × 3 seeds, 1D f(t) fitting, 100 epochs)**:

| model | mse   | **weight_eff_rank** | hidden_eff_rank |
|-------|-------|----------------------|------------------|
| **MLP**   | 0.1721 | **3.61** (lowest) | 1.55 |
| CfC   | 0.2591 | **8.36** (HIGHEST) | 1.93 |
| LSTM  | 0.3366 | 4.73            | 1.73 |
| GRU   | 0.2982 | 3.85            | 2.07 |

**Verdict**:
- **H1 ✗ (paper prediction)**: CfC has the HIGHEST weight_eff_rank
  (8.36), not the lowest. Smoothness is NOT a low-rank bias.
- **H2 ✗ (correlation with smoothness)**: rank and smoothness
  rankings are **inverted** — smoothest model has highest rank.
- **H3 PARTIAL**: CfC hidden_eff_rank = 1.93 < 4 ✓, but LSTM/GRU/MLP
  are in the same range.
- **H4 ✓ (no collapse)**: All models have eff_rank > 1.5.

**The 4-round smoothness audit (rounds 91-94) is now complete**:
smoothness is a **property of the function class CfC learns** but
NOT a predictor of robustness (rounds 92, 93) or low rank (round 94).
CfC's stack should be chosen for tasks where smooth interpolation
matters (3DGS, irregular time-series with smooth priors), not for
tasks where robustness or parameter efficiency are the primary
metrics.

**Verdict on arXiv:2606.00243**: the paper's theory is specific to
**discrete-time linear RNNs with locality-restricted learning rules**
(RFLO, tBPTT). It does NOT generalize to continuous-time CfC cells
with full BPTT. CfC is smooth but high-rank — a distinct regime.

See `docs/research/2026-06-15_cfc_effective_rank_report.md` and
the unit tests in `tests/test_effective_rank.py` (27/27 pass).

## Per-Expert Effective Rank (FAME/MR-MoE diversity test, round 95)

`per_expert_effective_rank`, `expert_diversity_ratio`, and
`expert_diversity_summary` (PRD #10-57, round 95) directly test the
**"diverse experts"** claim of the FAME paper (arXiv:2606.08896)
and the **"multi-rate specialization"** claim of MR-MoE
(arXiv:2606.12240) by measuring per-expert weight effective rank
in a FAME/MR-MoE cell.

```python
from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.effective_rank import expert_diversity_summary

cell = FAMECfCCell(input_size=1, hidden_size=8, n_experts=5, top_k=2)
# ... train for N epochs ...
summary = expert_diversity_summary(cell)
# {
#   'per_expert': [5.04, 4.39, 4.86, 5.13, 4.27],  # K=5 eff_ranks
#   'diversity_ratio': 1.32,                       # max/min
#   'mean': 4.74, 'min': 4.27, 'max': 5.13, 'std': 0.32,
#   'n_experts': 5, 'n_dead': 0
# }
```

**Round 95 bench (3 datasets × 2 models × 2 conditions × 3 seeds,
100 epochs)**:

| dataset    | FAME trained div | MR-MoE trained div | FAME > MR-MoE? |
|------------|------------------|--------------------|-----------------|
| toy_sin    | **1.32 ± 0.08**  | 1.08 ± 0.01        | ✓ (Δ=0.24)     |
| structured | 1.15 ± 0.04      | 1.12 ± 0.04        | ✓ (Δ=0.03)     |
| random     | **1.31 ± 0.08**  | 1.13 ± 0.01        | ✓ (Δ=0.18)     |

**Verdict**:
- **H1 (FAME develops > 1.5 diversity) REJECTED**: FAME trains to
  1.15-1.32 (not > 1.5), but **FAME is consistently more diverse
  than MR-MoE** (Δ = 0.03-0.24).
- **H2 (utilization correlates with eff_rank) REJECTED**: no
  correlation between router utilization and weight rank.
- **H3 (dead experts collapse) REJECTED**: dead experts stay at
  init eff_rank (~5-6), don't collapse to 0. Good news: router
  correctly gates gradient.
- **H4 (orthogonality boosts diversity) NOT TESTED** — direct test
  of round 80 mechanism, deferred to backlog.

**Verdict on FAME & MR-MoE papers**: the FAME "diverse experts"
claim is **modestly supported** in our cell-level instantiation;
the MR-MoE "multi-rate specialization" claim is **NOT supported**
(dense softmax mixes experts too uniformly).

**Implication for the LNN stack**:
- **FAME is the better MoE choice when expert diversity matters**
  (top_k routing creates real differentiation)
- **MR-MoE is closer to a soft attention ensemble** (dense routing
  averages experts)
- The 5-round smoothness + diversity audit (rounds 91-95) is
  complete: smoothness, low rank, robustness, and diversity are
  **independent properties** of CfC and the MoE stack — not
  interchangeable.

See `docs/research/2026-06-15_per_expert_effective_rank_report.md`
and the unit tests in `tests/test_effective_rank.py` (27/27 pass).

## FAME + Orthogonality Diversity Test (round 80 mechanism, round 96)

`scripts/bench_fame_orth_diversity.py` (PRD #10-58) tests the open
H4 from round 95: does the round 80 `orthogonality_loss`
(arXiv:2606.03631 AnchorMoE) at the safe λ=0.001 setting
actually increase FAME's expert diversity?

```python
# Compare baseline vs +orth at λ=0.001, 100 epochs, 3 seeds
from lnn.core.orthogonality import orthogonality_loss
from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.effective_rank import expert_diversity_summary

cell = FAMECfCCell(input_size=1, hidden_size=8, n_experts=5, top_k=2)
# Train with:  h_new, expert_outs = cell.forward_with_aux(x, h, dt=1.0)
#              orth = orthogonality_loss(expert_outs, lambda_coeff=0.001)
#              loss = task_loss + orth
summary = expert_diversity_summary(cell)
# 'diversity_ratio' = max/min per-expert eff_rank
```

**Round 96 bench (3 datasets × {baseline, +orth λ=0.001} × 3 seeds,
100 epochs)**:

| dataset    | cond     | div_ratio   | act_cos       |
|------------|----------|-------------|---------------|
| toy_sin    | baseline | 1.32 ± 0.08 | 0.7555        |
| toy_sin    | +orth    | 1.31 ± 0.06 | 0.7337        |
| structured | baseline | 1.15 ± 0.04 | 0.4238        |
| structured | +orth    | 1.16 ± 0.03 | **0.2600**    |
| random     | baseline | 1.31 ± 0.08 | 0.3319        |
| random     | +orth    | 1.24 ± 0.04 | **0.2472**    |

**Verdict**:
- **H1 (orth increases weight diversity) REJECTED**: Δ div_ratio = -0.07 to +0.01, all within noise
- **H2 (task loss safe at λ=0.001) CONFIRMED**: Δ = 0% / -2% / +3%
- **H3 (orth reduces activation cos_sim) PARTIAL**: structured -0.164, random -0.085, toy_sin -0.022

**Key insight — activation vs weight diversity**:
- **Activation diversity** = expert hidden states are decorrelated on the same input
- **Weight diversity** = expert weight matrices have different singular value spectra
- `orthogonality_loss` targets **activation** diversity directly. It does NOT target weight diversity.
- To boost weight diversity, we'd need a penalty on the **Gram matrix of weight matrices** (e.g. `||W_i W_j^T||_F^2`).

**Verdict on arXiv:2606.03631 (AnchorMoE)**:
- Orthogonality constraint decorrelates expert representations: **CONFIRMED** (H3, structured/random)
- Constraint is safe for task loss at small λ: **CONFIRMED** (H2, ±3%)
- Constraint increases weight diversity: **REJECTED** (H1) — orthogonality is activation-level only

**The 6-round audit (rounds 91-96) is now complete**:
smoothness, robustness, rank, weight diversity, activation diversity
are **independent properties** of CfC and the MoE stack. Each
mechanism targets its own level — they do not cross.

See `docs/research/2026-06-15_fame_orth_diversity_report.md`.

## Weight-Level Orthogonality (round 97)

`weight_orthogonality_loss` and `FAMECfCCell.compute_weight_orth_loss`
(PRD #10-59) target the **weight matrices** directly — the
**weight-level** counterpart of `orthogonality_loss` (round 80)
which targets activations.

```python
from lnn.core.orthogonality import weight_orthogonality_loss
import torch

W_list = [torch.randn(4, 4) for _ in range(3)]
aux = weight_orthogonality_loss(W_list, lambda_coeff=0.001)
# L = λ * Σ_{i<j} ||W_i W_j^T||_F^2 / (||W_i||_F · ||W_j||_F)
# Normalized so the penalty is dimensionless and bounded.
```

**Round 97 bench (3 datasets × {baseline, act, wt, both} × 3 seeds,
100 epochs, λ=0.001)**:

| dataset    | mode     | div_ratio   | mean_eff   | act_cos       |
|------------|----------|-------------|------------|---------------|
| toy_sin    | baseline | 1.32 ± 0.08 | 5.13       | 0.7555        |
| toy_sin    | wt       | 1.30 ± 0.10 | **4.13**   | 0.7266        |
| toy_sin    | **both** | **1.33** ± 0.07 | **4.06** | 0.7458        |
| structured | baseline | 1.15 ± 0.04 | 5.31       | 0.4238        |
| structured | wt       | 1.17 ± 0.07 | **4.38**   | 0.3870        |
| structured | both     | 1.15 ± 0.04 | **4.44**   | 0.2406        |
| random     | baseline | 1.31 ± 0.08 | 5.49       | 0.3319        |
| random     | wt       | 1.15 ± 0.02 | **4.28**   | 0.4092        |
| random     | both     | 1.18 ± 0.02 | **4.24**   | 0.2018        |

**Verdict**:
- **H1 (wt orth increases weight diversity) REJECTED**: Δ div_ratio = -0.02 to +0.02 (essentially zero)
- **H2 (task loss safe) CONFIRMED**: ±3% across all datasets
- **H3 (wt orth reduces act_cos) PARTIAL**: marginal on toy_sin/structured, wrong direction on random
- **Side finding (the headline)**: weight_orth reduces **mean_eff_rank by ~20%** (5.13→4.13, 5.31→4.38, 5.49→4.28)

**Key insight**:
- `orthogonality_loss` (round 80) = **activation-level** tool (decorrelates hidden states)
- `weight_orthogonality_loss` (round 97) = **weight-level** tool (reduces expert complexity)
- The **"both" combination** gives activation diversity + weight regularization at ±3% task cost

**The 7-round audit is now complete (rounds 91-97)**: smoothness,
robustness, rank, weight diversity, activation diversity, **weight
regularization** are independent properties. Each mechanism targets
its own level — they do not cross.

See `docs/research/2026-06-15_fame_weight_orth_report.md` and the
unit tests in `tests/test_orthogonality.py` (20/20 pass).

## What Is Stable?

| Area | Path | Status |
|---|---|---|
| Core sequence models | `lnn/core/cfc.py`, `lnn/core/ltc.py`, `lnn/core/liquid_neuron.py` | Reusable, tested package code |
| Sequence utilities | `lnn/core/sequence_utils.py`, `lnn/core/trainer.py`, `lnn/data/timeseries.py` | Reusable helpers for experiments |
| Research backbones | `lnn/core/noise_adaptive_cfc.py`, `lnn/core/multimodal_physreg.py`, `lnn/core/dynpmnn.py`, `lnn/core/variants.py` | Tested, but APIs may change with new papers |
| Benchmarks and recipes | `scripts/`, `configs/` | Reproducibility entry points; CLI flags may evolve |
| Research archive | `docs/research/`, `docs/reports/`, `analysis/` | Timestamped evidence and iteration history |
| Knowledge workflow | `AGENTS.md`, `skills/` | Automation and paper-analysis workflow docs |

Use `lnn/` for library code. Use `docs/research/` and `analysis/` when you want
to inspect how a result was produced.

## Quick Paths

| Goal | Start here |
|---|---|
| Understand the current benchmark result | [LNN_TLDR.md](LNN_TLDR.md) |
| Learn LNN principles from zero | [docs/guides/LNN_PRINCIPLES_FOR_BEGINNERS.md](docs/guides/LNN_PRINCIPLES_FOR_BEGINNERS.md) |
| Reproduce the EMMA rover recipe | [LNN_QUICKSTART.md](LNN_QUICKSTART.md) |
| Compare supported model families | [LNN_MODEL_GUIDE.md](LNN_MODEL_GUIDE.md) |
| Audit LFM/LNN active-3B vs 30B+ LLM claims | [analysis/llm_battlecard/2026-06-04_llm_battlecard.md](analysis/llm_battlecard/2026-06-04_llm_battlecard.md) |
| Run the local LFM2.5 micro-eval | [analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md](analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md) |
| Run the LFM2.5 HTTP endpoint micro-eval | [analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.md](analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.md) |
| Inspect the LFM2.5 DPO Q4 regression result | [analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.md](analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.md) |
| Inspect the LLM micro-eval leaderboard | [analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.md](analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.md) |
| Read the product/research roadmap | [docs/PRD_LNN_Edge_Research.md](docs/PRD_LNN_Edge_Research.md) |
| Inspect multimodal design decisions | [docs/guides/LNN_MULTIMODAL_DESIGN.md](docs/guides/LNN_MULTIMODAL_DESIGN.md) |
| Understand the automation agents | [AGENTS.md](AGENTS.md) |

## Benchmark Snapshot

The current headline EMMA rover result is an adaptive freeze recipe using a
Bi-CfC-NAD style backbone:

```bash
python lnn/data/emma_rover_features.py
python scripts/benchmark_adaptive_freeze.py \
    --epochs 80 \
    --warmup-epochs 40 \
    --freeze-targets audio_only \
    --num-samples 200 \
    --hidden-size 64
```

The latest recorded run reports roughly `MSE ~= 0.31`, compared with a
`video_only` baseline around `0.87`. Treat this as an actively maintained
research benchmark; inspect `analysis/emma_rover/` and the linked research docs
for the full ablation trail.

For Jetson smoke checks:

```bash
RUN_BENCHMARK=1 COMMIT_AND_PUSH=0 ./scripts/run_daily_lnn_task.sh
python scripts/jetson_lnn_benchmark.py --quick --pareto
```

## Repository Layout

```text
LNN/
├── lnn/                  # Python package: models, datasets, utilities
├── tests/                # Unit and regime tests
├── scripts/              # Benchmarks, ablations, automation entry points
├── configs/              # Experiment configs
├── analysis/             # Generated benchmark outputs and plots
├── docs/                 # Research reports, PRD, living-review notes
├── papers/               # Paper tracking and archives
├── projects/             # External repo clones and reproduction work
└── skills/               # Vercel Skills-compatible research agents
```

This repository can also be opened as an Obsidian vault. The GitHub README is
intentionally shorter and code-first; the vault-style notes remain in `docs/`.

## Automation

Generate the daily LNN research digest without committing:

```bash
COMMIT_AND_PUSH=0 ./scripts/run_daily_lnn_task.sh
```

Install the local user-level systemd timer:

```bash
./scripts/install_daily_lnn_timer.sh
```

GitHub Actions also runs `.github/workflows/daily-lnn-research.yml` to generate
daily research summaries.

## Related Implementations

- [raminmh/liquid_time_constant_networks](https://github.com/raminmh/liquid_time_constant_networks)
- [raminmh/CfC](https://github.com/raminmh/CfC)
- [mlech26l/ncps](https://github.com/mlech26l/ncps)
- [emilierp/exact_lnn](https://github.com/emilierp/exact_lnn)
- [makramchahine/drone_causality](https://github.com/makramchahine/drone_causality)

## License

MIT. See [LICENSE](LICENSE).
