---
title: "PRD #10-102 — Round 265 — STE-NeuronWiseCfCCell (hard top-k + soft gradients)"
date: 2026-06-28
round: 265
branch: master
audit_context: "64 strictly positive + 28 target-dep + 60 negatives = 152 mechanism classes"
arxiv_anchor: "arXiv:2606.21295v2 (Cai & Zhao, Topological Neural Dynamics, 2026-06-19)"
predecessor: "Round 263 (NeuronWiseCfCCell, hard top-k) + Round 264 (SoftNeuronAttentionCfCCell, soft — HONEST NEGATIVE)"
response_paper: "papers/daily/2026-06-24/2026-06-19_Topological_Neural_Dynamics_2606.21295.pdf"
report_anchor: "docs/research/2026-06-28_round264_soft_neuron_attention_report.md"
---

# PRD #10-102 — Round 265 — STE-NeuronWiseCfCCell (hard top-k + soft gradients)

## Background

Round 264 (SoftNeuronAttentionCfCCell) was a **HONEST NEGATIVE**:
soft attention over per-neuron neighborhoods *underperformed* the
hard top-k of r263 on all 3 datasets (structured 6.6× worse,
toy_sin 4.2× worse, random ~equal). The core finding:

> r263's hand-coded top-k imposes a STRONGER inductive bias
> than learnable continuous structure in 1D toy regime.

The r264 report identified the H4 partial result as a key
insight:

> `cold τ_attn + small L1 → near-r263 performance`. Soft
> attention CAN match r263, but doesn't BEAT it. The H4 partial
> result (cold temperature ≈ r263) is a "superset in the limit".

This round implements the **fix** identified by the r264 report:
use a **straight-through estimator (STE)** to combine the
benefits of both:

  - **Forward pass**: hard top-k (binary mask, true sparsity)
  - **Backward pass**: soft mask (sigmoid, gradients flow)

The classic STE identity is:

  mask_STE = (hard - soft).detach() + soft

In the forward direction this equals `hard` (binary, sparse).
In the backward direction the gradient flows through `soft`
(differentiable). This is the standard trick for binary latent
variables (Jang et al. 2016 Gumbel-Softmax, Courbariaux
BinaryNet, etc.).

## Goal

Test if STE-NeuronWiseCfCCell **beats both** r263 (hard top-k,
non-learnable) AND r264 (soft attention, fully learnable but
soft mixing). The hypothesis is that STE gives:

  - r263's true sparsity (binary mask in forward)
  - r264's differentiability (gradients flow to neighbor_logits)
  - Structural learning without soft-mixing degradation

## Mechanism

```python
class STENeuronWiseCfCCell(NeuronWiseCfCCell):
    def get_neighborhood_mask(self) -> torch.Tensor:
        # Hard top-k (binary, true sparsity) — like r263.
        hard = sparse_topk_mask(self.neighbor_logits, self.density)
        # Soft mask (sigmoid, differentiable).
        soft = torch.sigmoid(self.neighbor_logits / self.ste_temperature)
        # Straight-through estimator: hard forward, soft backward.
        return (hard - soft).detach() + soft
```

The forward pass uses `hard` (binary mask, true sparsity like
r263). The backward pass computes gradient via `soft` (which
flows to `neighbor_logits`).

Key design choices:
- **Inherit all of r263's per-neuron dynamics**: per-neuron τ,
  per-neuron α, per-neuron input strength, per-neuron bias.
- **Hard top-k forward**: exactly `k = round(density * d_h)`
  edges per row (true sparsity).
- **Soft sigmoid backward**: gradients flow to neighbor_logits
  via the soft mask.
- **Temperature τ_ste**: controls the *softness* of the backward
  gradient. Lower → sharper (closer to r263's gradient behavior);
  higher → smoother (closer to r264's gradient behavior).
- **ste_temperature is a hyperparameter**, not learned. We
  ablate across a few values to find the right one.

## Hypotheses (PRD #10-102)

- **H1**: STE-NeuronWiseCfCCell beats r263 (hard top-k,
  non-learnable) on at least one dataset because learnable
  structure (with STE gradients) outperforms hand-coded
  structure.
- **H2**: STE-NeuronWiseCfCCell beats r264 (soft attention)
  on at least one dataset because hard top-k forward avoids
  the soft-mixing degradation.
- **H3**: The learned neighbor_logits become CORRELATED with
  useful structure: pre-training logits = 0, post-training
  logits have structure (e.g., self-edge logits > off-diag
  mean). Evidence the gradient signal is meaningful.
- **H4**: STE-NeuronWiseCfCCell is a strict superset of both
  r263 and r264: with ste_temperature → 0, the soft mask
  approaches a one-hot (steep sigmoid, gradient ≈ 0 — no
  learning); with ste_temperature → ∞, the soft mask
  approaches uniform (like r264's no-L1 mode).

## Configurations (5 modes × 3 datasets × 2 seeds = 30 cells)

1. `r263_baseline` (round 263, density=0.3) — hard top-k
   reference (non-learnable)
2. `ste_cold` (τ_ste=0.1) — sharp backward gradient (close to
   r263's behavior)
3. `ste_default` (τ_ste=1.0) — moderate
4. `ste_warm` (τ_ste=5.0) — soft backward gradient
5. `ste_no_init` (τ_ste=1.0, neighbor_logits init=0.0) —
   control for initialization effect

The 5 modes give a temperature ablation. We compare against
r263_baseline (the strongest non-learnable baseline) and
implicitly against r264's softattn_default (in the report
discussion, since they share the same bench protocol).

## Expected Pattern

If H1+H2 hold (STE beats both): **STRICTLY POSITIVE**, the
64+1=65 SP bucket grows.

If H1 holds but H2 doesn't (only beats r263): TARGET-DEP,
the 28 bucket grows.

If neither holds: HONEST NEGATIVE, the 60+1=61 NEG bucket
grows.

Given the r264 result (soft mixing is the failure mode), the
most likely outcome is **STRICTLY POSITIVE** or **TARGET-DEP**:
the hard forward mask removes the soft-mixing degradation
while the soft backward provides useful gradient signal.

## Files to add

1. `lnn/core/ste_neuron_wise_cfc.py` (~120 LOC) — subclass of
   r263 with STE mask
2. `tests/test_ste_neuron_wise_cfc.py` (~150 LOC) — 12 unit tests
3. `scripts/bench_ste_neuron_wise_cfc.py` (~250 LOC, copy/adapt
   from r264)
4. `lnn/core/__init__.py` — re-export STENeuronWiseCfCCell
5. `docs/research/2026-06-28_round265_ste_neuron_wise_report.md` —
   bench report

## Bench config

- 3 datasets: toy_sin, structured, random
- hidden_size = 16
- 100 epochs, lr=1e-2, batch=16, 2 seeds
- 5 modes (above)
- Loss: MSE only (no auxiliary L1 needed; sparsity is enforced
  by the hard mask in forward)
- Metrics: test_mse, neighbor_logits stats (mean, std, pre vs
  post), sparsity (binary mask entropy), tau stats

## Why This Round

1. **Direct fix for r264**: r264 failed because soft mixing
   degraded task loss. STE preserves the hard mask in forward.
2. **Natural extension of r263**: r263's structure was
   non-learnable. STE makes it learnable with minimal code
   change (subclass).
3. **Tests H1+H2 (falsifiable)**: clear head-to-head against
   both predecessors.
4. **Cheap experiment**: ~120 LOC + 150 LOC tests, no
   external deps.
5. **Strategic**: closes the r263-r264 gap in the LNN+MoE stack.
   r263 = hand-coded structure. r264 = continuous structure
   (failed). r265 = STE-structured (best of both).

## Risk Assessment

- **Risk: STE gradient too noisy at low temperature**: low —
  sigmoid gradients are well-behaved, especially at small
  τ_ste where the soft mask is close to the hard mask.
- **Risk: STE gradient too uniform at high temperature**: low
  — at large τ_ste the sigmoid saturates and gradients vanish,
  but we ablate across temperature to find the right regime.
- **Risk: neighbor_logits diverges**: low — the soft mask is
  bounded, the hard mask is binary, the gradient is bounded
  by sigmoid derivative.
- **Risk: hidden=16 too small**: acknowledged. Like r264,
  STE may help more at larger hidden sizes.

## Pattern Update Expectation

After r265:
- **65 strictly positive** (if H1+H2 confirmed, beats both r263 and r264)
- 28 target-dep (if H1 only)
- 60 negatives (unchanged)
- Total: **152 → 153 mechanism classes** (most likely outcome:
  65 SP + 28 TD + 60 NEG)

## Caveats / Pre-registered Decisions

- **Pre-registered**: H1 PASS = ste_default beats r263_baseline
  on ≥ 1 dataset.
- **Pre-registered**: H2 PASS = ste_default beats
  r264's softattn_default on ≥ 1 dataset (using the r264 bench
  JSON for reference).
- **Pre-registered**: H3 PASS = std(neighbor_logits) > 0.05
  after training (evidence the gradient is doing something).
- **Pre-registered**: τ_ste final value reported for each
  cell.
- **Pre-registered**: binary mask entropy reported for each
  cell (should be ~0 since mask is exactly top-k per row).