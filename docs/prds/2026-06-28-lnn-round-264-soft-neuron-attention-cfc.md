---
title: "PRD #10-101 — Round 264 — SoftNeuronAttentionCfCCell (STE-learnable structure)"
date: 2026-06-28
round: 264
branch: master
audit_context: "64 strictly positive + 28 target-dep + 59 negatives = 151 mechanism classes"
arxiv_anchor: "arXiv:2606.21295v2 (Cai & Zhao, Topological Neural Dynamics, 2026-06-19)"
response_paper: "papers/daily/2026-06-24/2026-06-19_Topological_Neural_Dynamics_2606.21295.pdf"
report_anchor: "docs/research/2026-06-28_round263_neuron_wise_tnd_report.md"
predecessor: "Round 263 (NeuronWiseCfCCell) — top-k hard sparsification"
---

# PRD #10-101 — Round 264 — SoftNeuronAttentionCfCCell (STE-learnable structure)

## Background

Round 263 (NeuronWiseCfCCell) introduced per-neuron dynamics with
**top-k hard sparsification** of the neighborhood mask. The top-k
operator is non-differentiable, so the structure (`neighbor_logits`)
is a *structural hyperparameter* rather than a learned parameter.

The r263 report explicitly flagged this limitation:

> "neighbor_logits is NOT learned via gradient (topk is not
>  differentiable). The structure is structural-hyperparameter-
>  driven; for structure learning, use evolutionary search,
>  REINFORCE, or replace with a soft-mask + L1 approximation."

This round implements the third option: **soft attention** over
neighbors. Replace the binary mask with a softmax over neighbor
logits, with a temperature τ_attn. The result:

  α_{ij} = softmax(neighbor_logits / τ_attn)[i, j]

The forward pass uses α (not the binary mask), and gradients flow
naturally to neighbor_logits. This makes the structure **fully
learnable via gradient descent**.

Why this is an improvement over r263:
1. **Differentiable**: structure can be learned, not just set.
2. **Soft → sparse via temperature**: at low τ_attn, attention
   becomes peaked (sparse); at high τ_attn, it becomes uniform
   (dense). Temperature is a knob.
3. **All-to-all connectivity**: every neuron can attend to every
   other; learning determines which connections matter.
4. **L1 sparsity bonus**: an auxiliary L1 penalty on attention
   weights encourages sparsity without hard top-k.

## Goal

Test if **soft attention** over per-neuron neighborhoods improves
over **hard top-k sparsification** (r263) and **plain CfC**
(baseline). The hypothesis is that learning the structure gives a
strict win over the hard-coded r263 design.

## Mechanism

```python
# Replaces r263's top-k sparse mask with softmax attention.
self.neighbor_logits = nn.Parameter(torch.randn(d_h, d_h) * 0.1)
self.tau_attn = nn.Parameter(torch.tensor(1.0))  # temperature
self.l1_lambda = 0.01  # sparsity penalty

def get_attention(self) -> torch.Tensor:
    """Soft attention mask via row-wise softmax."""
    return torch.softmax(self.neighbor_logits / self.tau_attn.clamp(min=0.01), dim=-1)

def forward(self, x, h0=None):
    h = h0 or zeros(B, d_h)
    alpha = self.get_attention()  # (d_h, d_h) row-stochastic
    outputs = []
    for t in range(T):
        # Per-neuron signal (B, d_h):
        rec = h @ (alpha * self.W_rec).T  # soft-masked recurrence
        in_proj = self.W_in(x[:, t]) * self.input_strength
        s = rec + in_proj + self.bias + self.alpha_per_neuron * h
        h = (1 - tau) * h + tau * tanh(s)
        outputs.append(h)
    return stack(outputs, dim=1), h

def sparsity_loss(self) -> torch.Tensor:
    """L1 penalty on attention weights (encourages sparsity)."""
    return self.l1_lambda * self.get_attention().abs().mean()
```

Key design choices:
- **Soft attention** is row-stochastic via softmax (no normalization
  bug).
- **Temperature τ_attn** is learned via a single scalar parameter
  clamped at ≥ 0.01.
- **L1 penalty** is opt-in (default λ=0.01) — encourages sparsity
  without hard top-k.
- **All-to-all connectivity** preserved — no top-k constraint.
- The forward pass is otherwise identical to r263 (per-neuron τ,
  per-neuron α, per-neuron input strength).

## Hypotheses (PRD #10-101)

- **H1**: SoftNeuronAttentionCfCCell beats r263 (hard top-k) on at
  least one dataset because learnable structure outperforms
  hand-coded structure.
- **H2**: Attention weights become SPARSE naturally (mean
  attention weight < 0.1) after training — soft → sparse
  without explicit top-k.
- **H3**: Different neurons attend to different sources (per-row
  attention entropy varies, std > 0.5) — evidence of
  specialization.
- **H4**: SoftNeuronAttentionCfCCell is a strict superset of r263:
  with τ_attn → 0, the soft mask approaches a hard top-k (r263).
  With τ_attn → ∞, the mask becomes uniform (dense).

## Configurations (5 modes × 3 datasets × 2 seeds = 30 cells)

1. `r263_baseline` (round 263, density=0.3) — hard top-k reference
2. `softattn_default` (τ_attn init=1.0, L1 λ=0.01) — main treatment
3. `softattn_cold` (τ_attn init=0.1, L1 λ=0.001) — sharper attention
4. `softattn_warm` (τ_attn init=5.0, L1 λ=0.1) — soft, sparse
5. `softattn_nopen` (τ_attn init=1.0, L1 λ=0.0) — no sparsity bonus

## Expected Pattern

Following the audit pattern (64 SP + 28 TD + 59 NEG), we expect
r264 to be **STRICTLY POSITIVE** or **TARGET-DEP** if H1 is
confirmed. If H1 fails (soft attention underperforms hard top-k),
this is a **HONEST NEGATIVE-WITH-NUANCE** that adds to the
audit's structural-vs-learned-structure axis.

## Files to add

1. `lnn/core/soft_neuron_attention_cfc.py` (~280 LOC) — new cell
   + get_attention helper + sparsity_loss helper
2. `tests/test_soft_neuron_attention_cfc.py` (~150 LOC) — 13 unit tests
3. `scripts/bench_soft_neuron_attention_cfc.py` (~280 LOC) — 30-cell bench
4. `lnn/core/__init__.py` — re-export SoftNeuronAttentionCfCCell
5. `docs/research/2026-06-28_round264_soft_neuron_attention_report.md` — bench report

## Bench config

- 3 datasets: toy_sin, structured, random
- hidden_size = 16
- 100 epochs, lr=1e-2, batch=16, 2 seeds
- Loss: MSE + L1 sparsity (only when l1_lambda > 0)
- Metrics: test_mse, attention entropy (per-row mean), attention
  sparsity (mean weight), tau stats (mean, std), sparsity loss
  magnitude

## Why This Round

1. **Completes r263**: r263's main weakness is non-learnable
   structure. r264 closes that gap with soft attention.
2. **Tests H1 (falsifiable)**: learnable structure vs hand-coded
   structure is a clean experiment.
3. **Cheap experiment**: ~280 LOC + 150 LOC tests, no external deps.
4. **Natural progression**: r263 added per-neuron dynamics; r264
   adds per-neuron learned routing.
5. **Aligns with TND**: TND's claim is that neurons should have
   different neighborhoods; soft attention lets the network
   discover them rather than imposing them via top-k.

## Risk Assessment

- **Risk: L1 penalty collapses attention to uniform**: medium —
  mitigated by warm-init and small λ default.
- **Risk: τ_attn → 0 too fast (gradient explosion)**: low — τ_attn
  clamped at ≥ 0.01.
- **Risk: gradient on softmax is small when peaked**: low — softmax
  gradients are well-behaved at low temperature.
- **Risk: hidden=16 too small for soft attention**: acknowledged.
  Soft attention should help MORE at larger hidden sizes (more
  candidates to attend to); 16 may not show the benefit.

## Pattern Update Expectation

After r264:
- **65 strictly positive** (if H1+H2 confirmed)
- 28 target-dep (if random fails as expected)
- 59 negatives (unchanged)
- Total: **152 mechanism classes**

If r264 fails on all 3, it's classified as **HONEST NEGATIVE**
(soft attention overfits in 1D / hidden=16) and contributes to a
new failure-mode class.

## Caveats / Pre-registered Decisions

- **Pre-registered**: H1 PASS = softattn_default beats
  r263_baseline on ≥ 1 dataset. H1 FAIL = softattn_default ≤
  r263_baseline on all 3.
- **Pre-registered**: H2 PASS = mean attention weight < 0.1
  (sparse).
- **Pre-registered**: H3 PASS = std of per-row attention entropy
  > 0.5.
- **Pre-registered**: temperature τ_attn final value reported for
  each cell (evidence the cell learns the right temperature).