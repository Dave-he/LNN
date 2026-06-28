---
title: "PRD #10-100 — Round 263 — NeuronWiseCfCCell (TND response)"
date: 2026-06-28
round: 263
branch: master
audit_context: "63 strictly positive + 28 target-dep + 59 negatives = 150 mechanism classes"
arxiv_anchor: "arXiv:2606.21295v2 (Cai & Zhao, Topological Neural Dynamics, 2026-06-19)"
response_paper: "papers/daily/2026-06-24/2026-06-19_Topological_Neural_Dynamics_2606.21295.pdf"
report_anchor: "docs/reports/Topological_Neural_Dynamics_2606.21295_研读报告.md"
---

# PRD #10-100 — Round 263 — NeuronWiseCfCCell (TND response)

## Background

Round 257's bridge document
(`docs/research/2026-06-25_round257_bridge_to_neuronwise_research.md`)
explicitly identified TND as the 2026 frontier and called for
"explicit per-neuron interaction operators". Subsequent rounds
implemented **basin-level** analogs:

- r257: inter-basin geometric repulsion (centers separated)
- r258: inter-basin learned sparse adjacency A ∈ ℝ^{K×K}
- r259: multi-hop message passing through the adjacency
- r260: per-step input-dependent adjacency A_t = softmax(MLP(x_t))
- r261: mix of static (r258) and per-step (r260) adjacency
- r262: learned channel projection c_t = LayerNorm(W_c @ x_t)
  before routing (helps with d_in > 1)

**Missing piece** (still): within each **basin**, all neurons share
a single hidden vector h_k ∈ ℝ^d_h. The cell does NOT have per-neuron
dynamics. TND's key claim is that neuron-wise dynamics with explicit
neighborhoods gives heterogeneous local trajectories + emergent
multi-time-scale propagation delay. r257-262 work on the basin axis
but not the neuron axis.

## Goal

Implement a CfC variant where each neuron has its own:

1. **Learnable neighborhood** N(i) ⊆ V with sparsity mask M_{ij}
2. **Local dynamics** h_i^{t+1} = (1-τ_i) h_i^t + τ_i tanh(∑_{j∈N(i)} M_{ij} W_{ij} v_j^t + w_i^in x_i^t + b_i + α_i h_i^t)
3. **Self-feedback** α_i per neuron (vs shared for the whole layer)
4. **Per-neuron tau** τ_i initialized from a small base set

This breaks the CfC assumption of a single τ (or n_tau groups) and
the assumption that all neurons in a layer share the same recurrent
operator. It also makes propagation delay emerge from topology
(neurons far from input integrate longer history) rather than from
explicit gating.

## Mechanism (CfC-style closed-form approximation)

We approximate the leaky integrator update

```
h_i^{t+1} = (1-τ_i) h_i^t + τ_i tanh(s_i^t)
```

with the CfC closed-form pattern: each neuron is its own tiny CfC
cell, with τ_i replaced by an "effective time" τ̃_i = τ_i * time_scale_i,
and s_i^t computed from the sparse neighborhood.

```python
# Per-neuron parameters (shape: d_h, )
self.tau_per_neuron = nn.Parameter(torch.full(d_h, base_tau))
self.alpha_per_neuron = nn.Parameter(torch.zeros(d_h))
self.bias_per_neuron = nn.Parameter(torch.zeros(d_h))
self.input_strength_per_neuron = nn.Parameter(torch.ones(d_h) * 0.1)

# Sparse neighborhood mask (d_h x d_h), learnable via sigmoid reparam
self.neighbor_logits = nn.Parameter(torch.randn(d_h, d_h) * 0.1)
self.neighbor_target_density = neighbor_density  # fraction of edges kept

# Per-neuron recurrent weights (d_h, d_h) but masked
self.W_rec = nn.Parameter(torch.randn(d_h, d_h) * (1.0 / math.sqrt(d_h)))

# Per-neuron input projection (d_in, d_h)
self.W_in = nn.Linear(d_in, d_h, bias=False)

def forward(self, x, h0=None):
    h = h0 or zeros(B, d_h)
    outputs = []
    for t in range(T):
        # Sparse mask: top-k per row, or soft threshold
        M = sparse_mask(self.neighbor_logits, density=self.neighbor_target_density)
        # Per-neuron signal: (B, d_h)
        s = (h @ (M * self.W_rec).T) + self.W_in(x[:, t]) + self.bias_per_neuron
        # Per-neuron CfC-style update with per-neuron tau
        gate = torch.sigmoid(self.tau_per_neuron)
        h = (1 - gate) * h + gate * torch.tanh(s + self.alpha_per_neuron * h)
        outputs.append(h)
    return stack(outputs, dim=1), h
```

Key design choices:
- **Sparse neighborhood** is enforced by `sparse_mask` (top-k per row),
  not soft thresholding. This guarantees a true per-neuron in-degree.
- **Per-neuron τ** is parameterized in log-space, sigmoid-bounded to
  [τ_min, τ_max] = [0.05, 5.0].
- **Self-feedback α** is clamped to [-0.5, 0.5] for stability.
- **CfC closed-form** is preserved per-neuron (each neuron is a
  leaky integrator with sigmoid time-constant).

## Hypotheses (PRD #10-100)

- **H1**: NeuronWiseCfCCell beats plain CfC on toy_sin and structured
  because per-neuron τ allows heterogeneous time-scales to emerge
  from topology, not from explicit gates.
- **H2**: The learned neighborhood mask becomes ASYMMETRIC (avg off-
  diagonal density > 0.3) — evidence the cell learns directed
  routing, not just symmetric smoothing.
- **H3**: Per-neuron τ values span a wide range after training
  (std(τ) > 0.3 * mean(τ)) — neurons develop heterogeneous time-
  scales. This is the TND prediction: "different neurons evolve at
  different rates".
- **H4**: NeuronWiseCfCCell is a strict superset of single-τ CfC:
  with neighbor_density=1.0 and uniform τ, it degenerates to a
  recurrent network equivalent to CfC's gate × tanh form.

## Configurations (5 modes × 3 datasets × 2 seeds = 30 cells)

1. `cf` (baseline)
2. `neuronwise_d03` (density 0.3 — sparse graph, key treatment)
3. `neuronwise_d05` (density 0.5 — moderate)
4. `neuronwise_d10` (density 1.0 — fully connected, α=τ=shared)
   This is the "superset" condition (H4).
5. `neuronwise_d03_shared` (density 0.3 but shared τ — control for
   per-neuron τ effect).

## Expected Pattern

Following the audit pattern (63 SP + 28 TD + 59 NEG), we expect
r263 (sparse + per-neuron τ) to be **STRICTLY POSITIVE** if TND's
claim transfers to 1D time-series. The all-to-all condition (d=1.0)
should match CfC (H4). The shared-τ control should match CfC (τ
alone isn't enough).

## Files to add

1. `lnn/core/neuron_wise_cfc.py` (~250 LOC) — new cell + sparse_mask helper
2. `tests/test_neuron_wise_cfc.py` (~150 LOC) — 13 unit tests
3. `scripts/bench_neuron_wise_cfc.py` (~250 LOC) — 30-cell bench
4. `lnn/core/__init__.py` — re-export NeuronWiseCfCCell
5. `docs/research/2026-06-28_round263_neuron_wise_tnd_report.md` — bench report

## Bench config

- 3 datasets: toy_sin, structured, random (same as r248-r262)
- hidden_size = 16 (small regime, matches prior rounds)
- 100 epochs, lr=1e-2, batch=16, 2 seeds
- Loss: MSE
- Metrics: test_mse, mean τ (final), std τ (final), neighbor asymmetry
  (||M - M^T||_F), propagation delay estimate (mean distance from
  input neuron)

## Why This Round

1. **Closes the TND gap**: r257-262 operate on the basin axis; r263
   finally operates on the neuron axis. After r263, the LNN+MoE
   stack has coverage on both axes.
2. **Tests a falsifiable TND claim**: per-neuron time-scales should
   emerge if TND's theory transfers to 1D time-series.
3. **Cheap experiment**: ~250 LOC + 150 LOC tests, no external deps.
4. **Falls within the "strict superset" pattern** (r262 explicitly
   tested H4 with success). Following the pattern, d=1.0 should
   reproduce CfC's behavior.
5. **Natural progression from r262**: r262 added input channel
   projection for routing; r263 adds structural per-neuron dynamics
   to USE that projection more deeply.

## Risk Assessment

- **Risk: per-neuron τ destabilizes**: low — each τ is sigmoid-bounded,
  α is clamped to [-0.5, 0.5], W_rec is init-small.
- **Risk: sparse mask is too restrictive**: tested via density sweep
  (d=0.3 / 0.5 / 1.0).
- **Risk: fails on random**: high — random data has no structure for
  per-neuron specialization to exploit. Expected to be TARGET-DEP.
- **Risk: hidden=16 too small to show heterogeneity**: acknowledged.
  We report std(τ) and std(α) as evidence.

## Pattern Update Expectation

After r263:
- **64 strictly positive** (if H1+H2 confirmed)
- **29 target-dep** (if random fails as expected)
- **59 negatives** (unchanged)
- Total: **152 mechanism classes**

If r263 fails on all 3, it's classified as **HONEST NEGATIVE** and
contributes to a new failure-mode class (per-neuron τ overfits in
1D).

## Caveats / Pre-registered Decisions

- **Pre-registered**: H4 (d=1.0 reproduces CfC) is a PASS/FAIL check
  before reading H1/H2/H3. If H4 fails, the cell has a bug.
- **Pre-registered**: τ-std > 0.3 × mean(τ) is the heterogeneity
  threshold for H3.
- **Pre-registered**: density sweep at d ∈ {0.3, 0.5, 1.0} gives
  the saturation curve. If d=0.5 and d=1.0 are equal, then d=0.3
  is the sweet spot. If d=0.5 > d=1.0, then per-neuron τ needs
  graph sparsity.
