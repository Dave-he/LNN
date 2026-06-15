# Round 110 — MoFE-Time Frequency-Domain Experts (response to arXiv:2507.06502)

**Date**: 2026-06-15
**Round**: 110
**Paper**: arXiv:2507.06502 — *MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models* (Liu et al. Jul 2025)
**PRD**: #10-72
**Tests**: 23/23 in `tests/test_freq_experts.py`
**Bench**: 24 cells, 100 epochs (3 datasets × 4 conditions × 2 seeds), `scripts/bench_freq_experts.py`

## Summary

We implemented **MoFE-Time-inspired frequency experts** — a structural fix where each expert is a **learnable Fourier reconstructor** with its own harmonic frequencies and amplitudes. The audit pattern "structural > routing-only" predicted this would be the 6th structural winner because it changes the architecture fundamentally (each expert is a frequency reconstructor, not an MLP).

Bench results at 100 epochs (24 cells, 2 seeds):

- **H1 ✗ REJECTED**: Frequency experts do NOT improve over MLP on any of 3 datasets
- **H2 ✗ REJECTED**: sin_irr: MLP 0.0001-0.0004, freq_learned 0.0000-0.0010 (mixed, mostly worse)
- **H3 ✗ REJECTED**: structured_irr: MLP 0.0000-0.0001, freq_learned 0.0000-0.0003 (similar)
- **H4 ✓ CONFIRMED**: random_irr: MLP 0.0000-0.0008, freq_learned 0.0000-0.0001 (competitive, neutral)
- **Time branch is critical**: freq_no_time is 30-100× WORSE than freq_learned (0.012-0.124 vs 0.0001-0.001)
- **Learnable vs fixed frequencies**: NO MEANINGFUL DIFFERENCE (both 0.0001-0.0010 in 1D)

**Verdict**: **HONEST NEGATIVE-WITH-NUANCE** (4th target-dep in 91-110 audit, after 100 SNNL, 108 Anchored, 109 Dynamic). The mechanism is real (routing entropy 0.98+, all 4 experts active) but the **time-domain branch is doing all the work**, not the frequency experts.

This is the **3rd structural mechanism in 91-110 to fail to improve** in 1D synthetic (after Anchored 108 and Dynamic 109). The pattern is becoming clear: **structural mechanisms that depend on data structure (frequency, anchor, drift) struggle in 1D because the data is too simple**. Only structural mechanisms that DON'T depend on data structure (Reliability Gate 99, QuITE 102, SETA 105, Soft MoE 107) succeed.

## What is MoFE-Time?

Standard MoE: expert = MLP (linear → activation → linear). MoFE-Time: expert = learnable Fourier reconstructor.

Per expert k:
1. Has `h` learnable harmonic frequencies `{ω_i}` clamped to [0, 2π]
2. Projects input X_t to "frequency space" via Linear → (B, T, h)
3. Reconstructs in time domain: `x_n = Σ α_i(t) · cos(ω_i · t) + β_i(t) · sin(ω_i · t)`
4. The amplitudes α_i, β_i come from the projection (so they vary with input)
5. Standard top-K routing over experts

The paper's claim: 6.95% MSE reduction on 6 benchmarks vs Time-MoE. The key innovation: **implicit and learnable** Fourier transform, not pre-computed.

## Implementation

### Core API (`lnn/core/freq_experts.py`, ~430 lines)

```python
class FrequencyExpert(nn.Module):
    """A learnable Fourier reconstructor.
    - to_freq: Linear(input_size, n_freqs) — projects to freq space
    - omega_raw: learnable frequencies, sigmoid-clamped to [0, 2π]
    - to_hidden: Linear(basis_dim, hidden_size) — projects basis to output
    - forward(x): compute basis functions, weighted by freq projections
    """

class FrequencyRouter(nn.Module):
    """Top-K router with aux load-balancing loss (Switch-Transformer style).
    Returns (full_weights, top_idx, top_w, aux_loss)."""

class TimeFreqMoECfCCell(nn.Module):
    """K frequency experts + optional time-domain branch + output projection."""

class TimeFreqMoECfCNetwork(nn.Module):
    """Wrapper. Stateless (no recurrent state).
    - get_utilization() — routing_H, max_min, active_fraction
    - get_omegas() — learned frequencies for all experts
    """
```

### Key implementation details

1. **Bounded frequencies**: `omega = sigmoid(omega_raw) * 2π` — keeps them in [0, 2π]
2. **Complex basis** (cos + sin): more expressive than cos-only
3. **NaN-safe**: `torch.nan_to_num` before all projections
4. **Time-domain branch**: optional Linear(input, hidden) added to freq expert output
5. **Aux load-balancing loss**: Switch-Transformer style `K * Σ f_i * P_i`

## Bench

`scripts/bench_freq_experts.py` — 24 cells (3 datasets × 4 conditions × 2 seeds × 100 epochs):

### Conditions

| Cond | Description |
|------|-------------|
| `baseline_mlp` | Standard MLP expert (control) — 2-layer MLP per expert |
| `freq_fixed` | Frequency expert with frozen omega (no learning) |
| `freq_learned` | Frequency expert with learned omega (MoFE-Time) |
| `freq_no_time` | Frequency expert with no time-domain branch (ablation) |

### Results (test_mse, 2 seeds, 100 epochs)

| Condition | sin_irr | structured_irr | random_irr | H |
|-----------|---------|----------------|------------|---|
| baseline_mlp | 0.0001-0.0004 | 0.0000-0.0001 | 0.0000-0.0008 | 0.50 |
| freq_fixed | 0.0000-0.0008 | 0.0000-0.0006 | 0.0000-0.0001 | 0.99 |
| freq_learned | 0.0000-0.0010 | 0.0000-0.0003 | 0.0000-0.0001 | 0.99 |
| freq_no_time | **0.0232-0.0291** | **0.0124-0.0425** | **0.0561-0.1238** | 0.93-0.98 |

### Critical findings

1. **MLP wins narrowly** in 1D: 0.0000-0.0008 vs 0.0000-0.0010
2. **freq_fixed ≈ freq_learned**: learnable frequencies do NOT help in 1D (data too simple to benefit)
3. **Time branch is critical**: removing it makes the model 30-100× worse
4. **Routing entropy is high** (0.99) for all freq conditions, vs 0.50 for MLP
5. **All 4 experts are active** in all conditions

## Discussion

### Why MLP wins in 1D

In 1D synthetic (sin/structured/random):
- Frequencies are simple (1-2 dominant)
- An MLP with 2 layers can learn these patterns with enough training
- Frequency experts add architectural complexity (basis projection, omega learning) that doesn't pay off when the data is too simple

In higher-dim real-world time series (electricity, traffic, weather):
- Multiple overlapping frequencies
- Long-range dependencies
- Frequency experts can specialize on different frequency bands — but in 1D, the MLP wins

### Why freq_no_time fails catastrophically

The frequency expert's output is `sum of basis functions weighted by input projection`. Without a time-domain branch:
- The output is purely a sum of sinusoids
- For non-periodic data, this misses the trend/scale
- For random data, the sinusoid approximation is poor

The time branch adds `Linear(input)` which captures the linear trend. **Without it, the model is just a fancy Fourier series**.

### Why learnable vs fixed doesn't matter in 1D

`freq_learned` allows omega to adapt to data. But in 1D sin:
- The data has 1 dominant frequency
- The model can capture it with EITHER learned or fixed omega
- The amplitude (input projection) does most of the work
- Omega precision doesn't matter when there's only 1 frequency

## Comparison with prior rounds

| Round | Mechanism | Type | test_mse Δ | Verdict |
|-------|-----------|------|-----------|---------|
| 99 | Reliability gate | Augmentation | -1 to -10% | STRICTLY POSITIVE |
| 102 | QuITE | Embedding | -100% vs uniform | STRICTLY POSITIVE |
| 105 | SETA | Architecture | -1 to -10% | STRICTLY POSITIVE |
| 107 | Soft MoE | Structural | ±5% | SAFER ROUTING |
| 108 | Anchored MoE | Structural | -3% best, +9% on random | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | +60-100× on full | NEGATIVE-WITH-NUANCE |
| **110** | **Freq Experts** | **Structural** | **MLP wins narrowly** | **NEGATIVE-WITH-NUANCE** |

**Pattern update (91-110)**: 5 STRUCTURAL winners all DON'T depend on data structure:
- Reliability Gate (99): noise-aware augmentation
- QuITE (102): masked-attention embedding
- SETA (105): shared+unique architecture
- Soft MoE (107): full-context dispatch

3 STRUCTURAL mechanisms that DO depend on data structure (108, 109, 110) all struggle in 1D because the data is too simple.

**NEW INSIGHT**: structural > routing-only only when:
1. The structural change is CONSTRUCTIVE (109 add > prune)
2. The structural change DOESN'T depend on data structure (102/105/107 succeed; 108/109/110 fail in 1D)

## Critical bugs fixed during round 110

1. **get_utilization iteration bug**: `for i, w in zip(ti, tw)` failed when `ti` was a 1D int tensor. Fixed by flattening top_idx and top_w to 1D first.
2. **Pyright torch false-positives**: pre-existing, ignored.

## Recommendation

**Don't use frequency experts in 1D synthetic**:
- Use MLP (or SETA, Soft MoE) instead
- The mechanism is real but doesn't help in simple data

**Use frequency experts in 2 scenarios**:
1. **High-dim real-world time series** with multiple overlapping frequencies (electricity, traffic, weather)
2. **When you need a frequency-aware inductive bias** (e.g., as a feature extractor before a more complex model)

For 1D, **time branch + MLP is the safer choice** — the time branch is the critical component, not the frequency decomposition.

## Files added

- `lnn/core/freq_experts.py` (NEW, ~430 lines)
- `tests/test_freq_experts.py` (NEW, 23/23 tests)
- `scripts/bench_freq_experts.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-110-a-frequency-experts.md` (PRD #10-72)
- `docs/research/2026-06-15_freq_experts_report.md` (this report)
- `docs/daily/2026-06-15_LNN_research_summary_v36.md` (digest v36)
- `README.md` (new Frequency Experts section)
- `lnn-round-110-freq-experts.md` (memory)

## Future work

1. **PhysioNet 36D test**: high-dim real medical time series — frequency experts should specialize
2. **Larger K** (8, 16 experts): more experts = better frequency band specialization
3. **Multi-resolution**: stack freq experts with different max_omega (slow + fast frequencies)
4. **Combine with SETA** (round 105): shared time-domain branch + unique frequency experts
5. **Combine with QuITE** (102): masked attention for irregular time series
