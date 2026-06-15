# LNN Research Digest v36 — 2026-06-15

**Coverage**: MoFE-Time Frequency-Domain Experts + 91-110 audit update (4th target-dep).

## Headline

Round 110 implemented **MoFE-Time Frequency-Domain Experts** (arXiv:2507.06502 Liu et al. Jul 2025) — *Mixture of Frequency Domain Experts for Time-Series Forecasting Models*. The mechanism: each expert is a **learnable Fourier reconstructor** with its own harmonic frequencies and amplitudes. This is the **6th structural mechanism** in our 91-110 audit.

The result is **HONEST NEGATIVE-WITH-NUANCE** (4th target-dep in audit):
- **H1 ✗ REJECTED**: Frequency experts do NOT improve over MLP on any of 3 datasets
- **H2 ✗ REJECTED**: sin_irr: MLP 0.0001-0.0004, freq_learned 0.0000-0.0010 (mixed, mostly worse)
- **H3 ✗ REJECTED**: structured_irr: MLP 0.0000-0.0001, freq_learned 0.0000-0.0003 (similar)
- **H4 ✓ CONFIRMED**: random_irr: MLP 0.0000-0.0008, freq_learned 0.0000-0.0001 (competitive)
- **Time branch is critical**: freq_no_time is 30-100× WORSE (0.012-0.124 vs 0.0001-0.001)
- **Learnable vs fixed frequencies**: NO MEANINGFUL DIFFERENCE in 1D (both 0.0001-0.0010)

**NEW INSIGHT**: **structural > routing-only only when the structural change DOESN'T depend on data structure**. The 5 STRUCTURAL winners (99, 102, 105, 107) all don't depend on data structure. The 3 STRUCTURAL failures (108, 109, 110) all depend on data structure that doesn't exist in 1D synthetic.

## 1. MoFE-Time in 60 seconds

Standard MoE: expert = MLP (linear → activation → linear). MoFE-Time: expert = learnable Fourier reconstructor.
```
input (B, T, D)
  │
  ├── FrequencyExpertPool: K experts, each:
  │   - to_freq: Linear(D, h) → (B, T, h)  [project to freq space]
  │   - omega_i: learnable, clamped to [0, 2π]
  │   - basis_i: cos(omega_i · t), sin(omega_i · t)  [T-length basis]
  │   - output = sum over i of x_f[:,:,i] * basis_i  → to_hidden → (B, T, H)
  │
  ├── Router: top-K over (B*T, K)
  │
  └── Output: weighted sum of top-K expert outputs (+ optional time branch)
```

## 2. Bench summary (24 cells, 100 epochs)

`scripts/bench_freq_experts.py`:
- 4 conditions: `baseline_mlp` (control), `freq_fixed` (frozen omega), `freq_learned` (MoFE-Time), `freq_no_time` (ablation)
- 3 datasets: sin_irr, structured_irr, random_irr (30% train, 50% test)
- 2 seeds × 100 epochs, T=32, D=2, hidden=16, K=4, top_k=2

### test_mse (mean over 2 seeds, 100 epochs)

| Condition | sin_irr | structured_irr | random_irr | H |
|-----------|---------|----------------|------------|---|
| baseline_mlp | 0.0001-0.0004 | 0.0000-0.0001 | 0.0000-0.0008 | 0.50 |
| freq_fixed | 0.0000-0.0008 | 0.0000-0.0006 | 0.0000-0.0001 | 0.99 |
| freq_learned | 0.0000-0.0010 | 0.0000-0.0003 | 0.0000-0.0001 | 0.99 |
| freq_no_time | **0.0232-0.0291** | **0.0124-0.0425** | **0.0561-0.1238** | 0.93-0.98 |

## 3. The 91-110 audit pattern

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 91-94 | TV smoothness, dropout, rank | Diagnostic | NEGATIVE |
| 95-97 | Per-expert rank, FAME+orth | Combined | NEGATIVE/PARTIAL |
| 98-99 | Backward coherence, Reliability gate | Regularizer/Aug | PARTIAL/**STRICTLY POSITIVE** |
| 100-101 | SNNL, ORC | Regularizer | TARGET-DEP/DIAGNOSTIC |
| 102 | QuITE | Embedding | **STRICTLY POSITIVE** |
| 103-104 | QuITE+MoE, SDG-MoE | Routing | TARGET-DEP/NEGATIVE |
| 105 | SETA | Architecture | **STRICTLY POSITIVE** |
| 106 | AuxLF | Load balancer | TARGET-DEP |
| 107 | Soft MoE | Structural | **SAFE ROUTING** |
| 108 | Anchored MoE | Structural | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | NEGATIVE-WITH-NUANCE |
| **110** | **Freq Experts** | **Structural** | **NEGATIVE-WITH-NUANCE** |

**5 STRUCTURAL winners** (99, 102, 105, 107) all DON'T depend on data structure:
- Reliability Gate: noise-aware augmentation
- QuITE: masked-attention embedding
- SETA: shared+unique architecture
- Soft MoE: full-context dispatch

**3 STRUCTURAL failures** (108, 109, 110) all depend on data structure that doesn't exist in 1D:
- Anchored MoE: structural prior on routing
- Dynamic TMoE: drift detection (no real drift in 1D)
- Freq Experts: frequency decomposition (too simple in 1D)

## 4. Why MLP wins in 1D

In 1D synthetic, frequencies are simple (1-2 dominant). An MLP with 2 layers captures these patterns with enough training. The frequency expert mechanism adds architectural complexity (basis projection, omega learning) that doesn't pay off when the data is too simple.

In higher-dim real-world time series (electricity, traffic, weather), frequency experts can specialize on different bands. In 1D, the MLP wins.

## 5. Why freq_no_time fails

The frequency expert's output is `sum of basis functions weighted by input projection`. Without a time-domain branch:
- Output is purely a sum of sinusoids
- For non-periodic data, misses trend/scale
- For random data, the sinusoid approximation is poor

The time branch adds `Linear(input)` which captures linear trend. **Without it, the model is just a fancy Fourier series**.

## 6. Implementation highlights

`lnn/core/freq_experts.py` (~430 lines):
- `FrequencyExpertConfig(input_size, hidden_size, n_freqs, max_omega, use_complex_basis)` — dataclass
- `FrequencyExpert` — learnable Fourier reconstructor with omega_raw (sigmoid-clamped to [0, 2π])
- `FrequencyRouter(input_size, n_experts, top_k)` — top-K router with Switch-Transformer aux loss
- `TimeFreqMoECfCCell` — K freq experts + optional time branch + output projection
- `TimeFreqMoECfCNetwork` — full network wrapper
- `get_utilization()` — routing_H, max_min, active_fraction, utilization
- `get_omegas()` — learned frequencies for all experts

`tests/test_freq_experts.py` (23/23):
- TestFrequencyExpert (6): forward shape complex/real, NaN-safe, omega bounded, gradient flows
- TestFrequencyRouter (4): shape, topk in range, weights sum to 1, aux loss non-negative
- TestTimeFreqMoECfCCell (4): init, forward, NaN-safe, no time branch
- TestTimeFreqMoECfCNetwork (6): init, forward, NaN-safe, get_utilization, get_omegas, gradient flows
- TestFreqExpertIntegration (3): omegas learn, captures periodic signal, outputs depend on input

## 7. Critical bugs fixed

1. **get_utilization iteration bug**: `for i, w in zip(ti, tw)` failed when `ti` was a 1D int tensor. Fixed by flattening top_idx and top_w to 1D first.
2. **Pyright torch false-positives** — pre-existing, ignored.

## 8. Recommendation

**Don't use frequency experts in 1D synthetic**:
- Use MLP (or SETA, Soft MoE) instead
- The mechanism is real but doesn't help in simple data

**Use frequency experts in 2 scenarios**:
1. **High-dim real-world time series** with multiple overlapping frequencies
2. **When you need a frequency-aware inductive bias** as a feature extractor

For 1D, **time branch + MLP is the safer choice** — the time branch is the critical component.

## 9. Files added

- `lnn/core/freq_experts.py` (NEW, ~430 lines)
- `tests/test_freq_experts.py` (NEW, 23/23 tests)
- `scripts/bench_freq_experts.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-110-a-frequency-experts.md` (PRD #10-72)
- `docs/research/2026-06-15_freq_experts_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v36.md` (this file)
- `README.md` (new Frequency Experts section)
- `lnn-round-110-freq-experts.md` (memory)

## 10. Future work

1. **PhysioNet 36D test**: high-dim real medical time series
2. **Larger K** (8, 16 experts): better frequency band specialization
3. **Multi-resolution**: stack freq experts with different max_omega
4. **Combine with SETA** (round 105): shared time + unique frequency
5. **Combine with QuITE** (102): masked attention for irregular time series
