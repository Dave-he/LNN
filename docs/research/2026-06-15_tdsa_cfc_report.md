# Round 152 — Time-Domain Self-Attention CfC (TDSA-CfC)

**Date**: 2026-06-15
**PRD**: #10-114
**Verdict**: **NEGATIVE (21st)** — all 3 variants WORSE on sin (+29-37%),
mixed on structured, neutral on random.

## Summary

Round 152 tests **Time-Domain Self-Attention CfC (TDSA-CfC)** —
a single-head self-attention over the time axis as a parallel
context stream, projected to input_size, then concatenated with x
as input to CfC. Inspired by the Transformer (Vaswani 2017) and
tests whether attention outperforms conv (MSDC 151) for parallel
context on time series::

    # Self-attention over time axis
    q = Linear_q(x)  # [B, T, attn_dim]
    k = Linear_k(x)  # [B, T, attn_dim]
    v = Linear_v(x)  # [B, T, attn_dim]
    attn = softmax(q @ k.transpose(-1, -2) / sqrt(attn_dim))  # [B, T, T]
    c = attn @ v  # [B, T, attn_dim]
    # Project to input_size
    c = Linear_o(c)  # [B, T, D]
    # Concat with x
    aug_x = concat([x, c], dim=-1)  # [B, T, 2D]
    # Standard CfC with augmented input
    h = CfCCell(aug_x, h)

**Verdict**: **NEGATIVE (21st in 91-152 audit)**:

- **tdsa**: sin **+36% WORSE**, structured +4% (worse), random -1%
- **tdsa_2head**: sin **+37% WORSE**, structured -9%, random -1%
- **tdsa_noncausal**: sin **+29% WORSE**, structured -14%, random -2%

All three TDSA variants are significantly WORSE on sin. This is
the FIRST parallel-context mechanism to be uniformly worse on the
sin dataset.

## 1. Hypothesis

- **H1** (Sin data): attention should help periodic data.
  **REJECTED** — all 3 variants are +29-37% WORSE.
- **H2** (Structured data): attention should help regime-change
  data. **MIXED** — tdsa +4% worse, tdsa_2head -9%, tdsa_noncausal
  -14%.
- **H3** (Random data): attention may overfit on noise.
  **REJECTED** — random is essentially neutral (-1 to -2%).
- **H4** (Multi-head vs single): 1 head vs 2 heads. **REJECTED** —
  2-head is essentially identical to 1-head.

## 2. Implementation

`lnn/core/tdsa_cfc.py` (~220 lines) — `TimeDomainSelfAttentionCfCCell` +
`TimeDomainSelfAttentionCfCStackedNetwork`.

Key design choices:

1. **Single-head self-attention** (default): attn_dim = input_size.
2. **Causal masking** (default): upper-triangular mask prevents
   attending to future.
3. **Q/K/V/O linear projections**: standard Transformer.
4. **Concat with x**: aug_x = concat([x, c], dim=-1).
5. **Standard CfC**: takes aug_x as input, h is unchanged.
6. **NaN handling**: zero-fill input before attention.

## 3. Bench results (24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0275±0.0028** | 0.1327±0.0183 | 0.1051±0.0029 | 2545 |
| tdsa | 0.0375±0.0005 | 0.1386±0.0474 | **0.1037±0.0028** | 4521 |
| tdsa_2head | 0.0376±0.0006 | 0.1207±0.0269 | 0.1037±0.0028 | 4521 |
| tdsa_noncausal | 0.0354±0.0036 | **0.1143±0.0121** | 0.1032±0.0027 | 4521 |

**Headline numbers (× change vs baseline)**:

- **tdsa**: sin **+36% WORSE**, structured +4% (worse), random -1%
- **tdsa_2head**: sin **+37% WORSE**, structured -9%, random -1%
- **tdsa_noncausal**: sin **+29% WORSE**, structured -14%, random -2%

## 4. Why self-attention fails on sin and helps slightly on structured

### 4.1 Sin (all variants +29-37% WORSE)

Sin is highly smooth and periodic. The CfC's local recurrence
already captures the local dynamics. Adding self-attention:

- **Pollutes the input**: the attention output c_t is a weighted
  sum of ALL previous x's. For sin, this adds noise (the attention
  weights are learned, but with only 30 epochs they don't have time
  to learn the right pattern).
- **Doubles the input dim**: aug_x has 2D features, so the CfC
  sees a different input distribution than baseline CfC. The CfC's
  gates are pre-trained for D-dim input.
- **Slow training**: attention has more params (Q/K/V/O = 4*D² per
  layer). With only 30 epochs and 32 training sequences, the
  attention is undertrained.

Result: the attention output is essentially noise, polluting the
CfC's input and causing the model to overfit to noise.

### 4.2 Structured (tdsa +4% worse, tdsa_2head -9%, tdsa_noncausal -14%)

Structured has a regime change at t=T/2. The non-causal variant
helps most (-14%) because it can attend to the FUTURE half of the
sequence to detect the regime boundary. The causal variants
(tdsa, tdsa_2head) have to wait until the boundary is past to
detect it, which is too late.

### 4.3 Random (all -1 to -2%, essentially neutral)

Random is cumulative noise. The attention weights don't learn
anything useful, so the output is essentially the mean of x over
all timesteps. This is a slight regularizer (smoothing) but not
significantly different from baseline.

## 5. Why this differs from MSDC 151 (strictly positive 14th)

Both rounds add a parallel context stream to CfC:

- **MSDC 151 (strictly positive)**: multi-scale dilated conv.
  Receptive fields 1, 3, 5. Wins all 3 datasets.
- **TDSA 152 (negative)**: self-attention. Receptive field
  full sequence (with causal mask). Loses on sin.

The crucial difference: **MSDC's conv has bounded, learnable
receptive fields** (1, 3, 5 steps). **TDSA's attention has unbounded
receptive field** but with weights learned from only 30 epochs
of training. For T=32 sequences, the attention can't learn good
weights fast enough — it adds noise to the input.

**Conv > Attention for parallel context on T=32 data.** MSDC's
fixed receptive field is more sample-efficient than attention's
flexible but data-hungry weighting.

## 6. Why this differs from TCC 149 (target-dep 8th)

Both rounds add a parallel 1D conv-or-attention context stream:

- **TCC 149**: single-K 1D conv. Receptive field K. Wins sin
  (K=3) and structured (K=7) but loses noise.
- **TDSA 152**: self-attention. Receptive field full sequence.
  Loses sin and structured.

Conv with bounded receptive field is more sample-efficient than
attention for T=32. Conv doesn't need to learn weights — it just
learns a single filter per output channel.

## 7. NEW INSIGHTS

1. **Self-attention is NOT a good parallel context for T=32
   data** — all 3 variants lose on sin.
2. **Conv > Attention for parallel context** — MSDC's bounded
   receptive field is more sample-efficient than attention's
   flexible but data-hungry weighting.
3. **Non-causal attention helps slightly on structured** (-14%)
   because it can peek at the future to detect regime boundaries.
4. **Multi-head attention is not helpful** — tdsa_2head is
   essentially identical to tdsa. For T=32, 1 head is sufficient.
5. **Pattern reinforced (14 + 9 + 21 = 44 mechanism classes)**:
   - 14 strictly positive (preserves recurrent step + adds structure)
   - 9 target-dep: input-side processing, bidi, SCRN, Time-Decay,
     TCC, LiNo
   - **21 negatives** (was 20): previous 20 + **TDSA (this round)**

**NEW RULE**: AVOID self-attention as parallel context for short
sequences (T≤32). Use conv (MSDC 151) instead. Attention's
flexibility is wasted when there isn't enough data to learn good
weights.

## 8. The 91-152 audit: 44 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip | 134 | STRICTLY POSITIVE (13th winner) |
| Multi-Scale Dilated Conv CfC | 151 | STRICTLY POSITIVE (14th winner) |
| Layer Normalization | 135 | TARGET-DEPENDENT |
| Conv Input Preprocessing | 137 | TARGET-DEPENDENT |
| GLU + Identity Skip | 139 | TARGET-DEPENDENT |
| Decoupled / IndRNN-CfC | 143 | TARGET-DEPENDENT |
| Bidirectional CfC (concat) | 144 | TARGET-DEPENDENT (5th) |
| SCRN-CfC (α=0.5) | 146 | TARGET-DEPENDENT (6th) |
| Time-Decay CfC (γ=0.5) | 148 | TARGET-DEPENDENT (7th) |
| TCC-CfC (K=3/5/7) | 149 | TARGET-DEPENDENT (8th) |
| LiNo-CfC (sum/concat) | 150 | TARGET-DEPENDENT (9th) |
| **Time-Domain Self-Attention CfC** | **152** | **NEGATIVE (21st)** |
| SCRN-CfC (α=0.8/0.95/0.99) | 146 | NEGATIVE (17-19th) |
| Diff Features (diff_only) | 145 | NEGATIVE (16th) |
| Multiplicative Integration | 142 | NEGATIVE (15th) |
| Adaptive Time-Constant | 141 | NEGATIVE (14th) |
| SE Channel Attention | 140 | NEGATIVE (13th) |
| GLU alone | 139 | NEGATIVE (12th) |
| Sinusoidal Time Embedding | 138 | NEGATIVE (11th) |
| Zoneout | 136 | NEGATIVE (10th) |
| Bidirectional CfC (weighted) | 144 | NEGATIVE (15th) |
| Clockwork CfC (K=2/3/4) | 147 | NEGATIVE (20th) |
| LiNo (concat mode on structured) | 150 | NEGATIVE (sub-verdict) |
| LiNo (lin_only) | 150 | NEGATIVE (sanity) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (14 + 9 + 21 = 44 tests)**:

- 14 winners preserve recurrent step + add useful structure
- 9 target-dep: input-side processing, bidi, SCRN, Time-Decay,
  TCC, LiNo
- **21 negatives** (was 20): per-step mods, alternatives,
  regularizers, bottlenecks, redundant info, replacements,
  long-α SCRN, Clockwork partition, **self-attention (this round)**

## 9. Recommendation

**TDSA-CfC is the 21st NEGATIVE in the 91-152 audit.**

- **DO NOT use TDSA-CfC** for time series data. Self-attention
  is not sample-efficient for T=32 sequences.
- **DO use MSDC-CfC** (multi-scale dilated conv) for time series.
  Wins all 3 datasets.
- **DO use conv (TCC, MSDC) NOT attention** for parallel context
  on T≤32 data.
- **Non-causal TDSA is a slight improvement on structured** (-14%)
  but a clear loss on sin (+29%). Not worth the sin regression.

## 10. Critical implementation details

1. **Causal masking** (default): upper-triangular mask prevents
   attending to future.
2. **Q/K/V/O linear projections**: standard Transformer.
3. **attn_dim = input_size** (default): 1 head with attn_dim=D.
4. **NaN handling**: zero-fill input.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.

## 11. Files

- `lnn/core/tdsa_cfc.py` (~220 lines)
- `tests/test_tdsa_cfc.py` (20 tests, all pass)
- `scripts/bench_tdsa_cfc.py` (24-cell bench)
- `results/bench_tdsa_cfc.json`
- `docs/prds/2026-06-15-lnn-round-152-a-tdsa-cfc.md`
- `docs/research/2026-06-15_tdsa_cfc_report.md`
