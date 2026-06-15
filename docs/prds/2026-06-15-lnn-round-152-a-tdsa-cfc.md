# PRD #10-114 — Time-Domain Self-Attention CfC (TDSA-CfC) (Round 152)

**Date**: 2026-06-15
**Round**: 152
**Verdict target**: TARGET-DEPENDENT (10th), STRICTLY POSITIVE (15th), or NEGATIVE (21st+)

## 1. Motivation

The 91-151 audit shows 14 strictly positive + 9 target-dep = 23
mechanisms that follow a clear pattern: **preserve x, add parallel
context**. Recent winners:
- **MSDC 151 (strictly positive 14th)**: multi-scale dilated conv,
  receptive fields 1/3/5
- **TCC 149 (target-dep 8th)**: single-K 1D conv
- **LiNo 150 (target-dep 9th)**: linear projection (no receptive field)

What's NOT been tested: **time-domain self-attention** as a parallel
context stream. Self-attention (Vaswani 2017) is the most flexible
context aggregator — it can learn to attend to whichever timestep
is most relevant, unconstrained by local windows (conv) or no
receptive field (linear).

Round 152 tests **Time-Domain Self-Attention CfC (TDSA-CfC)** —
a single-head self-attention over the time axis, output projected
to input_size, then concatenated with x as input to CfC::

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

This is **different from**:
- **MSDC 151 (strictly positive 14th)**: MSDC uses multi-scale conv
  (local windows). TDSA uses self-attention (full sequence).
- **TCC 149 (target-dep 8th)**: TCC uses single-K conv. TDSA uses
  attention.
- **LiNo 150 (target-dep 9th)**: LiNo uses linear projection (no
  receptive field). TDSA uses attention (full sequence).
- **Transformer (Vaswani 2017)**: Pure attention replaces recurrence.
  TDSA ADDS attention as parallel context, KEEPS the recurrent step.

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

TDSA-CfC: self-attention over time axis as parallel context::

    # Self-attention
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

This is a **structural addition** (preserves x, adds attention-based
context).

## 3. Hypotheses

- **H1** (Sin data): attention should help periodic data (can
  attend to distant phases of the period). **EXPECTED: positive.**
- **H2** (Structured data): attention should help regime-change
  data (can attend across the regime boundary). **EXPECTED: positive.**
- **H3** (Random data): attention may overfit on noise. **EXPECTED:
  neutral or negative.**
- **H4** (Multi-head): 1 head vs 2 heads vs 4 heads. **EXPECTED:
  1 head is sufficient for T=32.**

## 4. Implementation

`lnn/core/tdsa_cfc.py` (~150 lines) — `TimeDomainSelfAttentionCfCCell` +
`TimeDomainSelfAttentionCfCStackedNetwork`.

Key design choices:

1. **Single-head self-attention** (default): attn_dim = input_size.
2. **Causal masking**: upper-triangular mask to prevent attending
   to future. This is the "causal self-attention" used in autoregressive
   models.
3. **Linear projections**: q, k, v, o are all nn.Linear.
4. **Concat with x**: aug_x = concat([x, c], dim=-1).
5. **Standard CfC**: takes aug_x as input, h is unchanged.
6. **NaN handling**: zero-fill input before attention.

## 5. Bench

24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs:
- cfc (baseline)
- tdsa (self-attention, 1 head, default attn_dim=8)
- tdsa_2head (self-attention, 2 heads)
- tdsa_residual (attention + x residual, then CfC with augmented input)

## 6. Why this might win (mechanism reasoning)

The audit pattern: input-side processing that PRESERVES x wins.
TDSA preserves x (concat), preserves h (CfC stream is recurrent
step), and adds an attention-based context stream.

Self-attention is the most flexible context aggregator:
- Can learn to attend to specific timesteps (unlike conv's fixed
  receptive field)
- Can model long-range dependencies (unlike conv's bounded window)
- Can adaptively weight context (unlike linear's static projection)

Risks:
- O(T²) memory and compute (manageable for T=32)
- May overfit on T=32 with only 30 epochs (less data than
  conv/linear)
- Attention output is dense (every timestep gets a non-zero
  contribution from all previous)

## 7. Critical implementation details

1. **Causal masking**: upper-triangular mask (-inf) so position t
   attends only to positions 0..t.
2. **attn_dim = input_size** (default): 1 head with attn_dim=D.
3. **Multi-head**: split attn_dim into h heads, compute attention
   per head, concat, project.
4. **NaN handling**: zero-fill input before attention.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.
