# PRD #10-108 — Slow Context RNN CfC (Round 146)

**Date**: 2026-06-15
**Round**: 146
**Verdict target**: TARGET-DEPENDENT (6th) or STRICTLY POSITIVE (14th)

## 1. Motivation

The 91-145 audit shows that 5 of 5 target-dep winners are input-side
processing that **PRESERVES** the original input (LN 135, conv 137,
GLU+skip 139, decoupled/IndRNN 143, bidi_concat 144). The 16
negatives include any input-side processing that **REPLACES** the
input (diff_only 145, multiplicative integration 142, etc.).

Round 146 tests a classic "slow context" idea from Mikolov et al.
2015 "LSTM with Working Memory" / "Slow Recurrent Neural Network"
(SCRN): a separate slow context unit that low-pass-filters the
input stream and is concatenated to the hidden state. This is a
**structural addition** that:

1. **Preserves x and h** (per the input-preserving rule from round 145).
2. **Adds a slow context unit** s_t that captures long-term trends
   via an exponential moving average of past inputs.
3. **Has only ONE extra parameter** (α, the EMA decay rate).
4. **Does not modify the recurrent step** (per the per-step modification
   rule from rounds 141/142/144).

## 2. Mechanism

Standard CfC hidden state: `h_t = CfCCell(x_t, h_{t-1})`.

SCRN-CfC adds a parallel slow context stream:

```
s_t = α * s_{t-1} + (1 - α) * (W_s x_t)   (slow context, EMA of input)
h_combined_t = [h_t, s_t]                  (concat hidden + slow)
```

Where:
- `α ∈ [0, 1)` is the EMA decay (init 0.95 = ~20-step memory).
- `W_s` is a Linear(input_size, slow_size) projection (slow_size = hidden_size).
- `s_0 = 0` (no past context at start).

The combined hidden state has dim `hidden_size + slow_size` (2 × hidden).

## 3. Hypothesis

**H1** (Smooth data): slow context captures long-term trend, helps sin_irr.
- sin_irr test_mse should drop from 0.0094 → <0.007.

**H2** (Structured data): slow context captures regime trends.
- structured_irr test_mse should drop from 0.0053 → <0.004.

**H3** (Random data): slow context is a no-op for noise (random walk has no long-term trend).
- random_irr test_mse should not regress by >20% (some loss expected due to 2x hidden dim).

**H4** (Different α): α=0.95 (long memory) helps smooth; α=0.5 (short memory) is closer to baseline.
- Test α ∈ {0.5, 0.8, 0.95, 0.99}.

## 4. Implementation plan

`lnn/core/scrn_cfc.py` — `SlowContextEncoder` + `SCRNCfCCell` + `SCRNCfCStackedNetwork`.

```python
class SlowContextEncoder(nn.Module):
    """EMA-based slow context unit.

    s_t = α * s_{t-1} + (1 - α) * W_s x_t
    """
    def __init__(self, input_size, slow_size, alpha_init=0.95):
        self.proj = nn.Linear(input_size, slow_size)
        # logit-alpha for unconstrained optimization
        self.logit_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def forward(self, x):
        # x: [B, T, D]
        s = torch.zeros(B, slow_size)
        outputs = []
        for t in range(T):
            x_proj = self.proj(x[:, t, :])
            s = self.alpha * s + (1 - self.alpha) * x_proj
            outputs.append(s)
        return torch.stack(outputs, dim=1)
```

Then `SCRNCfCStackedNetwork` wraps CfCNetwork with slow context added.

## 5. Variants to test

| Cond | Description | Params |
|------|-------------|--------|
| cfc (baseline) | Vanilla CfC | 2545 |
| scrn_alpha_05 | α=0.5 (short memory) | ~3000 |
| scrn_alpha_08 | α=0.8 | ~3000 |
| scrn_alpha_095 | α=0.95 (default) | ~3000 |
| scrn_alpha_099 | α=0.99 (long memory) | ~3000 |

## 6. Bench plan

- 5 conditions × 3 datasets × 2 seeds = 30 cells
- 30 epochs, B=8, T=32, hidden=16 (matching round 144 setup)
- 3 datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 7. Risk

- **2x hidden dim downstream** — concat means the cell's input gets 2*hidden
  in subsequent layers, growing params.
- **α optimization** — logit-alpha parameterization avoids sigmoid saturation.
  Init at 0.95 with logit_alpha = log(0.95/0.05) ≈ 2.94.
- **NaN handling** — zero-fill input before slow context (s_t stays bounded).
- **T=32 sequence** — α=0.95 means effective memory is ~20 steps, sufficient for
  our 32-step sequences. α=0.99 gives ~100-step memory but slower adaptation.

## 8. Why this is likely TARGET-DEPENDENT (6th) or POSITIVE (14th)

Per the audit pattern:
- 5 input-side winners all preserve x (LN/conv/GLU/decoupled/bidi_concat)
- SCRN preserves x (the EMA is on W_s @ x_t, not replacing x)
- It adds a slow context unit (structural addition)
- It does NOT modify the recurrent step (h_t = CfCCell as before)
- α=0.95 is the "long memory" sweet spot for smooth/structured data

**Prior**: 
- TARGET-DEPENDENT 50% (5/5 input-side winners, similar pattern)
- STRICTLY POSITIVE 30% (13 winners in audit, but most are not input-side)
- NEGATIVE 20% (16 negatives, but most are per-step or replacements)

## 9. Mechanism notes

- **EMA (Exponential Moving Average)** is a classical signal processing tool
  (Oppenheim & Schafer 1975). The "slow context" idea is widely used in
  signal processing for trend extraction.
- **SCRN is NOT a new idea** (Mikolov 2015) but its application to CfC
  is novel. The audit's question is whether the slow context EMA provides
  useful long-term memory for CfC.
- **Key invariant**: the slow context NEVER replaces the hidden state. It
  is concatenated to the hidden, which means the recurrent step is unchanged.
