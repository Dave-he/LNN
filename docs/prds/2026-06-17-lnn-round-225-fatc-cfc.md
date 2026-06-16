# PRD #10-187 — Round 225 — Frequency-Adaptive Time Constants on CfC

**Date**: 2026-06-17
**Round**: 225
**Branch**: master
**Audit context (91-224)**: 60 strictly positive + 28 target-dep
+ 59 negatives = 147 mechanism classes.

## Background

13 rounds of spectral processing (r210-r224) explored FFT-based
gating. Question: can the same FFT be used to adapt the core
CfC time constants?

## Goal

Test if coupling the FFT magnitude of the hidden state with
the per-feature time constant τ_eff provides a benefit
beyond fixed time constants.

## Mechanism

```python
H = torch.fft.rfft(h, dim=-1)
mag = torch.abs(H)  # (B, n_freq)
scale_factor = torch.sigmoid(self.freq_to_scale(mag))  # (B, hidden)
mix = torch.sigmoid(self.fatc_mix)
time_scale = (1 - mix) * base + mix * base * scale_factor + 0.1
tau_eff = torch.exp(-f * dt / time_scale)
h_new = tau_eff * g + (1 - tau_eff) * h_branch
```

## Configurations (4 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (4-scale uniform)
3. `adaptive_scale`: r224 (3-branch per-step)
4. `fatc`: r225 (frequency-adaptive time constants)

## Result (24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0877 | 0.0038 | 0.0965 | 0.0627 |
| 4spectralbiasdrop (r216) | 0.0426 | 0.0011 | 0.0838 | 0.0425 |
| adaptive_scale (r224) | 0.0386 | 0.0027 | 0.0833 | 0.0416 |
| **fatc (r225)** | **0.0430** | **0.0009** | **0.0836** | **0.0425** |

Per-dataset (r225 vs cf):
- sin: -51.0% ✓
- structured: -74.9% ✓
- random: -13.4% ✓
- mean: -32.1%

## Verdict

**STRICTLY POSITIVE (61st)** 🎉 — ALL 3 datasets improve vs cf.
Ties r216 (mean +0.1%), has **best structured (0.0009)** of all
spectral variants.

## Pattern (60 + 28 + 59 = 147 → **61 + 28 + 59 = 148**)

- **61 strictly positive (UP from 60, +1)** 🎉
- 28 target-dep (unchanged)
- 59 negatives (unchanged)
- Total: **148 mechanism classes**

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- ~2x slower than cf, ~2x faster than r224

## Lesson

**FATC is strictly positive** — wins all 3 datasets vs cf,
ties r216 on mean, has best structured result of all spectral
variants. The frequency-time coupling is a novel mechanism
dimension.

## Next ideas

1. **FATC + adaptive_scale hybrid** — combine frequency-aware
   time constants with 3-branch scale selection
2. **Phase-aware FATC** — use FFT phase (not just magnitude)
   for time constant adaptation
3. **Per-layer FATC mix** — different mix values per layer
4. **FATC without spectral** — just frequency-aware time
   constants on plain CfC

**Why:** Round 225 is **STRICTLY POSITIVE 61st** — FATC
couples spectral analysis with time constants, wins all 3
datasets vs cf, has best structured result of spectral axis.

**How to apply:** Use FATC when you have clear frequency
content and want best structured performance.