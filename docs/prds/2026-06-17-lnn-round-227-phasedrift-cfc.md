# PRD #10-189 — Round 227 — PhaseDriftFATC on CfC

**Date**: 2026-06-17
**Round**: 227
**Branch**: master
**Audit context (91-226)**: 62 strictly positive + 28 target-dep
+ 59 negatives = 149 mechanism classes.

## Background

r225 (FATC) used FFT magnitude for time-constant adaptation.
r226 (PhaseFATC) added phase (cos, sin) to the FATC signal.
Question: does adding FFT *drift* (per-bin diff |H_diff|) provide
additional info that improves over static phase?

PhaseDriftFATC tests if "instantaneous frequency" / "group delay"
(d/dt of spectrum) beats static phase.

## Goal

Test if frequency-drift signal (|H_diff| = |H[i+1] - H[i]|)
added to magnitude improves over mag-only (r225) and
mag+phase (r226) for time-constant adaptation.

## Mechanism

```python
H = torch.fft.rfft(h, dim=-1)  # (B, n_freq)
mag = torch.abs(H)  # (B, n_freq)
# Per-bin drift: H[i+1] - H[i] (complex)
H_diff = H[:, 1:] - H[:, :-1]  # (B, n_freq - 1)
mag_diff = torch.abs(H_diff)  # (B, n_freq - 1) — rotation-invariant
mag_diff_padded = F.pad(mag_diff, (0, 1))  # (B, n_freq)
feat = torch.cat([mag, mag_diff_padded], dim=-1)  # (B, 2*n_freq)
scale_factor = torch.sigmoid(self.freq_to_scale(feat))  # (B, hidden)
mix = torch.sigmoid(self.fatc_mix)
time_scale = (1 - mix) * base + mix * base * scale_factor + 0.1
```

## Configurations (5 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (4-scale uniform)
3. `fatc`: r225 (magnitude-only)
4. `phasefatc`: r226 (mag + cos + sin)
5. `phasedrift`: r227 (mag + |H_diff|)

## Result (30 cells: 5 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0674 | 0.0057 | 0.1038 | 0.0589 |
| 4spectralbiasdrop (r216) | 0.0417 | 0.0015 | 0.0835 | 0.0422 |
| fatc (r225) | 0.0424 | 0.0029 | 0.0845 | 0.0433 |
| phasefatc (r226) | 0.0427 | 0.0012 | 0.0839 | 0.0426 |
| **phasedrift (r227)** | **0.0390** | **0.0035** | **0.0848** | **0.0425** |

Per-dataset (r227 vs cf):
- sin: -42.0% ✓
- structured: -38.4% ✓
- random: -18.2% ✓
- mean: -27.8%

## Verdict

**STRICTLY POSITIVE 63rd** 🎉 — ALL 3 datasets improve vs cf.

**TD nuance vs r226**: 
- Better on sin (-8.6%) — drift helps periodic
- Worse on structured (+183%) — phase helps clean breakpoints
- ~Tied on random (+1%) and mean (~tie)

## Pattern (62 + 28 + 59 = 149 → **63 + 28 + 59 = 150**)

- **63 strictly positive (UP from 62, +1)** 🎉
- 28 target-dep (unchanged)
- 59 negatives (unchanged)
- Total: **150 mechanism classes**

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- ~2-3x slower than cf, similar to r225/r226
- High variance on structured (0.0043 vs 0.0027)

## Lesson

**PhaseDriftFATC is strictly positive** — wins all 3 vs cf.
Drift signal is useful for periodic data (sin) but static
phase is better for clean breakpoints (structured).

**Tradeoff vs r226**: 
- Drift wins on sin (drift captures rate of change)
- Phase wins on structured (phase captures position)

## Next ideas

1. **Combine r226 + r227**: mag + cos + sin + |H_diff| (4*n_freq input)
2. **Drift-only FATC**: drop magnitude, use only |H_diff|
3. **Pivot to non-spectral axis** — 16 SPs is a lot

**Why:** Round 227 is **STRICTLY POSITIVE 63rd** (with TD nuance
vs r226) — PhaseDriftFATC adds |H_diff| to magnitude, wins all 3
vs cf, better than r226 on sin but worse on structured.

**How to apply:** Use PhaseDriftFATC for periodic/sinusoidal data.
Use r226 (PhaseFATC) for structured data with clean breakpoints.
Use r216 (4spectralbiasdrop) for best overall mean.
