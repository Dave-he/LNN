# PRD #10-107 — Difference Features CfC (Round 145)

**Date**: 2026-06-15
**Round**: 145
**Verdict target**: TARGET-DEPENDENT (6th) or HONEST NEGATIVE (16th) or STRICTLY POSITIVE (14th)

## 1. Motivation

The 91-144 audit shows a clear pattern: **5 of 5 target-dep winners are input-side
processing** (LN 135, conv 137, GLU+skip 139, decoupled/IndRNN 143, bidi_concat 144).
None are recurrent-step modifications, none are alternative recurrent cores.

The simplest untested input-side processing in classical time series is **finite
differences** (Box-Jenkins 1976, Hamilton 1994 "Time Series Analysis"):
- Δx_t = x_t - x_{t-1} (1st order)
- Δ²x_t = x_t - 2·x_{t-1} + x_{t-2} (2nd order)

These are textbook features for ARIMA models. They:
- Expose **local slope** (Δx) and **local curvature** (Δ²x)
- Are **parameter-free** (no learnable weights)
- Are **rotationally/translationally invariant** (don't depend on absolute value)
- Are **scale-normalizing** (smooth data has small Δ, random walk has large Δ)

## 2. Hypothesis

**H1** (Smooth data): Δx helps because sin(t) has predictable slopes.
- `sin_irr` test_mse should drop from baseline 0.0094 → <0.007.

**H2** (Structured data): Δx helps regime boundaries (slope change at boundary).
- `structured_irr` test_mse should drop from 0.0053 → <0.003.

**H3** (Random data): Δx HURTS because random walk has unpredictable slopes.
- `random_irr` test_mse should not regress by >10%.

**H4** (Combined): [x, Δx, Δ²x] helps smooth, neutral on others.
- 3x input dim, but no extra learnable params in the projection (we use a single Linear(D*3 → hidden)).

## 3. Implementation plan

`lnn/core/diff_cfc.py` — wraps a CfC cell/network with a DifferenceInputEncoder:

```python
class DifferenceInputEncoder(nn.Module):
    """Pre-computes Δx, Δ²x and concatenates with x.

    Args:
        input_size: original D.
        n_diff: number of finite differences (0=none, 1=Δx, 2=Δx+Δ²x).
        mode: "concat" ([x, Δx, ...]) or "diff_only" ([Δx, Δ²x]).
    """
    def __init__(self, input_size: int, n_diff: int = 1, mode: str = "concat"):
        self.input_size = input_size
        self.n_diff = n_diff
        self.mode = mode
        # Output dim
        if mode == "concat":
            self.output_size = input_size * (1 + n_diff)
        else:  # diff_only
            self.output_size = input_size * n_diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # Build n_diff difference features via cumulative subtraction
        feats = [x] if self.mode == "concat" else []
        prev = x
        for _ in range(self.n_diff):
            # Δx_t = x_t - x_{t-1} (Δx_0 = 0 or x_0)
            dx = torch.zeros_like(x)
            dx[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            feats.append(dx)
            prev = dx
        if self.mode == "diff_only":
            return torch.cat(feats, dim=-1)  # [B, T, n_diff * D]
        return torch.cat(feats, dim=-1)  # [B, T, (1+n_diff) * D]
```

Then `DiffCfCNetwork` wraps `CfCNetwork` and uses the encoder.

## 4. Variants to test

| Cond | Description | Input dim multiplier |
|------|-------------|---------------------|
| cfc (baseline) | Vanilla CfC | 1× D |
| diff_concat_1 | [x, Δx] | 2× D |
| diff_concat_2 | [x, Δx, Δ²x] | 3× D |
| diff_only_1 | [Δx] | 1× D |
| diff_only_2 | [Δx, Δ²x] | 2× D |

## 5. Bench plan

- 5 conditions × 3 datasets × 2 seeds = 30 cells
- 30 epochs, B=8, T=32, hidden=16 (matching round 144 setup)
- 3 datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 6. Risk

- **Param explosion** for `diff_concat_2`: input Linear is D*3 → hidden, so 3x input params.
  CfC baseline has 2545 params; diff_concat_2 may have ~2700 params. Acceptable.
- **Random walk 1.62× regression expected** per the audit pattern (smooth wins, noisy loses).
- **NaN handling**: difference of NaN is NaN. We zero-fill x BEFORE computing differences,
  so the difference features are well-defined.

## 7. Why this is likely TARGET-DEPENDENT

Per round 135 (LN), 137 (conv), 139 (GLU), 143 (decoupled/IndRNN), 144 (bidi_concat):
- All 5 input-side processing winners help smooth/structured and HURT or NEUTRAL on random.
- Difference features are pure input-side processing.
- High prior on TARGET-DEPENDENT verdict (5/5 same class = 100% target-dep).
- LOW prior on STRICTLY POSITIVE (only 1/5 input-side winners was strictly positive, the rest were
  per-step or recurrent — different class).
- LOW prior on NEGATIVE (input-side processing is generally safe).
