# Round 276 — STE × Batch Size Sweep — Results

**PRD**: #10-113 · **Date**: 2026-07-03 (completed) · **Session**:
finished as background job during /loop (r278 session)
**Verdict**: **PRODUCTION CONFIRMED** — batch=16 is optimal on the
production-critical structured dataset.

## Context

The (τ, λ, hidden, T, d_in, density) sweep of the STE line was complete
after r275. r276 closes the final hyperparameter axis: **batch size**.
The prior session left this bench 1/45; it was completed as a background
job.

## Results (45 cells: 5 batch sizes × 3 datasets × 3 seeds, 100 epochs)

### Mean test_mse
| batch | toy_sin  | structured | random   |
|-------|---------:|-----------:|---------:|
| 4     | 0.000051 | 0.001762   | 1.071268 |
| 8     | 0.000005 | 0.000855   | 1.004578 |
| **16**| 0.000031 | **0.000171** | 1.002469 |
| 32    | 0.000026 | 0.000734   | 1.003011 |
| 64    | 0.000012 | 0.007419   | 1.003196 |

### Seed variance (std across 3 seeds)
| batch | toy_sin  | structured | random   |
|-------|---------:|-----------:|---------:|
| 4     | 0.000070 | 0.002267   | 0.079221 |
| **16**| 0.000037 | **0.000021** | 0.016894 |
| 64    | 0.000005 | 0.005292   | 0.016841 |

## Hypothesis scorecard

- **H1 (batch=16 optimal on structured)**: ✅ **CONFIRMED** — 0.000171
  is best by 4-43× over all other batch sizes. Production locked.
- **H2 (small batch 4,8 doesn't hurt structured)**: ❌ REJECTED —
  batch=4 is **10× worse** (0.001762), batch=8 is 5× worse (0.000855).
- **H3 (large batch 32,64 ≈ 16 on structured)**: ❌ REJECTED —
  batch=32 is 4× worse, batch=64 is **43× worse** (0.007419).
- **H5 (smaller batch reduces seed variance)**: ❌ REJECTED — batch=16
  has the LOWEST structured variance (0.000021); batch=4 is 100× noisier.

## Interpretation

batch=16 is a genuine sweet spot on the production-critical structured
dataset — both lowest mean error AND lowest seed variance. The
structured task's piecewise-constant segments need enough gradient
updates per epoch (256/16 = 16 updates) to resolve the segment
boundaries; too few (b64 = 4 updates/epoch) underfits badly, too many
(b4 = 64 updates/epoch) injects gradient noise that destabilises the STE
soft mask.

toy_sin and random are batch-insensitive (toy_sin is trivial at any
batch; random is unlearnable at any batch, all ≈1.0).

**Production unchanged**: batch=16 stays. The full STE hyperparameter
sweep (τ, λ, hidden, T, d_in, density, batch) is now complete —
r267-r276.

## Files
- `scripts/bench_ste_batch_size.py` (fixed summary IndexError for
  <3-dataset runs, committed in r277)
- `analysis/ste_batch_size_bench.json` (45 cells, complete)

## Pattern audit
Hyperparameter confirmation (production-locked), no class change.
Closes the STE sweep line; r277-r278 then pivoted to architectural
changes (liquid τ).


## 本仓具体实现路径 (in-house, 合成数据, 2026-09-28 批量化补丁)

### 适配度
- **高**: 已在 r303 落地

### 实施步骤
1. **数据**: 沿用 `lnn/data/timeseries.py`
2. **模型**: 沿用 `lnn/core/ste_cfc.py`
3. **实验**: 已完成, 见 r303 report
**合规边界** (沿用 2026-06-09 用户偏好 critical 级 + AGENTS §约束):
- 仅合成数据 (`lnn/data/<synth>.py` 新建), 不接真机 / ROS / CAN / Modbus / mavlink / BMS / 真实电网
- 仅 in-house 模型 (基于本仓 `lnn/core/` 现有 ODE / CfC / LTC / 守恒 / 蒸馏栈)
- 任何负结果 (rollout fold / F1 < baseline / 长尾塌缩) → 进 `analysis/negative_results/` 而非默认报告
- 严禁触碰 8 条不可重复 claim ([[AGENTS]] §约束), 严禁宣称"AGI / 意识 / SOTA 横扫"

**维护说明**:
- 本实现路径段为 **standardized 模板**, grounding 到本报告核心方法论
- 实施时需按本报告 grounding 数字调整 λ, hidden, solver, seed 等超参
- 一旦实验落地, 把落地结果附在 `analysis/<新域>/<日期>_results.md` 并在本段维护交叉引用
