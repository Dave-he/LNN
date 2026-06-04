# BiCfCEnsemble Quickstart Recipe

The `BiCfCEnsemble` class is the v15 FINAL production recipe for LNN cross-modal tasks, encoded as a reusable class. It packages the 65+ round ablation program's findings (rounds 56, 65, 67, 70, 71) into a single import.

## What BiCfCEnsemble does

In 4 lines of code, you can:

1. **Train 30 models** with different random seeds
2. **Rank them by validation MSE** (smart selection)
3. **Ensemble the top 20** predictions
4. **Get an honest LOO MSE** of ~0.24 (47× better than single-seed)

## Why this works

The recipe is the FINAL convergence point of 65+ rounds of ablation:

| Round | Finding |
|---|---|
| 56 | `freeze=audio_only` is the best freezing strategy |
| 65 | 30 seeds + K=20 by val = 0.24 honest LOO (NEW BEST) |
| 67 | 30 seeds is the FINAL sweet spot (40+ does NOT improve) |
| 70 | BiCfCEnsemble FULL reproduction = 0.24 (validated) |
| 71 | v15 recipe generalizes to vanilla_cfc (Bi-CfC still 20× better) |

All these findings are encoded in the class defaults. Any future PR can use `BiCfCEnsemble` directly without re-running 65+ rounds of ablation.

## Quickstart (5 minutes)

```python
from lnn.core.ensemble import BiCfCEnsemble
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

# 1. Load EMMA rover dataset (or your own)
ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
tl_full, te = create_segment_loo_dataloaders(ds, held_out_fold=0, batch_size=8)
train_dataset = tl_full.dataset

# 2. Instantiate BiCfCEnsemble with v15 defaults
ensemble = BiCfCEnsemble()  # 30 seeds, K=20, h=96, ep=80, etc.

# 3. Train 30 seeds (this takes ~5 min on CPU)
ensemble.fit(train_dataset)

# 4. Evaluate
metrics = ensemble.evaluate(te)
# Returns: ensemble_mse, per_seed_mean_mse, per_seed_std_mse

# 5. Predict
preds = ensemble.predict(te)  # shape [N_samples, output_size]
```

**Expected result**: `ensemble_mse ≈ 0.24` (verified round 70 reproduction).

## Full reproduction (~25 minutes)

For the FULL 30-seed K=20 reproduction that validates the v15 recipe:

```bash
python scripts/probe_bicfc_30seed_reproduction.py
```

This runs 120 fold runs (30 seeds × 4 folds) and prints the per-fold breakdown plus aggregate.

## Quickstart examples

`examples/quickstart_bicfc_ensemble.py` contains three runnable examples:

1. **v15 default (30 seeds, K=20)**: full production recipe (~5 min)
2. **Budget-constrained (5 seeds, K=2)**: 6× faster, slightly higher MSE
3. **Predict-only (3 seeds, K=2, tiny model)**: demonstration of the API

```bash
python examples/quickstart_bicfc_ensemble.py
```

## Customizing the recipe

The default args match the v15 FINAL recipe. Override any of them:

```python
ensemble = BiCfCEnsemble(
    n_seeds=50,                    # try larger pool (NOT recommended, round 67)
    K=30,                          # try larger K
    hidden_size=128,               # try larger h
    epochs=100,                    # try longer training
    warmup_epochs=50,              # try longer warmup
    phase2_inject_sigma=0.15,      # try different inject
    freeze="audio_only",           # recommended
    val_frac=0.20,                 # 80/20 train/val
    lr=5e-3,
)
```

⚠️ **Caveat**: changing the defaults may give *worse* results. The defaults are the FINAL optimal from 65+ rounds of ablation.

## API

### `BiCfCEnsemble(...)`

Constructor with v15 defaults. All parameters are keyword arguments.

| Param | Default | v15 source |
|---|---|---|
| `n_seeds` | 30 | round 67 (FINAL) |
| `K` | 20 | round 65 (optimal) |
| `hidden_size` | 96 | round 38 (SOTA) |
| `epochs` | 80 | round 25-26 |
| `warmup_epochs` | 40 | round 25-26 (half) |
| `phase2_inject_sigma` | 0.10 | round 54-65 |
| `freeze` | "audio_only" | round 56 |
| `val_frac` | 0.20 | round 64-65 |
| `lr` | 5e-3 | standard |

### `ensemble.fit(train_dataset, seed_values=None)`

Train the ensemble.

- `train_dataset`: a `torch.utils.data.Dataset`
- `seed_values`: list of int (length ≥ n_seeds), or None (uses `range(1, n_seeds+1)`)

Returns `self`.

### `ensemble.predict(test_loader)`

Predict on a test DataLoader by averaging top-K models.

Returns: `torch.Tensor` of shape `[N_samples, output_size]`.

### `ensemble.evaluate(test_loader)`

Compute ensemble metrics.

Returns: `dict` with:
- `ensemble_mse`: K-model ensemble MSE on test set
- `per_seed_mean_mse`: mean of per-seed test MSEs
- `per_seed_std_mse`: std of per-seed test MSEs
- `n_seeds`: number of seeds trained
- `K`: number of models in ensemble

## Where to find more info

| File | What it has |
|---|---|
| `lnn/core/ensemble.py` | The class implementation (270 lines) |
| `tests/test_ensemble.py` | 15 unit tests (CI protection) |
| `scripts/probe_bicfc_30seed_reproduction.py` | 30-seed K=20 reproduction (~25 min) |
| `docs/research/2026-06-04_bicfc_ensemble_class_report.md` | Class implementation report |
| `docs/research/2026-06-04_bicfc_ensemble_unit_tests_report.md` | Unit tests report |
| `docs/research/2026-06-04_bicfc_30seed_reproduction_report.md` | Full reproduction report (0.24 validated) |
| `docs/research/2026-06-04_vanilla_cfc_30seed_report.md` | Cross-model validation (vanilla_cfc 4.97) |

## Why Bi-CfC + v15 recipe is the FINAL production expectation

- **Recipe generalizes**: works for vanilla_cfc too (4.97, -76% vs per-seed mean)
- **Bi-CfC is 20× better** than vanilla_cfc (0.24 vs 4.97) because of:
  - **Noise-adaptive EMA gate**: handles input noise
  - **Bidirectional context**: sees both past and future
  - **Independent noise aggregation**: each direction
- **47× better than single-seed baseline** (0.24 vs 11.63)
- **Reproducible across 4 folds**: variance < per-seed variance

## What to do if your MSE is higher

1. **Check your data**: BiCfCEnsemble assumes `train_dataset[i]` returns `(batch_dict, target_dict)` with `batch_dict["video"]` (B, T, 3), `batch_dict["audio"]` (B, T, 1), and `target_dict["params"]` (B, output_size=5). For the EMMA rover dataset, this is the case.
2. **Try adjusting `freeze`**: try `"none"` (no freezing) — usually worse for production, but may help on some data.
3. **Check your val_frac**: too small → unreliable val ranking; too large → less training data.
4. **Run a sanity check**: with `n_seeds=1, K=1`, you should get a single-seed MSE matching what you had before. If not, there's a data format issue.

## Examples in this repository

- `examples/quickstart_bicfc_ensemble.py` — three runnable examples
- `scripts/probe_bicfc_30seed_reproduction.py` — full 30-seed reproduction
- `scripts/probe_bicfc_ensemble_reproduction.py` — 10-seed K=5 reproduction (faster)

## Next steps

1. **Run the quickstart** (`python examples/quickstart_bicfc_ensemble.py`)
2. **Run the full reproduction** (`python scripts/probe_bicfc_30seed_reproduction.py`)
3. **Use BiCfCEnsemble in your own code** with the API above
4. **Report issues** if your MSE doesn't match expected
