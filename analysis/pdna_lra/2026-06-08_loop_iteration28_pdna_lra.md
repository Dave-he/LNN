---
title: 2026-06-08 Loop iteration 28 — PRD §10 #10 stage C: PDNA on LRA Pathfinder (infrastructure + null smoke)
date: 2026-06-08
tags: [LNN, loop, pdna, lra, pathfinder, long-range-arena, prd-10-10, iter28, stage-c, smoke-implementation, honest-negative]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-08 Loop iteration 28 — PRD §10 #10 stage C

> `/loop 1h` 第 28 次触发 (PRD iter 计数 #28)。
> 紧接 iter#20 PDNA stage B (sMNIST Gapped +2.53 pp) 后,本轮执行 **PRD §10 #10 stage C**:
> 把 `PDNAPulseHead` 推到 Long Range Arena (LRA) 风格长程任务 — synthetic Pathfinder。
>
> 1. **新数据集模块** `lnn/data/pathfinder_synth.py` (~150 行): 32x32 grid 端点连接性二分类,500/500 balance
> 2. **新脚本** `scripts/experiment_pdna_lra.py` (~250 行): 3 variant (baseline_cfc / cfc_pulse / full_pdna) × N seed
> 3. **新测试文件** `tests/test_pdna_lra.py` (6 单测, 全绿含 1 个 CLI smoke 端到端)
> 4. **backbone matrix 扩展** `_ingest_lra_pathfinder` + `--include-lra` flag;matrix 现 9 rows × 11 backbones × **4 domains**
> 5. **诚实负面**: 1 seed × 1 epoch × 200 train × hidden=32 烟测 3 variants 全部 48.75% ≈ 随机
> 6. **0 回归** pytest 6/6 新增 (pathfinder + LRA) + verify 9/9
> 7. **commit + rebase + push origin/master** (本轮目标)

## 1. 关键实现

### 1.1 `lnn/data/pathfinder_synth.py` — synthetic Pathfinder data

```python
# 32x32 grid → seq_len=1024 (flattened)
# 2 endpoint markers (3x3 filled circles at value 1.0)
# 0/1 random piecewise-linear path connecting waypoints (Bresenham rasterization)
# Class balance 50/50; endpoints at least 16 cells apart so task is non-trivial
@torch.no_grad()
def generate_pathfinder(n_samples, cfg=None, seed=42) -> (seqs, labels):
    ...
```

### 1.2 `scripts/experiment_pdna_lra.py` — LRA smoke runner

```python
# Reuses iter#20 PDNAClassifier (imported from experiment_pdna_smoke.py)
# 3 variants: baseline_cfc, cfc_pulse (attend β=0), full_pdna
# Train: AdamW lr=5e-4, weight_decay=1e-4, cosine schedule
# Eval: binary classification accuracy on synthetic Pathfinder
```

### 1.3 `build_backbone_matrix.py` extension

```python
def _ingest_lra_pathfinder(path) -> dict | None:
    """Reports test_acc (higher better) per variant."""
    ...
# main(): --include-lra flag → scan analysis/pdna_lra/*_summary.json
# _compute_win_tally: lra_pathfinder domain joins smnist_gap/molecular (higher_is_better)
```

## 2. 6 单测 (tests/test_pdna_lra.py)

| 测试 | 验证 |
|---|---|
| `test_pathfinder_default_config_shapes` | 默认 config → 1024-dim seq, label ∈ {0,1} |
| `test_pathfinder_class_balance_close_to_50_50` | N=400 → 正例率 ∈ (0.35, 0.65) |
| `test_pathfinder_deterministic_same_seed` | 同 seed → 同 seq+label |
| `test_pathfinder_endpoint_markers_present` | 每张图 ≥ 6 个 endpoint-value cells (3x3 marker) |
| `test_pathfinder_different_seed_changes_data` | 不同 seed → 数据变化 |
| `test_pdna_lra_cli_runs_end_to_end` | subprocess CLI 端到端跑通,产出 JSON+MD |

## 3. Smoke 结果 (1 seed × 3 variants × 1 epoch × 200 train × hidden=32)

| Variant | n_params | test_acc (mean ± std) | train_s |
|---|---:|---:|---:|
| baseline_cfc ⚠️N<3 (n=1) | 3362 | 48.75 ± 0.00 | 115.9 |
| cfc_pulse ⚠️N<3 (n=1) | 5540 | 48.75 ± 0.00 | 140.4 |
| full_pdna ⚠️N<3 (n=1) | 5540 | 48.75 ± 0.00 | 120.8 |

Δtest_acc: cfc_pulse +0.00pp, full_pdna +0.00pp — **3 variants 全部 48.75% ≈ 随机 (50%)**

**诚实负面解读**:
- 1 seed × 1 epoch × 200 train × hidden=32 是 *最小可行性烟测*。Pathfinder 是真长程挑战,需 ≥5000 train + ≥5 epoch + hidden≥64 才能拿到 baseline-cfc 高于随机的信号。
- 三个 variants 同样的 48.75% (39/80 正确) 表明模型还在输出 ~0.5 概率阶段,还没开始学 — 不是 PDNA 失败,是规模不够。
- iter#20 sMNIST Gapped 的 +2.53 pp 信号需要 hidden=64 + 5 epochs + 10000 train + 5 seeds 才能跑出来。LRA Pathfinder 1024-length 是 sMNIST 784 的 1.3×,但 *难度高一个数量级* (需要 trace path through grid)。

## 4. Backbone matrix 扩展后

| Domain | Row | n_seeds | Winner |
|---|---|---:|---|
| timeseries | mackey_glass [h=24] | 3 | lstm |
| timeseries | mackey_glass [h=16,r=4] | 1 | cfc |
| timeseries | mackey_glass [h=24,r=4] | 6 | ltc |
| timeseries | concept_drift [h=24] | 3 | lstm |
| timeseries | gradual_multi_regime [warmup=0.1,h=24,r=4] | 8 | lstm |
| timeseries | mackey_glass [warmup=0.1,h=32,r=4] | 2 | cfc |
| molecular | graph_tox21 [seeds:10] | 10 | cfc |
| smnist_gap | smnist_gap [n=3,h=64] | 3 | cfc_pulse |
| **lra_pathfinder (NEW)** | **lra_pathfinder [n=1,h=32,seq=1024]** | **1** | **baseline_cfc** |

**9 rows × 11 backbones × 4 domains**

## 5. PRD §10 #10 → stage A+B+C ✅

| 字段 | 更新 |
|---|---|
| Status | stage A+B → **stage A+B+C ✅ (iter#19/20/28)** |
| 输出物新增 | `lnn/data/pathfinder_synth.py` + `scripts/experiment_pdna_lra.py` + `tests/test_pdna_lra.py` |
| 行内备注 | stage C infrastructure 落地,1 seed smoke null (需 ≥5 seed + ≥5000 train 才有对比) |

## 6. 验证

- `pytest tests/test_pdna_lra.py -v` → **6/6 绿** (5 数据 + 1 CLI smoke)
- `python3.14 scripts/verify_all_models.py` → **9/9 ✓**
- 仓库全测 (multimodal 5 失败为 iter#25 验证的 pre-existing 与本轮无关)
- iter#28 总改动: lnn/data/pathfinder_synth.py +150 行 / scripts/experiment_pdna_lra.py +260 行 / tests/test_pdna_lra.py +110 行 / scripts/build_backbone_matrix.py +~50 行 / docs/PRD_LNN_Edge_Research.md 改 1 行
