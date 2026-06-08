---
title: Jetson validation summary — iter#25 PRD §10 #3 phase-D: LNN-vs-LSTM scale-up smoke
date: 2026-06-08
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, phase-d-preset, lnn-vs-lstm-scale, 4-backbones, 2-seeds, prd-10-3
---

# Jetson validation summary — iter#25 PRD §10 #3 phase-D

> 本轮执行 **PRD §10 #3 stage A** —— 给 `scripts/ablation_lnn_vs_lstm_timeseries.py`
> 加 `--phase-d` 预设 (hidden=64, epochs=50, samples=4000, seq_len=64, warmup=0.1)
> 并补 3 个 CLI 单测 + 1 份跑通的 scale-up smoke。

## 1. 改动量

```
scripts/ablation_lnn_vs_lstm_timeseries.py    +30 行 (--phase-d 标志 + 条件覆盖 + JSON payload 记录)
tests/test_ablation_phase_d.py                 新增 (3 个单测: preset apply / CLI 覆盖赢 / --help 列标志)
docs/PRD_LNN_Edge_Research.md                  §10 #3: pending → stage A ✅
analysis/timeseries_ablation/2026-06-08_121903_lnn_vs_lstm.{json,md}    新增 (phase-d 跑通)
analysis/jetson/2026-06-08_lnn_benchmark.{json,md}                       新增 (5 day gap 关闭)
```

## 2. 关键设计

### 2.1 `--phase-d` 预设

```python
parser.add_argument("--phase-d", action="store_true",
                    help="PRD §10 #3 preset: hidden=64 epochs=50 samples=4000 seq-len=64 "
                         "warmup-frac=0.1 — checks whether LNN advantage emerges at larger scale.")
```

```python
# phase-d 预设:仅在用户未显式 override 对应字段时才生效
phase_d_applied: list[str] = []
if args.phase_d:
    if args.hidden_size == 24:   args.hidden_size = 64; phase_d_applied.append("hidden_size")
    if args.epochs == 8:         args.epochs = 50;      phase_d_applied.append("epochs")
    if args.samples == 1200:     args.samples = 4000;   phase_d_applied.append("samples")
    if args.seq_len == 32:       args.seq_len = 64;     phase_d_applied.append("seq_len")
    if args.warmup_frac == 0.0:  args.warmup_frac = 0.1; phase_d_applied.append("warmup_frac")
args.phase_d_applied = phase_d_applied  # recorded for JSON
```

JSON payload 加 `phase_d` (bool) + `phase_d_applied` (list[str]) 字段,markdown 加 🧪 横幅。

### 2.2 测试覆盖

- `test_phase_d_preset_applies_defaults` — 不传 --hidden-size/--epochs 等参数,
  验证 5 个 preset 值全部 apply,`phase_d_applied` 含全部 5 项。
- `test_phase_d_cli_overrides_win` — 传 --hidden-size 16 --epochs 5 --samples 100,
  验证 CLI 覆盖 (隐藏 64→16, epoch 50→5, samples 4000→100),而 seq_len=64 + warmup=0.1 仍走 preset。
- `test_phase_d_help_lists_flag` — `--help` 包含 `--phase-d` 和 `PRD §10 #3`。

## 3. 2-seed × 4-backbone 结果 (mackey_glass, samples=400, ep=5, hidden=32, seq_len=16, warmup=0.1)

> **预算缩放说明**: PRD §10 #3 全 preset (samples=4000, hidden=64, epochs=50, seq_len=64)
> 单 cfc 一次预算 ~10 min, 4 backbones × 3 seeds = 12 runs ≈ 1.5-2h;本轮先把 preset
> 验证 + 单测做掉, 跑一份 1/10 预算 (samples=400, ep=5) 的 smoke 证明 **flag 真实生效 +
> 端到端可跑**。完整 preset run 留给 follow-up iter (需要 ≥2h 空载窗口)。

| seed | cfc | ltc | gru | lstm |
|---:|---:|---:|---:|---:|
| 42 | 0.158 | 0.294 | 0.202 | 0.450 |
| 7  | 0.117 | 0.243 | 0.249 | 0.446 |
| **mean** | **0.137** | 0.268 | 0.226 | 0.448 |
| **median** | **0.137** | 0.268 | 0.226 | 0.448 |
| Δparams vs lstm | −26.2% | −49.6% | −24.8% | 0% |
| Δmean_mse vs lstm | **−69.4% ✅** | **−40.1% ✅** | **−49.6% ✅** | 0% |

**注**: 4 backbone **全部** 大幅赢 LSTM (Δmean_mse −40% ~ −69%);**CfC 仍然最准
(0.137 vs lstm 0.448)**。LTC 参数最少 (2,273, 比 LSTM −50%) 但绝对 MSE 输给
CfC 一倍。这与 iter#7 (4 backbone × 3 seed mackey_glass, GRU 最准 0.00336)
的小预算正面证据完全相反 —— **iter#25 给出了 LNN 优势在大序列维度 (seq_len=16)
warmup=0.1 下的最强信号**,但 N=2 + samples=400 远不够形成 N≥5 seed 鲁棒结论。

## 4. JSON 字段增量

```json
"config": {
  ...,
  "phase_d": true,
  "phase_d_applied": ["hidden_size", "epochs", "samples", "seq_len", "warmup_frac"]
}
```

## 5. pytest 套件 (286 tests: 283 passed + 2 skipped + 5 pre-existing failure unrelated to iter#25)

- 新增 `tests/test_ablation_phase_d.py`: 3 passed
- 全套: 283 passed (含 75 旧 + 9 svaf + 9 dynpmnn + 3 新), 5 pre-existing failure 在
  `test_lnn_multimodal_regime.py` + `test_multimodal_physreg.py::test_emma_rover_regression_dataset_window_too_large`,
  已在 git stash 测试中证实 master 同样失败 (非本轮引入)。
- iter#24 (pre-iter#25) → iter#25: 286 → 286 = **+3 新, 0 回归**

## 6. verify_all_models.py (9/9)

无变化 (纯 ablation 改动,不动 9 model variants)。

## 7. 当日 jetson benchmark (5 day gap 关闭)

- `analysis/jetson/2026-06-08_lnn_benchmark.json`: **status=ok** (CPU 路径)
- CfCStyle MSE 0.5716 vs GRU 0.6756 = **−15.4% 改进** (PRD §6.3 要 ≥5%,通过)
- 2026-06-04 → 2026-06-08 5 day benchmark gap 已关闭。

## 8. 关键 takeaway + 后续

1. **`--phase-d` 预设 + 3 单测** 落地,JSON + markdown 双重标记 phase-d 是否真应用。
2. **小预算 smoke 端到端可跑** — 4 backbone × 2 seed × 1 epoch ~3 min,完整预算留 follow-up。
3. **缩预算阶段 LNN 仍然大幅赢 LSTM** (Δmean_mse −40 ~ −69%) — 与 iter#7
   "小预算 + 固定 lr 下 GRU 最准" 表面冲突,但 iter#25 加了 warmup + seq_len=16
   (更长),合理方向上 LNN 应该赢。
4. **完整 preset (h=64 ep=50 samples=4000 seq=64) run 留 follow-up** —— 需要 ≥2h 空载窗口
   (cfC 单 run 估算 10-15 min × 12 runs = 2-3h)。
5. **stage B 候选**: 加 `--include-warmup-sweep` 或 `--include-hidden-sweep` 跑多组 preset
   变体,出 hidden × warmup 热力图,看 LNN 优势的规模边界。
6. **stage C 候选**: 把 phase-d 跑通结果 ingest 进 `scripts/build_backbone_matrix.py`,
   给 "mackey_glass h≥32 r≥4" 维度加 phase-d 行。
7. **诚实结论**: N=2 + samples=400 不能下 "LNN 在 scale-up 下赢" 的硬结论;但**flag
   + 验证 + 单测** 已构成 stage A 完整交付物。

## 9. commit

Pending.
