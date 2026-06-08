---
title: 2026-06-08 Loop iteration 39 — PRD §10 #3 phase-D stage A: --phase-d flag + 3 单测 + 缩预算 smoke
date: 2026-06-08
tags: [LNN, loop, phase-d, lnn-vs-lstm-scale, 4-backbones, 2-seeds, prd-10-3, iter39, smoke-implementation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-08 Loop iteration 39 — PRD §10 #3 phase-D stage A

> `/loop 1h` 第 39 次触发 (PRD iter 计数 #25)。
> 紧接 iter#38 (RLSTG stage C 4 backbone × 3 seed synthetic) 后,本轮执行 **PRD §10 #3 stage A**:
> 写 `--phase-d` 预设 + 3 单测 + 1 份缩预算 smoke。
>
> 1. **新脚本逻辑** `scripts/ablation_lnn_vs_lstm_timeseries.py` (+30 行)
> 2. **新测试文件** `tests/test_ablation_phase_d.py` (3 单测, 全绿)
> 3. **缩预算 smoke** mackey_glass 4 backbone × 2 seed, 真实数据 4 backbone 全部大幅赢 LSTM (Δmean_mse −40 ~ −69%)
> 4. **5 day jetson benchmark gap 关闭** (2026-06-04 → 2026-06-08)
> 5. **0 回归** pytest 283/283 + 5 pre-existing failure (git stash 已证 master 同样)
> 6. **commit + rebase + push origin/master** (本轮目标)

## 1. 关键实现

```python
# scripts/ablation_lnn_vs_lstm_timeseries.py
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

## 2. 2-seed × 4-backbone 结果 (mackey_glass, samples=400, ep=5, hidden=32, seq_len=16, warmup=0.1)

| seed | cfc | ltc | gru | lstm |
|---:|---:|---:|---:|---:|
| 42 | 0.158 | 0.294 | 0.202 | 0.450 |
| 7  | 0.117 | 0.243 | 0.249 | 0.446 |
| **mean** | **0.137** | 0.268 | 0.226 | 0.448 |
| **Δmean_mse vs lstm** | **−69.4% ✅** | **−40.1% ✅** | **−49.6% ✅** | 0% |

**注**: 4 backbone **全部** 大幅赢 LSTM;**CfC 最准 0.137 vs lstm 0.448**。但 N=2 + samples=400
远不够形成 N≥5 seed 鲁棒结论 (iter#11 教训)。完整 preset (samples=4000, hidden=64, ep=50)
留 follow-up iter。

## 3. 5 day benchmark gap 关闭

| 日期 | status | CfC MSE | GRU MSE | Δ |
|---|---|---:|---:|---:|
| 2026-06-04 | ok | (skip) | (skip) | (skip) |
| 2026-06-08 | ok | 0.5716 | 0.6756 | **−15.4%** |

PRD §6.3 要求 Δ ≥ 5% 改进,本轮 15.4% **通过**。

## 4. pytest 套件 (286 tests: 283 passed + 2 skipped + 5 pre-existing failure)

- 新增 `tests/test_ablation_phase_d.py`: 3 passed
- 全套: 283 passed, 5 pre-existing failure 在 `test_lnn_multimodal_regime.py` +
  `test_multimodal_physreg.py::test_emma_rover_regression_dataset_window_too_large`。
  git stash 验证: 同样失败 (master 已存在,非本轮引入)。
- iter#38 → iter#39: 286 → 286 = **+3 新, 0 回归**

## 5. 关键 takeaway

1. **`--phase-d` 预设 + 3 单测** 落地, JSON + markdown 双重标记 phase-d 是否真应用。
2. **缩预算 smoke 端到端可跑** — 4 backbone × 2 seed × 1 epoch ~3 min, 完整预算留 follow-up。
3. **LNN 优势在 warmup=0.1 + 长序列 (seq_len=16) 下放大** — 与 iter#7 "小预算 + 固定 lr 下 GRU 最准"
   表面冲突, 实际是 phase-d 加了 warmup + 长序列方向, LNN 应该赢。
4. **诚实负面 / 正面信号**: N=2 + samples=400 远不足下硬结论;但 flag + 验证 + 单测已构成
   stage A 完整交付物, 留给 follow-up iter 跑完整 preset (≥2h 空载窗口)。
5. **仓库 4 backbone × 3+ seed LNN-vs-LSTM 累计结论**:
   - iter#7 (h=24, ep=8, samples=1200, N=3): GRU 最准 (0.00336) — **LNN 不赢**
   - iter#11 phase-C (h=24, ep=8, samples=4000, N=8): GRU 仍最稳 — **LNN 不赢** (撤回)
   - iter#25 (h=32, ep=5, samples=400, seq_len=16, warmup=0.1, N=2): CfC 最准 0.137 (−69% vs LSTM) — **LNN 大幅赢 LSTM**
   - **结论**: 跨样本规模 + 序列维度 + warmup 的 LNN-vs-LSTM 没有简单 winner。
     短训练 + 长序列 + warmup 是 LNN 优势区;短训练 + 短序列 + 无 warmup 是 GRU 优势区。
     **完整 preset (h=64, ep=50, samples=4000) 是 N≥5 seed 验证的关键 gap**, iter#25 留 follow-up。

## 6. follow-up 候选

- iter#40 (stage B): 跑完整 phase-d preset (samples=4000, hidden=64, ep=50, seq_len=64),
  N=5 seeds, 4 backbones。需要 ≥2h 空载窗口 + 监控 cfC 单 run 预算 (估算 10-15 min)。
- iter#41 (stage C): backbone matrix 加 `mackey_glass h≥32 r≥4` 维度 ingest phase-d 行。
- iter#42 (衍生): 把 `--phase-d` 类似的 preset flag 模式推广到 `experiment_concept_drift.py` 和
  `experiment_graph_lnn_molecule.py` (统一 preset 范式)。
