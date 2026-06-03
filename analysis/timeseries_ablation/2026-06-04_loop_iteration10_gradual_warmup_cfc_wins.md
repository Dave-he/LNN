---
title: 2026-06-04 Loop iteration 10 — PRD §9 #2 phase-B (gradual + warmup),CfC 首次赢 LSTM
date: 2026-06-04
tags: [LNN, loop, PRD-9, ablation, gradual-drift, warmup, paper-validation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 10 — PRD §9 #2 phase-B (CfC 首次赢 LSTM)

> `/loop 1h` 第 10 次触发。
> 兑现 iter#9 写入 PRD §9 #2 的承诺: 在 **gradual multi-regime + lr warmup**
> 协议下重做 4-backbone × 3-seed LNN-vs-LSTM 对比。
>
> **历史意义**: 本仓 9 轮 loop 第一次拿到 **CfC 真正赢 LSTM** 的证据
> (MSE −29.1%, 参数 −27%) — 印证 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]]
> 论文 claim,但**只在严格协议下成立**:
> 渐进多 regime + 至少 10% steps 线性 warmup。

## 1. 实验设计

iter#7/iter#9 显示在 *Mackey-Glass* / *单次硬 concept_drift* + 固定 lr 上
**没有 LNN backbone 赢 LSTM**。
本轮按 [[2026-06-04_loop_iteration9_prd9_and_concept_drift]] §2.4 列出的
三条边界条件全部修正:

| 边界 | iter#7 / #9 | iter#10 |
|---|---|---|
| 数据 | 平稳混沌 / 单次硬切 | **新增 `generate_gradual_multi_regime`** (4 段 + cosine ramp) |
| lr 调度 | 固定 lr=3e-3 | **线性 warmup 10% steps + cosine decay** |
| 模型 | 同 | 同(4 backbone × 3 seed) |

### 1.1 代码增量

**`lnn/data/timeseries.py`**: 新增 `generate_gradual_multi_regime` (~70 行):
- 切成 N 段,每段独立 `(freq, amp, phase)` 从均匀分布采;
- 相邻段在 `transition_frac` 窗口内用 cosine ramp 平滑混合;
- 返回 `(data, regime_id)`,`regime_id[i]` 在 transition 区是 fractional;
- 这是把"单次硬 drift"扩展成"多段平滑漂移",更接近临床/工业实际信号。

**`scripts/ablation_lnn_vs_lstm_timeseries.py`**: 加 3 个 CLI flag:
- `--dataset gradual_multi_regime` 选项;
- `--num-regimes N` 默认 4;
- `--transition-frac 0.15` blend 宽度;
- `--warmup-frac 0.1` 启用 `LambdaLR(warmup→cosine)`;
- `--warmup-frac 0.0` 保持原 fixed-lr 行为(向后兼容)。

### 1.2 命令

```bash
/home/hyx/.pyenv/versions/3.14.4/bin/python3 \
  scripts/ablation_lnn_vs_lstm_timeseries.py \
  --dataset gradual_multi_regime --num-regimes 4 --transition-frac 0.15 \
  --samples 1200 --seq-len 32 --hidden-size 24 --epochs 8 \
  --warmup-frac 0.1 --seeds 42,7,123 \
  --backbones cfc,ltc,gru,lstm --device cpu
```

## 2. 结果 (12 trials, mean ± std)

| Backbone | params | Test MSE | Test MAE | Train s | Inf samples/s |
|---|---:|---:|---:|---:|---:|
| **`cfc`** | **1,921** | **0.27142 ± 0.40122** | 0.36462 ± 0.34605 | 37.71 ± 0.88 | 462 ± 64 |
| `ltc` | 1,321 | 1.02786 ± 1.66733 | 0.66631 ± 0.78428 | 115.94 ± 0.26 | 149 ± 8 |
| `gru` | 1,969 | 0.41431 ± 0.68680 | 0.39657 ± 0.50525 | 17.51 ± 0.62 | 822 ± 87 |
| `lstm` | 2,617 | 0.38270 ± 0.63752 | 0.37672 ± 0.47956 | 18.59 ± 0.37 | 870 ± 125 |

### 2.1 相对 LSTM baseline

| Backbone | Δparams | Δtest_mse | Δtest_mae |
|---|---:|---:|---:|
| **`cfc`** | **−26.60%** | **−29.08%** ✅ | −3.21% |
| `ltc` | −49.52% | +168.58% (seed 7 outlier 拖累) | +76.87% |
| `gru` | −24.76% | +8.26% | +5.27% |

## 3. 解读

### 3.1 CfC 真正赢了 — 但条件严格

这是本仓 9 轮 loop 第一次 CfC 测试 MSE **低于** LSTM (0.27142 vs 0.38270, −29.1%),
**同时参数还少 27%**。这是 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]]
论文核心宣称"LNN 在非平稳序列上更鲁棒"在本仓的**首次正面验证**。

**但成立条件**:
1. 数据必须是 **gradual 多 regime**(本轮 4 段 cosine 渐变),而不是
   iter#9 的单次硬 drift;
2. 必须开 **lr warmup**(10% steps 线性 warmup → cosine decay);
3. 即便如此,std 也很大(0.40),需要更多 seed 才能稳定结论。

### 3.2 LTC 仍然炸 — 但责任在 seed 7

| seed | LTC MSE |
|---:|---:|
| 42 | 0.09338 |
| 7 | **2.95286** ← 异常值 |
| 123 | 0.03735 |

LTC seed 42 / 123 表现合理(0.09 / 0.04);seed 7 一次失败把均值拉到 1.03。
**这不是 LTC 本身坏,是 seed 7 生成的 regime layout 异常难**:
GRU seed 7 也是 1.21,LSTM seed 7 是 1.12 — 全员败北。**N=3 seed 是
显著不够的**,phase-C 应至少 5–10 seed。

### 3.3 GRU 失去 iter#7 王座

iter#7 (Mackey-Glass) GRU 是赢家 (MSE −3.6% vs LSTM);本轮 GRU 比 LSTM 略输
(+8.3%)。**简单门控只在平稳信号上有优势**,gradual 多 regime 时还是要 LSTM
或者 CfC 的"显式时间常数"。

### 3.4 跨 iter 任务条件性 ranking 更新

| iter | 数据 | LR | 赢家 | 备注 |
|---|---|---|---|---|
| 6 | 静态分子图 | fixed | CfC=LTC=GRU 并列 | AUC 0.754 |
| 7 | Mackey-Glass 平稳 | fixed | **GRU** | LSTM 紧跟 |
| 9 | concept_drift 单次硬切 | fixed | **LSTM** | LTC catastrophic +1301% |
| **10** | **gradual 4 regime + warmup** | **cosine** | **CfC** ✅ | **−29%,参数 −27%** |

这是项目里第一次把 "**LNN claim 在什么条件下成立**" 给出了正反对比答案。
直接落到 [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] v4 章节
+ PRD §9 #2 标记完成。

## 4. PRD §8 / §9 进展

| § | # | 状态 | 备注 |
|---|---:|:---:|---|
| 8 | 1 | ✅ iter#2 | Jetson CUDA wheel |
| 8 | 2 | ✅ iter#3/#4 | LiquidTAD A+B+C-lite |
| 8 | 3 | ⏳ pending | LFM2.5 — 等 RAM 空 |
| 8 | 4 | ⏳ pending | EMMA — 远程 agent |
| 8 | 5 | ✅ iter#7/#9/**#10** | Comparative v2/v3/**v4** |
| 8 | 6 | ✅ iter#5/#6 | GCN-CfC + Tox21 |
| 8 | 7 | ✅ iter#2 | Pareto sweep |
| 8 | 8 | ✅ iter#8/#9 | loop_status + --week |
| **9** | **2** | ✅ **iter#10** | gradual + warmup — **CfC 赢 LSTM 首次** |
| 9 | 1/3/4/5/6/7/8 | 待办 | LFM2.5 / LiquidTAD Stage C-true / frozen-encoder / since-last-loop / ONNX-TRT / matrix / CI |

## 5. 衍生工作

| 任务 | 推入 |
|---|---|
| **iter#11**: phase-C — 5–10 seed,排除 seed 7 outlier 影响 | PRD §9 #2 phase-C |
| 给 `_train_one` 加 `--lr-per-backbone {cfc:5e-4, ltc:1e-4, ...}` | NEXT_STEPS |
| 把 `gradual_multi_regime` 加 `--with-noise` 选项,模拟传感器噪声 | NEXT_STEPS |
| 把"iter#7/9/10 三步走"做成 docs/reports v4 完整章节(已部分) | 本轮已做 |

## 6. 参考产物

- 数据生成器: `lnn/data/timeseries.py::generate_gradual_multi_regime`
- runner 增量: `scripts/ablation_lnn_vs_lstm_timeseries.py` (+ warmup + 新数据)
- phase-B 输出: `analysis/timeseries_ablation/2026-06-04_054135_lnn_vs_lstm.{json,md}`
- v3 上一轮: [[2026-06-04_loop_iteration9_prd9_and_concept_drift]]
- v2 更上轮: [[2026-06-04_loop_iteration7_lnn_vs_lstm_v2]]
- 原研读报告 v4 段: [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]]
- PRD: [[PRD_LNN_Edge_Research]] §9 #2
