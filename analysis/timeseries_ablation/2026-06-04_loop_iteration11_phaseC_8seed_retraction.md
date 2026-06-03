---
title: 2026-06-04 Loop iteration 11 — phase-C 8-seed retraction of iter#10 "CfC wins"
date: 2026-06-04
tags: [LNN, loop, PRD-9, ablation, multiseed, negative-finding, retraction]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 11 — PRD §9 #2 phase-C(8 seed),撤回 iter#10 "CfC 赢" 的庆祝

> `/loop 1h` 第 11 次触发。
> Iter#10 用 N=3 seed 拿到 "CfC −29.1% vs LSTM" 的强结论时,
> 我自己在报告里就写过"N=3 不够 — phase-C 应该 5–10 seed"。
> 本轮兑现该承诺,用 8 个 seed 复测。
>
> **结论反转**: CfC 在 8 seed 上 mean **+15.1%** / median **+116%** vs LSTM。
> 之前的"CfC 赢"是 **N=3 seed 的统计假象**,印证 EMMA agent commits
> `1bb78af` / `b521f86` 反复发现的 "single-seed-is-seed-lucky" 教训扩展到
> "small-N-seed-also-lucky"。**这是科学诚信级的回撤**,直接落到
> [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]] v5 章节。

## 1. 实验配置

与 iter#10 phase-B **完全一致** 的协议,只把 seed 数从 3 扩到 8:

- 数据: `generate_gradual_multi_regime(num_regimes=4, transition_frac=0.15)`
- 模型: hidden=24, 3 blocks, batch=32, **warmup_frac=0.1**, AdamW lr=3e-3
- Seeds: **{42, 7, 123, 2026, 11, 313, 777, 1337}** (8 个)
- Backbone: cfc, ltc, gru, lstm
- 共 4 × 8 = **32 trials**, ~ 18 min wall-clock on Jetson CPU

## 2. 全套 per-seed 原始数据 (test MSE)

| seed | CfC | LTC | GRU | LSTM |
|---:|---:|---:|---:|---:|
| 42 | 0.01920 | 0.09338 | 0.02656 | **0.01695** |
| **7** | **0.73408** | **2.95286** | **1.20729** | **1.11884** |
| 123 | 0.06097 | 0.03735 | 0.00908 | **0.01232** |
| 2026 | 0.04176 | 0.30629 | **0.01199** | 0.03057 |
| 11 | 0.03945 | 0.46645 | 0.04292 | **0.01654** |
| 313 | 0.06103 | 0.09499 | 0.07554 | **0.06439** |
| **777** | **0.70943** | 0.29374 | **0.21213** | **0.19993** |
| 1337 | 0.02335 | 0.00795 | 0.00942 | **0.00816** |

Bold = 最佳列。

### 2.1 Seed 7 和 777 是"数据集硬点",不是 backbone 问题

- seed 7 上 LSTM 都跑到 1.12,GRU 1.21,CfC 0.73,LTC 2.95;
- seed 777 上 LSTM 0.20,GRU 0.21,CfC 0.71,LTC 0.29。

**两个 seed 同时把所有 backbone 的 MSE 拉到平时的 10–100×**。
说明这两个 seed 生成的 4 段 regime layout 极度难预测(可能 freq/amp 相对
范围更大,test set 全在最后一个 regime 上)。**问题在数据集生成的随机性,
不在哪个模型上**。

## 3. 跨 backbone 的鲁棒统计

| backbone | params | mean | **median** ⭐ | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| cfc | 1,921 | 0.21116 | 0.05136 | 0.31558 | 0.01920 | 0.73408 |
| ltc | 1,321 | 0.53163 | 0.19437 | 0.99105 | 0.00795 | 2.95286 |
| gru | 1,969 | 0.19937 | 0.03474 | 0.41286 | 0.00908 | 1.20729 |
| **lstm** | 2,617 | **0.18346** | **0.02376** ⭐ | 0.38329 | 0.00816 | 1.11884 |

### 3.1 mean ranking (受 seed 7 / 777 outlier 拖累)

LSTM (0.183) < GRU (0.199) < CfC (0.211) < LTC (0.532)

### 3.2 median ranking (outlier-resistant)

**LSTM (0.024) < GRU (0.035) < CfC (0.051) < LTC (0.194)**

两种 ranking **结论一致**: LSTM 最准,GRU 次之,CfC 第三,LTC 最差。

### 3.3 与 phase-B (N=3, seeds={42,7,123}) 的对比

| 协议 | seeds | CfC vs LSTM (mean) | 解读 |
|---|---|---|---|
| phase-B | {42, 7, 123} | **−29.1%** ✅ | "CfC 赢" — 庆祝信号 |
| phase-C | {42, 7, 123, 2026, 11, 313, 777, 1337} | **+15.1%** ❌ | 实际平手或微输 |

**为什么 phase-B 那么乐观?** 因为 N=3 时 seed 7 的 CfC MSE 0.73 被 seed 42/123
的 0.02/0.06 略微稀释,LSTM seed 7 的 1.12 拖累更大 → CfC 看起来赢。
加进 seed 777 后(对 CfC 也是 0.71 异常,但对 LSTM 只是 0.20)整体反转。

**这是 N=3 seed 在 high-variance 任务上的典型陷阱。**

## 4. 撤回 iter#10 的庆祝 — 但不必撤掉所有信号

iter#10 的 README/PRD/研读报告 v4 应该都被本轮的 v5 章节"上 disclaimer 但不删"
处理:

- **保留**: 数据生成器 `generate_gradual_multi_regime` 和 `--warmup-frac` flag
  这两个工程产物本身依然有用,**它们没有被 phase-C 反驳**。
- **保留**: 协议讨论(gradual ≠ sharp,warmup vs no-warmup)依然成立。
- **撤回**: "CfC 真正赢 LSTM" 这个 headline。改成 "CfC 在该协议下与 LSTM 持平
  到微输,std 略小但 N=8 仍未能转正"。
- **新增结论**: 仓库累计 11 轮 loop 在合成时序回归任务上 **没有任何 LNN backbone
  跨 8 seed 稳定赢 LSTM**。论文 claim 大概率需要 (a) 真实临床数据 (b) 更长训练
  (c) hidden ≥ 64 才有机会成立。

## 5. 工程教训 — 加进 PRD §6 验证指标

| 教训 | PRD §6 第几条 |
|---|---|
| 任何 "Δ vs baseline" 报告 N=3 seed 不可信,默认 N≥5 | 加 #6 (新增) |
| Δ% 报告必须**同时**写 mean 和 median(outlier 警报) | 加 #7 (新增) |
| std/mean > 1 的结果应自动加 "⚠️ high variance" 标记 | 加 #8 (新增) |
| 数据集生成 seed 必须固定,但用多个 model seed 才能区分 backbone vs data 噪声 | 加 #9 (新增) |

## 6. PRD §8 / §9 进展更新

| § | # | 状态 | 备注 |
|---|---:|:---:|---|
| 8 | 5 | ✅ iter#7/#9/#10/#11 | Comparative v2→v5(本轮 v5 = 8-seed retraction) |
| **9** | **2** | ✅ **iter#10 phase-B + iter#11 phase-C** | 任务全部完成,结论是 CfC 在合成 1200-sample/h=24/8-epoch 上不赢 LSTM,**N≥5 seed required** |
| 9 | 其他 | 待办 | LFM2.5 / LiquidTAD Stage C-true / frozen-encoder / since-last-loop / ONNX-TRT / matrix / CI |

## 7. 下一步候选

| 任务 | 推入 |
|---|---|
| iter#12: phase-D — hidden=64 + ep=50 + samples=4000,看规模带来的 LNN 优势 | PRD §9 #2 phase-D |
| iter#13: 在 phase-C 数据上用 `HierarchicalDecayLiquidTADHead` 替 CfC,看更优 LNN 变体 | PRD §9 #6 衍生 |
| iter#14: 真实 PhysioNet sepsis 子集复现 — 验证论文 clinical claim 的非合成版本 | PRD §9 (新 #9) |

## 8. 参考产物

- 32 trials 原始 JSON: `analysis/timeseries_ablation/2026-06-04_065730_lnn_vs_lstm.json`
- 8-seed MD: `analysis/timeseries_ablation/2026-06-04_065730_lnn_vs_lstm.md`
- 上一轮 phase-B (N=3): [[2026-06-04_loop_iteration10_gradual_warmup_cfc_wins]]
- 之前 v3 (concept_drift): [[2026-06-04_loop_iteration9_prd9_and_concept_drift]]
- 之前 v2 (Mackey-Glass): [[2026-06-04_loop_iteration7_lnn_vs_lstm_v2]]
- 研读 v5 增补: [[Comparative_Analysis_of_LNN_and_LSTM_研读报告]]
- PRD: [[PRD_LNN_Edge_Research]] §6 / §9 #2
