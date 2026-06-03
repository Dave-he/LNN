---
title: LNN Multimodal TL;DR (v4)
date: 2026-06-03
tags: [LNN, multimodal, TLDR, SOTA, adaptive-freeze, regime, random-window-specific, seed-lucky, v4]
related:
  - "[[LNN_QUICKSTART]]"
  - "[[docs/guides/LNN_MULTIMODAL_DESIGN]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
  - "[[docs/research/2026-06-03_loop_research_report]]"
---

# 🚀 LNN 多模态系统 — TL;DR v4 (30 秒读完)

> **TL;DR (v4)**: 跨 **43 轮** ablation + 多轮 cron session 后,本仓库在 **真实 EMMA rover LOO 数据** 上达到 **adaptive freeze SOTA: single-seed MSE 0.42 (h=96, K=10, seed=42)**,但 round 43 (commit 1bb78af) **refuted** 单一 seed 报告:**5-seed mean = 8.16 ± 6.78** (3 of 4 new seeds 27-37× worse)。**所有 round 26/34/38 SOTA 数字都是 single-seed,均需作 advisory 看待**;**生产可用应走 seed ensemble**。v3 的 SOTA 0.31 (random-window-specific) 在严格 LOO 下甚至 **LOO mean 14.89, 17× 差于 random-window 0.87** — *regime × seed 二重限定* 是头号前提。Round 21 同时给出 **第二 encoder 必须是 Bi-CfC-NAD family** (GRU +3.9% < frozen random +24.5% < LSTM +36.1% ≈ Bi-CfC +35.2%) 的强约束。

## 5 句话核心结论

1. **regime 决定一切**:小预算 (h≤16, ep≤20) → cross_attn 赢 (+50%);大预算 (h≥64, ep≥80) → cross_attn 输 (反而 −755%);regime 翻转是 *convergence-driven* (video_only 接近收敛时, cross_attn 优化负担 > 正则化收益)
2. **新 SOTA: adaptive freeze-after-warmup** (h=64, ep=80, K=40, freeze=audio_only) 拿到 MSE 0.31, 2.8× 优于纯 video_only — **首次跨过 video_only 基准**;**但 SOTA recipe 是 rover-specific, 推到合成 burst / audio=random / 不同 hidden 都会失败**
3. **第二 encoder 必要条件** (任一缺失 → gain 大跌): recurrent + trainable + 输入有变化;**family 选择稳健**: LSTM / vanilla CfC / Bi-CfC-NAD 几乎并列 (+32~+36%);GRU 单 seed 不可靠但多 seed 平均 OK
4. **audio 信息内容 ≤ 5pp 贡献** — 跨模态 "信息融合" 的实际作用极小;Bi-CfC-NAD vs vanilla CfC 仅 +2.7pp 区别
5. **hidden ≥ 8** 是 LNN 普遍容量门槛;**hidden=8 在真实数据上有反常曲线** (self-xattn 优于 cross_attn), 合成数据上没有 — task-dependent

## 5 行 production recipe (★) — ★ regime-限定 ★

```python
hidden_size = 64
epochs = 80
warmup_epochs = 40       # 0.5 × total
freeze_targets = "audio_only"  # 冻结 audio_encoder;cross-attn projections 继续更新
# After warmup: requires_grad=False on audio_encoder; rebuild Adam.
```

*期望*: 在 **EMMA rover 滑窗 dataset (n=200, window=16)** 上 **MSE ≈ 0.31** (★ SOTA, ***random-window-specific***)。

## ★ **random-window-specific 警告** (round 35)

0.31 数字是 *random-window dataset* (200 windows from sliding across all 60 frames) 上测的。**当用 `TemporalSegmentRegressionDataset` (4 disjoint 15-frame segments) 做 4-fold LOO 时,video_only baseline 升到 mean 14.89, std 11.18 (vs random-window 0.87 = 17× 差)**。Adaptive-freeze 在 LOO 上仅 +6.3% over video_only (marginal)。

**含义**: 任何未来 LNN 多模态 PR 声称 "新 SOTA" 必须 *同时* 在:
- `EmmaRoverRegressionDataset` (random-window, *段泄漏*) — 报告 *仅作 reference*
- `TemporalSegmentRegressionDataset` (4-fold LOO, *跨段严格*) — **作为主指标**

`TemporalSegmentRegressionDataset` 是 *新 baseline*: 任何"新 SOTA"必须 < 14.89 LOO mean 才有跨段泛化价值。

## ★ regime 限定 (round 32)

- ✅ **适用**: 真实 EMMA rover 数据, h=64, ep=80, audio=normal/zero
- ❌ **不适用**: 合成 burst / h=32 / h≥128 / audio=random — 会灾难性失败 (round 27, 31, 33)
- ⚠️ **需重新调参**: 不同物理系统 / 不同数据 → K=0.5 经验可能不成立
- ⚠️ **random-window vs LOO**: random-window 0.31 *不是* 跨段泛化; 真跨段 mean 14.89 (本节)

## ★ **seed-lucky 警告** (round 43, 1bb78af) — 24th meta-conclusion refinement

Round 38 single-seed 报告的 LOO SOTA 0.42 (h=96, K=10, **seed=42**) **被 5-seed 多 seed 验证推翻**:

| seed | LOO mean | 倍数 vs SOTA | std |
|---:|---:|---:|---:|
| **42** | **0.42** | 1.0× (★ original) | 0.55 |
| 1 | 15.36 | 37× worse | 20.92 |
| 2 | 0.72 | 1.7× | 0.59 |
| 3 | 12.80 | 30× worse | 4.71 |
| 100 | 11.49 | 27× worse | 11.81 |
| **mean** | **8.16** | **19× worse** | **6.78** |

**含义**:
- 任何后续 LNN 多模态 PR 声称 "SOTA" 必须 *同时* 报告 **≥3 seeds 的 mean ± std**;单 seed 报告仅作 advisory
- 生产推荐: **seed ensemble** (K=5 seeds, 推理时均值) — 比 SOTA 单一 seed 更可靠
- 旧 SOTA 数字 (0.31 v3, 0.42 v3.5) **都作 advisory 看待**;它们的 single-seed 本质是缺陷
- 已就位: `scripts/benchmark_multiseed_encoder_families.py` (本轮新增) — 5 seeds × 3 family probe

## CI 强制双 regime 测 (★ §32)

```bash
pytest tests/ -q                  # 默认: small_budget only (140 测试, ~19 秒)
pytest tests/ -q -m large_budget   # 仅 large_budget regime (2 测试, ~80 秒)
pytest tests/ -q -m 'not large_budget'  # 等同默认
pytest tests/ -q                  # 跑全部 142 测试 (~85 秒)
```

任何 LNN 多模态 PR 必须 *通过两 regime 都跑* 才能 merge。

## 必读清单 (新 PR 作者 30 秒内)

1. **`LNN_TLDR.md`** (本文) — 1 页摘要
2. **`LNN_QUICKSTART.md`** — 5 分钟跑 SOTA 教程
3. **`docs/guides/LNN_MULTIMODAL_DESIGN.md`** v3 — 完整设计指南 (241 行, 含决策树 + regime 表 + 失败模式)
4. **`docs/research/2026-06-02_multimodal_physreg_appendix.md`** — 33 轮 ablation 完整历史 (想溯源必读)

## 关键仓库资产

- `lnn/core/multimodal_physreg.py` — 9 个 ablation 模型类 (Multimodal / CrossModalAttn / UniVideo / RegisterToken / VanillaCfC / LSTM / GRU / NonRecurrent / 等)
- `lnn/data/emma_rover_features.py` + `emma_rover_regression.py` — 真实 rover 数据 (零重型依赖)
- `scripts/benchmark_adaptive_freeze.py` — **SOTA recipe 训练流程 (★)**
- `scripts/benchmark_emma_rover.py` + `benchmark_register_token.py` — 两个核心 benchmark
- `scripts/scan_*.py` — 6 个扫描工具 (hidden_size, video channels, budget sweep, GRU recovery 等)
- `analysis/emma_rover/` + `analysis/multimodal_physreg/` — 全部 JSON 数据
- `tests/test_lnn_multimodal_regime.py` — **5 个 regime-conditional 单测 (★ CI 强制)**

## 新 PR 作者的 5 步操作 (含 regime)

```bash
# 1. 跑完整测试 (142 个, ~85 秒, 含 large_budget)
python -m pytest tests/ -q

# 2. 在小预算 + 大预算下分别跑新模型
python scripts/benchmark_emma_rover.py --epochs 20 --hidden-size 16
python scripts/benchmark_emma_rover.py --epochs 80 --hidden-size 64

# 3. 与 5 个 baseline 比较 (video_only / register_token / LSTM / cross_attn)
# 任何声称 "信息融合" 的工作必须 *超过* register_token +27.5%

# 4. (若声明新 SOTA) 必须 *超越* adaptive-freeze recipe baseline (MSE 0.31)
python scripts/benchmark_adaptive_freeze.py --epochs 80 --warmup-epochs 40 --freeze-targets audio_only

# 5. 至少 3 random seeds 报告 mean ± std (避免 §30 GRU 那种 single-seed 异常)
```

## 一句话备忘

> **LNN 多模态系统的最优架构 *不是* 跨模态 attention,而是 *adaptive freeze* 的单流 Bi-CfC-NAD (h=96, ep=80, K=10, freeze=audio_only, **LOO single-seed MSE 0.42 / 5-seed mean 8.16 ± 6.78**) — 比纯 video_only 优 ~2× (mean over seeds);regime 是 hidden_size × epochs × seed 三维空间, 任何 "+X% gain" 报告都 *必须* 注明 regime、seed 集合、且 mean±std;**第二 encoder 必须是 Bi-CfC-NAD family** (GRU +3.9%, LSTM +36.1% 仍 family-internal),random-window 0.31 *不是* 跨段泛化 (真 LOO mean 14.89);SOTA recipe 在 regime 外灾难性失败 — 跨任务迁移需重新调参。**生产推荐: 5-seed ensemble**。**
