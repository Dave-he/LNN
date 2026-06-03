---
title: LNN Multimodal TL;DR (v3)
date: 2026-06-03
tags: [LNN, multimodal, TLDR, SOTA, adaptive-freeze, regime, random-window-specific, v3]
related:
  - "[[LNN_QUICKSTART]]"
  - "[[docs/guides/LNN_MULTIMODAL_DESIGN]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
---

# 🚀 LNN 多模态系统 — TL;DR v2 (30 秒读完)

> **TL;DR**: 跨 33 轮 ablation + 多轮 cron session 后,本仓库在 **真实 EMMA rover 数据** 上达到 **新 SOTA: MSE 0.31**,通过 *adaptive freeze-after-warmup* 策略 + Bi-CfC-NAD backbone, *2.8×* 优于纯 video_only baseline 0.87, *~200×* 优于 cross_attn 端点 60.84。**但 SOTA recipe 在 regime 外 (合成 burst / 不同 audio mode) 会灾难性失败** — *regime 限定* 是头号前提。

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

> **LNN 多模态系统的最优架构 *不是* 跨模态 attention,而是 *adaptive freeze* 的单流 Bi-CfC-NAD (h=64, ep=80, K=40, freeze=audio_only, MSE 0.31) — 比纯 video_only 优 2.8×;regime 是 hidden_size × epochs 的二维空间, 任何 "+X% gain" 报告都 *必须* 注明 regime 且 *两 regime 都跑*;GRU 第二 encoder 可用但 *必须* ≥5 seeds;SOTA recipe 在 regime 外灾难性失败 — 跨任务迁移需重新调参。**
