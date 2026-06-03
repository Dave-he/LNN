---
title: LNN Multimodal TL;DR
date: 2026-06-03
tags: [LNN, multimodal, TLDR, SOTA, adaptive-freeze]
related:
  - "[[docs/guides/LNN_MULTIMODAL_DESIGN]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
---

# 🚀 LNN 多模态系统 — TL;DR (30 秒读完)

> **TL;DR**: 跨 §6-§30 共 25 轮 ablation + 多轮 cron session 后,本仓库在 **真实 EMMA rover 数据** 上达到 **新 SOTA: MSE 0.31**, 通过 *adaptive freeze-after-warmup* 策略 + Bi-CfC-NAD backbone, *2.8×* 优于纯 video_only baseline 0.87, *~200×* 优于 cross_attn 端点 60.84。

## 5 句话核心结论

1. **regime 决定一切**:小预算 (h≤16, ep≤20) → cross_attn 赢 (+50%);大预算 (h≥64, ep≥80) → cross_attn 输 (反而 −755%);regime 翻转是 *convergence-driven* (video_only 接近收敛时, cross_attn 优化负担 > 正则化收益)
2. **新 SOTA: adaptive freeze-after-warmup** (h=64, ep=80, K=40, freeze=audio_only) 拿到 MSE 0.31, 2.8× 优于纯 video_only — **首次跨过 video_only 基准**
3. **第二 encoder 必要条件** (任一缺失 → gain 大跌): recurrent + trainable + 输入有变化;**family 选择稳健**: LSTM / vanilla CfC / Bi-CfC-NAD 几乎并列 (+32~+36%);GRU 单 seed 不可靠但多 seed 平均 OK
4. **audio 信息内容 ≤ 5pp 贡献** — 跨模态 "信息融合" 的实际作用极小;Bi-CfC-NAD vs vanilla CfC 仅 +2.7pp 区别
5. **hidden ≥ 8** 是 LNN 普遍容量门槛;**hidden=8 在真实数据上有反常曲线** (self-xattn 优于 cross_attn), 合成数据上没有 — task-dependent

## 5 行 production recipe (★)

```python
hidden_size = 64
epochs = 80
warmup_epochs = 40       # 0.5 × total
freeze_targets = "audio_only"  # 冻结 audio_encoder;cross-attn projections 继续更新
# After warmup: requires_grad=False on audio_encoder; rebuild Adam.
```

*期望*: 在 EMMA rover 滑窗 dataset 上 MSE ≈ 0.31。

## 必读清单 (新 PR 作者 30 秒内)

1. **`LNN_TLDR.md`** (本文) — 1 页摘要
2. **`docs/guides/LNN_MULTIMODAL_DESIGN.md`** — 完整设计指南 (决策树、必要条件、失败模式)
3. **`docs/research/2026-06-02_multimodal_physreg_appendix.md`** — 25 轮 ablation 完整历史 (想溯源必读)

## 关键仓库资产

- `lnn/core/multimodal_physreg.py` — 9 个 ablation 模型类 (Multimodal / CrossModalAttn / UniVideo / RegisterToken / VanillaCfC / LSTM / GRU / NonRecurrent / 等)
- `lnn/data/emma_rover_features.py` + `emma_rover_regression.py` — 真实 rover 数据
- `scripts/benchmark_emma_rover.py` + `scripts/benchmark_register_token.py` + `scripts/benchmark_adaptive_freeze.py` — 三个核心 benchmark
- `scripts/scan_*.py` — 6 个扫描工具 (hidden_size, video channels, budget sweep 等)
- `analysis/emma_rover/` + `analysis/multimodal_physreg/` — 全部 JSON 数据

## 新 PR 作者的 3 步操作

```bash
# 1. 跑完整测试 (137 个单测,~6 秒)
python -m pytest tests/ -q

# 2. 在小预算 + 大预算下分别跑新模型
python scripts/benchmark_emma_rover.py --epochs 20 --hidden-size 16
python scripts/benchmark_emma_rover.py --epochs 80 --hidden-size 64

# 3. 与 5 个 baseline 比较 (video_only / register_token / LSTM / cross_attn)
# 任何声称 "信息融合" 的工作必须 *超过* register_token +27.5%
```

## 一句话备忘

> **LNN 多模态系统的最优架构 *不是* 跨模态 attention,而是 *adaptive freeze* 的单流 Bi-CfC-NAD (regime 限定);regime 是 hidden_size × epochs 的二维空间, 任何 "+X% gain" 报告都 *必须* 注明 regime。**
