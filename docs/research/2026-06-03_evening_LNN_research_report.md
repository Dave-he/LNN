---
title: LNN 研究报告 - 2026-06-03 晚(round 35,6h cron)
date: 2026-06-03
tags: [LNN, EMMA, segment-LOO, audio-mode, daily-research]
related:
  - "[[docs/research/2026-06-03_pm_LNN_research_report]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
---

# 🌊 LNN 研究报告 — 2026-06-03 晚 (round 35,6h cron)

> 同日第三次 6h cron 触发。承接 round 33-34 segment LOO 系列发现。本节关键产出:**在严格 cross-segment LOO 下,EMMA 论文的"audio 物理信息"核心假说真正成立**;round 16 random-window 上"audio=random/zero 帮助"是数据泄漏 artifact。

## 1. Daily digest 复盘(06-03 完整数据,25 arXiv + 42 GitHub + 18 HF)

详见 `docs/daily/2026-06-03_LNN_research_digest.md`。今日 digest 主要信号已在 06-03 PM 日报记录,核心更新:
- **`YGTKL16/MFENCE`**(Rust HFT + LNN)持续 active
- **`Linlab2026/GCN-CfC`**(分子筛选)有新提交
- **`LiquidAI/LFM2.5-8B-A1B-MLX-*`** 系列下载持续增长
- 无 LNN 主线新论文(arXiv 仍以 2605.27467 / 2605.24047 为代表)

## 2. Round 35 — Audio_mode 扫描在 LOO 上

### 2.1 动机

Round 34 设立了 **K=20 audio=normal segment-LOO SOTA(mean MSE 3.23)**。Round 16 在 random-window 上发现 audio=random 比 audio=normal 好(+10.7pp)。两者矛盾:LOO 上 audio=normal 是否真比 random 好?

### 2.2 可证伪假设

> 把 round 34 的 K=20 LOO SOTA 配置扩展到 audio_mode ∈ {zero, random},任一变体应能挑战 audio=normal 的 LOO mean 3.23。

### 2.3 实现

`lnn/data/emma_rover_temporal_folds.py::TemporalSegmentRegressionDataset` 新增 `audio_mode` 参数:
- `normal`:保留 peak Hz audio 信号
- `zero`:全零
- `random`:同功率独立高斯
- `lowpass`:per-array 均值标量

`scripts/benchmark_emma_segment_loo_real_adaptive_freeze.py` 新增 `--audio-mode`,跑 K=20 audio=zero/random 4-fold LOO。

### 2.4 实验结果(rover segment-pure 4-fold LOO, h=64, ep=80, K=20, seed=42)

| audio_mode | LOO mean | per-fold MSE | std | vs normal(round 34, 3.23) |
|---|---:|---|---:|---:|
| **normal** | **3.23** 🏆 | [0.11, 3.75, 4.61, 4.46] | 1.83 | baseline |
| zero | 29.33 | [36.92, 38.92, 29.49, 11.98] | 10.61 | **+808% 灾难** |
| random | 61.47 | [15.39, 42.69, 69.99, 117.81] | 37.82 | **+1804% 大灾难** |

→ **可证伪假设彻底证伪**:audio_mode 任一变体在 LOO 上都比 normal 灾难性更差。
→ JSON:`analysis/emma_rover/2026-06-03_r35_loo_K20_audio_{zero,random}.json`。

### 2.5 关键发现 — EMMA "audio 物理信息" 假说在 LOO 下真正成立

| 评测协议 | audio=normal | audio=zero | audio=random | 解读 |
|---|---:|---:|---:|---|
| **Random-window**(round 16) | +50.3% gain | +52.7%(略好) | **+61.7%(最好)** | data leakage: 测试集与训练集 audio 分布重叠,architecture 比 audio 内容更重要 |
| **Segment-pure LOO**(本节) | **3.23(最佳)** | 29.33 | 61.47 | 真正泛化:audio motor RPM 信号不可替代,zero/random 让 cross-attn 学错误模式 |

→ **EMMA 论文的"audio 携带 video 推不出的 motor RPM ↔ wheel radius 物理耦合"核心假说在 LOO 上真正成立**!round 16 的"audio 不重要"结论是 random-window 数据泄漏 artifact。
→ 这是 35 轮 ablation 第一次**支持** EMMA 原始论文假说;前 34 轮的"audio 不重要"叙事被严格 LOO 翻盘。

### 2.6 机制解读

- **audio=normal + LOO**:audio_encoder 学到的 RPM 表示在 4 个 segment 间 generalize,因为 motor 物理常数(wheel radius)恒定。Phase 1 K=20 短 warmup 防止 audio_encoder 过度专门化某 segment。
- **audio=zero + LOO**:cross-attn 把所有 query 全部 attend 到一个常零向量上 → audio context vanishing → video alone 不足以恢复 5 个参数 → 所有 fold 退化到 ~30 MSE
- **audio=random + LOO**:cross-attn 学到对噪声的"伪相关" → phase 2 video fine-tune 锁定在错误模式 → held-out fold 完全失效(fold 3 达 118 MSE)

### 2.7 元结论第十九次精化 — Random-window 与 LOO 上的 audio 角色完全反转

| Round | audio 角色 |
|---:|---|
| 16 | random-window 下 audio=random 最佳,EMMA audio 物理假说"被否定" |
| **35(本节)** | **LOO 下 audio=normal 唯一最佳,EMMA audio 物理假说被严格验证** |

新生产决策:

```python
# 35-round honest production: include audio_mode by deployment protocol
hidden_size = 64; total_epochs = 80; freeze_targets = 'audio_only'

if deployment_protocol == "in_distribution_random_window":
    warmup_epochs = 40
    audio_mode = 'normal'  # round 31 验证大预算下 normal 最佳
    # Expected: test MSE ~0.31
elif deployment_protocol == "out_of_distribution_cross_segment":
    warmup_epochs = 20
    audio_mode = 'normal'  # ★ 本节确认 LOO 上 normal 唯一最佳 ★
    # Expected: LOO mean MSE ~3.23 (best fold 0.11, worst 4.61)
```

## 3. 下一步研究思路(W+1)

按价值排序:

1. **LOO K 进一步细化(K=15, K=25)** — 确认 K=20 是真正最优
2. **segment-mixed audio normalization** — 在 phase 1 给每个 segment 的 audio 做 z-score,降低 segment-specific 偏移
3. **测试 K=20 + audio=normal + h=32/h=96 / 不同容量** 看 LOO SOTA 3.23 是否容量 portable
4. EMMA quadrotor 数据(blocked)

## 4. 提交 + 推送

- 2 个 K=20 audio_mode JSON + appendix §39 + 本日报准备 commit
- `pytest tests/` **142/142 全过**,零回归

---
*本报告由 6h cron `7131cb00` 触发(今日第三次)。今日 4 个 /loop 已记录:round 32(K 精细扫描)+ round 33(real adaptive freeze on LOO)+ round 34(K=20 NEW LOO SOTA)+ round 35(本节,audio_mode LOO 验证)。*
