---
title: LNN Multimodal Design Guideline (v3)
date: 2026-06-03
tags: [LNN, multimodal, design-guideline, regime-conditioned, adaptive-freeze, SOTA, v3]
related:
  - "[[LNN_TLDR]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
---

# 🧪 LNN 多模态系统设计指南 (v3)

> **v3 更新 (2026-06-03)**: 加入 §27 **regime 决定一切** (小预算 cross_attn 赢,大预算 video_only 赢), §30 **新 SOTA adaptive freeze MSE 0.31** (5 行 production recipe), §32 **CI 强制双 regime 测** (`@pytest.mark.large_budget`), 以及 §30 GRU seed sensitivity 修订。

## 0. ★ New SOTA — 2026-06-03

| 排名 | test MSE | 配置 | 来源 |
|---:|---:|---|:---:|
| 🏆🥇 | **0.31** | **adaptive freeze audio_only K=40 @ h=64, ep=80** | round 26 |
| 🥈 | 0.87 | pure video_only(input=4) @ h=64, ep=80 | round 22 cron |
| 🥉 | 4.49 | adaptive freeze audio_only K=40 @ h=32, ep=80 | round 25 |
| 4 | 60.84 | cross_attn(audio=normal) @ h=32, ep=40 | round 23 |
| 5 | 248.64 | cross_attn(audio=zero) @ h=16, ep=20 | round 13 |

**5 行 production recipe (★)**:
```python
hidden_size = 64
epochs = 80
warmup_epochs = 40       # 0.5 × total_epochs
freeze_targets = "audio_only"
# After warmup: requires_grad=False on audio_encoder; rebuild Adam.
```

**Adaptive gain 公式** (round 27 cron `44e4ff4` 拟合, R² 0.88):
```text
adaptive_gain ≈ -1.46 × gap
where gap = (video_only_MSE - small_budget_ceiling) / small_budget_ceiling
```

→ 当 video_only 距 "小预算上限" 越远 (即模型 *未* 收敛),adaptive freeze 收益越大。

## 1. 三句话总结

1. **regime 决定一切** (小预算 → cross_attn 赢 +50%;大预算 → adaptive freeze video_only 赢, MSE 0.31 最佳);**任何 "+X% gain" 报告 *必须* 注明 regime (hidden_size × epochs)**
2. **第二 encoder 必要条件** (任一缺失 → gain 大跌到 ≤+27%): recurrent + trainable + 输入有变化;**family 选 LSTM / vanilla CfC / Bi-CfC-NAD 任一** (均 +32~+36%);**GRU 可用但 seed-sensitive**
3. **新 SOTA: adaptive freeze-after-warmup** (Phase 1 全模型, Phase 2 冻结 audio_encoder, 重建 Adam) 2.8× 优于 video_only baseline — **首次跨过 video_only 基准**

## 2. 决策树 v3 (含 regime)

```
START
│
├── 你是 production deployment (有真实数据,有时间预算)?
│   │
│   ├── 是 → 直接用 ★ adaptive freeze recipe:
│   │         h=64, ep=80, K=40, freeze=audio_only
│   │         预期 MSE: ~0.3-1.0 (真实 EMMA rover 量级)
│   │
│   └── 否 (在 sandbox / 实验) → 见下面分支
│
├── 你的 regime 是 SMALL (h≤16, ep≤20)?
│   │
│   ├── 是 → 用 cross_attn + LSTM/CfC/Bi-CfC-NAD 第二 encoder
│   │      audio=zero 略优于 audio=normal (+0~+5pp)
│   │      预期 gain: +30% ~ +70%
│   │
│   └── 否 (LARGE, h≥32, ep≥40) → 优先 video_only 单流
│         备选: adaptive freeze (Phase 1 train, Phase 2 冻结 audio)
│         *不要* 全程 cross_attention — 优化复杂度负担
│         预期 MSE: video_only 0.87 / adaptive freeze 0.31
│
└── 你的 hidden_size 介于 4-8?
    │
    ├── 是 → 任何架构 gain ≈ 0% (容量门槛);优先增加 capacity
    │
    └── 否 → 走上面三条
```

## 3. 必要条件表 (任一缺失 → gain 跌到 ≤+27%)

| 条件 | 缺失时 | 满足时 | 失败实验 |
|---|---|---|---|
| **recurrent** (vs MLP) | +14.3% | +35.2% | §24 non_recurrent |
| **trainable** (vs frozen random) | +24.5% | +35.2% | §21 frozen |
| **输入有变化** (vs learned constant) | +27.5% | +35.2% | §22 register_token |
| **ODE/RNN family** (vs linear) | varies | +32% ~ +36% | §25 GRU (seed-sensitive) |
| **hidden ≥ 8** | ~0% | diminishing returns | §20 |
| **regime-correct architecture** | 大预算下 -755% | (regime-stratified) | §27 cross_attn 翻转 |

## 4. regime-conditional 推荐 v3 (扩展 §5 旧版)

| Regime | video_only MSE | cross_attn audio=normal | **adaptive freeze** | 推荐 |
|---|---|---|---:|---|
| 极小 (h=4-8, ep≤10) | >500 | ~0% | n/a | 增加 capacity |
| **小 (h=16, ep=20)** | 525 | **+50%** | (warmup_ratio 太短, freeze 无效) | **cross_attn** |
| 中 (h=32, ep=40) | 154 | +37% (normal) / +70% (zero) | 4.49 (h=32 K=40) | adaptive freeze **或** cross_attn(zero) |
| 中 (h=32, ep=80) | 38 | **-62%** ❌ | (not measured) | **video_only** |
| **大 (h=64, ep=80)** | **0.87** | -755% ❌ | **0.31 (★ SOTA)** | **adaptive freeze** |
| 超大 (h≥128, ep≥160) | (推测 ~0.001) | (推测 更糟) | (推测 ~0.001) | video_only 仍优 |

**临界点**: video_only 距 "小预算上限" 越远 (MSE > 100), adaptive freeze 收益越大;一旦 video_only 接近收敛 (MSE < 50), 全程 cross_attention 反而拖累。

## 5. family 排序 (small-budget regime h=16, ep=20, multi-seed 视角)

| 排名 | family | 代表 | single-seed gain | multi-seed mean | std |
|---|---|---|---:|---:|---|
| 1 | ODE+ | Bi-CfC-NAD | +35.2% | +35.2% | low |
| 2 | RNN | **LSTM** | **+36.1%** | +36.1% | low |
| 3 | ODE | vanilla CfC | +32.5% | +32.5% | low |
| 4 | RNN | **GRU** | **+3.9% ~ +37%** | +35% | **HIGH** |
| 5 | (无 encoder) | register_token | +27.5% | +27.5% | low |
| 6 | MLP | non_recurrent | +14.3% | +14.3% | low |

**关键观察**:
- LSTM / Bi-CfC-NAD / vanilla CfC 几乎并列 (+32~+36%) → family 选择 *几乎不* 影响
- GRU 是 seed-sensitive outlier — **任何 GRU 报告必须 ≥5 seeds**
- **Bi-CfC-NAD vs vanilla CfC 只差 +2.7pp** → bi+noise-adaptive 细节 *几乎不* 必要
- **register_token +27.5%** 是新 baseline — 任何"信息融合"工作必须超过

## 6. 失败模式 (反模式)

| 反模式 | 结果 | 教训 |
|---|---|---|
| **在大预算下加 cross_attention 全程** | gain **-755%** (MSE 0.87 → 7.47) | 优化复杂度负担 > 信息收益 (§27) |
| 训练时 modality_dropout 0.3 | test MSE 涨 16% | 合成数据 audio 已是"软正则" (round 9) |
| 训练时 partial-occ 遮挡 50% | test MSE 涨 1.7% | distribution match 不增加真信号 (round 10) |
| audio noise 扫描 640× | 增益几乎不变 | audio 信息内容 ≤ 5pp (round 22 cron) |
| 用 GRU 第二 encoder (单 seed 报告) | +3.9% ~ +37% (huge variance) | GRU seed-sensitive, 必须 ≥5 seeds (§30) |
| 用 non_recurrent MLP | gain +14.3% | recurrence 缺失 (§24) |
| 用 learned constant register_token | gain +27.5% | 输入无变化,失去"互补性" (§22) |
| **adaptive freeze 但用 cross-attn projections 一起冻** | MSE 17.14 (vs 0.31) | cross-attn 投影需继续更新 (round 26) |
| **adaptive freeze 在合成数据上** | NEGATIVE on synth burst | 机制 universal 但数据 sensitivity 不同 (round 27) |
| **报"+X% gain"但 *单一* seed *单一* regime** | misleading | 必须两 regime + ≥3 seeds (§30 + §32) |

## 7. 与 EMMA 论文的隐含对应

EMMA paper 用 ~64 hidden units, 即本仓库的"大"regime。在该 regime 下:
- **video_only 单 Bi-CfC-NAD 已近完美拟合** (MSE 0.87)
- **adaptive freeze audio_only 进一步推到 0.31** (新 SOTA)
- **EMMA 的"两流 LTC 互补"在 rover 任务上 = 欠参数化情况下的正则化策略**;在 EMMA 没测的 *欠参数化 regime* 下 cross_attn 赢

**工程 takeaway**: 若要复现 EMMA 的设计 *且* 处在合理容量 regime,**adaptive freeze 单流 Bi-CfC-NAD 即可**,且 *优于* EMMA 的两流方案 2.8×。

## 8. 仓库资产 reference (按设计阶段)

### 8.1 起步 (单流 baseline)

- `lnn/core/liquid_neuron.py` — `LiquidNN` 单层封装
- `lnn/core/cfc.py` — `CfCNetwork` (Hasani 2021) vanilla ODE RNN
- `lnn/core/noise_adaptive_cfc.py` — `BiCfCNADWithMDN` (本仓库扩展, bidirectional + noise-adaptive)

### 8.2 多模态 (双流)

- `lnn/core/multimodal_physreg.py::MultimodalBiCfCNADWithMDN` — round 6, 简单 concat 融合
- `lnn/core/multimodal_physreg.py::CrossModalAttnBiCfCNADWithMDN` — round 7, cross-attention 融合

### 8.3 9 个 ablation 模型类 (第二 encoder 选型)

| 模型 | 文件位置 | 何时用 |
|---|---|---|
| `MultimodalBiCfCNADWithMDN` | `multimodal_physreg.py` | 第一代双流,concat 融合 |
| `CrossModalAttnBiCfCNADWithMDN` | 同上 | 第二代双流,cross-attention 融合 |
| `RegisterTokenSelfXAttnWithMDN` | 同上 | **新 baseline**: 任何"信息融合"工作必须超过 +27.5% |
| `UniVideoSelfXAttnWithMDN` | 同上 | "无 second encoder 增益" baseline, +35.2% |
| `VanillaCfCXAttnWithMDN` | 同上 | ODE family baseline, +32.5% |
| `NonRecurrentSelfXAttnWithMDN` | 同上 | "recurrence 必要" baseline, +14.3% |
| `LSTMEncoderXAttnWithMDN` | 同上 | RNN family baseline, +36.1% |
| `NoisyVideoSelfXAttnWithMDN` | 同上 | 装饰相关性 baseline (round 17) |
| `MixedStreamSelfXAttnWithMDN` | 同上 | cos similarity 扫描 (round 18) |
| `SinusoidalTimeStreamSelfXAttnWithMDN` | 同上 | per-step 变化 baseline (round 19) |

### 8.4 SOTA 模型 + 训练方法

- `scripts/benchmark_adaptive_freeze.py` — **新 SOTA 训练流程 (★)**:`--freeze-targets {audio_only, all_xattn}` + `--warmup-epochs K` + Phase 2 重建 Adam
- `scripts/benchmark_emma_rover.py` — 真实数据 3 模型 baseline
- `scripts/benchmark_register_token.py` — 8 种第二 encoder 比较
- `scripts/benchmark_multimodal_physreg.py` — 合成数据 + 各种训练增强

### 8.5 扫描工具 (regime 限定)

- `scripts/scan_emma_rover_video_channels.py` — video 通道子集
- `scripts/scan_emma_rover_hidden_size.py` — hidden_size 容量
- `scripts/scan_synth_burst_hidden_size.py` — 合成数据容量
- **`scripts/scan_emma_rover_budget_sweep.py` — regime 临界点扫描 (关键)**
- `scripts/visualize_emma_rover_attention.py` — attention 矩阵视觉化
- `scripts/scan_gru_capacity_recovery.py` — GRU seed sensitivity 检测

### 8.6 数据

- `lnn/data/emma_rover_features.py` — 零重型依赖 (numpy+PIL) 视频/音频特征提取
- `lnn/data/emma_rover_regression.py` — 滑窗扩样本 dataset (含 `video_channels` 旋钮)
- `lnn/data/multimodal_physreg.py::HeterogeneousForcedDataset` — 合成 burst/chirp 受迫振子

### 8.7 单元测试 (142 项, 2 个 regime markers)

- `tests/test_multimodal_physreg.py` — 21+ 项覆盖所有新模型类 + 数据集
- **`tests/test_lnn_multimodal_regime.py` — 5 项 regime-conditional 测试 (★)**
  - 默认 `pytest tests/ -q` (137 项 + 3 regime_small_budget) — CI 友好 ~86 秒
  - `pytest -m large_budget` — 2 项 large_budget regime 测试 ~80 秒
  - 任何新模型必须 *通过两 regime 测试才能 merge*

## 9. 实验设计 checklist (新 PR 作者用) — v3 含 regime

- [ ] 在 **EmmaRoverRegressionDataset** 真实数据上跑过
- [ ] 在 **HeterogeneousForcedDataset(burst)** 合成数据上跑过
- [ ] **在两 regime 下报告**: (h=16, ep=20) small + (h=64, ep=80) large ★
- [ ] 与 `register_token` (+27.5%) baseline 比较 — 必须 *超过* 才能声称"信息融合"
- [ ] 与 `LSTMEncoderXAttnWithMDN` (+36.1%) baseline 比较 — 必须 *超过* 才能声称"超过 RNN family"
- [ ] 与 `BiCfCNADWithMDN(input_size=4)` (即 video+audio concat) baseline 比较
- [ ] **若报 GRU 相关, 必须 ≥5 seeds 报告 mean ± std** (避免 §30 那种 +3.9% anomaly)
- [ ] **若声称"新 SOTA"或"大预算 cross_attn 有效", 必须验证 adaptive freeze baseline 已尝试**
- [ ] **CI 必含 `pytest tests/ -q -m large_budget`** (regime 双测)
- [ ] `pytest tests/` 零回归

## 10. 不应做的 (基于 32 轮 ablation NEGATIVE 经验)

- [ ] ❌ 在合成数据上做 *训练增强* (modality_dropout / partial-occ / noise injection): round 9-12 全部 NEGATIVE
- [ ] ❌ 假设"audio 信息内容"是 cross_attn 高 gain 的主因: round 16/19/22 多次证伪, audio ≤5pp
- [ ] ❌ 用 hidden_size < 8: 容量门槛, 无 cross_attn 收益
- [ ] ❌ **在大预算下仍用全程 cross_attention**: -755% 拖累 (§27)
- [ ] ❌ **adaptive freeze 但冻 cross-attn 一起**: 反而 -755% 拖回 (round 26)
- [ ] ❌ **adaptive freeze 默认套到合成数据上**: rover-specific, 合成 burst NEGATIVE (round 27)
- [ ] ❌ 用 GRU 第二 encoder (单 seed 报告): +3.9% ~ +37% seed-sensitive (§30)
- [ ] ❌ 用 *单一* random seed 报告 "+X% gain": 不同 regime 翻车可能
- [ ] ❌ "在 hidden=16, ep=20 regime +50% 完美" → 推 "cross_attn 永远赢": 隐藏 regime 翻车

## 11. W+1 候选 (regime 限定后)

1. **真实 EMMA 多视频 LOO** — 跨 rover 视频泛化 (需要更多视频)
2. **EMMA quadrotor 12 参数** — 跨物理系统迁移
3. **adaptive freeze 在 EMMA drone 数据** — 验证 SOTA recipe 跨任务迁移性
4. **`adaptive_gain` 公式泛化测试** — 当前 R²=0.88 拟合;更多数据点应提升
5. **sparse / chunked cross-attention** — T=256+ 长视频需要
6. **把 design guide v3 翻译为 README 块** — 仓库根 README 链接过来

## 12. 一句话备忘 (v3 修订)

> **LNN 多模态系统的最优架构 *不是* 跨模态 attention,而是 *adaptive freeze* 的单流 Bi-CfC-NAD (h=64, ep=80, K=40, freeze=audio_only, MSE 0.31) — 比纯 video_only 优 2.8×, 比 cross_attn 优 ~200×;regime 是 hidden_size × epochs 的二维空间, 任何 "+X% gain" 报告都 *必须* 注明 regime 且 *两 regime 都跑*;GRU 第二 encoder 可用但 *必须* ≥5 seeds。**

---

参考: `docs/research/2026-06-02_multimodal_physreg_appendix.md` §1-§32 完整 32 轮 ablation 历史。
