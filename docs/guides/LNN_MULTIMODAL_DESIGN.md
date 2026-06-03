---
title: LNN Multimodal Design Guideline
date: 2026-06-03
tags: [LNN, multimodal, design-guideline, cross-attention, Bi-CfC-NAD, regime-conditioned]
related:
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
  - "[[docs/daily/2026-06-03_LNN_research_digest]]"
---

# 🧪 LNN 多模态系统设计指南

> **核心信息**: 跨 §6-§28 十五轮 ablation 后,本仓库对 *LNN 多模态参数回归* 的所有结论是 **regime 限定**的。**任何 "+X% gain" 报告都必须注明 regime** (hidden_size × epochs)。本指南给出"按 regime 选架构"的决策树 + 失败模式 + 仓库资产 reference。

## 1. 三句话总结

1. **recurrent + trainable + 输入有变化** 是 *第二 encoder* 的必要三条件 (任一缺失 → gain 跌到 ≤+27%)
2. **family 选 LSTM / vanilla CfC / Bi-CfC-NAD 任一** (gain +32~+36%);**避免 GRU** (catastrophic +3.9%) 和 non_recurrent MLP (+14.3%)
3. **regime 决定推荐架构**: 小预算 (h≤16, ep≤20) → **cross_attn**;大预算 (h≥64, ep≥80) → **video_only** 单流 (cross_attn 反而 -755% 拖累)

## 2. 决策树

```
START
│
├── 你在欠参数化 regime (hidden ≤ 32, epochs ≤ 40)?
│   │
│   ├── 是 → 用 cross_attn + LSTM/CfC/Bi-CfC-NAD 第二 encoder
│   │      audio=zero 通常比 audio=normal 更好(在欠拟合下 audio=normal 反而拖累)
│   │      预期 gain: +30% ~ +70%
│   │
│   └── 否 (h ≥ 64, ep ≥ 80) → 用 video_only 单 Bi-CfC-NAD
│         *不要* 加 cross_attention — 优化复杂度负担
│         预期 MSE: video_only 0.87 vs cross_attn 7.47
│
└── 你的 hidden_size 介于 4-8?
    │
    ├── 是 → cross_attn 增益 ≈ 0% (容量门槛); 优先考虑加 capacity 而非加第二 encoder
    │
    └── 否 → 走上面两条
```

## 3. 必要条件表 (任一缺失 → gain 跌)

| 条件 | 缺失时 | 满足时 | 失败实验 |
|---|---|---|---|
| **recurrent** (vs MLP) | +14.3% | +35.2% | §24 non_recurrent |
| **trainable** (vs frozen random) | +24.5% | +35.2% | §21 frozen |
| **输入有变化** (vs learned constant) | +27.5% | +35.2% | §22 register_token |
| **ODE/RNN family** (vs random init / linear) | varies | +32% ~ +36% | §25 GRU outlier (specific) |
| **hidden ≥ 8** | ~0% | (≥8 gives diminishing returns) | §20 |

**注意**: 必要条件是 *regime-independent*;推荐架构 (cross_attn vs video_only) 是 *regime-dependent*。

## 4. family 排序 (小预算 regime h=16, ep=20)

| 排名 | 模型 | gain | 关键特性 |
|---|---|---:|---|
| 1 | cross_attn(audio=zero) | +52.7% | 完整架构 + zero audio |
| 2 | cross_attn(audio=normal) | +50.3% | 完整架构 + real audio |
| 3 | cross_attn(audio=random) | +61.7% | 完整架构 + i.i.d. noise |
| 4 | LSTM | +36.1% | bidirectional RNN |
| 5 | Bi-CfC-NAD uni_video | +35.2% | recurrent Bi-CfC + cross-attn (same video) |
| 6 | vanilla CfC | +32.5% | unidirectional CfC + cross-attn |
| 7 | **register_token** | +27.5% | **新 baseline**: 任何工作必须超过 |
| 8 | GRU | +3.9% | **避免** (catastrophic outlier) |
| 9 | non_recurrent MLP | +14.3% | 避免 (recurrence 缺失) |

**关键观察**:
- LSTM / Bi-CfC-NAD / vanilla CfC 几乎并列 (+32~+36%) → family 选择 *几乎不* 影响
- GRU 是 outlier,不是 RNN family 通用失败
- **Bi-CfC-NAD vs vanilla CfC 只差 +2.7pp** (§26) → Bi-CfC-NAD 的 bi+noise-adaptive 细节 *几乎不* 必要

## 5. regime-conditional 推荐

| Regime | video_only MSE 量级 | cross_attn 增益 | 推荐 |
|---|---|---|---|
| 极小 (h=4-8, ep≤10) | >500 | ~0% | 增加 capacity,而非加 cross_attn |
| **小 (h=16, ep=20)** — 本文 13 轮 ablation 标准 | ~525 | **+50%** ✅ | **cross_attn** (正则化收益) |
| 中 (h=32, ep=40) | ~150 | +36.5% (zero +70%最佳) | cross_attn(audio=zero) |
| 中 (h=32, ep=80) | ~38 | **-62%** ❌ | **video_only** (交叉 regime 反转!) |
| **大 (h=64, ep=80)** | **0.87** | **-755%** ❌❌ | **video_only** 单流最优 |
| 超大 (h≥128, ep≥160) | (未测) | (推测 -800% 或更糟) | video_only 仍优 |

**临界点**: cross_attn 优于 video_only 仅在 *video_only 未充分收敛* 时 (MSE > 100);一旦 video_only 接近收敛 (MSE < 50),cross_attn 立即翻车。

## 6. 失败模式 (反模式)

| 反模式 | 结果 | 教训 |
|---|---|---|
| 在大预算下加 cross_attention | gain **-755%** | cross-attention 优化复杂度 > 信息收益 |
| 训练时 modality_dropout 0.3 | test MSE 涨 16% | 合成数据 audio 已是"软正则",人为 dropout 破坏平衡 |
| 训练时 partial-occ 遮挡 50% | test MSE 涨 1.7% (noise) | distribution match 不增加真信号 |
| audio noise 扫描 640× | 增益几乎不变 | audio 内容贡献 ≤ 5pp |
| 用 GRU 第二 encoder | gain +3.9% | GRU 是 specific outlier,不是 family 通用 |
| 用 non_recurrent MLP | gain +14.3% | recurrence 缺失,无法学到时间结构 |
| 用 learned constant register_token | gain +27.5% | 输入无变化,失去"互补性"基础 |

## 7. EMMA 论文的隐含对应

EMMA paper 用 ~64 hidden units, 即本指南的"大"regime。在该 regime 下:
- **video_only 单 LTC 已近完美拟合** (本仓库验证 MSE 0.87)
- **EMMA 的"两流 LTC 互补"在 rover 任务上本质是欠参数化情况下的正则化策略**
- 充足容量下,单 LTC 就够;EMMA 没测这一 regime

**工程 takeaway**: 若要复现 EMMA 的设计 *且* 处在合理容量 regime,**单流 LTC 即可**;只在 *欠参数化* 情况下加 cross_attention。

## 8. 仓库资产 reference (按设计阶段)

### 8.1 起步(单流 baseline)

- `lnn/core/liquid_neuron.py` — `LiquidNN` 单层封装
- `lnn/core/cfc.py` — `CfCNetwork` (Hasani 2021) vanilla ODE RNN
- `lnn/core/noise_adaptive_cfc.py` — `BiCfCNADWithMDN` (本仓库扩展, bidirectional + noise-adaptive)

### 8.2 多模态(双流)

- `lnn/core/multimodal_physreg.py::MultimodalBiCfCNADWithMDN` — round 6, 简单 concat 融合
- `lnn/core/multimodal_physreg.py::CrossModalAttnBiCfCNADWithMDN` — round 7, cross-attention 融合

### 8.3 Ablation 工具 (8 个备选第二 encoder)

| 模型 | 何时用 |
|---|---|
| `RegisterTokenSelfXAttnWithMDN` | 最低 baseline, 任何"信息融合"工作必须超过 +27.5% |
| `UniVideoSelfXAttnWithMDN` | "无 second encoder 增益" baseline, +35.2% (recurrent self-xattn) |
| `VanillaCfCXAttnWithMDN` | ODE family baseline, +32.5% |
| `NonRecurrentSelfXAttnWithMDN` | "recurrence 必要" baseline, +14.3% |
| `LSTMEncoderXAttnWithMDN` | RNN family baseline, +36.1% |
| `NoisyVideoSelfXAttnWithMDN` (round 17) | 装饰相关性 baseline |
| `MixedStreamSelfXAttnWithMDN` (round 18) | cos similarity 扫描 |
| `SinusoidalTimeStreamSelfXAttnWithMDN` (round 19) | per-step 变化 baseline |

### 8.4 benchmark 脚本

- `scripts/benchmark_emma_rover.py` — 真实数据 3 模型比较
- `scripts/benchmark_multimodal_physreg.py` — 合成数据 + 各种训练增强
- `scripts/scan_emma_rover_video_channels.py` — video 通道子集
- `scripts/scan_emma_rover_hidden_size.py` — hidden_size 容量
- `scripts/scan_synth_burst_hidden_size.py` — 合成数据容量
- `scripts/scan_emma_rover_budget_sweep.py` — **regime 临界点扫描**(关键)
- `scripts/visualize_emma_rover_attention.py` — attention 矩阵视觉化
- `scripts/benchmark_register_token.py` — 8 种第二 encoder 比较

### 8.5 数据

- `lnn/data/emma_rover_features.py` — 零重型依赖 (numpy+PIL) 视频/音频特征提取
- `lnn/data/emma_rover_regression.py` — 滑窗扩样本 dataset (含 `video_channels` 旋钮)
- `lnn/data/multimodal_physreg.py::HeterogeneousForcedDataset` — 合成 burst/chirp 受迫振子

### 8.6 单元测试

- `tests/test_multimodal_physreg.py` — 21+ 项覆盖所有新模型类 + 数据集 + 拒绝非法值

## 9. 实验设计 checklist (新 PR 作者用)

任何新 LNN 多模态 PR 提交前必须确认:

- [ ] 在 **EmmaRoverRegressionDataset** 真实数据上跑过
- [ ] 在 **HeterogeneousForcedDataset(burst)** 合成数据上跑过
- [ ] 至少在 *两个 regime* 下报告: (h=16, ep=20) + (h=64, ep=80)
- [ ] 与 `register_token` (+27.5%) baseline 比较 — 必须 *超过* 才能声称"信息融合"
- [ ] 与 `LSTMEncoderXAttnWithMDN` (+36.1%) baseline 比较 — 必须 *超过* 才能声称"超过 RNN family"
- [ ] 与 `BiCfCNADWithMDN(input_size=2)` (即 video+audio concat) baseline 比较
- [ ] 至少 3 random seeds 报告 mean ± std
- [ ] `pytest tests/` 零回归

## 10. 不应做的 (基于 15 轮 ablation NEGATIVE 经验)

- [ ] ❌ 在合成数据上做 *训练增强* (modality_dropout / partial-occ / noise injection): round 9-12 全部 NEGATIVE
- [ ] ❌ 假设"audio 信息内容"是 cross_attn 高 gain 的主因: round 16/19/22 多次证伪
- [ ] ❌ 用 hidden_size < 8: 容量门槛, 无 cross_attn 收益
- [ ] ❌ 在大预算下仍加 cross_attention: 反而 -755% 拖累
- [ ] ❌ 用 GRU 第二 encoder: +3.9% catastrophic
- [ ] ❌ 用 *单一* random seed 报告 "+X% gain": 不同 regime 翻车可能

## 11. W+1 候选 (regime 限定后)

1. **真实 EMMA 多视频 LOO 测试** — 跨 rover 视频的泛化能力
2. **EMMA quadrotor 12 参数** — 验证设计在 *不同物理系统* 上的迁移性
3. **GRU 反常根因诊断** — 为什么 GRU 在 h=16 失败,在 h=32 怎么样?
4. **adaptive training** — round cron 测过 two-phase cross→video,但 NEGATIVE; *自适应切换时机* 可能需要更细的设计
5. **写 pytest 标记** `@pytest.mark.large_budget` — 强制 future PR 在两 regime 跑

## 12. 一句话备忘

**EMMA 论文的"两流 LTC 互补"在 rover 任务上:小预算下 = 正则化策略,大预算下 = 不必要。LNN 多模态设计的核心约束是 *regime*(hidden_size × epochs),不是 family 选型。**

---

参考: `docs/research/2026-06-02_multimodal_physreg_appendix.md` §1-§28 完整 ablation 历史。
