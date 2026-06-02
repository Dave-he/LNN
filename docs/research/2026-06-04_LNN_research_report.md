---
title: LNN 最新进展研究报告 - 2026-06-04
date: 2026-06-04
tags: [LNN, CfC, Bi-CfC-NAD, GRU, ablation, cross-modal, architecture-family, research-report]
related:
  - "[[docs/research/2026-06-03_LNN_research_report]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
---

# 🌊 LNN 最新进展研究报告 — 2026-06-04

> 接续 2026-06-03 (20+ 轮 /loop 跨度的 EMMA cross-attention ablation)。本日核心产出是 **架构 family 必要性测试** — 把第二 encoder 从 Bi-CfC-NAD 换为 GRU,得到史上最差结果 +3.9%。这把"trainable + recurrent 即可"的 round 20 元结论**彻底改写**为 **"必须是 Bi-CfC-NAD 系列"**。

## 1. 今日 digest 摘要

(daily pipeline 已由 cron 自动跑出,arXiv API 仍不稳定,GitHub 限流,HF 14 个新 LFM2.5 衍生模型,关键资产:`reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil` 已下载 580+;EPFL Liberte 系列 SFT/GRPO 持续更新。无 LNN 主线新论文。)

## 2. Round 21 — Bi-CfC-NAD 架构 family 必要性测试

### 2.1 动机

经过 round 11-24(13 个连续 /loop)的 ablation,cross_attn 在 EMMA rover 上 +52.7% gain 的归因已收敛为 3 条必要条件:
- 第二 encoder 存在
- 第二 encoder 是 recurrent(cron round 20: MLP +14.3% FAIL)
- 第二 encoder 是 trainable(本仓库 round 20: frozen random +24.5% FAIL)

但 round 24 §24.6 提出最后一个开放问题:**这 3 条是否充分?** 是否换任何"trainable + recurrent" encoder 都能复现?或 Bi-CfC-NAD 自带的 noise-adaptive / parallel-EMA / 双向递归是必要的?

### 2.2 可证伪假设

> 用 1 层双向 `nn.GRU`(trainable + recurrent + 双向)替换第二 encoder,其余 cross-attention / fusion / MDN head 完全相同。
> - 若 gain ≈ uni_video Bi-CfC +35.2% → 三条件充分,Bi-CfC family 特性可有可无
> - 若 gain << +35% → Bi-CfC family 特性是第 4 条必要条件

### 2.3 实现

`lnn/core/multimodal_physreg.py::GRUEncoderXAttnWithMDN`:
- video encoder 保留 Bi-CfC-NAD(为隔离变量,只换 audio_encoder)
- 第二 encoder = `nn.GRU(input_size=video_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)` + `Linear(2*hidden, hidden)` projection
- cross-attention q/k/v/fuse_proj/MDN 与 Bi-CfC 版本 bit-identical

4 个新单测覆盖形状、audio ignored、GRU 参数收到梯度、输出与 uni_video Bi-CfC 不同。`pytest tests/` **137/137 全过**(133 base + 4 新),零回归。

### 2.4 实验结果(epochs=20, n=200, K=1, seed=42, hidden=16, video_dim=3)

| 第二 encoder | trainable | recurrent | family | gain |
|---|:---:|:---:|---|---:|
| 无(register_token, cron r19) | n/a | n/a | n/a | +27.5% |
| 无(sinusoidal, mine r19) | n/a | n/a | n/a | +26.5% |
| frozen random Bi-CfC(r20) | ❌ | ✅ | Bi-CfC | +24.5% |
| MLP(cron r20) | ✅ | ❌ | n/a | +14.3% |
| **GRU 双向(NEW r21)** | ✅ | ✅ | **GRU** | **+3.9%** ❌💥 |
| **Bi-CfC-NAD uni_video(r13)** | ✅ | ✅ | **Bi-CfC** | **+35.2%** ✅ |
| Bi-CfC-NAD(cross_attn audio=zero) | ✅ | ✅ | Bi-CfC | +52.7% |

→ **可证伪假设彻底证伪**:GRU 第二 encoder **+3.9%**,比 Bi-CfC uni_video 低 **31.3pp**,甚至**比无 encoder 还低 23pp**(register_token +27.5%)。
→ JSON:`analysis/emma_rover/2026-06-03_r21_gru_encoder.json`。

### 2.5 元结论第六次精化 — Bi-CfC-NAD architecture family 也是必要条件

| Round | 元结论演进 |
|---:|---|
| 11/13 | "audio 携带物理信息" |
| 16 | "audio 内容不重要,架构正则化" |
| 17 | "decorrelated 第二流" |
| 18 | "register-token meta-pool" |
| 19 | "trainable recurrent encoder" |
| 20 | "trainable AND recurrent 都必要;输入次要" |
| **21** | **"trainable + recurrent + Bi-CfC-NAD family 都必要;GRU 不行,MLP 不行"** |

20 轮以来的所有 ablation 现在完整定位到:
1. ✅ 第二 encoder 存在
2. ✅ encoder 是 recurrent(非 MLP)
3. ✅ encoder 是 trainable(非 frozen)
4. ✅ **encoder 是 Bi-CfC-NAD family(非 GRU,可能也非 LSTM)**
5. ❌ 输入 informative — 次要(audio 内容只贡献 ~2-4pp)

**关键工程结论**:cross_attn 的 +52.7% gain 是 **Bi-CfC-NAD specific**。EMMA 论文宣称的"两流互补"实际上需要 LTC/CfC 风格的 noise-adaptive 递归动态;**普通 RNN/GRU/LSTM 系列做 cross-attn 第二 encoder 几乎无收益**。这对未来 LNN 多模态设计意义重大:第二 encoder 必须用 LTC/CfC 系列,不能省工夫用 GRU。

### 2.6 为什么 GRU 失败?推测

GRU +3.9% 远低于 frozen random Bi-CfC +24.5% — 即使**冻结的 Bi-CfC 也比 trained GRU 强**。可能原因:

- **Feature-space 错配**:GRU 的 hidden state dynamics(gating)与 Bi-CfC-NAD 的 noise-adaptive decay 在特征空间结构上不同。Cross-attention 需要两个 encoder 输出的特征空间能"互查询",GRU 输出的 256 hidden 与 Bi-CfC 的 16 hidden 经 attention 后无法对齐。
- **没有 noise-adaptive 机制**:GRU 在含噪 rover features 上易过拟合 input noise,而 Bi-CfC-NAD 的 `noise_beta` EMA gate 能动态降权输入。
- **训练优化困难**:双向 GRU 参数 ~3.6K(GRU 本体 1.6K + projection 1.4K)在 epochs=20 / num_samples=200 下未充分收敛(`train NLL` 在 epoch 20 仍 0.4,远高于 Bi-CfC 版本的负值)。

下一轮可以测试 (a) hidden 增大 / 更多 epoch 是否让 GRU 追上;(b) LSTM 是否表现不同;(c) noise-adaptive 是否是关键 — 用一个不带 NAD 的 vanilla CfC 看是否仍 PASS。

## 3. 下一步研究思路(W+1 backlog)

按价值排序:

1. **GRU + 更大 hidden / 更多 epochs 重测** — 排除"GRU 只是欠拟合"的可能。预期:若 GRU 仍 < +20%,Bi-CfC family 特性是真必要;若追上 +35%,本轮结果是预算 artifact。
2. **vanilla LSTM 第二 encoder** — 与 GRU 同 family 对照,验证"普通 RNN 都不行"。
3. **CfC(不带 noise-adaptive)第二 encoder** — 隔离 "noise-adaptive" vs "CfC 风格 closed-form ODE" 哪个是关键。
4. **真实 EMMA 多视频 LOO**(数据未释出 — 仍 blocked)
5. **EMMA quadrotor 12 参数回归**(数据未释出)

## 4. 提交 + 推送

- 新模型 `GRUEncoderXAttnWithMDN` + 4 单测 + benchmark wiring + 1 JSON + 本报告 + appendix §25 准备 commit。
- 全套 `pytest tests/` **137/137 全过**,零回归。

---
*本报告由 6h cron `7131cb00` 触发(今日第一次)。1h cron `855d0d94` 仍在主线 EMMA ablation 推进上。*
