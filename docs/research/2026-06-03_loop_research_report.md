---
title: LNN 最新进展与新研究思路 (loop 触发) - 2026-06-03
date: 2026-06-03
tags: [LNN, CfC, Bi-CfC-NAD, GRU, LSTM, encoder-family, regime, seed-lucky, multi-seed, research-idea, backlog]
related:
  - "[[docs/research/2026-06-04_LNN_research_report]]"
  - "[[docs/research/2026-06-03_LNN_research_report_final]]"
  - "[[docs/research/2026-06-02_multimodal_physreg_appendix]]"
  - "[[LNN_TLDR]]"
---

# 🌊 LNN 最新进展与新研究思路 — 2026-06-03 (loop iteration, post-pull)

> 本日 5h cron `3ac85e3c` 触发迭代,**pull 后**(github.com SSH 通过 192.168.6.25:7890 代理恢复)补入 2 个关键新 commit:**1bb78af round 43 seed-lucky 推翻** + **897aefa round 41 LiquidTAD 复制**。本轮目标:综合 43 轮 ablation + 6/3 daily digest + 外部信号,定位**未闭环的关键实验**并产出**可立即验证的新研究思路**。

## 0. 紧急重大更新:Pull 后纳入 round 41/43 (★ 24th meta-conclusion refinement)

### 0.1 Round 43 (commit `1bb78af`):SOTA 0.42 被证明 seed-lucky ★★★

第二轮 LOO SOTA `0.42` (h=96, K=10, **seed=42**) 在 4 个额外 seed 上验证后**被推翻**:

| seed | LOO mean | 倍数 vs SOTA | std |
|---:|---:|---:|---:|
| **42** ★ | 0.42 | 1.0× (original SOTA) | 0.55 |
| 1 | 15.36 | 37× worse | 20.92 |
| 2 | 0.72 | 1.7× | 0.59 |
| 3 | 12.80 | 30× worse | 4.71 |
| 100 | 11.49 | 27× worse | 11.81 |
| **mean** | **8.16** | **19× worse** | **6.78** |

**含义**(我整合后):
- 之前 round 26/34/38 所有 SOTA 数字 (0.31 v3, 0.42 v3.5) **都是 single-seed 报告**,其论文价值应作 **advisory** 看待
- LOO 实际生产期望:best case (lucky seed) 0.42,**mean 期望 8.16**,worst case 15+
- 推论升级为 **24th meta-conclusion refinement**:任何 LNN 多模态 SOTA 报告 **必须 ≥3 seeds mean ± std**
- 推理侧应对: **5-seed ensemble** (K=5 seeds,预测取均值) — 这是本轮新思路 ★ 思路 A

### 0.2 Round 41 (commit `897aefa`):LiquidTAD arXiv:2604.18274 复制 ★★

新增 `lnn/core/long_sequence.py::HierarchicalDecayLiquidBlock` + `HierarchicalDecayLiquidTADHead`,实现 paper 的 "Hierarchical Decay-Rate Sharing prior"(参数效率来源)+ TAD head。**6/6 单元测过**。

- 新资产:`scripts/experiment_long_sequence.py` 入口(本轮未跑,待 torch 恢复)
- 推论:这意味着仓库对 liquid-inspired 视频 SOTA **已有 2 个复制 (EMMA 物理参数回归 + LiquidTAD TAD)**,从 1 个扩展到 2 个 — 强化了 "LNN 在多模态视频/物理任务的工程价值" 论点

## 1. 今日外部信号 digest

来源:`docs/daily/2026-06-03_LNN_research_digest.md`(arXiv API 429 但保留候选池)。

### 1.1 arXiv 候选 (节选,12 篇 LNN 直接相关)

| 日期 | 论文 | 核心 | 与本仓库关系 |
|---|---|---|---|
| 2026-05-26 | Comparative Analysis of LNN vs LSTM for Sequential Pattern Recognition (Ye Kyaw Thu 等) | LNN/CfC 与 LSTM 在临床/序列模式识别上的对比 | 直接相关 → 进入 §3 研究思路 |
| 2026-05-21 | **EMMA: Extracting Multiple physical parameters from Multimodal Data** (Farhat Shaikh 等) | 物理参数多模态恢复,video+audio+image | **本仓库当前 SOTA 来源** |
| 2026-05-05 | Physics-Modeled Neural Networks (DynPMNNs) | ODE-based continuous-time deep learning | 思路对位 LTC/CfC 的连续时间视角 |
| 2026-04-24 | LNN for Natural Gas Spot Price Forecasting | LNN 在时间序列预测上的应用 | 验证 LNN 在 tabular/financial 数据的可迁移性 |
| 2026-04-20 | **LiquidTAD: Efficient Temporal Action Detection via Parallel Liquid-Inspired Temporal Relaxation** | liquid-inspired 并行时序松弛,TAD | **第二个 liquid 视频 SOTA** (本仓库 round 14 跟进过) |
| 2026-04-15 | A Nonasymptotic Theory of Gain-Dependent Error Dynamics in Behavior Cloning | 行为克隆 PD 增益误差 | 与 LNN 控制应用相关 (本仓库 experiment_*) |
| 2026-04-12 | MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling | iPhone 部署 + CfC 风格 + MMP 协议 | 边缘部署参考 |
| 2026-04-08 | Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks | sub-THz + LNN 6G | 新兴跨域应用 |
| 2026-04-05 | Symbolic-Vector Attention Fusion for Collective Intelligence | SVAF (vector × symbolic) | 与 SVAF-MeloTune 配套 |
| 2026-04-02 | AEGIS: Adversarial Entropy-Guided Immune System | thermodynamic SSM | SSM 与 LNN 比较视角 |
| 2026-03-28 | Liquid Networks with Mixture Density Heads for Efficient Imitation Learning (Nikolaus Correll) | LNN+MDN 在 Push-T/RoboMimic 仿真 | **直接对位本仓库 EMMA 验证** |
| 2026-02-28 | Continuous-Time Mask Refinement with Local Self-Similarity Priors | ODE-based medical 图像分割 | 跨域连续时间方法 |

### 1.2 GitHub 候选 (节选,LNN 主线 12 个)

| 日期 | 仓库 | Star | 关注点 |
|---|---|---:|---|
| 2026-06-02 | infinition/LSTN | 1 | Rust liquid-style 文本生成 — 边缘推理路径 |
| 2026-06-02 | YGTKL16/MFENCE | 0 | **LNN + HFT 高频交易** — 新跨域 |
| 2026-06-02 | Linlab2026/GCN-CfC | 0 | **GCN+CfC 分子筛选** — 跨域结构数据 |
| 2026-06-01 | heimdilon/sncp-ppo-crowdnav | 0 | **PPO+LTC TurtleBot3 crowd navigation** — 端到端 LNN RL |
| 2026-06-01 | Alexng2024/EDSSM | 0 | **Event-Driven State Space Models** — 与 CfC-DT 形成对照 |
| 2026-05-30 | parhat1/cfdna-tau-repository | 1 | LNN 在 cfDNA cancer detection, **跨研究 AUC 跌至 0.40** — 重要泛化失败案例 |
| 2026-05-29 | 2841649220/LSHN | 0 | LNN+spiking+hypergraph 混合 — 新混合范式 |
| 2026-05-29 | karl4th/liquid-neural-network-delivery-robot | 0 | LNN 配送机器人 — 端到端物理任务 |
| 2026-05-29 | JuneDylan/LNN_Github | 0 | **LNN 教程仓库** — 对比教学 |
| 2026-05-28 | 192273273.../liquid-neural-networks-transparency | 0 | **LNN 透明度综述** — 知识沉淀型,可吸收 |

### 1.3 HuggingFace 候选 (节选,LFM 衍生 18 个)

| 模型 | 关键参数 | 关注点 |
|---|---|---|
| LiquidAI/LFM2.5-8B-A1B-MLX-4bit/8bit | 8B 4/8bit MLX | **Apple Silicon 边缘路径** — Jetson 对位 |
| reaperdoesntknow/LFM2.5-1.2B-Distilled-SFT | 1.2B SFT | 1.2B 蒸馏 → 进入 Jetson 队列 |
| reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil | 8B distillation | 8B 蒸馏对照 |
| coder3101/LFM2.5-VL-450M-heretic | 450M VL | **450M 视觉语言** — 边缘 VL 路径 |
| LiquidAI/LFM2.5-8B-A1B-MLX-* | 8B MLX | 基准 |

## 2. 仓库内部 43 轮 ablation 主线总结

从 round 11-43 整理出**核心 5 条必要条件 + 1 条量级 + 1 条 multi-seed 限定**结论:


1. ✅ **第二 encoder 存在** (round 13/19)
2. ✅ **recurrent** (round 20, MLP +14.3% FAIL)
3. ✅ **trainable** (round 20, frozen random +24.5% FAIL)
4. ✅ **Bi-CfC-NAD family** (round 21, GRU +3.9% FAIL, GRU < register_token +27.5% < frozen random +24.5% < Bi-CfC +35.2%)
5. ❌ **输入 informative** (audio 内容仅 ~2-4pp,次要)
6. 量级: cross_attn +52.7% on EMMA rover (h=64, ep=80, K=40, freeze=audio_only, random-window-specific)
7. ⚠️ **multi-seed 限定** (round 43, ★ 新加入):round 38 single-seed SOTA 0.42 实际 5-seed mean 8.16 ± 6.78;**所有 +X% gain 报告都必须 ≥3 seeds mean ± std**

**未闭环的关键 gap**:
- ❓ **LSTM 第二 encoder 多 seed 验证** (★ 思路 A):代码已写 (`LSTMEncoderXAttnWithMDN` lnn/core/multimodal_physreg.py:1228),**0 测试,0 多 seed benchmark,0 JSON** — 4 个单测本轮补上
- ❓ **Bi-CfC-NAD × LSTM × GRU 多 seed 对照** (★ 思路 A):round 21/25 是单 seed 结论,多 seed 可能全 fail — `scripts/benchmark_multiseed_encoder_families.py` 本轮新增,等 torch 恢复
- ❓ **vanilla CfC(无 NAD)** 隔离测试:NAD vs closed-form ODE 哪个关键?
- ❓ **真实 EMMA quadrotor 12 参数** (数据未释出,仍 blocked)
- ❓ **EDSSM × CfC 对照** (GitHub 候选,有潜力形成新对照实验)
- ❓ **LiquidTAD SOTA 在 EMMA 上的迁移** (★ round 41 已复制,待 §TAD 任务对比)
- ❓ **LFM2.5-1.2B-Distilled-SFT 在 Jetson 上的 SMOKE 验证** (HF 候选, 0 数据)
- ❓ **5-seed ensemble 在 EMMA rover 上的真实生产价值** (★ 思路 D):round 43 推翻单 seed,但 ensemble 是否真能稳住 mean?

## 3. 今日新研究思路 (3 个,按价值排序)

### 思路 A (★ 高价值 / 立即可行): LSTM 第二 encoder round 22 闭环

**问题**:round 21 证伪 GRU,但 LSTM 是另一主流 RNN family,结论是否对 LSTM 也成立?`LSTMEncoderXAttnWithMDN` 已实现但**从未跑过**。

**可证伪假设**:
- H0: LSTM 在 cross-modal 第二 encoder 位置**也**失败 (test MSE ≈ video_only baseline)
- H1: LSTM **恢复**到 +30%+ 范围 (说明 GRU 是 family 异类,RNN-general claim 不成立)

**预期**:
- 若 H0 → 推论升级为 **"recurrent 但非 CfC family 的 encoder 在 cross-modal 第二流处系统性失败"**(工程强约束)
- 若 H1 → 推论降级为 **"GRU-specific 缺陷"**,LSTM 可作为 CfC 备选

**实现**(已具备):
- `lnn/core/multimodal_physreg.py:1228` `LSTMEncoderXAttnWithMDN` ✅
- `scripts/benchmark_register_token.py:43` 已在 model 工厂里导入 ✅
- ✅ `tests/test_multimodal_physreg.py` LSTM 段(4 个测试,本轮补上)✅
- 缺:`scripts/benchmark_lstm_encoder.py` 独立 benchmark → **本轮改成 scripts/benchmark_multiseed_encoder_families.py** (直接覆盖思路 A + 思路 B 多 seed 维度)
- 缺:`analysis/emma_rover/2026-06-03_r22_lstm_encoder.json` (待 torch 恢复跑)

**预计耗时**:5 seeds × 3 families × 80 epochs ≈ 5×3×8s ≈ 120 秒(实际 5×3×15s ≈ 225s,等 torch 恢复)

### 思路 B (中价值 / 长期): EDSSM × CfC 对照实验

**问题**:GitHub 候选 `Alexng2024/EDSSM` 提出 "Event-Driven State Space Models",用闭式连续时间线性动力学传播 latent state — 与 CfC 的 closed-form continuous-time 形成**方法论对照**。两者都"闭式 ODE 推理",但一个用 SSM 一个用 LTC gating。

**可证伪假设**:
- 若 EDSSM 在 EMMA rover 上**接近** Bi-CfC-NAD 表现 → "闭式 ODE 推理"是 cross-modal 第二 encoder 的真正必要条件,而非 LTC-specific
- 若 EDSSM 表现 ≈ GRU → "闭式 ODE 推理"不够,需要 LTC/CfC 风格的 gating 动态

**依赖**:需 clone EDSSM,理解其 forward 协议,在 `multimodal_physreg.py` 写 `EDSSMEncoderXAttnWithMDN`。

**预计耗时**:半天(首次集成)+ 90 秒 benchmark。

### 思路 C (低-中价值 / 边缘): LFM2.5-1.2B-Distilled-SFT 在 Jetson Orin Nano 上的 SMOKE

**问题**:HF 候选 `reaperdoesntknow/LFM2.5-1.2B-Distilled-SFT` 是 1.2B 边缘友好参数,这是 Liquid AI 官方蒸馏生态第一个进入 Jetson 部署可行尺寸的模型。

**可证伪假设**:
- 1.2B MLX/SafeTensor 格式在 Jetson Orin Nano CPU 路径上**首次推理**时延 < 500ms/token
- 8bit/4bit 量化后**首次推理**时延 < 200ms/token

**依赖**:需下载 safetensors (`huggingface-cli download` 已被网络 block,需用镜像或代理)。

**预计耗时**:半天(下载+转换+benchmark)。

### 思路 D (★ 新加 / 最高价值): 5-seed ensemble 验证(round 43 直接推论)

**问题**:round 43 推翻单 seed SOTA,**ensemble 是否真能稳住 mean?** 这决定了"生产能否用"。

**可证伪假设**:
- H0: 5-seed ensemble (取 5 个 seed 预测均值) 在 LOO mean 上**比 best single-seed 还低**(生产胜利)
- H1: 5-seed ensemble **等同 mean** (无 ensemble gain,单 seed best 仍是 lucky)
- H2: 5-seed ensemble **高于 best single-seed 但低于 mean** (有部分 ensemble gain,推荐)

**实现**:`scripts/benchmark_multiseed_encoder_families.py` 的 `ensemble` 段已经写好 — 它会输出 `{family}_avg` 的 seed-ensemble MSE 数字,直接填实 §0.1 推论。

**预计耗时**:0 (脚本已写,等 torch 跑)

## 4. 今日执行(立即可做,不依赖 torch)

由于本机 torch 缺失,**runtime benchmark 不可执行**,但**研究框架层**仍可推进:

1. ✅ 写研究综合报告(本文件, post-pull 版本)
2. ✅ **Pull 成功**(github.com SSH 走 192.168.6.25:7890 代理) → 拉入 1bb78af + 897aefa 2 个新 commit
3. ✅ 补 `LSTMEncoderXAttnWithMDN` 在 `tests/test_multimodal_physreg.py` 的 4 个 unit test(与 GRU 测试对位)
4. ✅ 写 `scripts/benchmark_multiseed_encoder_families.py` (思路 A + 思路 D 合并)— 5 seeds × 3 family,带 ensemble 段
5. ✅ 更新 `LNN_TLDR.md` v3 → v4,加入 round 43 seed-lucky 警告 + 5-seed 表格 + 一句话备忘更新
6. ✅ git add + commit(网络已恢复, push 一次成功即可)


## 5. 今日执行(本机可写代码,无 torch 仍 OK)

由于 **unit test 全部依赖 `import torch`**,补 unit test 会**触发 collection 失败**(与 2026-06-03 final report §5.3 同样的环境阻塞)。故本轮**只交付代码 + 文档 + 报告**;待 torch 恢复后由下一轮 cron 补跑测试 + 填 JSON。

## 6. 下一步 backlog(W+1,W+2)

| 优先级 | 项目 | 状态 | 依赖 |
|---|---|---|---|
| ★★★ | LSTM + GRU + Bi-CfC-NAD **5-seed × 3 family** (思路 A + D) | **脚本 + 测试就绪**,JSON 待 torch | torch 恢复 |
| ★★★ | LNN_TLDR v4 (本轮) | **本轮已完成**(seed-lucky 警告已加入) | — |
| ★★ | EDSSM × CfC 对照 (思路 B) | 思路,未启动 | EDSSM 仓库 clone |
| ★★ | LiquidTAD 在 EMMA 上迁移 (★ round 41 复制基础) | 部分:round 41 复制已就绪,EMMA 上待跑 | torch |
| ★ | LFM2.5-1.2B Jetson smoke (思路 C) | 思路,未启动 | 网络恢复 + 下载 |
| ★ | cfDNA 跨研究 AUC 0.40 案例复现 | 思路,未启动 | 数据获取 |
| ★ | 5-seed ensemble 在 EMMA rover 上的真实生产价值 (思路 D) | 脚本已就绪(同思路 A),JSON 待跑 | torch |

---
*本报告由 5h cron `3ac85e3c` 触发(本次迭代)。所有代码/文档就绪;runtime benchmark 与 JSON 填实等 torch 恢复后由下一轮 cron 补跑。*
