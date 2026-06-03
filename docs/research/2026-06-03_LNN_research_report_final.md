---
title: LNN 今日总报告与验收记录 - 2026-06-03
date: 2026-06-03
tags: [LNN, daily-report, acceptance, EMMA, CfC, Jetson, research-report]
related:
  - "[[docs/daily/2026-06-03_LNN_research_digest]]"
  - "[[docs/research/2026-06-03_LNN_research_report]]"
  - "[[docs/research/2026-06-03_pm_LNN_research_report]]"
  - "[[docs/research/2026-06-03_evening_LNN_research_report]]"
  - "[[docs/research/2026-06-04_LNN_research_report]]"
  - "[[analysis/jetson/2026-06-03_loop_validation_summary]]"
  - "[[docs/reports/2026-06-03_loop_iteration_research_brief]]"
---

# LNN 今日总报告与验收记录 - 2026-06-03

> 本文为 2026-06-03 的正式汇总版，整合当日早报、PM 报告、晚间报告与 loop 验证记录，作为“今日研究报告 + 验收记录”的统一入口。原始分阶段报告保留，不做覆盖。

## 1. 今日输入与范围

### 1.1 数据输入

- 每日追踪摘要：`docs/daily/2026-06-03_LNN_research_digest.md`
- 研究分报告：
  - `docs/research/2026-06-03_LNN_research_report.md`
  - `docs/research/2026-06-03_pm_LNN_research_report.md`
  - `docs/research/2026-06-03_evening_LNN_research_report.md`
  - `docs/research/2026-06-04_LNN_research_report.md`（文件日期为 06-04，但内容承接 06-03 的 round 21 主线，作为同日链路补充保留）
- 辅助验证：
  - `analysis/jetson/2026-06-03_loop_validation_summary.md`
  - `docs/reports/2026-06-03_loop_iteration_research_brief.md`
  - `analysis/repo_watchlist/2026-06-03_lnn_open_source_watchlist.md`

### 1.1.1 纳入与排除说明

- **纳入**：06-03 三份阶段性报告的主线结论，以及落盘为 06-04、但实际承接 06-03 round 21 的架构 family 必要性测试。
- **不纳入本次主结论**：未在上述文档中形成闭环论证、或仅处于探索中的零散实验文件。
- **口径说明**：本文是“研究汇总完成”的正式入口，不等同于“本机代码复现实测已闭环”。

### 1.2 今日外部信号概览

根据 `docs/daily/2026-06-03_LNN_research_digest.md`：

- arXiv 候选论文 25 篇
- GitHub 候选仓库 42 个
- Hugging Face 候选模型 18 个
- 当日 arXiv API 返回 429，脚本保留上一轮成功候选池，避免日报被清空

当日最值得关注的外部信号包括：

1. `reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil` 与 `LFM2.5-1.2B-Distilled-SFT`，表明 Liquid Foundation Models 的蒸馏生态开始成形。
2. `YGTKL16/MFENCE`，把 LNN 从机器人/物理任务扩展到高频交易。
3. `Linlab2026/GCN-CfC`，把 CfC 带入分子筛选与药物发现。
4. `infinition/LSTN`，展示“liquid”思想在 Rust 文本生成引擎中的工程化尝试。

## 2. 今日研究主线结论

### 2.1 Round 13：首次量化“架构贡献”与“信息贡献”

早报的核心工作是 `UniVideoSelfXAttnWithMDN` 消融：

- 在合成 burst 任务上，`uni_video_xattn` 与 `cross_attn` 几乎持平。
- 结论是：该任务中的收益几乎全部来自双编码器 + cross-attention 架构本身，audio 信息贡献极小。
- 在真实 EMMA rover 上，`uni_video_xattn` 仍优于 `video_only`，但明显落后于真实 `cross_attn`。
- 这把“架构 vs 信息”的讨论从定性推进到定量：真实任务里 audio 物理信息确实贡献了一部分不可替代收益。

该轮形成的关键量化结论：

| 任务 | 架构贡献 | audio 信息贡献 | 结论 |
|---|---:|---:|---|
| 合成 burst | ~26.6 pp | ~1.0 pp | 增益几乎都来自架构 |
| 真实 EMMA rover | ~32.2 pp | ~18.8 pp | 架构与音频信息共同起作用 |

### 2.2 Round 28：adaptive freeze 的适用条件是 gap-driven

PM 报告把“adaptive freeze 是否只对某类任务有效”进一步收敛为机制结论：

- 当 burst 任务的 `audio_noise_std=0.05` 时，端点 gap 不足，adaptive freeze 失败。
- 当把噪声拉高到 `2.0 / 4.0` 后，pure cross-attn 与 pure video-only 的性能 gap 迅速拉大，adaptive freeze 转为稳定 PASS。
- 结论不是“这个 recipe 只适合某个任务”，而是“只要端点 gap 足够大，它就会工作”。

因此，今日形成的生产化经验法则是：

```text
if gap < 0%: 直接用 pure video_only
if 0% < gap < 5%: adaptive 可能只有边际收益
if gap >= 20%: adaptive freeze audio_only 高概率 PASS
```

### 2.3 Round 35：严格 LOO 下，EMMA 的 audio 物理信息假说成立

晚间报告解决了此前最关键的叙事冲突：

- 在 random-window 协议下，`audio=random` 一度优于 `audio=normal`，看起来像是“audio 内容不重要”。
- 但在更严格的 cross-segment leave-one-out 协议下，结论完全反转：
  - `audio=normal`：LOO mean MSE = 3.23，为最佳
  - `audio=zero`：29.33
  - `audio=random`：61.47
- 这说明此前 random-window 下的优势主要来自分布泄漏与架构正则化假象，而不是可泛化的物理信息。

今日最终收敛出的结论是：

> 在严格泛化协议下，EMMA 中“audio 携带 video 推不出的 motor RPM 物理信息”的核心假说成立；此前“audio 不重要”的结论不能作为生产决策依据。

## 3. 今日综合判断

把三轮主线合起来，06-03 的核心研究增量可以概括为三句话：

1. **先拆分贡献**：架构收益和信息收益不是一回事，必须分别估计。
2. **再抽象机制**：adaptive freeze 的有效性取决于端点 gap，而不是任务名字。
3. **最后用严格协议验真**：真正的泛化评估下，audio 物理信息是必要信号，不是噪声。

这使得仓库对 EMMA/LNN 多模态路线的认识从“经验上可行”推进到“可以写成方法论”：

- 架构层：双编码器 + cross-attention 仍是主干；
- 训练层：是否启用 adaptive freeze，先看 gap；
- 评估层：必须使用 LOO / cross-segment 这类避免泄漏的协议；
- 部署层：对 out-of-distribution 场景，必须保留真实 audio，而不能用 zero/random 近似替代。

同时，补入同日链路的 round 21 后，还可以再加一条工程约束：

- **编码器 family 层**：第二 encoder 不能被“任意 trainable recurrent 模型”替代，至少在当前证据下，Bi-CfC-NAD family 仍是必要条件之一。

## 4. 今日生态与工程补充

### 4.1 开源生态

来自 `docs/reports/2026-06-03_loop_iteration_research_brief.md` 与 watchlist 的结论：

- `GCN-CfC` 值得进入图结构任务 smoke 验证队列；
- `EDSSM` 值得与 CfC-DT 做闭式连续时间对照；
- `LSTN` 适合跟踪“非 PyTorch、Rust 侧 liquid 推理路径”；
- `LFM2.5-1.2B-Distilled-SFT` 是最适合进入 Jetson 边缘推理队列的蒸馏模型。

### 4.2 Jetson loop 验证

来自 `analysis/jetson/2026-06-03_loop_validation_summary.md`：

- 9 个 LNN 变体在 Jetson Orin Nano CPU 路径上全部完成创建、前向、反向与优化器 step 验证；
- CPU smoke benchmark 下，CfCStyle 测试 MSE 优于 GRU，但 CUDA 仍存在驱动 / wheel 版本错配问题；
- 因此边缘部署主线继续有效，但 GPU 路径仍需要单独修复。

## 5. 验收记录

### 5.1 文档整合验收

- 结果：**通过**
- 说明：已将 06-03 的三份研究报告整合为本统一入口，且不覆盖原始阶段性报告。

### 5.2 关键产物抽样完整性验收

- 结果：**通过**
- 已核对存在的关键产物：
  - `docs/daily/2026-06-03_LNN_research_digest.md`
  - `analysis/multimodal_physreg/2026-06-03_uni_video_xattn_synthetic_burst.json`
  - `analysis/emma_rover/2026-06-03_005615_emma_rover.json`
  - `analysis/emma_rover/2026-06-03_r21_gru_encoder.json`
  - `analysis/multimodal_physreg/2026-06-03_r27_adaptive_freeze_burst_h32.json`
  - `analysis/multimodal_physreg/2026-06-03_r28_burst_noise2.0.json`
  - `analysis/multimodal_physreg/2026-06-03_r28_burst_noise4.0.json`
  - `analysis/emma_rover/2026-06-03_r34_segment_loo_K20.json`
  - `analysis/emma_rover/2026-06-03_r35_loo_K20_audio_zero.json`
  - `analysis/emma_rover/2026-06-03_r35_loo_K20_audio_random.json`
  - `analysis/jetson/2026-06-03_loop_validation_summary.md`
- 说明：这里验证的是“主结论依赖的关键产物存在”，不是对所有实验文件做全量逐项审计。

### 5.3 本机自动化测试验收

- 结果：**通过**
- base 环境直接执行 `pytest -q` 时失败，原因是默认解释器为 Python 3.13，且未安装 `torch`。
- 在仓库推荐的 conda 环境中执行：

```bash
conda run -n lnn python -m pytest tests/ -q
```

- 实际结果：`142 passed in 495.38s (0:08:15)`
- 复现环境：
  - Python 3.11.13
  - torch 2.2.2
  - numpy 1.26.4
  - ncps 1.0.1
- 结论：报告中晚间阶段提及的 `142/142` 测试结论已在本机 `lnn` 环境成功复现；此前失败属于解释器与依赖环境选错，不是仓库本身回归。

### 5.4 验收结论

综合判断：

- **研究汇总：通过**  
  今日研究报告已完成整合，主线结论清晰，关键实验产物存在，且补齐了同日 round 21 的日期错位链路。

- **本机复现实测：通过**  
  在 `conda` 的 `lnn` 环境中，`pytest tests/ -q` 已实测 `142 passed`，自动化测试验收闭环完成。

## 6. 后续动作

建议按优先级执行：

1. 为避免再次误用 base 环境，后续默认使用：
   - `conda run -n lnn python -m pytest tests/ -q`
   - 或先 `conda activate lnn` 再执行实验与测试
2. 若要进一步做“强验收”，可继续补一轮关键 benchmark 脚本 smoke 与 JSON 指标抽样核对。
3. 后续若需要对外汇报，可直接以本文作为 2026-06-03 的正式日报入口，原三份分报告作为附录与审计链路保留。

## 7. 最终结论

2026-06-03 是这条研究线的重要收敛日：

- 它把“cross-attn 为什么有效”拆成了可量化的架构贡献与信息贡献；
- 把 adaptive freeze 的适用条件抽象成了可操作的 gap-driven 规则；
- 又用更严格的 LOO 协议纠正了先前可能受泄漏影响的结论，重新确认 audio 物理信息在泛化场景中的必要性。

如果只保留一个当日结论，那就是：

> **LNN 多模态路线在 06-03 不只是“跑出了更好结果”，而是第一次同时收敛了机制解释、评估协议与生产决策规则。**
