---
title: "LNN 技术路线分叉与突破评估 — 2026-08/09 月窗口 (r301-r307 + 当日补抓)"
date: 2026-09-24
source: |
  docs/reports/ 下 2026-08/09 月 30+ 研读 + 9/24 今日 systemd timer
  抓取 (papers/daily/2026-09-24_lnn_research.json, 因 arXiv Varnish throttle 暂返 0 篇,
  已复测 200 OK 待补抓) + analysis/repo_watchlist/2026-09-24_lnn_open_source_watchlist.md
  (41 repos + 24 HF models)
related_digest: "docs/daily/2026-09-24_LNN_research_digest.md"
lineage_reports:
  - "docs/reports/PLAN_Parallel_Liquid_CfC_研读报告_r301_2026-08-07.md"
  - "docs/reports/PLAN_Parallel_Liquid_CfC_Sharp_Validation_r302_2026-08-07.md"
  - "docs/reports/STE_Parallel_CfC_r303_2026-08-07.md"
  - "docs/reports/LFM2_5_Parallel_CfC_Integration_r304_2026-08-07.md"
  - "docs/reports/Midpoint_Parallel_CfC_r305_2026-08-07.md"
  - "docs/reports/SDE_CfC_r306_2026-09-21.md"
  - "docs/reports/MDN_CfC_r307_2026-09-22.md"
  - "docs/reports/2026-09-24_review_Stochastic_Liquid_Deformation_Fields_SDE_CfC_D3DGS_2608.28702_研读报告.md"
  - "docs/reports/2026-09-24_review_Liquid_NN_Drop_in_CfC_Deformation_Field_D3DGS_2606.07670_研读报告.md"
  - "docs/reports/AwareLiquid_M1_MT-LNN_研读报告.md"
tag: technical-landscape, breakthrough-assessment, route-divergence
prior_landscape: "docs/research/2026-09-08_technical_route_landscape.md"
---

# LNN 技术路线分叉与突破评估 (2026-09-24 时点)

> 本文档是 [[2026-09-08_technical_route_landscape]] 的**深度收窄 + 突破性评估**版:
> 9/8 那份枚举了 9 条横向技术路线, 本文聚焦**2026-08/09 月窗口内**新出现的 **4 条路线分叉点** 与 **"是否算重大突破"** 的诚实判断。
>
> 与 [[LNN_深度研读报告]] §0 (项目定位) 保持一致: **LNN 不是 GPT/Claude 级 dense LLM 替代品**, 本文评估在该边界内进行。

## 0. 一句话判断

> **没有"Sora 时刻"式颠覆突破**, 但出现了 **4 条互不收敛的技术路线分叉**, 以及**方法论级突破 ——"诚实负结果"成为主流** (r306 SDE-CfC、r307 MDN-CfC 均经 toy benchmark 严苛复现才公开撤回)。
> LNN 正从"一种 RNN 变体"演化成**多个互不收敛的技术家族**。

## 一、4 条技术路线分叉

### 路线 A · Parallel-CfC 边缘推理化 (本仓 r301-r305 系列, 工业向)

| 轮次 | 论文 / 方案 | 核心 |
|---|---|---|
| **r301** | PLAN Parallel Liquid CfC | **first-order Taylor 近似** 替代闭式 CfC, 允许 `torch.scan` 并行扫描 |
| r302 | PLAN Sharp Validation | 在 irregular timestep + 长序列上确认近似误差 < 5% |
| **r303** | STE Parallel CfC | per-neuron straight-through mask, backbone 走 parallel anchor vs sequential path 二选一 |
| **r304** | LFM2.5 Parallel CfC Integration | **与 LiquidAI 商业 LFM2.5-1.2B-Instruct 对齐**, LSTM 头换成 Parallel-CfC, Jetson Orin Nano 实测可跑 |
| r305 | Midpoint Parallel CfC | Midpoint 方法替代 Taylor, 精度更好但吞吐略降 |

**路线意义 (工程最强)**: 这是**最成熟的工程路线** —— 把 CfC 从"理论优雅但难并行" 变成"可编译到 Triton / CUDA graph 的边缘友好核"。LFM2.5 整合代表 LNN 进入了**真正的商业模型栈**, 而不仅仅是 benchmark 玩具。

**落地优先级**: ★★★★★ (本仓 [`lnn/core/parallel_cfc.py`] 是首选生产路径)

### 路线 B · SDE / 概率论重解释 (SDE-CfC, arXiv:2608.28702 + 2606.07670)

- **起点 (2606.07670)**: CfC 单元作为 D-3DGS 形变场的 drop-in 替代品 (NeRF-DS 配置 0.32M params, 比 MLP 小 39%)
- **SDE 拓展 (2608.28702, APSIPA ASC 2026)**: 把 LTC 单元视为 Itô 过程, 在门预激活加 $\lambda \varepsilon$ 单步 Euler-Maruyama 噪声
- **关键性质**:
  - 推理期**严格等价于确定性 CfC** (0 推理开销, 0 额外参数)
  - $\lambda = 0$ 时严格退化为确定性 CfC (size-matched 对比)
  - 单步 EM 在 $\sigma$ 之前加 $\lambda \varepsilon$ 即可 (1 行代码改动)

**但 toy benchmark 复现结果 (r306) 是诚实负结果**:
- D-NeRF mean PSNR: SDE-LNN 38.28 vs MLP 38.19 → **+0.09 dB (在 run-to-run 方差内, σ≈0.05 dB)**
- NeRF-DS mean PSNR: SDE-LNN 23.73 vs MLP 23.82 → **-0.09 dB (噪声略有害)**
- 噪声水平扫描 (λ ∈ {0, 0.05, 0.10, 0.20}): **确定性 CfC (λ=0) 已是甜蜜点**, 加噪声无收益

**方法论意义 ≫ SOTA 意义**: 这是首次**对"LNN 鲁棒性来自 SDE 随机性" 这个根深蒂固的叙事做实证检验**, 发现确定性 CfC 在干净基准上**已经接近甜蜜点**, 鲁棒性需要显式 SDE 拓展才能拿到。**LNN "鲁棒性"叙事的微妙纠正**: CfC 文献反复强调 LNN 对噪声和抖动鲁棒, 本文的负结果说明 — 这种鲁棒性**在确定性 CfC 的实际部署中可能并未真正激活**。

**落地优先级**: ★★ (架构上有 0 改动 + 0 推理成本优势, 但实证无收益, 仅作训练期 jitter robustness 备援)

### 路线 C · 多模态动作分布 / 模仿学习 (MDN-CfC, arXiv:2603.27058 / r307)

- MDN head 替换 Linear 头, Bishop-style Gaussian mixture
- 论文在 Push-T / RoboMimic Can / PointMaze 上报 **2.4× lower error / 1.8× faster / half params**

**但 toy benchmark 复现失败 (r307) — 诚实负结果**:

| cond | toy_sin | structured_irr | random_irr | params |
|---|---|---|---|---|
| mse_cfc (base) | 0.0349 | 0.2454 | 1.0275 | 945 |
| mdn_k1 | **+25.7%** | -0.4% | -0.2% | 979 |
| mdn_k3 | **+36.8%** | -0.7% | -0.2% | 1081 |
| mdn_k5 (paper) | **+56.0%** | -0.5% | +0.1% | 1183 |

**结论**:
- toy_sin 上 **K 越大退化越大** (K=1→5: +25.7%→+56.0%) —— 典型 over-parameterisation
- structured_irr (paper-style bimodal target) 上几乎打平 —— mode 互为相反数, **混合均值塌缩成 MSE 头预测零**
- 论文 2.4× 改善依赖**真正的"对称但不互为相反"多模动作分布** (Push-T 的"左转/右转"), toy 上的"互为相反数" 模式触发不了 MDN 优势

**落地优先级**: ★ (`MDNCfCNetwork` + `MDNHead` 实现留在 repo, 进 negative-result 目录; 仅在真实多模动作分布任务上重新启用)

### 路线 D · 类脑 / 生物启发路线 (MT-LNN, AwareLiquid/M1, 9/14 研读)

- 借鉴微管 13 根 protofilament, 把 transformer block 换成 13 路 MultiScaleResonance
- **attention-free O(1) 工作记忆**: hidden=832 时仅 **0.381 MB**, 与上下文长度无关
  - 128k context: 比 fp16 GQA=1 小 1008×, 比 2-bit + GQA=8 小 1070.9×
  - 1M context: 小 8563× / 8567×
  - **crossover: 17–976 tokens** (低于此区间 Transformer 仍占优)
- 流式 state-only 解码: 1000-token decode footprint `~1020 KB → 4.1 KB`
- Continual learning 零新增参数 (`replay.py` bounded reservoir-replay: forgetting≈0, acc≈1.0)

**⚠️ 重要 retraction** (README 已自我撤回):
- **Orch-OR / 意识门**: AVP failed, Φ̂ sign inverted, 训练路径 inert
- **MT-LNN 125M PPL 88.93 vs modern Transformer 78.86** (同 20K 步 / fp32 / WikiText-103): 严格控制变量下 LNN 仍输 ~10 PPL, "LNN 替代 LLM" 在 PPL 维度**不成立**
- **M-series 不是 O(1)**: 在 transformer block 上叠 MT-adapter, 训练记忆 > vanilla transformer

**⚠️ 1★ 单人项目**: 不作 baseline 引用, 只作"架构思想 + 概念验证" 来源

**落地优先级**: ★★ (架构思想可借鉴: 13-protofilament + MultiScaleResonance 是少数与主流 transformer 平行的 attention-replacement 路径, 但生产用须自实现)

## 二、是否算"重大突破"?

### ✅ 算"突破" 的:

1. **方法论突破: 诚实负结果的常态化 (r306 / r307)**
   - LNN 社区开始对 paper claim 做系统性 toy benchmark 复现
   - 负结果进 `negative-result` 目录而非默认 stack
   - 这是 ML 整体都罕见的科研态度, 值得记入知识库

2. **工程突破: Parallel-CfC + LFM2.5 整合 (r301-r304)**
   - 首次让 LNN 进入商业模型栈 (LiquidAI LFM2.5)
   - 从"论文 toy" 跨到 "Jetson Orin Nano 可跑"
   - PLAN / STE / Midpoint 三种 parallel scan 策略让 CfC 永远退出"无法编译到 CUDA graph" 的窘境

3. **架构突破: MT-LNN attention-free O(1) 记忆 (条件性)**
   - 仅限 O-series (M-series 不是 O(1))
   - 仍 1★ 单人项目, 但**架构思想可借鉴**

### ❌ 不算"突破" 的 (需要警惕):

1. **没有 SOTA 横扫**: SDE-CfC、MDN-CfC 都只是"接近 MLP / 持平 MSE", **没有任何 LNN 变体在标准 ML 任务上显著超过 transformer**
2. **商业主线依然是 LiquidAI 内部**: LFM2.5-1.2B/2.6B/8B 仍是 closed-weight, LNN 社区多在 toy / 边缘 / 个性化任务
3. **D-3DGS / FJSP / EEG 等场景**: LNN 在这些**长尾工业任务**上是稳定选择, 但都没到 "GPT-3 moment" 级别的影响力
4. **MT-LNN PPL 不敌 Transformer**: "LNN 替代 LLM" 在 PPL 维度已被实证否定, 引用 M-series 做 LLM 替代 claim 是误用

## 三、与本仓 `liquid_*` 路径的映射

| 本仓工作 | 对应路线 | 状态 |
|---|---|---|
| [`lnn/core/parallel_cfc.py`] | A (PLAN) | **首选生产路径** (r301-r305 已落地) |
| [`lnn/core/mdn_cfc.py`] | C (MDN-CfC) | 进 negative-result 目录 (r307) |
| [`scripts/bench_mdn_cfc.py`] | C | 100 epochs × 3 seeds, 52s |
| `Binary Gated Pulse` / `BlendGatedCfC` | 路线 1 (continuous-time substitution) | 持续优化 |
| [`scripts/jetson_lnn_benchmark.py`] | A + 边缘部署 | 9/22/23 因 libcudss.so.0 退化, 9/24 CPU fallback 兜住 |
| (未实现) SNCP-PPO-Lite 13-protofilament actor | D (MT-LNN 思想借鉴) | 候选实验队列 |

## 四、给未来的建议 (落地导向)

| 你的目标 | 推荐路径 | 入口 |
|---|---|---|
| **可立即落地的工程成果** | 抓 r304 LFM2.5-Parallel-CfC | 把 [`lnn/core/parallel_cfc.py`] 接进你现有的边缘推理 pipeline |
| **学术研究题材** | 跟 r307 MDN-CfC 负结果延伸 | 构造"对称但不互为相反" 的多模 toy target, 验证 Diffusion Policy on CfC 是否能拿到 paper 报的效果 |
| **长期视野** | 观察 MT-LNN O-series 2027 发展 | 这是少数可能挑战 transformer 长上下文 O(n²) 的非 attention 路径 |

## 五、诚实约束 (Future Claude Sessions 必读)

> **不要在文档 / 报告 / 答辩中重复以下未经验证的 claim** (本仓已有反例支撑):

1. ❌ "MDN-CfC 在模仿学习上 2.4× 改善" — r307 toy benchmark 复现失败, 进 negative-result
3. ❌ "SDE-LNN 比确定性 CfC 更鲁棒" — r306 toy benchmark 噪声反而略有害, 仅在训练期 jitter robustness 备援场景有效
4. ❌ "LNN 可替代 LLM (PPL 维度)" — MT-LNN 125M WikiText-103 PPL 88.93 vs Transformer 78.86, 严格控制变量下输 ~10 PPL
5. ❌ "MT-LNN Orch-OR / 意识门带来 AGI" — README 已自撤回 (AVP failed, Φ̂ sign inverted, 训练路径 inert)
6. ❌ "LFM2.5 是纯液态模型" — LiquidAI 自己把 ODE + RK4/Euler solver 替换为 double-gated conv + GQA, 是"Liquid + X 混合" 而非纯 LNN
7. ❌ "LNN 在标准 ML 任务上 SOTA 横扫 transformer" — 任何 LNN 变体均未在 NLP / CV 主基准上显著超过 transformer
8. ❌ "MT-LNN M-series 是 O(1) 记忆" — M-series 在 transformer block 上叠 MT-adapter, **不是 O(1)**, 仅 O-series 是

> 任何未来 claim 必须先在 r301-r307 + MT-LNN 研读中找到对应 grounding, 否则标为"未验证" 并走诚实负结果路径。

---

**维护说明**: 本文为 2026-09-24 时点快照, 每月由 `lnn-daily-research.timer` 触发更新, 或新 scheme 出现时即时追加。
**关联索引**: [[LNN_深度研读报告]] §0 (项目定位) 与本文"§0 一句话判断" 保持一致。