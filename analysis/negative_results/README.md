---
title: LNN 负结果台账 (Negative Results Ledger)
date: 2026-10-16
tags: [LNN, negative-result, honest-constraints, ledger]
---

# LNN 负结果台账

> **本目录是 `AGENTS.md` 约束 #4 的落地实体。**
> 约束 #4 原文：「toy benchmark 复现失败的负结果, 必须进 `negative-result` 目录而非默认 stack」。
> 在本目录建成之前（2026-10-16 之前），该约束只存在于文字里 —— 负结果实际散落在
> `docs/reports/`、`analysis/research/`、`analysis/timeseries_ablation/` 等处，无统一台账、无法机械执行。

## 0. 两条设计原则

**① 指针式，不搬移、不复制。** 每条台账只存 *判定 + grounding 路径*，权威证据留在原文件。
理由：搬移会打断 `docs/LNN_深度研读报告` 与各研读报告里已有的 `[[wikilink]]`；
复制内容则必然与原文漂移，而**漂移后的负结果台账是比没有台账更危险的东西** —— 它会变成
一个看起来有据、实际已过期的 claim 来源。

**② 台账本身可校验。** `scripts/check_negative_results.py` 校验每条 `grounding:` 路径真实存在，
且 `AGENTS.md` 约束表每行都能在台账中找到对应条目。台账失效会**硬失败**，不会静默烂掉。

```bash
python3 scripts/check_negative_results.py     # 退出码 0 = 台账与证据一致
```

## 1. 什么算负结果（收录门槛）

同时满足才收录：

1. 有**具体可证伪的主张**（不是"效果一般"这类模糊表述）；
2. 有**实测或可核查的证据**（本项目跑的 benchmark，或可复核的外部来源）；
3. 判定**已经收敛**（不是"还在跑"）。

不收敛的中间态（如某轮 sweep 暂时落后）**不收录** —— 台账收录即代表"这条已被判定为负"。

## 2. 引用纪律（硬约束）

- 引用本台账任何条目时，**必须同时给出其 grounding 路径**，让读者能自己核。
- **不得反向升级**：负结果只否证它证伪的那条主张，不得被改写为更强的结论
  （例：NR-06 不得被引为"LNN 已被证伪表达力"）。
- 外部负结果（NR-09）我方**未复现**，引用时必须标注未验证，并带上来源方自述的局限。

## 3. 台账

| ID | 被否证的主张 | 判定 | 证据来源 | 我方复现 |
|---|---|---|---|---|
| [NR-01](#nr-01) | MDN-CfC 在模仿学习上 2.4× 改善 | **证伪** | 本项目 toy benchmark (r307) | ✅ 已复现 |
| [NR-02](#nr-02) | SDE-LNN 比确定性 CfC 更鲁棒 / SOTA 横扫 | **证伪** | 本项目 toy benchmark (r306) | ✅ 已复现 |
| [NR-03](#nr-03) | LNN 可替代 LLM (PPL 维度) | **不成立** | MT-LNN 严格控制变量对比 | ⚠️ 引上游数据 |
| [NR-04](#nr-04) | MT-LNN Orch-OR / 意识门带来 AGI 路径 | **上游已自撤回** | MT-LNN README retraction 段 | ✅ 撤稿可核 |
| [NR-05](#nr-05) | LFM2.5 是纯液态模型 | **事实错误** | LiquidAI 架构搜索实测否决液态算子 + 24B-A2B config | ✅ 可核 |
| [NR-06](#nr-06) | LNN 在标准 ML 主基准上 SOTA 横扫 transformer | **无证据支持** | 多来源 (NR-01/02/03/09) | ➖ 综合判定 |
| [NR-07](#nr-07) | MT-LNN M-series 是 O(1) 工作记忆 | **事实错误** | MT-LNN 架构说明 | ✅ 可核 |
| [NR-08](#nr-08) | SDE-CfC 改进确定性 CfC 形变场 | **持平（非改进）** | 本项目 toy benchmark (r306) | ✅ 已复现 |
| [NR-09](#nr-09) | LNN 优于经典模型（金融时序预测） | **证伪（外部）** | 第三方本科论文，120 次校正比较（数据级已复核）+ 第二独立来源 | ❌ 未复现 |

---

<a id="nr-01"></a>
### NR-01 · MDN-CfC 在模仿学习上 2.4× 改善 → **证伪**

- **原始主张**：arXiv:2603.27058（Liquid Networks with MDN Heads）称 liquid+MDN 头在
  Push-T / RoboMimic / PointMaze 上 2.4× lower error、1.8× faster inference。
- **实测结论**：落到本项目 toy benchmark 后 **toy_sin 上 MDN 反而 −25% ~ −56% 退化**
  （K=1,3,5 全 HURT）；structured_irr 上 MDN ≈ MSE，仅 −0.4% ~ −0.7% 边际；
  random_irr 中性。
- **失败机理**：这些 toy benchmark 的 target 是 single-mode 平滑函数或纯噪声，
  **不存在**论文场景中"多模动作分布必须由混合表示"的结构 —— MDN 找不到东西可混合。
- **引用纪律**：不得反向表述为"MDN 头在 LNN 上无效"；这是**任务结构不匹配**，
  在真实多模态动作分布上未证否。
- `grounding: docs/reports/MDN_CfC_r307_2026-09-22.md`

<a id="nr-02"></a>
### NR-02 · SDE-LNN 比确定性 CfC 更鲁棒 / SOTA 横扫 → **证伪**

- **原始主张**：arXiv:2608.28702（Stochastic Liquid Deformation Fields）建议把
  CfC 放到 SDE 视角下，隐含 gate-level 噪声带来鲁棒性增益。
- **实测结论**：**eval 完全退化为 vanilla CfC**（这反而验证了论文 §3.2 的精确极限声称）；
  训练期 gate 高斯扰动在 toy_sin / structured_irr / random_irr 上**基本中性**；
  只有 structured_irr 在 σ=0.20 处有 −0.2% 边际改善。
- **关键反例**：learnable σ 从 0.05 漂移到 0.72（**14×**），并把 toy_sin 搞坏 +2.5%、
  structured_irr +8.3% —— 典型"梯度过正则化"信号，即噪声反而**略有害**。
- **适用范围**：该扰动仅在**训练期 jitter robustness 备援场景**有效，不支持"推理更鲁棒"。
- `grounding: docs/reports/SDE_CfC_r306_2026-09-21.md`

<a id="nr-03"></a>
### NR-03 · LNN 可替代 LLM (PPL 维度) → **不成立**

- **对照数据**（MT-LNN，125M，WikiText-103，20K 步，3 seeds，fp32，严格同条件）：

  | 模型 | PPL |
  |---|---|
  | modern Transformer | **78.86 ± 0.25** |
  | MT-LNN | 88.93 ± 0.33 |
  | simple Transformer | 94.14 ± 0.78 |

- **结论**：即便**最优**的 LNN-O 系列仍**输 ~10 PPL**；比 simple Transformer 略好，
  比 modern Transformer 明显差。"LNN 替代 LLM" 在 PPL 维度不成立。
- **引用纪律**：本条数据来自上游 MT-LNN 的受控对比，**我方未独立复现**；
  引用须标注为引上游。注意 MT-LNN 另有 5 条已验证主张（见 grounding §3.1），
  本条只否证 PPL 维度的替代性。
- `grounding: docs/reports/AwareLiquid_M1_MT-LNN_研读报告.md`

<a id="nr-04"></a>
### NR-04 · MT-LNN Orch-OR / 意识门带来 AGI / "new path to general intelligence" → **上游已自撤回**

- **状态**：MT-LNN README **自行撤回**该组主张 —— AVP failed、Φ̂ sign inverted、
  训练路径 inert。
- **结论**：任何 "LNN = consciousness / AGI 新路径" 论调在 MT-LNN 上**无任何支撑**。
- **引用纪律**：引用 MT-LNN 任何数据前**必须**先查其 README 的 retraction 段
  （grounding §3.2 列出全部 4 条撤回）。该仓库 retractions 数量多，
  **唯一安全引用**是 §3.1 的 5 条 validated。
- `grounding: docs/reports/AwareLiquid_M1_MT-LNN_研读报告.md`

<a id="nr-05"></a>
### NR-05 · LFM2.5 是纯液态模型 → **事实错误**

- **事实（2026-10-22 按 LFM2 Technical Report 原文核对后修订）**：
  LiquidAI 在 **LFM2 (arXiv:2511.23404, 2025-11-28) 的硬件在环架构搜索**里，
  把 **CfC / Liquid-S4 / S4 / S5 / Mamba / Mamba2 / 线性注意力显式放进候选空间实测**，
  在边缘延迟–内存预算下**选择了纯 gated short conv + 少量 GQA**。
  原文 §2.1 Outcomes："augmenting these stacks … with linear-attention, state-space, or
  additional convolution operators **does not improve aggregate quality** … and
  **typically worsens device metrics**."
- **⚠️ 版本 off-by-one 纠正**（本条旧版写作"LFM2 → LFM2.5 替换 ODE"，**错**）：
  - **LFM2 自己的 abstract 已是** "gated short convolutions with a small number of
    grouped query attention blocks" —— **LFM2 里就没有 ODE**，切换发生在
    **LFM1 → LFM2**，比旧版记录早一代。
  - LFM2 = **gated** short conv（单门控）；"double-gated" 是 **LFM2.5** 的 conv 变体。
  - 不是"悄悄替换"，是**架构搜索里正面测过并否决**（且输在**质量**轴，不只延迟轴）。
- **2026-10-21 新证据**：`LiquidAI/LFM2.5-24B-A2B` 的 `config.json` 核对 ——
  40 层中 **30 层 `conv` + 10 层 `full_attention`**，
  **零层是 ODE / LTC / CfC**（`model_type: lfm2_moe`, 64 experts top-4）。
- **结论**：**LiquidAI 把"液态"从 backbone 降级为组件**。LFM2.5 是
  **"Liquid + X 混合"**，不是纯 LNN —— 这不是外部质疑，是厂商自己的演进路径，
  且是**上游一手证据**（作者自述 + 搜索空间描述 + 量化结论）。
  原文 §8.2 把 LTC 定位为 **"inspired"**（启发）而非架构组件。
- **引用纪律**：本条只否证"LFM2.5 是纯液态模型"这**一个**主张；
  **不得**被引为"LNN 无用"或"液态在所有形态下都无效"——
  液态作为**局部时序组件**（本仓 CfC head / r304 路线）**不受本条否定**。
- `grounding: docs/reports/LFM2_Technical_Report_2511.23404_研读报告.md`

<a id="nr-06"></a>
### NR-06 · LNN 在标准 ML 任务 (NLP / CV 主基准) 上 SOTA 横扫 transformer → **无证据支持**

- **判定**：综合判定，**不是**来自单一实验。至今**任何 LNN 变体均未在标准主基准上显著超过
  transformer**；LNN 稳定胜出只发生在**长尾工业任务**（D-3DGS / FJSP / EEG 等）。
- **支撑**：NR-01（模仿学习退化）、NR-02（噪声反有害）、NR-03（PPL 输 ~10）、
  NR-09（外部金融时序 0/120）。
- **正向机理**（比结论更重要）：NR-09 作者把失败归因于**日频收益率信号太稀疏**，
  与本项目"只在信号密集的长尾任务胜出"一致 ——
  **判据是信号密度，不是任务名称。**
- **引用纪律**：本条不得被引为"LNN 已被证伪表达力"或"LNN 无用"；
  只否证"标准主基准 SOTA 横扫"这**一个**主张。
- `grounding: docs/LNN_深度研读报告.md`

<a id="nr-07"></a>
### NR-07 · MT-LNN M-series 是 O(1) 工作记忆 → **事实错误**

- **事实**：M-series 在 **transformer block 之上叠 MT-adapter**，其训练记忆
  **大于** vanilla transformer，**不是 O(1)**。
- **仅 O-series（attention-free）才是 O(1)**（`tests/test_long_context_memory.py` 钉住，
  0.381 MB state @ hidden=832）。
- **混淆来源**：M 与 O 两条线在宣传中常被混称"MT-LNN"。
- `grounding: docs/reports/AwareLiquid_M1_MT-LNN_研读报告.md`

<a id="nr-08"></a>
### NR-08 · SDE-CfC 改进确定性 CfC 形变场 → **持平，不是"改进"**

- **数据**：mean PSNR **+0.09 dB**，落在 run-to-run 方差内（σ ≈ 0.05 dB 量级）。
- **结论**：严格表述是**持平**。任何"改进 / 提升"的说法都不成立。
- **与 NR-02 的区别**：NR-02 讲噪声**鲁棒性**主张被否；本条讲**形变场画质**主张
  不成立。同源实验，不同主张 —— **一条实验可以否证两条主张，但两条必须分开记**，
  否则容易把"持平"读成"部分改进"。
- `grounding: docs/reports/SDE_CfC_r306_2026-09-21.md`

<a id="nr-09"></a>
### NR-09 · LNN 优于经典模型（金融时序预测） → **证伪（外部，我方未复现）**

- **来源**：`julschmidtJJL/lnn-vs-classical-forecasting`（2026-10-15 更新，
  Leuphana University Lüneburg 本科毕业论文，附 89 页 PDF，评级 1.0）
- **结论**：11 个液态变体 × 8 个 baseline × 3 个美股标的，**120 次校正比较中 0 次**
  显示液态模型显著优于传统 baseline。
- **方法学**：expanding-window walk-forward（750 日初始窗 / 21 日 fold），
  5 seeds；显著性用 **Diebold–Mariano（Harvey–Leybourne–Newbold 小样本校正）+ Holm**；
  baseline 含 LSTM、CT-RNN、Neural ODE、Naive、四个 ARMA 规格。
- **三条子结论**：
  1. 无显著优势（DM + Holm 校正后）；
  2. "最优模型"不是真赢 —— 方向命中率与"永远预测上涨"持平，其边缘是市场漂移；
  3. 液态模型确实与其他神经模型可分（轨迹长度能对神经模型场排序），
     **但表达度不转化为对传统基准的技能**。
- **作者归因**：**限制来自数据而非模型** —— 日频收益率信号太少。
- **来源方自述局限（引用必须一并带上）**：ARMA baseline 拟合时未加截距
  （statsmodels SARIMAX 默认）；论文声明补截距重拟合后零结果不变。
  另有三条（2026-10-22 核对 `src/docs/methods_dm_test_limitations.md` +
  `src/docs/thesis_results.md` 时确认）：
  (a) **DM 的 HAC 带宽问题** —— 不重叠 walk-forward fold 的损失差分存在自相关，
      朴素方差估计会**高估**显著性（p 偏乐观）；
  (b) **"k of 5 seeds" 不是独立重复** —— seed 间不构成独立重采样；
  (c) **"未检测到差异" ≠ "等效"** —— 缺等效性检验。
- **复现状态（2026-10-22 更新）**：
  - ✅ **数据级独立复核通过** —— 来源仓库现已提交结果表，从 `main` 分支拉取
    `src/docs/figures/data/dm_cfc_vs_sarimax_audit.csv` **逐行核对 120 行**：
    `significant` 全 0、`p_corrected` 全为 **1.0**，与来源方 "0 of 120" 声明吻合。
    另核对 `mae_run2a.csv`（`cfc_pure_ncp` 三标的均第 1 但领先幅度仅 ~0.4%）与
    `directional_skill.csv`（`edge_vs_always_long` 三标的全为负）。
  - ✅ **方法学实现已核对源码** —— `src/evaluation/evaluator.py`:
    `dm_test_stat(e1,e2,h=1)` 用平方误差损失差分 + Newey-West HAC（h=1 退化为普通方差）
    + HLN 小样本校正 $\sqrt{(T+1-2h+h(h-1)/T)/T}$，$t(T-1)$ 分布；
    `apply_holm_bonferroni` 用 `statsmodels.multipletests(method="holm")`；
    显著性族为 **30 个 (arm, ticker, seed) 族 × 4 baseline = 120**。
  - ✅ **第二独立来源印证** —— `niharmaddina/Liquidity`（2026-10-12）独立用
    walk-forward（502 交易日 × 4 标的）得出同向结论：LNN+ODE 在**全部 4 标的**上
    RMSE 劣于 persistence（AAPL 7.03 vs 2.54 等），方向命中率**一致低于"永远预测上涨"**。
  - ❌ **我方仍未复现其 grid**（未 clone 全仓、未重跑 11 变体 × 8 baseline ×
    3 标的 × 5 seed）。故本条维持 **"证伪（外部，我方未复现）"** 判定，
    **引用时必须标注"我方未复现"**；若要升级为一手负结果，须先本地复现。
- `grounding: analysis/research/2026-10-22/2026-10-22_lnn_research_note.md`

---

## 4. 新增一条负结果

1. 跑出的 benchmark 有**可证伪主张 + 收敛判定**（见 §1 门槛）；
2. 在本文件 §3 表格加一行 + 新增一个 `<a id="nr-NN"></a>` 条目；
3. 条目**必须**带 `grounding: <仓库相对路径>` 一行（校验脚本据此核对）；
4. 若该主张已被 `AGENTS.md` 约束表收录，同步更新约束表的 grounding 描述；
5. 跑 `python3 scripts/check_negative_results.py` 必须通过才能提交。

## 关联

- 约束全文与执行边界：`AGENTS.md` §Agent 约束
- 路线图与诚实约束论证：`docs/research/2026-09-24_technical_route_landscape.md` §五
- 全局研读索引：`docs/LNN_深度研读报告.md` §0
