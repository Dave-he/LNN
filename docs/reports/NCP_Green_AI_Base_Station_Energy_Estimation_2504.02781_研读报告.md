---
title: "Towards Green AI-Native Networks: Evaluation of Neural Circuit Policy for Estimating Energy Consumption of Base Stations"
date: 2025-04
tags: [LNN, LTC, NCP, Neural-Circuit-Policy, green-AI, energy-efficiency, telecom, RAN, MLOps, sparsity]
---

# 研读报告：NCP 绿色 AI 网络 — 用稀疏神经回路策略估算基站能耗，训练能耗比 LSTM 低一个数量级

## 1. 元数据

- **论文标题**：Towards Green AI-Native Networks: Evaluation of Neural Circuit Policy for Estimating Energy Consumption of Base Stations
- **作者**：Selim Ickin, Shruti Bothe（Ericsson Research）；Aman Raparia, Nitin Khanna, Erik Sanders（Ericsson AB）
- **机构**：Ericsson Research / Ericsson AB（瑞典 爱立信）
- **发表时间**：2025-04-03 (arXiv: 2504.02781v1, cs.LG)；15 页，含 IEEE 风格模板（会议名未填）
- **代码**：未给出公开仓库；数据与模型为电信运营商内部数据（Ericsson 自有 KPI 计数器）
- **本地 PDF**：[papers/2504.02781/2504.02781.pdf](../../papers/2504.02781/2504.02781.pdf)
- **本地文本**：[papers/2504.02781/2504.02781.txt](../../papers/2504.02781/2504.02781.txt)
- **数据**：基站（cell / radio unit / site）级多变量时序 KPI + 实测能耗，15 分钟粒度；训练集前 65%，测试集最后 30%（含合成噪声/漂移变体）
- **关联概念**：LTC (Hasani 2021) / NCP (Lechner 2020) / 稀疏连线 / 连续时间 ODE / 电信 RAN 能耗 / MLOps / Green AI / Kepler 能耗测量

## 2. 核心问题

AI-Native 网络架构里，ML 模型被用来做基站节能决策（推荐休眠/降配等动作），**但推荐动作本身要靠跑 ML 模型来产生**——这是一个**自我消耗悖论**：为省电而部署的模型自己也要耗电、占内存、且需要持续维护（尤其是要适应数据分布漂移而反复调参）。

作者把背景数字摆得很清楚：**RAN（无线接入网）约占移动网络总能耗的 75%**。同时他们指出 ML 模型侧的两个放大器：

1. **模型数量爆炸**：每个基站/小区/无线单元各有多个模型，每个模型针对不同任务与不同数据分布；
2. **分布漂移 → 手工调参**：基站重配置或用户画像变化会改变输入分布，模型需要持续重训，而**每次试错式调参都是一次训练能耗**。

传统解法是**网络连接剪枝（edge pruning）**：识别并迭代删除不贡献的参数。这能省参数，但作者指出它**不改变架构的连接拓扑**，而 NCP 的稀疏性是**结构性的**（连接拓扑本身就是稀疏的、且被设计成可解释可审计的）。

作者由此提出本文的问题：

> **用稀疏结构化的 NCP（而非 LSTM）来估算基站能耗，能否在维持可接受精度的前提下，同时降低训练能耗、内存占用与 MLOps 维护成本；并且——能否用"超参不敏感"来取代"试错式调参"，从而把调参过程的能耗一并省掉？**

## 3. 方法论与核心思路

### 3.1 总体架构

```
基站侧 KPI 计数器（15 分钟粒度）
  资源块数 / 数据量 / 连接用户数 / 活跃会话数 / 传输量 / 活跃小区与无线单元数
    │
    ▼
  预处理 + 聚类（按站点/无线单元/时间窗聚合；用 coherence score 定簇数）
    │
    ▼
  ┌──────────────────────┬──────────────────────┐
  │ LSTM (baseline)      │ NCP (90% sparse)      │   ← 多组超参扫描
  │ 全连接                │ AutoNCP 稀疏中间层      │
  │ epochs ∈ {50..800}   │ epochs ∈ {50..800}     │
  │ neurons ∈ {16,32,64} │ neurons ∈ {16,32,64}   │
  └──────────────────────┴──────────────────────┘
    │                          │
    ▼                          ▼
  估算基站能耗（回归）      估算基站能耗（回归）
    │
    ▼
  评测三件事：
    (1) 精度：R² / MSE（含 >90 百分位尾部 MSE）
    (2) 资源：参数量 + 训练能耗（Kepler 工具实测，Kubernetes 集群）
    (3) 鲁棒：超参敏感性 + 合成噪声/漂移（KS 统计量）
```

**方法论上最值得学习的一点**：作者没有只比"最佳配置"，而是**扫描完整的超参网格**（epochs × 中间层神经元数 × 两种模型），用**同一套指标同时评精度、能耗、参数量**，并把每个点画成"均值 + 95 置信区间"的圆点图。这使得"能耗"成为一个**与精度同等的优化目标**，而不是事后附加的脚注。

### 3.2 LTC 连续时间动力学（论文 Eq. 1–4）

论文先用 LTC 的标准形式做背景铺垫。CT-RNN 的形式：

$$
\frac{dx(t)}{dt} = -\frac{x(t)}{\tau} + S(t), \quad S(t) \in \mathbb{R}^M \tag{1}
$$

其中输入项由隐状态与输入共同决定：

$$
S(t) = \left( f\big(x(t), I(t), t, \theta\big) \right)\big(A - x(t)\big) \tag{2}
$$

把 Eq.1 与 Eq.2 合并，得到**系数不再为常数**的变系数形式：

$$
\frac{dx(t)}{dt} = -\left[\frac{1}{\tau} + f\big(x(t), I(t), t, \theta\big)\right] x(t) + f\big(x(t), I(t), t, \theta\big) A \tag{3}
$$

对应的**系统时间常数**为：

$$
\tau_{sys} = \frac{\tau}{1 + \tau\, f\big(x(t), I(t), t, \theta\big)} \tag{4}
$$

**这是全文的机制支点**：$\tau_{sys}$ 随输入与状态变化，因此即使输入出现极端高/低异常值，$\tau_{sys}$ 仍**有界且自适应**，神经元状态也随之有界——输出不会发散。论文由此归纳 LTC 相对 CT-RNN 的三条优势：**输入依赖的可变时间常数**、**状态有界（因此更不易过拟合）**、**大规模稀疏性使模型更小**。

### 3.3 NCP：LTC 作为积木的稀疏架构

NCP 受 *C. elegans* 秀丽隐杆线虫的神经系统启发（仅 302 个神经元却能产生复杂行为）。其关键结构特征：

- **中间层 90% 稀疏**，且中间神经元↔command 神经元之间有**唯一的连接拓扑**；
- 每个神经元是 **LTC 系统**，NCP 是**四层**特定架构（sensory / intermediate / command / motor），中间神经元高度稀疏且**循环连接**；
- 突触电流 $I^s_{ij}$ 由**输入依赖的非线性** $f$ 决定，$f$ 是突触前状态 $x_j(t)$ 与突触参数的函数；外部输入电流 $I_{in}$ 与漏电流 $I_{leakage}$ 共同作用于突触后状态 $x_i(t)$。

### 3.4 与 LSTM 的架构差异（作者自述）

| 维度 | LSTM | NCP |
|---|---|---|
| 连接 | **全连接** | **稀疏（90%），唯一拓扑** |
| 动力学 | 离散时间门控（memory cell + gates） | **一阶连续时间系统 + 非线性门** |
| 超参敏感性 | **高**（神经元数/epochs 增加即过拟合） | **低**（提高超参反而更稳） |
| 数据漂移处理 | 作者称"struggle with handling data drift" | 作者称"handle data drift more effectively" |
| 参数量（本论文 best case） | **89,129** | **26,470** |

**注意最后两行**：作者声称 NCP 在数据漂移上更鲁棒，但 §4 的实验结果**直接推翻了这个说法**（见 §5.2）。这类"机制描述先于实验、且被自己的实验否定"的写法在引用时必须特别小心。

### 3.5 评测设计

- **能耗测量**：用 **Kepler 工具**（Kubernetes 生态的容器级能耗计量）在 Kubernetes 集群中实测训练过程能耗，最大资源配额 40Gi / 10 CPUs；
- **分布漂移量化**：用 **Kolmogorov–Smirnov 双样本检验**（Python `ks_2samp`）量化合成测试集与原测试集的差异，作为"漂移程度"的横轴；
- **尾部性能**：除总体 R² 外，还专门报 **>90 百分位样本的 MSE**——因为**低估高能耗基站的代价大于高估**（漏掉节能动作更糟），这是电信场景的领域正确取舍。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $\dfrac{dx(t)}{dt} = -\dfrac{x(t)}{\tau} + S(t)$ | CT-RNN 基础形式（常数时间常数） |
| Eq. 2 | $S(t) = \big(f(x(t), I(t), t, \theta)\big)\big(A - x(t)\big)$ | 输入项：由隐状态/输入决定的非线性调制 |
| Eq. 3 | $\dfrac{dx(t)}{dt} = -\left[\dfrac{1}{\tau} + f(x(t), I(t), t, \theta)\right]x(t) + f(x(t), I(t), t, \theta)A$ | LTC 变系数状态方程（系数不再是常数） |
| Eq. 4 | $\tau_{sys} = \dfrac{\tau}{1 + \tau\, f(x(t), I(t), t, \theta)}$ | **输入依赖的可变系统时间常数**（全文机制支点，保证状态有界） |

## 5. 关键成果与贡献

### 5.1 主结果（Fig. 4 / Fig. 5）

**精度（Fig. 4）**：

| 模型 | 配置 | R² | >90 百分位 MSE |
|---|---|---|---|
| NCP | 64 神经元 / 400 epochs | **0.736**（最高） | 0.34 |
| NCP | 16 神经元 / 400 epochs | 0.729 | **0.33（最低尾部 MSE）** |
| NCP | 32 神经元 / 400 epochs | 0.736 | 0.36 |
| NCP | 16 神经元 / 200 epochs | 0.734 | — |
| **LSTM** | **16 神经元 / 100 epochs** | **0.748（最高）** | **0.31** |
| LSTM | 16 神经元 / 50 epochs | ≈ 0.748（相近） | — |
| LSTM | 16 神经元 / **800 epochs（过拟合）** | **0.56（崩塌）** | **0.77** |
| NCP | 64 神经元 / **800 epochs** | **0.724（几乎不降）** | **0.32** |

**这是全文最关键的对比**：**最优 LSTM 的 R² 0.748 略高于最优 NCP 的 0.736**，尾部 MSE 也是 LSTM（0.31）略优于 NCP（0.33）。**作者自己明确承认**："In general, the R² score and MSE differences between the best case models were not considered significant, still LSTM performed marginally better in overall"（Table 1 汇总栏亦写明 LSTM "Marginally higher Accuracy"、NCP "Marginally Lower Accuracy"）。

**能耗与参数量（Fig. 5）**：

| 场景 | 能耗 (J) | R² | 参数量 |
|---|---|---|---|
| LSTM, 16 神经元 / 100 epochs（最佳精度） | **4,458** | 0.748 | **89,129** |
| LSTM, 16 神经元 / 50 epochs | 2,229 | ≈0.748 | 较小 |
| NCP, 64 神经元 / 400 epochs | 3,423 | 0.736 | **26,470** |
| NCP, 16 神经元 / 400 epochs | 2,116 | 0.729 | 更小 |
| NCP, 16 神经元 / 200 epochs | **1,058** | 0.734 | 更小 |

**关键结论**：NCP 在**同精度档位下的训练能耗显著更低**——最佳 NCP（1,058 J）只有最佳 LSTM（4,458 J）的 **23.7%**，即约 **4.2× 能耗优势**；参数量 **26,470 vs 89,129**（**3.4× 更少**）。作者进一步指出：**等配置下（LSTM 同参数/同 epochs）能耗差一个数量级**。

### 5.2 鲁棒性（Fig. 6）——作者的自我否定

作者对两个最佳模型注入合成噪声与合成漂移，用 KS 统计量横轴量化漂移程度。结论出人意料：

> "both models are **equally sensitive** to the noise and drift in the testset. Hence, **we can not conclude that NCPs are robust** to additional noise and drift in the testset."

即：**NCP 在噪声/漂移上并不比 LSTM 更鲁棒**。这**直接否定了 §1.4 中"handle data drift more effectively"的机制性宣称**——是全文内部一致性问题最突出的一处。

### 5.3 真正成立的结论：超参不敏感 → 省掉试错调参的能耗

这才是本文的核心贡献，且它是一个**间接**但可量化的机制：

- LSTM：精度对超参**高度敏感**，需多次试错 → **每次试错都是一次完整训练能耗**（实验中 LSTM 全网格的能耗总和远大于 NCP）；
- NCP：把 epochs 和神经元数**直接拉高**（400 epochs / 64 神经元）反而**精度更稳、能耗仍低于 LSTM**——**"一次到位"取代"迭代试探"**。

作者的表述："it would be reasonable to keep the number of parameters high in the first attempt to guarantee both high performance, reasonable energy consumption as well as reducing the time spent for model hyper-parameters in ML operations."

### 5.4 贡献清单

1. **首次把 NCP 用于电信基站能耗估算**这一真实运营商场景（真实 KPI + 实测能耗数据，Ericsson 内部）。
2. **给出"NCP vs LSTM"的完整超参网格对照**：精度相当（LSTM 微优）、**能耗低约 4×（最佳档 1,058 J vs 4,458 J）、参数量少约 3.4×**。
3. **证明 NCP 对超参（epochs / 神经元数）鲁棒**：LSTM 800 epochs 崩塌至 R² 0.56，NCP 同设置仍保持 0.724——**这一点是全文最扎实的机制级结论**。
4. **量化了调参过程的能耗隐忧**（MLOps 侧成本），并给出"一次到位"的工程建议。
5. **负面结论同样有价值**：NCP 在噪声/漂移下**不比 LSTM 鲁棒**（作者明确写出）。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **未观察到 NCP 相对 LSTM 的额外噪声/漂移鲁棒性**，留待未来工作；
- 未来方向：不同**稀疏水平**对性能的影响、**SDE vs ODE 的性能对比**、为 NCP 嵌入提升抗噪/抗漂移的新方法。

### 6.2 隐含局限（本仓视角）

- **(a) 精度上 NCP 实际略输 LSTM**（0.736 vs 0.748，尾部 MSE 0.33 vs 0.31）。摘要只说 "sustaining high model accuracy"、结论说 "superior performance ... while sustaining high model accuracy"，**均未明说精度略输**（只在正文与 Table 1 承认）。**引用时必须写明"NCP 精度与 LSTM 相当但略低"**，不能写成"NCP 更准"。
- **(b) 能耗对比存在"选点偏差"风险**：NCP 的 1,058 J 场景（16 神经元/200 epochs，R² 0.734）与 LSTM 的 4,458 J 场景（16 神经元/100 epochs，R² 0.748）**不是同一精度点**——严格对齐精度后 NCP 仍更省（3,423 J @ R² 0.736 vs 4,458 J @ 0.748，约 23% 省），但"4.2×" 这个数字来自**取两个模型各自最优点的比较**，属于**最优-最优对比**而非**同超参对比**。论文自己也说等配置下"order of magnitude higher"，但没给该对比的完整数字表。
- **(c) 能耗测量口径不可外推**：Kepler 在 Kubernetes 容器中测的是**整机/节点级能耗**，不是纯模型计算能耗；且训练在**云端 Kubernetes 集群**完成，与"边缘基站部署"不是同一环境。**这与本仓 Jetson 边缘场景不是一回事**——不能拿本文的焦耳数去推边缘设备。
- **(d) 关键超参的"免费午餐"存疑**：NCP 表现最好的配置是 **400–800 epochs**，而 LSTM 在同样 epochs 下过拟合。也就是说 **NCP 的优势部分来自"它能在高 epochs 下不崩"**，若 NCP 也需要早停，其能耗优势会被压缩。论文未做**等 epochs** 的直接对照（只给了各自最优）。
- **(e) 无方差/重复实验**：R² 与能耗都是单次测量，Fig. 4/5 的圆点虽有 95 百分位区间，但**没有说明是多次 run 的置信区间还是 bootstrap**；能损测量的重复性未讨论。
- **(f) 数据集不可复现**：Ericsson 内部 RAN 数据，**外部研究者无法复现**；且未说明站点数量、无线单元数量、簇数、最终样本量（只说 15 分钟粒度聚合）。
- **(g) §1.4 的机制宣称与 §4 实验矛盾**（漂移鲁棒性），如前所述；这是**引用该文时最需要警惕的一点**。
- **(h) "Green AI"的因果链未闭合**：本文只测了"训练能耗"，**没有测推理能耗**，而 RAN 基站侧的**推理**才是 7×24 持续运行的真实能耗来源。作者的目标是"估算能耗的模型"，其推理能耗在实际部署中同样重要，但本文未涉及。

### 6.3 未来方向（作者 + 本仓补充）

- 作者：不同稀疏水平的影响、**SDE vs ODE 对比**、提升抗噪/抗漂移的方法；
- 本仓补充：**推理能耗**（本仓 Jetson benchmark 正好有 `analysis/jetson/` 产物可承接）、**同 epochs 严格对照**、**多站点外部验证**。

## 7. 对本仓的意义

**可作 `lnn/core/variants.py` 的什么变体**：本文的核心价值不是新架构，而是**"NCP 稀疏性 + LTC 变系数时间常数"的组合配方**与**"把能耗当作一等优化目标"的评测范式**。本仓 `experiment_autoncp.py` 已有 AutoNCP 线路，可新增一个 **"sparsity 水平扫描"消融轴**（10% / 50% / 90% / 95% 稀疏率 → 精度 / 参数量 / 训练能耗 / 推理延迟），直接对应作者列出的未来方向"understanding the impact of different sparsity levels on the performance"。

**可嫁接到哪个 bench 脚本**：`experiment_autoncp.py`（NCP 稀疏线路）、`experiment_all_variants.py`（LTC vs CfC vs LSTM vs GRU 统一对照，可直接改造成本文的 Table 1 口径）、`experiment_timeseries.py`（多变量时序回归）、以及本仓的 Jetson benchmark 流水线（承接 §6.2(h) 的推理能耗缺口）。

**提供哪个新 ablation axis**：
1. **"等 epochs / 等参数量"严格对照轴**：本仓所有跨架构对比**必须同时报"同超参"与"各自最优"两组数字**——本文的教训是"最优-最优"会掩盖机制，且在能耗维度上会放大优势；
2. **"超参敏感性扫描"轴**：把 epochs × 神经元数做成网格，**重点看"高 epochs 是否崩塌"**——这是 NCP 最有说服力的机制级证据（LSTM 0.56 vs NCP 0.724），比单点精度更能说明问题；
3. **"训练能耗 + 推理能耗 + 内存"三合一轴**：本仓已有 Jetson 产物，可把能耗测量接入（Kepler 思路 → 边缘侧改用 `tegrastats`）；
4. **"稀疏率 vs 精度"轴**（作者明确的 future work，本仓可先做）；
5. **"变系数时间常数"机制轴**：直接检验 $\tau_{sys}$ 的输入依赖是否真的带来过拟合抵抗力（本文只给了相关性，未隔离 $\tau_{sys}$ 变与不变）。

**Verdict**：
- **"NCP/LNN 在资源受限的电信边缘场景可显著降低训练能耗与内存（1,058 J vs 4,458 J；26,470 vs 89,129 参数）" → TARGET-POSITIVE**（本文最扎实的结论，且是**能耗维度**而非精度维度）；
- **"NCP 精度优于 LSTM" → TARGET-NEGATIVE**（**方向相反**：最优 LSTM R² 0.748 > NCP 0.736，尾部 MSE 亦略优；本文只能支持"精度相当"）；
- **"NCP 对数据漂移/噪声更鲁棒" → TARGET-NEGATIVE**（**作者自己用实验否定了该宣称**，§4 明确写"we can not conclude that NCPs are robust"；§1.4 的机制性表述与实验矛盾，引用必须带此警告）；
- **"NCP 超参鲁棒性 → 省掉试错调参的 MLOps 能耗" → TARGET-POSITIVE-WITH-NUANCE**（机制清晰且有数据支撑，但能耗口径为云端 Kubernetes、不可外推到边缘；"免费午餐"部分来自"高 epochs 不崩"）；
- **"NCP 适合作为电信基站能耗估算的推荐架构" → TARGET-DEPENDENT**（数据集为 Ericsson 内部数据、不可外部复现，本文只覆盖**训练**能耗，未覆盖推理能耗）。
