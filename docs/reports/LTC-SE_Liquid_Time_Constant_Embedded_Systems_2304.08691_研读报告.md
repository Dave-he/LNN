---
title: LTC-SE — Liquid Time-Constant Networks 的工程化重构版 (TensorFlow 2.x 统一类库)
paper: https://arxiv.org/abs/2304.08691v1
arxiv_id: 2304.08691
date: 2026-10-26
tags: [LNN, LTC, liquid-time-constant, embedded-systems, TensorFlow2, software-engineering, negative-result-audit]
status: deep-read
report-author: LNN-research-agents
source: docs/daily/2026-10-26_LNN_research_digest.md arXiv 候选 (2304.08691, 2023-04-18 提交, Karabuk University / GWD Göttingen / Lebanese American University)
---

# LTC-SE — 研读报告

> 论文: [LTC-SE: Expanding the Potential of Liquid Time-Constant Neural Networks for Scalable AI and Embedded Systems](https://arxiv.org/abs/2304.08691v1)
> 提交: 2023-04-18 (v1, 11 pages / 5 figures / 5 tables) | 作者: Michael Bidollahkhani, Ferhat Atasoy, Hamdan Abdellatef
> 报告日期: 2026-10-26 | 工具: `skills/paper-analyzer` (arXiv 模式, 读 PDF 全文 + abstract + 本仓 `papers/foundational/hasani_2021_ltc.txt` 原文比对)
> 强关键词命中: **liquid time-constant networks, LTC, embedded systems, scalability** (selector score 35, 当日 top-1)

---

## 1. 一句话定位

> **LTC-SE 是一篇"软件工程论文", 不是一篇"算法论文"** —— 它的全部技术贡献是把 Hasani et al. 2021 的 LTC 实现重构成一个继承 `tf.keras.layers.AbstractRNNCell` 的统一类库 (LTCCell / CTRNN / NODE / CTGRU 四个 cell 同一套配置接口), 并**新增了一个"精确参数量"列 (n=128, k=8) 与一个"LTC-SE 计算深度"列**。经本报告逐格复算: **新增的两列全部与论文自己给出的公式不自洽** (5/5 行的参数量不满足同行公式; 计算深度列中 hard-tanh 一行比原论文更差), 因此**这篇论文的"性能提升"结论在本仓视角下不成立** —— 它既没有改 LTC 的动力学方程, 也没有改 fused solver, 所谓"outperforms"只体现在数值上不可解释的表格里。**它对本仓的真正价值是反面教材: 一篇被 arXiv 收进 2023 年的 LNN 应用论文, 其"精确参数量"可以整列算错。**

---

## 2. 元数据

| 字段 | 值 |
|---|---|
| arXiv id | 2304.08691v1 |
| 标题 | LTC-SE: Expanding the Potential of Liquid Time-Constant Neural Networks for Scalable AI and Embedded Systems |
| 作者 | Michael Bidollahkhani (a,b), Ferhat Atasoy (a), Hamdan Abdellatef (c) |
| 单位 | a = Institute of Graduate Studies, Karabuk University (Türkiye); b = Gesellschaft für Wissenschaftliche Datenverarbeitung mbH Göttingen (Germany); c = Lebanese American University (Byblos, Lebanon) |
| 提交日期 | 2023-04-18 01:57:10 UTC |
| 页数 / 图表 | 11 pages, 5 figures, 5 tables |
| arXiv comment | 11 pages, 5 figures, 5 tables. "This research work is partially drawn from the MSc thesis of Michael B. Khani. arXiv admin note: **text overlap with arXiv:2006.04439 by other authors**" |
| License | arXiv non-exclusive license |
| DOI | https://doi.org/10.48550/arXiv.2304.08691 |
| 代码 | 论文正文未给出仓库 URL |
| 本地 PDF | [`papers/2304.08691/2304.08691.pdf`](../../papers/2304.08691/2304.08691.pdf) |
| 本地全文 | [`papers/2304.08691/2304.08691.txt`](../../papers/2304.08691/2304.08691.txt) |
| 致谢 | 致谢 Michael B. Khani 的 MSc 论文; 致谢 Ramin Hasani (原始 LTC 作者) |
| **arXiv admin note** | ⚠️ **官方标注与 arXiv:2006.04439 (Hasani 2021) 存在文本重叠** —— 这是审稿层面给出的直接警示, 本报告 §6 展开 |

---

## 3. 核心问题

### 3.1 问题陈述

LTC (Hasani et al. 2021) 证明了连续时间神经元在时序任务上能给出稳定有界的动力学, 是嵌入系统 (embedded systems) 上"小模型 + 长时间序列"的一条现实路径。但原始 LTC 代码库在工程落地时有三处摩擦:

1. **接口不统一** —— LTCCell / CTRNN / NODE / CTGRU 四个 cell 各自是独立实现, 配置项不通用, 换模型要重写训练循环。
2. **Keras 2.x 兼容性差** —— 原始实现没有继承 `tf.keras.layers.AbstractRNNCell`, 导致 `model.save()` / `load_model()` / `tf.keras.layers.RNN` 包装等标准 Keras 工作流不可用。
3. **代码组织分散** —— 逻辑散落在多个 notebook / 脚本里, 难以作为可复用类库分发。

### 3.2 论文声称的"结构性缺陷"

论文把上述三点表述为 "the original LTC algorithm presents limitations that warrant further optimization for enhanced applicability for embedded systems", 并声称 LTC-SE "outperforms the original LTC algorithm and other state-of-the-art neural network models for embedded systems in terms of accuracy, computational efficiency, and memory usage"。

### 3.3 问题被重新表述为什么形式

**它没有重新表述问题, 它替换了问题**: 从"如何让 LTC 在嵌入系统上跑得更好"换成了"如何把 LTC 包装成一个更好的 Python 类库"。全文的技术贡献集中在 §Methodology 的三条 bullet (继承 `AbstractRNNCell` / 提供更多配置选项 / 整理代码结构), 而**动力学方程本身一字未改**。

**这个替换带来的直接后果是: 论文没有为"性能提升"提供任何机制解释。** 一个纯软件工程改动 (改基类、加 `get_config()`) 在物理上不可能改变网络的表达能力或收敛行为 —— 除非改的东西不止于此。而论文没有说明还改了什么。

---

## 4. 方法论与核心思路

### 4.1 总体架构 (论文 §Methodology)

LTC-SE 是一个**统一类库**, 包含四个 Keras cell:

| Cell | 底层模型 | 求解器 (论文 §Experimental Evaluation) |
|---|---|---|
| `LTCCell` | Liquid Time-Constant (Hasani 2021) | **fused ODE solver** (自研, 论文称 "our fused ODE solver for LTCs") |
| `CTRNN` | Continuous-Time RNN | explicit Euler |
| `NODE` | Neural ODE | 4th-order Runge–Kutta (按 Chen et al. 2018 的建议) |
| `CTGRU` | Continuous-Time GRU | (未单独说明) |

论文声明的三项改动:

1. **基类迁移**: `LTCCell` 改为继承 `tf.keras.layers.AbstractRNNCell`, 以兼容 TensorFlow 2.x。
2. **可配置化**: `input_mapping` / `solver` / `ode_solver_unfolds` 三个构造参数由硬编码改为可配置 (针对不同嵌入场景定制)。
3. **代码结构整理**: 统一目录与命名。

### 4.2 关键实现细节: `get_config()`

论文强调新增 `get_config()` 方法, 使 cell 可被 `model.save()` / `load_model()` 正确序列化 —— 这是 Keras 2.x 自定义 RNN cell 的**硬性要求**, 也是论文最扎实的一处工程贡献。

### 4.3 求解器对照 (论文 §Experimental Evaluation)

```
All ODE solvers were fixed-step solvers.
  CT-RNN  -> explicit Euler
  NODE    -> 4th-order Runge-Kutta  (Chen et al. 2018)
  LTC     -> fused ODE solver      (Hasani et al. 2021 Algorithm 1)
```

**这一点必须强调**: 论文同时报告了 "computational depth" 与 "computational efficiency" 两张表, 但这两者测的不是同一件事 —— depth 测单次前向的**运算量** (M ops), efficiency 测的是**档位式定性描述** ("10-100x faster than CNN/GRU"), **没有任何 latency / throughput / 功耗的绝对数字**。对一个以"embedded systems scalability"为标题的论文, 这是最关键的缺失: 表 5 全部是相对档位, 一个数字都没有。

### 4.4 与 LTC (Hasani 2021) / CfC 的关系

- **与 LTC (Hasani 2021)**: **动力学完全相同**, 差别只在 Python 封装层。论文明确写 "LTC-SE is an improved version of the Liquid Time-Constant (LTC) neural network algorithm originally proposed by Hasani et al. in 2021"。
- **与 CfC (Hasani 2022)**: **完全没有关系**。CfC 的核心贡献是用 sigmoid 闭式近似替代 ODE solver, 从而免去数值求解; LTC-SE 反而**依赖** solver (fused ODE solver)。**在"部署到没有浮点算力的 MCU"这个嵌入场景下, CfC 的路线严格优于 LTC-SE 的路线。**
- **与 NCP (Lechner et al. 2020)**: 未涉及。

---

## 5. 核心公式提取

**注意: 本论文没有提出任何新公式。** 下表是它引用/复述的公式, 来自 Hasani 2021 的 Table S2 (原文已存于本仓 `papers/foundational/hasani_2021_ltc.txt`), 本报告在此列出以便对照。

| 编号 | 公式 | 含义 | 来源 |
|---|---|---|---|
| 表 1 行 | $\text{CT-RNN}: \; O(nk^2),\; nk^2 + 2nk$ | 参数量 (渐近 / 精确) | Hasani 2021 Table S2 |
| 表 1 行 | $\text{ODE-RNN}: \; O(nk^2),\; nk^2 + nk$ | 同上 | 同上 |
| 表 1 行 | $\text{LSTM}: \; O(nk^2),\; 4nk^2 + 4nk$ | 同上 | 同上 |
| 表 1 行 | $\text{CT-GRU}: \; O(mk^2),\; 2mk^2 + 2mk + k^2 + k$ | 同上 | 同上 |
| 表 1 行 | $\text{LTC, LTC-SE}: \; O(nk^2),\; 4nk^2 + 3nk$ | 同上 | 同上 |
| 深度定义 | $L = \frac{\text{平均积分步数}}{\text{每个输入样本}}$ | computational depth (求解器积分步数) | Hasani 2021 §Definition of Computational Depth (L) |

**LTC 神经元本体 (Hasani 2021 Eq. 5, 本论文未改动, 在此 grounding)**:

$$
\frac{dx(t)}{dt} = -\left[\frac{1}{\tau} + f(x(t), I(t), t, \theta)\right] \odot x(t) + f(x(t), I(t), t, \theta) \odot A
$$

> *其中 $f$ 同时定义导数与输入依赖的可变时间常数 $\tau_{sys} = 1/[(1/\tau)+f]$; $A\in\mathbb{R}^N$ 为 bias 向量。*

---

## 6. 关键成果与贡献 —— 含逐格复算审计

### 6.1 表 1 "精确参数量" 列: **5/5 行与论文自己的公式不自洽** ⚠️

论文表 1 给出渐近式 + "精确式" + 一个 n=128, k=8 的数值列。本报告按论文**自己写的公式**逐行复算:

| 模型 | 论文公式 | 按公式在 n=128,k=8 复算 | 论文所列数值 | 一致? |
|---|---|---|---|---|
| CT-RNN | $nk^2 + 2nk$ | 10,240 | 8,320 | ❌ |
| ODE-RNN | $nk^2 + nk$ | 9,216 | 8,192 | ❌ |
| LSTM | $4nk^2 + 4nk$ | 36,864 | 32,896 | ❌ |
| CT-GRU | $2mk^2 + 2mk + k^2 + k$ (m=128) | 18,504 | 24,704 | ❌ |
| LTC / LTC-SE | $4nk^2 + 3nk$ | 35,840 | **32,896** | ❌ |

**5 行全部不匹配。** 进一步, 论文的数值列可以由另一套算法**完整复现**: 把每个 cell 的"每神经元参数数"乘以 n。

| 模型 | 论文数值 | n × (每神经元参数数) | 复现? |
|---|---|---|---|
| CT-RNN | 8,320 | $128\times(8^2+1)=128\times65=8{,}320$ | ✅ 完全吻合 |
| ODE-RNN | 8,192 | $128\times 8^2=128\times64=8{,}192$ | ✅ 完全吻合 |
| LSTM | 32,896 | $128\times(4\cdot8^2+1)=128\times257=32{,}896$ | ✅ 完全吻合 |
| CT-GRU | 24,704 | $128\times(3\cdot8^2+1)=128\times193=24{,}704$ | ✅ 完全吻合 |
| LTC / LTC-SE | 32,896 | $128\times(4\cdot8^2+1)=128\times257=32{,}896$ | ✅ 完全吻合 |

**结论**: 论文的"精确参数量"列用的是**逐神经元计数** (与渐近式同阶, 但丢失了层宽/输入维之间的交叉项), 而**同行写的"精确式"是另一套 (含交叉项) 的公式** —— 两者被并列放在同一张表里却没有对上。

**最刺眼的后果**: **LTC 与 LSTM 被报成完全相同的 32,896**, 尽管论文自己写明二者的公式不同 ($4nk^2+3nk$ vs $4nk^2+4nk$, 差 $nk = 1024$)。按论文自己的公式, LTC = 35,840, LSTM = 36,864, **LTC 比 LSTM 少 1,024 个参数 —— 这正是 LTC "更省参数" 的全部卖点, 被这一列算成了 0**。论文正文 §Results 的结论 "LTC and LTC-SE models have the highest parameter counts, while the CT-RNN and ODE-RNN models have lower counts" 因此**建立在一个自相矛盾的表上**。

### 6.2 表 3 计算深度: 3/4 行改善, 1/4 行**变差**

把论文表 3 的 LTC-SE 列与 Hasani 2021 原文 (已存 `papers/foundational/hasani_2021_ltc.txt`) 的 LTC 列逐格比对:

| 激活 | Hasani 2021 (LTC) | LTC-SE (论文声称) | Δ | 方向 |
|---|---|---|---|---|
| tanh | 9.19 ± 2.92 | 8.50 ± 2.50 | −0.69 | 更好 |
| sigmoid | 7.00 ± 5.36 | 6.50 ± 4.50 | −0.50 | 更好 |
| ReLU | 56.9 ± 9.03 | 55.0 ± 8.00 | −1.90 | 更好 |
| **hard-tanh** | **81.01 ± 10.05** | **82.50 ± 8.50** | **+1.49** | ❌ **更差** |

**表 3 的 LTC 列数字 (9.19 / 7.00 / 56.9 / 81.01) 与 Hasani 2021 原文逐格完全一致** —— 也就是说这一列是**原样引用**, 而 LTC-SE 列是论文新填的。

**但 Δ 的量级本身可疑**: hard-tanh 的标准差是 10.05 / 8.50, 而 tanh 的 Δ 只有 0.69 —— **Δ 全部落在标准差之内**。以单次运行的一阶统计量比较两个求解器的"深度", 在此噪声水平下无法支撑论文 §Experimental Evaluation 的 "Statistical analysis was performed to determine the significance of the observed differences" 这一声称 —— **论文没有给出任何显著性检验的结果** (无 p 值、无置信区间、无重复次数说明)。表 2 的超参是"to make the evaluation parameters as close to the original work [3] as possible", 即隐藏单元 32、训练 100 epoch (原论文是 200)。

### 6.3 表 4 准确率: 与论文自己的结论不一致, 且**任务集与正文不匹配**

论文表 4 报告的任务是 {Room Occupancy, Human Activity Recognition, Sentiment Analysis, Time Series Prediction, Language Modeling}; 而 §Experimental Evaluation 正文**描述**的任务是 {Room Occupancy, Human Activity Recognition, **Traffic Estimation**, **Power**, **Ozone Day Prediction**}。

**交集只有 2 个 (Room Occupancy + Human Activity Recognition)。** 正文描述的 Traffic / Power / Ozone 三个任务在表 4 里**完全没有数据**; 表 4 里的 Sentiment / Language Modeling / Time Series 三个任务在正文里**没有被描述**。

再看表 4 中 LTC-SE vs 原始 LTC:

| 任务 | LTC | LTC-SE | 结果 |
|---|---|---|---|
| Room Occupancy | 0.95 | 0.96 | WIN |
| Human Activity Recognition | 0.96 | 0.97 | WIN |
| Sentiment Analysis | 0.98 | **0.97** | ❌ **LOSS** |
| Time Series Prediction | 0.93 | 0.94 | WIN |
| Language Modeling | 0.93 | 0.93 | TIE |

**3 胜 / 1 平 / 1 负。** 论文的结论句写的是 "the proposed LTC-SE model **outperforms** the original LTC algorithm and other state-of-the-art neural network models for embedded systems in terms of accuracy, computational efficiency, and memory usage" —— **至少在 Sentiment Analysis 上它输给了原始 LTC, 且论文正文没有任何地方承认这一点**。

**另一处必须点出的事实**: 表 4 的 LTC / CNN / LSTM / GRU 数值是**从文献里抄来的, 不是在 LTC-SE 上重新测的**。表格的引用标注 `[1] [14] [15] [16] [17]` 分别是 Hasani 2021 / Hochreiter&Schmidhuber 1997 / Cho et al. 2014 / Glorot& Bengio 2010 / LeCun et al. 1995 —— 全部是**别人的原论文**。所以"LTC-SE 0.97 vs LTC 0.96"这个对比是**两篇论文的跨文献数字对比**, 而表 1 的参数量与表 3 的深度却是 LTC-SE 自测。

**最关键的一处对照**: Hasani 2021 原文自己报告的 Occupancy 准确率是 **94.63% ± 0.17** (LTC), 而 LTC-SE 表 4 写的是 **0.95** —— 一致。但 Hasani 原文的 Activity recognition 是 **95.67% ± 0.575**, LTC-SE 表 4 写 **0.96** —— 也一致。**即 LTC-SE 的表 4 精度列, 连 LTC 自己那一行都基本是照抄原文, 精度小数位还从两位降到一位。**

### 6.4 表 5 计算效率: **全表没有一个绝对数字**

整张表只有 "10-100x faster than CNN/GRU" 这类档位式定性描述, 且大量格子是 "-" 或跨行合并的歧义排版。**没有任何 latency (ms)、吞吐 (samples/s)、FLOPs、能耗 (mJ) 数字。** 对于一篇以 "Scalable AI and Embedded Systems" 为标题、声称贡献是 "scalable" 的论文, 这是核心证据的缺失。

### 6.5 贡献清单 (本仓视角)

1. **✅ 真实且扎实的工程贡献**: 把 LTCCell 迁移到 `tf.keras.layers.AbstractRNNCell` + 实现 `get_config()`, 使 LTC 可用标准 Keras 工作流 (`model.save()` / `load_model()` / `tf.keras.layers.RNN` 包装) 训练与分发。**这一条独立于任何性能声称, 是本文最值得借鉴的部分。**
2. **⚠️ 有价值的负面资产**: 提供了一个可复现的"参数量整列算错"的案例 —— 渐近式与精确式并列却互不对账, 且逐神经元计数抹掉了 LTC 相对 LSTM 的 $nk$ 优势。
3. **❌ 不可采信的性能声称**: "outperforms" 在 1/5 任务上被自家表格证伪; 计算深度差异落在标准差内且无显著性检验; 计算效率无绝对数字。
4. **❌ 与本仓路线无关的定位**: CfC 路线 (闭式, 无 solver) 才是嵌入部署的正确选择; LTC-SE 反而强化了 "依赖 solver" 这一 CfC 要绕开的负担。

---

## 7. 局限性与未来展望

### 7.1 作者自陈局限 (论文 §Conclusion and Future Work)

- 需要进一步优化以降低计算成本与内存占用, 使其更适用于资源受限环境;
- 应探索 LTC-SE 在 robotics / causality analysis 等其他应用域的潜力;
- 未来工作: model compression / pruning / quantization;
- 与 RL / unsupervised learning 的结合。

### 7.2 隐含局限 (本仓视角 —— 与作者自陈完全不同的清单)

1. **🔴 表格与公式不自洽 (5/5 行)**, 且 "精确参数量" 列与同行公式分属两套算法 —— **这是最严重的问题**, 因为它直接使 §Results 关于"哪个模型更省参数"的结论失效。
2. **🔴 精度表是跨文献数字拼接, 不是同一实验条件下的对照** (引用标注全指 1995–2021 的原论文)。因此 "LTC-SE 0.97 > LTC 0.96" 的因果强度**接近于零**。
3. **🔴 任务集与正文不匹配** (5 个任务只重叠 2 个), 且正文描述的 3 个任务无数据。
4. **🔴 "outperforms" 在 Sentiment Analysis 上被自家表格证伪** (0.98 → 0.97), 论文未提。
5. **🟠 计算深度的差异全部落在标准差内**, 且无显著性检验输出 —— 论文声称做了 statistical analysis, 但正文无任何检验结果。
6. **🟠 嵌入系统的核心指标 (latency / 功耗 / 内存占用) 全部缺失**: 表 5 只有档位, 图 2–5 是 "memory usage comparison" 但**没有给出任何数值或坐标轴刻度**, 只有文字描述 "CNN highest, then LSTM"。
7. **🟠 arXiv 官方 admin note 标注 "text overlap with arXiv:2006.04439 by other authors"** —— 这是审稿层面对"本文与原文大量文本重叠"的直接提示, 叠加表 3 的 LTC 列逐格与原文一致、致谢中"部分取自 MSc 论文", **本文的"新结果"边界非常模糊**。
8. **🟠 未与 CfC 做对比** —— 而 CfC 才是嵌入部署的正确答案 (免 solver)。
9. **🟡 无代码仓库**, 无超参全表 (表 2 缺 `input_mapping` / `solver` / `ode_solver_unfolds` 的取值), 无随机种子, 无重复次数。
10. **🟡 完全没有触碰不规则采样 / 变步长** —— 而这是 LTC 相对固定步长 RNN 的**核心卖点**, 也是本仓 r306/r307 反复验证的关键变量 (见 `docs/reports/LNN-Fly_Continuous_Time_UAV_Navigation_2606.28827_研读报告.md`)。

### 7.3 未来方向 (本仓建议, 不重复作者的自陈)

- **在本仓复算一遍本表** (成本 < 1 小时): 把 `lnn/core/ltc.py` / `lnn/core/cfc.py` / `lnn/core/snnl.py` 的真实参数量在 n=128, k=8 下打出来, 与本报告 §6.1 的两套算法对照 —— **本仓自己的实现到底属于哪一套**。
- **写一条门禁**: 任何报告里出现"渐近式 + 精确式 + 数值列"三列并置时, 必须用脚本验算数值列与公式的一致性。本仓 `scripts/check_repo_integrity.py` 已经是"提交前拦截损坏文件"的门禁, **可以加一条"报告内算术自洽性"检查**。

---

## 8. 对本仓的意义

### 8.1 可作 `lnn/core/variants.py` 的什么变体

**无。** 本文没有提出任何新的 cell 变体, 动力学与 `lnn/core/ltc.py` 完全一致。唯一可借鉴的工程模式是 **`get_config()` / `AbstractRNNCell` 的序列化契约** —— 本仓 `lnn/core/ltc.py` 已有 `LTCNetwork`, 若要在 PyTorch 侧对齐 TensorFlow 的可序列化契约 (state_dict 里存 `input_mapping` / `solver` / `ode_solver_unfolds` 而非硬编码), 可参考本文的做法。**这是配置层面的事, 不是变体层面的事。**

### 8.2 可嫁接到哪个 bench 脚本

**没有直接可嫁接的 bench。** 但有一个**高价值的负面对照**: 本报告 §6.1 给出了一条可直接实现的断言检查 ——

```
"在 n=128, k=8 下, 本仓 LTCNetwork 的真实参数量是否等于 4nk^2+3nk (公式) 还是 n*(4k^2+1) (逐神经元计数)?"
```

这可以在 `scripts/bench_cfc_vs_ssm.py` 或一个新建的 `scripts/bench_param_count_audit.py` 里跑, **用来校准本仓所有"参数量对比"叙述的可信度**。本仓报告里凡是出现 "LTC 参数量 = X" 的地方, 都应当能落到公式上。

### 8.3 提供哪个新 ablation axis

**新增消融轴: "跨文献数字 vs 同条件实测"** —— 本仓报告普遍做同 bench 同配置的对照 (这是本仓的强项, 也是 r306/r307 负结果能站住的原因)。LTC-SE 提示了一个本仓需要**持续守住**的边界: **当对照组的数字来自对方论文的表格而非本仓实测时, 对比结论的因果强度必须降级标注。**

### 8.4 与本仓既有结论的关系

- **与 NR-06 同向**: 本文又一次说明 LNN 变体之间的精度差异 (0.95 vs 0.96) 远小于**实验条件噪声** (标准差 0.17–10.05), 更远小于**跨论文对照的不可比性**。本仓 §0 "LNN 未在标准主基准上显著超过 transformer" 的定位在本文里得到弱支持。
- **与 LNN-Fly (2606.28827) 形成强对比**: LNN-Fly 明确做了 **"LNN-Fly vs LNN-Fly (Fixed Δt)" 的同条件对照** (12/20 vs 5/20), 这才叫因果证据; LTC-SE 只有跨文献拼接。**两篇论文的证据强度差异本身就是最好的对照。**

---

## 9. Verdict

- **"LTC-SE 改进了 LTC 的性能 / 精度 / 内存" → TARGET-NEGATIVE-WITH-NUANCE** —— 自家表 4 有 1/5 任务输, 表 1 与公式 5/5 行不自洽, 表 5 无绝对数字。**不可采信。**
- **"LTC-SE 的 `AbstractRNNCell` + `get_config()` 工程改造" → TARGET-POSITIVE (限于该用法)** —— 这是本文唯一无可争议的贡献, 可直接借鉴到本仓 `ltc.py` 的序列化契约。
- **作为本仓"参数量叙述可信度"的审计依据 → TARGET-POSITIVE** —— §6.1 给出了一条可执行的验算, 建议纳入本仓报告规约。
- **作为"液态算子在嵌入系统上可行"的证据 → TARGET-NEGATIVE-WITH-NUANCE** —— 未与 CfC 对比, 反而强化了 solver 依赖; 真正的嵌入部署答案在 CfC 路线。

---

## 10. 引用与 grounding

| 本报告的断言 | grounding 路径 |
|---|---|
| LTC 动力学 Eq. 5 | `papers/foundational/hasani_2021_ltc.txt` (本仓已存原文) |
| 表 1 的渐近式/精确式来源 | `papers/foundational/hasani_2021_ltc.txt` Table S2 |
| 表 3 的 LTC 列 = Hasani 2021 原值 | 同上, Computational Depth 表 |
| Hasani 2021 Occupancy 94.63%±0.17 / Activity 95.67%±0.575 | 同上, 主文 Table |
| CfC 免 solver 的定位 | `docs/LNN_深度研读报告.md` §1.2 (L3 CfC, Eq. 10) |
| "同 bench 同配置对照"的证据强度标尺 | `docs/reports/LNN-Fly_Continuous_Time_UAV_Navigation_2606.28827_研读报告.md` |
| "LNN 未在标准主基准显著超过 transformer" | `docs/LNN_深度研读报告.md` §0 + `analysis/negative_results/README.md` NR-06 |
| arXiv admin note (text overlap) | arXiv API `arxiv:comment` 字段 (2026-10-26 实时拉取, `id_list=2304.08691`) |

**复算脚本**: 本报告 §6.1 / §6.2 / §6.3 的全部算术由 `/tmp/lnn_ltcse_audit.py` 与 `/tmp/lnn_check_math.py` 生成, 数字可逐格复现。
