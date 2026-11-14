---
title: CPS-LLM — Large Language Model based Safe Usage Plan Generator for Human-in-the-Loop Human-in-the-Plant Cyber-Physical System
date: 2026-11
tags: [LNN, LTC, system-identification, partial-observability, parameter-estimation, safety-critical, medical-CPS]
arxiv_id: 2405.11458
---

# 研读报告：CPS-LLM — 把 **LTC 当"部分可观测系统的参数辨识器"**用，以及一个**很诚实的负结果**

## 1. 元数据
- **论文标题**：CPS-LLM: Large Language Model based Safe Usage Plan Generator for Human-in-the-Loop Human-in-the-Plant Cyber-Physical System
- **作者**：Ayan Banerjee, Aranyak Maity, Payal Kamboj, Sandeep K. S. Gupta —— **IMPACT Lab, School of Computing and Augmented Intelligence, Arizona State University**
- **发表时间**：arXiv 2024-05-19（cs.AI / eess.SY）；**已发表于 AAAI 2024**（版权页标注 "Copyright © 2024, Association for the Advancement of Artificial Intelligence"）
- **代码**：**论文未给出仓库 URL**；LTC 部分引用 `github.com/topics/liquid-neural-networks` 通用实现
- **本地 PDF**：[papers/2405.11458/2405.11458.pdf](../../papers/2405.11458/2405.11458.pdf) ／ 文本 [2405.11458.txt](../../papers/2405.11458/2405.11458.txt)
- **arXiv 分类**：cs.AI, eess.SY
- **关联概念**：LTC（Hasani 2021）/ 系统辨识（system identification）/ 部分可观测 / Bergman 最小血糖模型 / 自动胰岛素输注（AID）/ STL 安全性规范 / LLM 指令微调 / DAGGER 式模仿学习
- **入选理由**：本仓**零覆盖**"液态网络做**系统辨识 / 参数估计**"这一方向。本仓 `lnn/core/physics.py`（`PhysicsInformedLNN`）已经有一个"LTC + 物理残差"的实现，但那是**需要真值参数**的 PINN 路线；本篇走的是**另一条路：只观测部分状态，反推出不可观测状态所依赖的物理系数**。这是一条本仓从未审过的工程范式。

## 2. 核心问题

**安全关键 CPS（cyber-physical system）的规划有两个层次**：高层是"人给一句自然语言意图"（比如"我吃了 30g 碳水"），低层是"控制器实际执行的动作序列"。中间这一层——**把意图翻译成一个物理上可执行、且对在环在体（human-in-the-plant）的人安全的动作计划**——传统上由领域专家手工完成。

作者指出，LLM 在这个位置有**两个具体失效模式**：

**失效模式 1（数值错误）**：未微调的 LLM 会把临床公式算反。论文给的例子：碳水 45g、碳水比 5、胰岛素在体（IOB）2U，正确剂量是 $45/5 - 2 = 7\text{U}$，而 LLM 算成 $(45/5) + 2 = 11\text{U}$ —— **多给 2U，足以造成严重低血糖**。

**失效模式 2（物理不自洽）**：即使数值算对，解释也是坏的。论文例子：碳水 7g、碳水比 5、IOB 3U，LLM 先用同样错误的加法得 1.4U，然后自圆其说"因为 1.4 小于最小可测剂量，所以建议不打"—— **数值与物理都不连贯**。

作者把根因归到一点（论文的 C3 论断）：

> "The main reason that LLM may generate infeasible and unsafe plans is that they are not trained with the knowledge of dynamical systems that govern the temporal evolution of the plant with the specific real world human user embedded in the plant."

**而在环在体（HIL-HIP）的特殊性在于：被控对象里嵌着一个真实的人** —— 同一个 CPS 方程里的系数**因人而异**且**随时间漂移**（胰岛素敏感性昼夜变化）。所以光有"通用物理"不够，必须有**这个人的个性化系数**。

于是问题被重述为：**当被控系统是"部分可观测"的（关键内部状态测不到）时，能不能用 LTC 从残缺观测里反推出物理模型的系数，再用这些系数去具身化地微调 LLM，从而生成既物理可行又安全的计划？**

## 3. 方法论与核心思路

### 3.1 总体架构（论文 Fig. 2）

```
┌── 预处理 ────────────────────────────────────────────────────────────┐
│ (a) CPS-LLM: LLAMA-2-7B 用 ALPACA 格式指令微调                       │
│     instruction: "从 f_ω(.) 的轨迹推导 ω"                              │
│     input      : 场景文本 + 交错嵌入的 f_ω(.) 数值轨迹                  │
│     response   : ω 的数值                                             │
│ (b) 系数提取器: LTC NN 编码器-解码器                                  │
│     encoder: ω = f_ω^{-1}(X)   decoder: 用 ω 重放轨迹                  │
│ (c) 上下文化: BARD chat RL 把用户查询映射为规划任务                     │
└──────────────────────────────────────────────────────────────────────┘
                              │
┌── 部署 ──────────────────────────────────────────────────────────────┐
│ ① trace τ → LTC NN → 个性化系数 ω_P                                   │
│ ② ω_P 嵌入 prompt → 微调后的 LLAMA 反推未来轨迹 X(t)                   │
│ ③ 轨迹 → BARD chat RL → 计划                                          │
│ ④ 计划 → 前向仿真评估安全性 (STL robustness)                          │
│ ⑤ 安全则执行并循环；不安全则反馈给用户                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 被辨识的物理系统：Bergman 最小血糖模型（Eq. 3）

$$\frac{dy}{dt} = z, \quad \frac{dz}{dt} = -2k_1 z - \frac{k_2}{1}y + \frac{k_2}{1}u_{ex}, \quad \frac{d\,iob}{dt} = -n\,iob + p_1(y + I_b) \tag{Eq. 3}$$

$X = (y, z, iob)$；$k_1$ 是胰岛素扩散系数，$n$ 与 $p_1$ 是患者特异指标。

**关键假设（也是本篇最有意思的地方）**：$y$（血糖）与 $z$（胰岛素动力学隐状态）**内部状态不可测**，只有 $iob$ 被（稀疏）采样到。**这是一个真正的部分可观测辨识问题**，不是"参数已知、只做拟合"。

### 3.3 LTC 系数估计器（Fig. 3）

论文描述的结构（原文措辞）：

- **输入**：20,000 条 IOB 轨迹（由 Eq. 3 按不同 $k_1$ 生成），每条 200 分钟，batch 32；
- **网络**：**LTC NN，32 个隐藏节点** → 接一个 **$3\times1$ dense layer + sigmoid**；
- **输出**：dense layer 的输出**就是** Eq. 3 的三个系数 $\omega = (k_1, n, p_1)$；
- **解码器**：用这些系数 + **Runge-Kutta 积分**重放 IOB 轨迹；
- **损失**：**重建的 IOB 与真实 IOB 的 RMSE**（注意：**损失里没有直接监督系数本身**，系数只通过"能否重建轨迹"被间接约束）；
- **训练**：200 epoch。

**这是一个"编码器-解码器 + 结构化物理"的组合**：encoder 学"从轨迹反推系数"（inverse problem），decoder 是**硬编码的 ODE 求解器**（不是神经网络），loss 是物理重建误差。**这与本仓 `2412.02215`（LTC 隐式物理模型反推）走的是相反方向** —— 2412.02215 是"把黑盒求解器放进 loss"，本篇是"把 ODE 求解器放在**解码器**"。

### 3.4 具身化微调（ALPACA 格式）

论文明确指出 **prompt 里必须嵌入物理量**才能让 LLM 算对。给的完整例子：

```
### Instruction: Find out the diffusion parameter from the Bergman Minimal Model
with the following time series. The 40 values corresponding to 400 seconds of IOB values
### Input: 1.0 0.99948 0.99747 0.99411 0.98975 ...
### Response: 0.015
```

部署时反问（正向生成）：

```
### Instruction: I took an insulin dosage now. What is my Insulin On Board percentage timeseries?
### Input: diffusion parameter = 0.025
### Response: Your timeseries is 1.0, 0.9995 ...
```

### 3.5 安全性评估：STL + 前向仿真

用 **UVA/Padova T1D 模拟器**做前向仿真，虚拟患者的 BMM 系数**就取自 LTC NN 估出来的值**。安全用 **STL 公式** $\varphi_t$ 表达（美国糖尿病协会标准：TBR < 4%），用 **robustness 函数** $\rho(\varphi, x, t)$ 判定。

### 3.6 与 LTC / 本仓其他工作的关系

| 维度 | 本篇 | 本仓 `2412.02215` | 本仓 `physics.py` |
|---|---|---|---|
| 目标 | 从部分观测反推**物理系数** | 隐式物理模型恢复 | PINN 式物理残差（**需真值 Θ**） |
| loss | 物理重建 RMSE（解码器=RK 求解器） | 黑盒求解器进 loss | 物理残差（**需真值参数**） |
| 可观测性 | **部分可观测**（$y,z$ 不可测） | 部分 | 需真值 |
| 闭环 | 系数嵌入 LLM prompt | 无 | 无 |

**本篇是本仓 `physics.py` 那条路线的"无真值"版本**，也是 `2412.02215` 的"解码器版"变体。**三者构成同一任务方向的三种实现风格，本仓至今只审过其中一篇。**

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 3 | $\frac{dy}{dt}=z$；$\frac{dz}{dt}=-2k_1z-\frac{k_2}{1}y+\frac{k_2}{1}u_{ex}$；$\frac{d\,iob}{dt}=-n\,iob+p_1(y+I_b)$ | **Bergman 最小模型**（被辨识对象；$y,z$ 不可测） |
| Eq. 4 | $\mathrm{MealBolus} = \mathrm{Carbohydrate}(g)/CR - IOB$ | **临床剂量公式**（LLM 算反的那一条） |
| Fig. 3 | $L = \frac{\|I - I_R\|_2}{}$，其中 $I_R$ = 用 dense 层输出 $\omega$ 经 RK 积分重建的 IOB | **LTC 系数估计的损失**（重建误差，**非系数监督**） |
| — | $\hat{\omega} = \mathrm{Dense}_{\text{sigmoid}}(\mathrm{LTC}_{32}(I_{\text{iob}}))$ | **系数读出层**（3 维，直接就是 $k_1,n,p_1$） |
| — | $\mathrm{RMSE}(\text{CPS-LLM 输出}, \text{BMM 解析解})$ | **LLM 侧物理一致性度量**（论文报 6% ± 3%） |

## 5. 关键成果与贡献

### 5.1 主结果

**（a）LTC 系数估计精度（Table 2）** —— 模拟真值 vs LTC 估计（单位 $10^{-2}$）：

| 数据类型 | $k_1$ | $n$ | $p_1$ |
|---|---|---|---|
| **Simulation（真值）** | 9.8 | 14.06 | 2.8 |
| Train（60%） | 9.78 | 14.06 | 2.62 |
| Test（40%） | [9.79, 9.81] | [14.05, 14.07] | [2.56, 2.75] |
| Overnight（无餐） | 9.8 | 14.06 | **4.0** |
| Afternoon（午餐） | 9.78 | 14.06 | 2.62 |
| Evening（晚餐） | 9.82 | 14.05 | **3.6** |

**论文的结论是"recover the dynamics coefficient with good accuracy and less variance despite having no measurements of $y$ and $z$ and only sampled measurements of iob"**。

**⚠️ 但这张表必须加三条限定**（本仓视角）：
1. **"Overnight $p_1 = 4.0$"与"Evening $p_1 = 3.6$"相对真值 2.8 是明显偏差（+43% / +29%）**，论文**没有解释**这个偏差。这是"良态"结论里的一个硬伤。
2. **表里的 $k_1$ 与 $n$ 几乎精确复现（9.78 vs 9.8、14.06 vs 14.06）** —— 精确到小数点后两位的复现在**真实辨识问题里不该发生**。合理怀疑是：**真值系数就是按这些值生成的，且网络有直接看到某些相关量的通道**（例如 $k_1$ 直接决定 IOB 衰减率，$n$ 决定线性衰减率，**两者在 IOB 轨迹里是可直接分离的**）。论文的 $k_1, n, p_1$ 三个量里，**只有 $p_1$ 真正依赖不可观测的 $y$**，**另两个是"可从 IOB 轨迹直接读出"的量**。**因此这张表真正证明的是"LTC 能从 IOB 轨迹反推 $p_1$"，而不是"能从部分观测做一般的非线性辨识"。**
3. **评估口径是"训练集 / 测试集 / 时段分组"三列，全部是同分布**（同一模拟器、同一 $k_1$ 生成机制），**没有任何分布外测试**。

**（b）LLM 侧的物理一致性**：CPS-LLM 生成的 IOB 序列与 T1D 模拟器解 BMM 方程的结果相比，**RMSE = 6% ± 3%**（Fig. 4 逐例 RMSE 从 0.012 到 0.149）。

**（c）安全关键主结果（Fig. 5）** —— 这是全文最硬的数字：

| CPS 计划生成方式 | TIR (70–180) | TAR (>180) | TBR (<70) | **TBR 危急 (<54)** |
|---|---|---|---|---|
| MPC + 人工（Bolus Wizard） | 78.2% (±21.1) | 17.4% (±17.3) | 4.4% (±4.2) | 1% (±1) |
| **MPC + 未微调 LLM（LLAMA-2）** | 81.2% (±18) | 9.9% (±4.2) | **8.8% (±1.1)** | **2.6% (±0.1)** |
| **MPC + CPS-LLM（具身微调）** | 81% (±12) | 16% (±7) | **3% (±2)** | **0% (0)** |

**这张表是全文最有价值的部分，且结论与直觉一致、方向清楚**：
- **未微调 LLM 比人工专家更危险**（危急低血糖 2.6% vs 人工 1%）—— **"LLM 在安全关键规划里会降低安全性"这一条被实测证实了**。
- **CPS-LLM 把危急低血糖压到 0%**，同时 TIR 与人工持平、TBR 从人工的 4.4% 降到 3%。
- **但 CPS-LLM 的 TAR 是 16%，比未微调 LLM 的 9.9% 高、比人工的 17.4% 略低** —— 也就是说 **CPS-LLM 是"用更高的 hyperglycemia 换了更低的 hypoglycemia"**，代价体现在 TAR 上。论文**没有讨论这个 trade-off**（只报了 TIR，看不出这个结构）。

**（d）模拟规模**：218 个进餐实例，碳水 7–50g，碳水比 CR 10–25；用 Tandem Control IQ（自适应 MPC）产生控制动作；先验胰岛素 30 分钟–3 小时。

### 5.2 机制分析
- **"contextualization + embodied fine-tuning 是必要条件"** 是论文的核心机制主张：未微调 LLM（哪怕接了 BARD 上下文）**仍然更危险**。论文原话："The untuned LLM is poorer than the manual determination of bolus and may even jeopardize safety since it has the highest hypoglycemia rate."
- **拟真保真的关键在"系数嵌入 prompt"**：论文展示，把扩散参数放进 prompt 后，**基础 LLAMA2 7B 依旧给出数值与物理都不连贯的回答**（"diffusion parameter 1.669 and the insulin intake of 0.005055"）—— 说明**光给系数不够，还得微调**。

### 5.3 贡献清单
1. **LTC NN 作为部分可观测系统的物理系数估计器**（32 节点 + 3 维 sigmoid 读出 + RK 解码器 + 重建 RMSE）。
2. **首个把 LLM 用于安全关键 HIL-HIP 规划的工作**（作者自陈）。
3. **一条有实测支撑的安全负面结论**：未微调 LLM 在安全关键胰岛素规划上**比人工更危险**。
4. **"具身化指令微调 + 上下文化 chat RL" 作为把物理知识注入 LLM 的可操作范式**。

## 6. 局限性与未来展望

### 6.1 作者自陈局限
- **"This is only an initial attempt ... although it has been shown for only one example, the methodology is general and its application to other examples such as autonomous cars and unmanned aerial vehicles is potentially feasible, albeit yet to be tested."** —— **只验证了一个案例（全自动胰岛素输注），其他场景未测试**。
- 应用到自动驾驶/无人机"potentially feasible, albeit yet to be tested"。

### 6.2 隐含局限（本仓视角）
- **🔴 全部实验都在仿真里**（UVA/Padova T1D 模拟器 + 虚拟患者），**没有任何真人数据**。而全篇的动机是"human-in-the-plant"。**"个性化"结论实际上是个性化到模拟器里的虚拟患者。**
- **🔴 系数估计的"泛化"未被证明**（见 §5.1(a) 的三条限定）：$k_1,n$ 近乎精确复现 ⇒ 真正难的只有 $p_1$；且 $p_1$ 在三个时段里有 29–43% 的偏差却未被讨论。
- **系数估计与 LLM 之间有一道未验证的接缝**：LTC 估的系数**同时**被用来（i）微调 LLM 的训练数据、(ii) **构造评估安全性的虚拟患者**（论文原话："we instantiated virtual patients with the same BMM model coefficients as obtained from the LTC NN"）。**⇒ 评估闭环用的是自己的估计值，"安全"结论因此存在结构性自证风险。**
- **方差被系统性低估**：Table 2 的 train/test 区间极窄（$k_1$: [9.79, 9.81]），而 Fig. 5 的 TIR 方差却很大（±21.1 / ±18 / ±12）—— **同一篇论文里两个方差口径不一致，无解释**。
- **未微调 LLM 的"更危险"结论只在一个指标上体现**（危急低血糖 2.6% vs 1%），样本量与统计检验**未报告**；2.6% vs 1% 在未给置信区间的情况下**不足以支撑强结论**。
- **工程可复现性低**：无代码、无真实数据、无硬件细节；LLM 部分依赖 BARD 的交互式 GUI 生成 prompt（不可脚本化复现）。

### 6.3 未来方向
作者：推广到自动驾驶与无人机。
**本仓可补的**：把"部分可观测系数辨识"做成**独立 bench**（见 §7.3），并与本仓已有的 `2412.02215` 路线做**同任务对照**——这两条路线在本仓从未被放在同一个指标下比较过。

## 7. 对本仓的意义

### 7.1 可作 `lnn/core/variants.py` 的什么变体
本篇对**网络结构**的贡献很小（就是一个 32 节点的 LTC + 3 维读出），但对**任务范式**的贡献不小。真正可落地的是一个新的 cell 角色：**"系数读出头"（coefficient readout head）** —— 从隐状态直接读出物理系数，且**不直接监督系数，只监督物理重建**。本仓 `lnn/core/physics.py` 的 `PhysicsInformedLNN` 有 `param_head`（Softplus 出 2 维），**但它的 loss 依赖真值 Θ**；本篇的路线是**用 ODE 解码器把真值需求消掉**。

建议：**在 `lnn/core/physics.py` 旁新增一个 `partial_observability` 变体**，把 `param_head` 从"直接监督"改成"经 RK 解码器重建监督"，并**显式构造部分可观测场景**（隐藏 $y,z$，只给 $iob$）。这是本仓 `physics.py` 从"PINN 式（需真值）"升级到"辨识式（不需真值）"的最小改动。

### 7.2 可嫁接到哪个 bench 脚本
- **`scripts/experiment_physics_lnn.py`**：本仓现有的物理 LNN 实验脚本，是本篇范式的直接落点（把已知真值的参数恢复任务改成**部分可观测**版本）。
- **新建 `scripts/bench_partial_obs_param_id.py`**：本篇的最小可复现版本其实很小——**32 节点 LTC + sigmoid 三维读出 + RK 解码器 + RMSE**，合成数据即可跑，**不依赖任何医疗数据**。这是本仓可以**当天复现**的少数外部论文之一。
- **`scripts/bench_cfc_input_dropout.py`**：本篇的"只有部分状态被采样"在机器学习里就是**掩码观测**；本仓已有 input dropout 的基础设施，可直接做"掩码率 vs 系数估计误差"的曲线。

### 7.3 提供哪个新 ablation 轴
1. **🔴 "可观测性比例"作为新轴**：本仓所有物理类 bench 默认状态全可测 + 参数已知。**本篇引入的是"部分可观测 + 参数未知"的二维新空间**，本仓至今**没有一个 bench 落在这个空间里**。具体轴：(a) 观测掩码率（保留 $iob$ 的比例）；(b) 不可观测状态的数量（1 个 vs 2 个）；(c) **待辨识参数的个数**（$k_1$ vs $p_1$ —— **按本篇的证据，这两个参数的辨识难度完全不同，混在一起评会掩盖真相**）。
2. **"系数监督 vs 重建监督"（直接 vs 间接）**：本篇用重建误差（间接），`2412.02215` 与 `physics.py` 用真值（直接）。**三者在同一数据上对照，是一个明确的、有价值的消融。**
3. **LTC vs GRU vs CfC 在辨识任务上的对比**：本篇只用了 LTC。**本仓的 `bench_cfc_vs_ssm.py` 正好可以加一个"参数辨识"子任务** —— 若 GRU 在这个任务上不输 CfC，则"LTC 的连续时间对辨识任务的价值"这一命题需要重新审视（本仓 §0 一直缺这种直接对照）。

**Verdict**：
- **"LTC 可作部分可观测系统的物理系数估计器" → TARGET-DEPENDENT-WITH-NUANCE**：**方法本身是合理且可复现的**（结构小、合成数据可生成），**但论文的证据强度低于其表述**：真正困难的只有 $p_1$，且 $p_1$ 在时段分组上有 29–43% 偏差未解释；$k_1/n$ 的近乎精确复现提示**它们在 IOB 轨迹里本就可直接读出**。**因此不能引用为"LTC 解决了部分可观测辨识问题"，只能引用为"LTC 可作为部分可观测系数估计的一个可行架构（在合成仿真上）"。**
- **"未微调 LLM 在安全关键规划上比人工更危险" → TARGET-POSITIVE（本轮最有价值的一条）**：Fig. 5 的危急低血糖 2.6% vs 1% 方向明确，**且与本仓 NR-06"不得宣称 LNN/新范式 SOTA"的纪律同向** —— 这是一条**反向的、约束性的**证据，对本仓的引用纪律有直接价值。**但引用时必须标"仿真、单一案例、无统计检验"。**
- **"CPS-LLM 达到 0% 危急低血糖" → TARGET-DEPENDENT-WITH-NUANCE**：数字漂亮，但**评估用的虚拟患者系数正是 LTC 估出来的同一个值**（自证风险），且**代价是 TAR 从 9.9% 升到 16%**（论文未讨论）。**不得引用为"CPS-LLM 优于人工专家"**，只能说"在同一仿真协议下把危急低血糖事件降到 0"。
- **本篇对"液态网络"家族的贡献定位**：它是 **LTC 作为"物理先验载体"的用法证据**，不是"LTC 作为时序预测器"的证据。**引用时不得把它算作"LNN 在时序建模上有效"的证据。**

## 8. 引用纪律（本篇专用）

1. **必须标注"全部为仿真（UVA/Padova T1D 模拟器 + 虚拟患者），无真人数据"**。
2. **必须标注"安全评估用的虚拟患者系数取自 LTC 估计值，存在自证风险"**。
3. **引用 Table 2 时必须同时给出 $p_1$ 在 Overnight(4.0)/Evening(3.6) 相对真值(2.8) 的偏差**，不得只报 $k_1$/$n$ 的近似精确值。
4. **不得引用为"液态网络在物理/辨识任务上达到 SOTA"**（本仓 NR-06）—— 这是一个单案例、单模拟器、无代码的论文。
5. **本篇是"LTC 谱系"证据**（Liquid Time-Constant），与 Lagrangian/Localized 无关。
6. **与本仓 `2412.02215`（LTC 隐式物理模型反推）不可混引**：两者路线相反（求解器在 loss 里 vs 求解器在解码器里），**任务相同但实现风格与结论强度都不同**。

---
*研读依据：`paper-analyzer` 技能 SOP + AGENTS.md §2 摘要 Agent 标准模块（6 段）+ 研读骨架模板。PDF 全文经 `scripts/extract_pdf_text.py` 提取（44,504 字节 / 979 行），所有公式编号与 PDF 原文对齐；Table 2 / Table 3 / Fig. 4 / Fig. 5 的数字均直接照录论文原文，未做换算。*
