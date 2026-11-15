---
title: Firebolt-VL - Efficient Vision-Language Understanding with Cross-Modality Modulation
date: 2026-11
tags: [LNN, LFM2, SSM, S4, FiLM, cross-modal, MLLM, edge-deployment, efficiency, duplicate-submission]
arxiv_id: 2604.04579
---

# 研读报告：Firebolt-VL — 用 S4 状态空间 + FiLM 调制做跨模态融合，取代交叉注意力

## 1. 元数据

- **论文标题**：Firebolt-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation
- **作者**：Quoc-Huy Trinh†, Mustapha Abdullahi†, Bo Zhao‡, Debesh Jha‡ —— Aalto University / University of South Dakota（† 共同一作，‡ 通讯）
- **发表时间**：2026-04-06（v1）→ 2026-04-07（v2），arXiv:2604.04579v2, cs.CV
- **代码**：https://fireboltvl.github.io（论文声明发布模型与代码）
- **本地 PDF**：[papers/2604.04579/2604.04579.pdf](../../papers/2604.04579/2604.04579.pdf) ／ 文本 [2604.04579.txt](../../papers/2604.04579/2604.04579.txt)
- **arXiv 分类**：cs.CV
- **关联概念**：LFM2（Liquid Foundation Model，**注意其真实架构是 conv+GQA 混合，不是连续时间 ODE**）/ S4 / S4D / Mamba / SSM / FiLM / 线性注意力 / SigLIP 视觉编码器 / MLLM 边缘部署
- **入选理由**：本仓**零覆盖**"液态/状态空间骨干做视觉语言模型"这一方向，且它与本仓 NR-05（LFM2 不是纯液态模型）**直接相关** —— 这是第一个把"LFM2 骨干 + 液态式跨模态融合"组合起来并给出可核对的效率数字的公开工作。

> 🔴 **arXiv 管理员注记（必须随引用一起标注）**：本论文的 arXiv comment 明确写着 **"substantial text overlap with arXiv:2511.11177"**。后者是 **Viper-F1**（同作者 Quoc-Huy Trinh 单作者，2025-11-14），标题为 "Fast and Fine-Grained Multimodal Understanding with Cross-Modal State-Space Modulation"，其方法描述与本文**逐句对应**（同样把 Transformer 解码器换成液态/状态空间，同样提出 Token-Grid Correlation Module + FiLM 调制，同样声称线性时间推理）。Viper-F1 的版本**已被 arXiv 管理员撤下**（"removed by arXiv administrators as the submitter did not have the rights to agree to the license at the time of submission"）。
> **因此本报告把它定性为「同一工作的重新提交（v2/v3）」，而非独立新工作**；其"首次提出 CMM"的优先权主张不可采信。本仓同时**未**研读 `2511.11177`（已撤稿），两者内容按同一份工作对待。

## 2. 核心问题

多模态大语言模型（MLLM）在视觉理解上进展迅速，但**计算与内存开销**限制了它们在资源受限场景（个人助理、文档理解、智能摄像头、机器人操作）的部署。

论文指出该领域存在**两个叠加的困难**：

1. **效率维度**：绝大多数 MLLM 建立在 Transformer LLM 之上，注意力对序列长度是**二次复杂度**。即便 FlashAttention 降低了计算量，**KV cache** 仍然存在，长上下文多模态输入的延迟急剧上升。已有高效路线（MobileVLM 用轻量骨干、MoE-LLaVA 只激活部分专家、SmolVLM2 用 pixel-shuffle 减少视觉 token）**都仍在 Transformer 框架内**，无法摆脱二次复杂度。
2. **感知维度**：小型 VLM 常**无法精确定位细粒度、任务相关的视觉区域**，导致细粒度推理任务（OCR、图表、计数）性能退化。这不是容量问题 —— 而是**结构问题**：多数工作用**简单线性投影**做跨模态对齐，对紧凑 VLM 而言"视觉编码器表征能力有限 → 对齐弱且不稳"；即便用 Dense Connector / Align-KD / Align-GPT 做蒸馏式对齐，仍**缺少视觉与文本之间的交互式融合**，无法把注意力导向与指令最相关的图像区域。Qwen-VL 引入跨注意力解决了交互，但跨注意力**又是二次复杂度**，与效率目标直接冲突。

于是问题被重述为：**能否用一种接近线性的序列算子（状态空间模型）+ 轻量的逐特征仿射调制（FiLM），既完成跨模态的细粒度对齐，又不付交叉注意力的二次代价？**

## 3. 方法论与核心思路

### 3.1 总体架构

```
输入图像 ──► SigLIP 视觉编码器 ──► 网格级视觉表示 X_v ∈ R^{G×D_v}
                                          │
输入指令 ──► Tokenizer ────────────► 文本嵌入 X_t ∈ R^{T×D_t}
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │        Cross-Modal Modulator (CMM)        │
                    │  投影 → 多头 token-grid 相关 → top-k 选取   │
                    │  → 加权视觉上下文 c → FiLM-in → SSM → FiLM-out│
                    │  → 残差 + FFN → X_out ∈ R^{T×D_t}            │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                     LFM2 (Liquid Foundation Model, 0.8B, 文本预训练)
                                          ▼  自回归解码
                                        输出 token
```

**两个阶段训练**：阶段 1 冻结视觉编码器与语言模型、只初始化 CMM（CC3M 数据集）；阶段 2 端到端训练（LLaVA-CoT + MMPR-v1.2 加工成 CoT 格式）。

### 3.2 跨模态调制器（CMM）：token-grid 相关 + FiLM

CMM 的核心思路是：**不需要显式的 token 级注意力，也能知道"每个文本 token 该看哪个图像网格"**。

先做共享空间投影并多头切分：

$$
X'_t = W_t X_t,\qquad X'_v = W_v X_v
$$

然后计算文本 token 与视觉网格之间的缩放点积相关矩阵，并对**网格维**做 softmax：

$$
S_{mm} = \sigma\!\left(\frac{X_t^{H} X_v^{H\top}}{\sqrt{D_H}}\right)
$$

其中 $\sigma(\cdot)$ 是**沿网格维度**的 softmax（把相关分数归一化到 $(0,1)$）。这一步给出"每个文本 token 对每个视觉网格的注意力权重"。随后**每个 token 只保留 top-$k$ 个网格位置**（论文实验得出最优 $k=4$）。

跨头平均后得到每个 token 的视觉上下文嵌入：

$$
\hat S_g = \frac{1}{H}\sum_{h=1}^{H} S_{mm}^{(h)},\qquad c = \sum_{g=1}^{G} \hat S_g X'_v
$$

然后用 **FiLM-in** 把文本嵌入与视觉上下文**线性融合**（不是拼接、不是注意力加权求和）：

$$
X_f = \mathrm{LN}(X_t)\odot(1 + \alpha\gamma_{in}) + \alpha\beta_{in},\qquad [\gamma_{in}, \beta_{in}] = W_f c
$$

其中 $\alpha$ 是**可学习的门控标量**，控制跨模态调制强度。$\gamma_{in}$ 按通道缩放文本表示，$\beta_{in}$ 引入由图像条件化的偏移。

### 3.3 SSM 时序混合 + FiLM-out + 残差

被视觉调制的表示 $X_f$ 送入 SSM：

$$
Y_f = \mathrm{SSM}(X_f)
$$

SSM 在 $O(T)$ 时间内建模序列交互。作者明确对比：与自注意力显式计算两两交互不同，**SSM 通过状态转移隐式传播信息**，同时捕获局部与全局依赖。

再做一次 **FiLM-out** 把 post-SSM 特征对齐回视觉上下文空间：

$$
Y_f = Y\odot(1 + \alpha\gamma_{out}) + \alpha\beta_{out},\qquad [\gamma_{out}, \beta_{out}] = W'_f c
$$

最后残差 + FFN：

$$
X_{out} = \mathrm{LN}\big((X_t + Y_f) + \mathrm{FFN}(X_t + Y_f)\big)
$$

并用均值池化得到全局多模态表示 $z = \frac{1}{T}\sum_t X_{out,t}$。

### 3.4 语言骨干：LFM2（**这里有一个必须点破的事实**）

论文用 **LFM2（Liquid Foundation Model）0.8B 文本模型**作为解码器骨干，理由是"Liquid-based model achieves substantially smaller time complexity than the attention-based model"。

🔴 **但本仓已在 `docs/reports/LFM2_Technical_Report_2511.23404_研读报告.md` 中核实**：LiquidAI 在 LFM2 的**硬件在环架构搜索**里显式实测并**否决**了 CfC / Liquid-S4 / S4 / S5 / Mamba / Mamba2 / 线性注意力，结论是它们"does not improve aggregate quality … and typically worsens device metrics"；**LFM2.5-24B-A2B 的 `config.json` 确认 40 层中 30 层 `conv` + 10 层 `full_attention`，无任何 ODE / LTC / CfC 层**。

⇒ **Firebolt-VL 的 "Liquid decoder" 在架构上并不是液态连续时间模型，而是一个 conv + GQA 的混合骨干**。这与本仓 **NR-05**（"LFM2.5 是纯液态模型" = 事实错误）**直接一致并进一步强化**。因此本篇的效率优势（线性时间）**来自 SSM/conv 骨干与 top-k 稀疏化，而不是来自"液态连续时间"**。论文自己在 §5.3 的 SSM 消融里也承认：**S4/S4D 优于 Mamba**，即**结构化 SSM 才是关键**。

### 3.5 复杂度分析（论文自述）

CMM 只做**一次** token-grid 相关（不像交叉注意力那样逐层重复），复杂度为 $O(TGD_t + TD_t^2 + TD_tf(T))$，其中 $f(T) \in \{1, \log T\}$。对比交叉注意力：每层 $O(TGD_t)$ 计算、内存正比于 $T \times G$。

### 3.6 与 LTC / CfC 的差异

| 维度 | Firebolt-VL | LTC | CfC |
|---|---|---|---|
| 连续时间性质 | ❌ 无 ODE / 无 $\tau$（LFM2 = conv+GQA；CMM 的 SSM 是 S4/S4D） | ✅ 显式 $\tau$，ODE 数值积分 | ✅ 闭式解 |
| 跨模态机制 | token-grid 相关 + FiLM 逐特征仿射 | 不适用（时序） | 不适用（时序） |
| 稀疏性来源 | top-$k$ 网格选取（空间稀疏） | NCP 式稀疏接线（结构稀疏） | 闭式解（计算稀疏） |
| 共同点 | **都用"结构化的、按位置/特征的调制"代替注意力** | | |

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. (1) | $\mathrm{FiLM}(x,c) = \gamma(c)\odot x + \beta(c)$ | 逐特征仿射条件化：$\gamma$ 缩放、$\beta$ 平移 |
| Eq. (2) | $p_\theta(Y\mid H) = \prod_{i=1}^{L_y} p_{\theta_i}(y_i \mid H, y_{<i})$ | 自回归解码，$H = [X_t; X_{mm}]$ |
| Eq. (3) | $X'_t = W_t X_t,\ X'_v = W_v X_v$ | 双模态投影到共享隐空间 |
| Eq. (4) | $S_{mm} = \sigma\!\left(\frac{X_t^H X_v^{H\top}}{\sqrt{D_H}}\right)$ | token-grid 相关矩阵，softmax 沿网格维 |
| Eq. (5) | $\hat S_g = \frac{1}{H}\sum_h S_{mm}^{(h)},\quad c = \sum_g \hat S_g X'_v$ | 跨头平均 + 加权视觉上下文 |
| Eq. (6) | $X_f = \mathrm{LN}(X_t)\odot(1+\alpha\gamma_{in}) + \alpha\beta_{in}$ | **FiLM-in**：视觉条件化文本，门控 $\alpha$ 可学习 |
| Eq. (7) | $Y_f = \mathrm{SSM}(X_f)$ | 线性时间时序混合 |
| Eq. (8) | $Y_f = Y\odot(1+\alpha\gamma_{out}) + \alpha\beta_{out}$ | **FiLM-out**：post-SSM 对齐回视觉空间 |
| Eq. (9) | $X_{out} = \mathrm{LN}\big((X_t + Y_f) + \mathrm{FFN}(X_t + Y_f)\big)$ | 残差 + FFN |
| Eq. (10) | $z = \frac{1}{T}\sum_t X_{out,t}$ | 均值池化得全局多模态表示 |

## 5. 关键成果与贡献

### 5.1 主结果（Table 1，Firebolt-VL = LFM2 0.8B）

| 方法 | LLM | 参数 | VQAv2 | POPE | AI2D | MMMU val | MME_p | SQA_I | MMB_dev |
|---|---|---|---|---|---|---|---|---|---|
| MobileVLM | Mobile-LLaMA | 1.7B | – | 84.5 | 36.6 | 25.3 | 1196.2 | 57.3 | 59.6 |
| MobileVLM V2 | Mobile-LLaMA | 1.7B | – | 84.3 | 38.1 | 19.0 | 1302.8 | 66.7 | 57.7 |
| MoE-LLaVA | Qwen2 | 2.2B | 76.2 | 87.0 | 42.1 | 26.6 | 1291.6 | 63.1 | 59.6 |
| LLaVA-Phi | Phi2 | 2.7B | 71.4 | 85.0 | – | – | 1335.1 | 68.4 | **59.8** |
| SmolVLM2 | Mobile-LLaMA | 0.3B | – | 54.3 | 39.2 | 28.9 | 1236.5 | 58.8 | – |
| **Firebolt-VL** | **LFM2** | **0.8B** | **76.6** | 69.4 | **46.2** | 26.4 | **1376.2** | 56.7 | **64.6** |

值得注意的**诚实之处**：论文自己指出 **POPE 上有性能下降**（69.4，低于 MoE-LLaVA 的 87.0），并归因于"SSM 的隐式状态传播在抑制幻觉内容上不如显式注意力"。这是 SSM 路线的一个**真实代价**，没有被掩盖。

### 5.2 效率（单张 H100，POPE 数据集，最大输出 256 token）

| 模型 | 吞吐 (tok/s) |
|---|---|
| **Firebolt-VL** | **46.67** |
| SmolVLM2 | 37.80 |
| MoE-LLaVA | 19.97 |
| MobileVLM V2 | 42.86 |
| MobileVLM | 39.99 |

### 5.3 消融一：SSM 变体选择（Table 2）

| SSM | POPE | AI2D | MMMU val | 均值 |
|---|---|---|---|---|
| Mamba | 57.4 | 39.9 | 23.6 | 40.3 |
| S4D | 69.9 | 45.6 | 23.7 | 46.4 |
| **S4** | **69.4** | **46.2** | **26.4** | **47.3** |

**结构化 SSM 明显优于 Mamba** —— 作者的解释是"视觉嵌入代表来自五个空间网格的信息，结构化状态空间架构能在状态转移过程中更精确地建模几何关系"。⇒ **这条证据对"液态/连续时间"命题是负面信号**：Mamba（更接近"液态"叙事里的连续时间变体）**输给** S4（经典结构化线性 SSM）。

### 5.4 消融二：跨模态融合机制（Table 3）

| 方案 | MME | AI2D | MMMU val |
|---|---|---|---|
| Prepend（线性投影后直接拼接） | 981.5 | 24.1 | 22.1 |
| Cross-attended（Q-Former 式） | 1036.5 | 45.1 | 24.4 |
| **CMM（本文）** | **1376.2** | **46.2** | **26.4** |

CMM 相对 Cross-attend 在 MME 上 **+339.7**（+32.8%），且**复杂度更低**（一次相关 vs 逐层交叉注意力）。

### 5.5 消融三：top-$k$ 网格数（Table 4）

| k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| MME | 1039.8 | 1192.0 | 1258.6 | **1376.2** | 1289.4 |
| AI2D | 43.9 | 44.1 | 45.4 | 46.2 | 45.6 |
| MMMU val | 22.2 | 24.3 | 25.8 | **26.4** | 26.0 |

**单调上升到 $k=4$，$k=5$ 反而略降** —— 作者的解释是"选太多网格会引入噪声或稀释最有信息区域的贡献"。这是一个干净的**稀疏度-性能拐点**证据。

### 5.6 贡献清单

1. **把 LFM2 骨干 + S4 式跨模态调制组合成一个端到端 0.8B 高效 MLLM**，在 7 个基准上与 7B–9B 大模型竞争。
2. **提出 CMM**：用一次 token-grid 相关 + 双向 FiLM（in/out）+ SSM 取代逐层交叉注意力，复杂度从每层 $O(TGD_t)$ 降为一次性 $O(TGD_t + TD_t^2 + TD_tf(T))$。
3. **给出可复现的 $k$ 稀疏度拐点**（$k=4$ 最优，$k=5$ 退化）与 **SSM 变体排序**（S4 > S4D > Mamba）。
4. **诚实报告了 POPE 的性能退化**并给出机制解释。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **只支持单图输入**；扩展到多图或视频序列**仍是开放挑战**，尚未开发针对视频输入优化的 CMM 版本。
- 未来方向是"能整合时序视觉信息与文本指令的高效跨模态连接器"。

### 6.2 隐含局限（本仓视角）

1. 🔴 **重复提交，且优先权不可采信**：与 `2511.11177`（Viper-F1）文本大幅重叠，后者已被 arXiv 管理员撤稿。**任何"CMM 是首个 SSM 跨模态融合"的主张都不能引用本篇作为首证**。
2. 🔴 **"Liquid" 一词在本文中的用法与本仓谱系不一致**：LFM2 是 conv+GQA 混合，**不含任何 ODE/CfC/LTC 成分**（本仓 NR-05 已核实）。把本篇记为"液态网络在 VLM 上的应用"是**范畴错误**；它真正属于"**结构化 SSM 在多模态上的应用**"。
3. **视觉网格只有 5 个**（论文自陈 "the vision encoder produces five grids"）—— 意味着 top-$k$ 的可选范围只有 1..5，**$k$ 消融的搜索空间被编码器的粗粒度限制住了**；结论"$k=4$ 最优"很可能只是"接近全选（5）时最稳健"这一有限网格上的表述，**不能外推**到高分辨率网格或多图场景。
4. **算子选择与骨干选择耦合**：CMM 内部用 S4，而语言骨干是 LFM2 的 conv+GQA —— 论文**没有做"CMM 换成 Mamba/线性注意力 + 同一骨干"的完整二维消融**，因此 CMM 增益与 SSM 变体增益**无法分离**。
5. **训练成本高**（2×H100 80GB，两阶段各 5 epoch），与"资源受限部署"的目标形成**讽刺性的张力** —— 论文只优化了**推理**，没有触及**训练**。
6. **无幻觉/鲁棒性专项评测**：POPE 退化被承认，但**没有针对视觉幻觉的机理分析**。
7. **无 Jetson/边缘实测**：本仓最关心的边缘部署（Orin Nano 量化、显存、实测 tok/s）**完全缺失** —— 只有单张 H100 上的吞吐数字。

### 6.3 未来方向

- 多图 / 视频输入的 CMM；
- 时序视觉信息与文本指令的高效融合；
- 更大视觉网格下的 $k$ 稀疏度再调优。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的什么变体**：**不能作为液态变体**（无 $\tau$、无闭式解），但它的 **FiLM-in → SSM → FiLM-out 夹心结构**是本仓可以借鉴的**通用"调制型残差块"模板**，且它与本仓 `lnn/core/variants.py` 里若有的 FiLM 相关实现路线天然兼容。**更重要的价值是一个反面教材**：这是本仓**第一次拿到"论文自称 Liquid、实际架构里零液态成分"的成规模公开案例**（不是 LFM2 的技术报告，而是第三方在它上面搭系统）。它说明**"Liquid" 这个词在 2026 年的 ML 论文里已经被泛化成"非 Transformer 时序骨干"的代名词**，本仓在追踪文献时**必须逐篇验证架构**，不能凭标题归类。

**可嫁接到哪个 bench 脚本**：本仓 `jetson_lnn_benchmark.py` 测的是 cell 级吞吐；本篇提示一个**本仓缺失的"结构化先验 vs 通用递归"的同参数量对照**。具体地，它的 SSM 消融（Mamba 40.3 vs S4 47.3，**S4 胜**）可以直接搬到本仓 toy benchmark：把 `lnn/core/ltc.py`（对 $\tau$ 加性）与一个 **S4 风格结构化线性循环**放在**同等参数量**下比 toy 任务精度 —— 本仓至今的所有变体都是"液态家族内部互比"（LTC vs CfC vs GRU），**从未与"非液态的结构化线性时序模型"同台**。这可能是本仓目前**最便宜、价值最高的一个新对照轴**。

**提供哪个新 ablation axis**：① **top-$k$ 稀疏度拐点**（对应本仓的"稀疏接线数量"轴 —— NCP 的稀疏度 vs 精度的权衡，本仓 `sncp_policy_lite.py` 已有类似维度但没有拐点分析）；② **FiLM 门控强度 $\alpha$ 的消融**（论文给了 $\alpha$ 但**没给它的消融表** —— 这是一个现成的空缺）；③ **同参数量下的"液态 vs 结构化 SSM"对照**。

**Verdict**：
- **对"LFM2 是纯液态模型"（NR-05）**：**TARGET-NEGATIVE-WITH-NUANCE** —— 本篇是**第三方独立印证**：LFM2 作骨干时，作者强调的是"linear time complexity"和"S4 结构化"（并实测 Mamba 输给 S4），**全文没有一处把 LFM2 描述成连续时间 ODE 模型**。NR-05 应当**再获一次外部强化**。
- **对"液态网络在多模态任务上更优"**：**TARGET-NEGATIVE** —— 本篇**不能**支持该命题，因为它的骨干与融合模块**都不含液态连续时间成分**。它证明的是"**结构化 SSM + 稀疏调制在效率-精度折中上很有竞争力**"。
- **对"本仓的液态谱系是否已到头"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 本篇的 $k=4$ 拐点与 S4>Mamba 的排序提示：在"**位置/结构化调制**"这条线上，**液态不是最优解**，经典结构化线性递推可能更合适。本仓若继续沿"液态 vs 结构化"的轴推进，**S4/Mamba 应当被加入对照组**，而不是继续只在液态家族内部比较。
