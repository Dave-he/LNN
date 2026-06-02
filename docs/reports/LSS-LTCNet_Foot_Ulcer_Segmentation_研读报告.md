---
title: Explainable Continuous-Time Mask Refinement with Local Self-Similarity Priors for Medical Image Segmentation
date: 2026-06
tags: [LNN, LTC, Liquid-Time-Constant, Continuous-Depth, Medical-Image-Segmentation, Explainable-AI, Foot-Ulcer, FUSeg]
---

# 研读报告：LSS-LTCNet — 基于局部自相似性先验与 LTC 连续时间边界精炼的可解释医学影像分割

## 1. 元数据

- **论文标题**：Explainable Continuous-Time Mask Refinement with Local Self-Similarity Priors for Medical Image Segmentation
- **作者**：Rajdeep Chatterjee, Sudip Chakrabarty, Trishaani Acharjee (Kalinga Institute of Industrial Technology, Bhubaneswar, India; AmygdalaAI-India Lab)
- **发表时间**：2026 年 2 月 (v1, 28 Feb 2026)
- **来源**：arXiv:2603.00459v1
- **本地 PDF**：[papers/daily/2026-06-03/2603.00459v1.pdf](../papers/daily/2026-06-03/2603.00459v1.pdf)
- **数据集**：MICCAI Foot Ulcer Segmentation (FUSeg) Challenge — 1,210 张 (810 train / 200 val / 200 test 私有)

## 2. 核心问题

糖尿病足溃疡 (foot ulcer) 的精准自动分割对临床愈合评估与治疗决策至关重要。现有 SOTA 分割器 (U-Net 家族、ViT-UNet) 在通用医学分割上表现良好，但在**糖尿病足溃疡**这类**组织异质 + 边界对比度低 + 形状不规则**的场景下，存在三类系统性问题：

1. **对比度与边界精度不足**：坏死组织、granulation 与健康表皮之间是低对比度、弥散渐变带，标准 intensity-based 卷积难以抓住尖锐过渡；
2. **ViT patchification 损失高频结构**：自注意力的 patch 化机制在平滑分割的同时削弱了细粒度形态学细节；
3. **缺乏可解释性**：现有模型依赖 Grad-CAM / SHAP 等 post-hoc 解释，对临床决策的可信度贡献有限。

论文的核心论断：**"分割边界的演化本质上是一个 ODE 驱动的连续过程"**。基于此，提出 LSS-LTCNet 框架，把**确定性的局部自相似性 (LSS) 结构性先验**注入到 encoder 早期，把 **Liquid Time-Constant (LTC) 连续深度循环**放在 bottleneck 做迭代式边界精炼，并以 **Boundary Alignment Loss (BAL)** 把 LSS 边信息与预测梯度对齐，实现 **ante-hoc (内置) 可解释 + SOTA 精度 + 25.70M 参数量**。

## 3. 方法论与核心思路

### 3.1 总体架构（4 模块流水线）

| 模块 | 角色 | 输出 |
|---|---|---|
| ResNet-34 Encoder | 提取多尺度语义特征 | $C_1, C_2, C_3, C_4, F_5$ |
| Additive LSS Fusion | 在 encoder 早期注入局部结构先验 | $F_1 = C_1 \oplus F_{LSS}$ |
| **LTC Refinement Loop (T=4 步)** | 在 bottleneck 迭代式精炼全局 spatial token | Refinement Token |
| Deep Supervision Decoder | 融合 skip + Refinement Token, 输出最终 mask + 2 个辅助头 | $\hat Y, \mathrm{Aux}_1, \mathrm{Aux}_2$ |

LSS 与 LTC 在物理意义上互为补充：LSS 给出"组织边界在哪"的结构性硬证据；LTC 在瓶颈处把这张边图当作 ODE 的初始条件，按连续时间步迭代地把分割轮廓**shrink-wrap**到真实组织边界。

### 3.2 局部自相似性 (LSS) Extractor

给定输入 $I$，对每个空间位置提取 K×K×3 局部 patch $p_i$。为消除临床成像的局部光照方差，先做 zero-center + 归一化：

$$
\hat p_i = \frac{p_i - \mu(p_i)}{\|p_i - \mu(p_i)\|_2 + \epsilon}
$$

再计算中心 patch $\hat p_c$ 与搜索半径 $R$ 内 $N$ 个邻居 $\hat p_n$ 的余弦相似度：

$$
S(\hat p_c, \hat p_n) = \hat p_c \cdot \hat p_n
$$

把每点邻域相似度集合 $S_c = \{S_1,\dots,S_N\}$ 聚合成 3 通道 LSS 图 $M_{LSS} \in \mathbb{R}^{3\times 512\times 512}$：

$$
M_{LSS}(x, y) = [\mu(S_c),\; \max(S_c),\; \sigma(S_c)]^T
$$

通过 3×3 卷积 + 双线性插值投影到 $F_{LSS} \in \mathbb{R}^{64\times 256\times 256}$，最后 element-wise 加到 encoder 早期特征 $C_1$ 上：

$$
F_1 = C_1 \oplus F_{LSS}
$$

三通道各自具有清晰的物理含义：

- **LSS Mean (μ)**：组织均匀性，识别稳定颗粒组织区，防止预测被光照伪迹碎片化；
- **LSS Max**：保留结构连续性，保护微结构 (上皮桥、细血管模式)；
- **LSS Std (σ)**：确定性边界检测器，定位坏死组织与健康皮肤的尖锐过渡带。

### 3.3 连续时间精炼 (LTC Bottleneck)

把 ResNet-34 最深层特征 $F_5 \in \mathbb{R}^{512\times 16\times 16}$ 与"初始粗 mask + 演化中 mask"做 GAP 拼接，得到 514 维输入向量 $x = [\mathrm{GAP}(F_5); \mathrm{GAP}(Y_{\text{masks}})]$。隐藏状态按 ODE 演化：

$$
\frac{dh}{dt} = \bigl[-h + \sigma_{rel}(W_h h + W_{in} x + b)\bigr] \odot \frac{1}{\tau(x)}
$$

其中 $\sigma_{rel}$ 是 ReLU，**输入依赖时间常数** $\tau(x) = \mathrm{softplus}(W_\tau x) + \epsilon$ 调节 cell 记忆时长。ODE 用 Euler 离散化迭代 $T=4$ 步：

$$
h_{t+1} = h_t + \Delta t \cdot \frac{dh}{dt}\bigg|_{h_t}
$$

最终状态被投射为 **Refinement Token**，广播到 decoder 的最高分辨率 stage，引导"轮廓 shrink-wrap"过程。

### 3.4 优化与 Boundary Alignment Loss

总损失为多尺度 deep supervision 加权和：

$$
L_{total} = \sum_{k \in \{m, a_1, a_2\}} \lambda_k \bigl(L^{(k)}_{BCE} + L^{(k)}_{Dice}\bigr) + \lambda_b L_{BAL}
$$

权重 $\lambda_m=1.0, \lambda_{a_1}=0.4, \lambda_{a_2}=0.2, \lambda_b=0.5$。

**Boundary Alignment Loss (BAL)** 是关键——它把 LSS 的 Mean 通道 $M^\mu_{LSS}$ 当作确定性边图，用 Sobel 算子对齐预测概率图梯度：

$$
L_{BAL} = \mathrm{MSE}\bigl(\nabla(\sigma(\hat Y)),\; \nabla(M^\mu_{LSS})\bigr)
$$

直觉上：标准 BCE/Dice 把"组织过渡带"当作普通像素优化，无法联合利用高频结构先验与 LTC 连续时间动力学。**BAL 强制让预测边界的梯度与 LSS 边图一致**，让 LSS 与 LTC 两个模块的能力被解锁。

### 3.5 上下文关系

- **与 LTC 经典论文 (Hasani 2021) 的关系**：本文是 LTC 在**视觉分割**领域的一次应用，遵循"ODE 驱动的连续深度循环"框架，但把时间常数 $\tau$ 显式变成输入依赖的 softplus 输出，强调"对组织边界的 ODE 演化"；
- **与 U-Net / ViT-UNet 的关系**：保留 ResNet-34 编码器与 skip connection，但在 encoder 早期 additive 注入 LSS 边图，在 bottleneck 替换 attention 为 LTC；
- **与可解释 AI 关系**：与 Grad-CAM、SHAP 等 post-hoc 解释不同，LSS 模块本身产生**确定性的三通道结构统计图**作为可读证据，属于 ante-hoc 内置可解释 (intrinsic explainability)。

## 4. 核心公式

### 4.1 LSS patch 归一化

$$
\hat p_i = \frac{p_i - \mu(p_i)}{\|p_i - \mu(p_i)\|_2 + \epsilon}
$$

### 4.2 patch 间余弦相似度

$$
S(\hat p_c, \hat p_n) = \hat p_c \cdot \hat p_n
$$

### 4.3 3 通道 LSS 图

$$
M_{LSS}(x, y) = [\mu(S_c),\; \max(S_c),\; \sigma(S_c)]^T \in \mathbb{R}^{3 \times 512 \times 512}
$$

### 4.4 早期加性融合

$$
F_1 = C_1 \oplus F_{LSS}, \quad F_{LSS} \in \mathbb{R}^{64 \times 256 \times 256}
$$

### 4.5 LTC 连续时间动力学 (Euler 离散, T=4)

$$
\frac{dh}{dt} = \bigl[-h + \sigma_{rel}(W_h h + W_{in} x + b)\bigr] \odot \frac{1}{\tau(x)}, \quad \tau(x) = \mathrm{softplus}(W_\tau x) + \epsilon
$$

### 4.6 Boundary Alignment Loss

$$
L_{BAL} = \mathrm{MSE}\bigl(\nabla(\sigma(\hat Y)),\; \nabla(M^\mu_{LSS})\bigr)
$$

### 4.7 总损失

$$
L_{total} = \sum_{k \in \{m, a_1, a_2\}} \lambda_k \bigl(L^{(k)}_{BCE} + L^{(k)}_{Dice}\bigr) + \lambda_b L_{BAL}
$$

## 5. 关键成果与贡献

### 5.1 与 SOTA 的对比（MICCAI FUSeg validation, Table 1）

| Method | Params (M) | FLOPs (G) | Dice ↑ (%) | IoU ↑ (%) | HD95 ↓ (px) |
|---|---:|---:|---:|---:|---:|
| VGG16-UNet | 69.31 | 42.19 | 84.88 | 77.29 | 14.72 |
| ViT-UNet | 130.22 | 102.00 | 80.16 | 71.32 | 15.32 |
| UNet | 31.78 | 374.34 | 84.04 | 76.23 | 13.31 |
| Mask R-CNN | 90.44 | 67.13 | 83.97 | 76.42 | 12.89 |
| ResNet101-UNet | 267.62 | 112.79 | 84.92 | 77.73 | 13.03 |
| U2Seg | 86.47 | 88.46 | 75.85 | 70.13 | 16.07 |
| SegNet | 69.21 | 42.19 | 85.32 | 78.11 | 12.71 |
| **LSS-LTCNet** | **25.70** | 82.45 | **86.96** | **79.54** | **8.91** |

要点：
- **Dice 86.96% / IoU 79.54%** 同时超过所有 SOTA；
- **HD95 8.91 px** 相对次优 SegNet (12.71 px) 改进 **30%**——这是临床最关心的边界精度；
- **参数量 25.70M**，相对 ResNet101-UNet 减少 **10×**，相对 ViT-UNet 减少 **5×**；
- **FLOPs 82.45G**，远低于 UNet (374G)。

### 5.2 消融研究 (Table 2)

| 配置 | Dice (%) |
|---|---:|
| ResNet-34 baseline | 85.22 |
| + LSS + LTC (without BAL) | 76.18 (退步 9%) |
| + LSS + LTC + BAL (完整) | **86.96** (+1.74 vs baseline) |

**BAL 是 LSS 与 LTC 协同的关键粘合剂**：没有 BAL，单纯堆叠 LSS 与 LTC 反而因优化不匹配退化 9%；BAL 强制预测梯度与 LSS 边图一致后，才释放了 LSS 与 LTC 的全部潜力。这条结论对后续所有"结构先验 + 连续时间动力学"组合工作都有方法论意义。

### 5.3 主要贡献

1. **LSS Fusion**：在 encoder 早期显式注入 patch 自相似性三通道结构先验 (μ/max/σ)，并对每个通道的物理意义给出临床可读解释；
2. **LTC Bottleneck Refinement Loop**：把边界精炼形式化为 ODE 演化，迭代 T=4 步把全局 spatial token shrink-wrap 到真实边界；
3. **Boundary Alignment Loss + Ante-hoc XAI**：用 Sobel 梯度对齐把 LSS 与 LTC 联合优化，并在结构上提供可视化审计轨迹 (LSS mean/max/std 三通道)；
4. **SOTA 边界精度 + 极致轻量**：25.70M 参数 + 86.96% Dice + 8.91 px HD95，且对 mHealth 部署高度友好。

## 6. 局限性与未来展望

### 6.1 局限

1. **单数据集验证**：只在 MICCAI FUSeg 一个内部数据集上做了对比；FUSeg test 集 ground truth 由挑战赛组织方保留，**所有量化指标都来自 200 张 validation set**，泛化性证据弱；
2. **ResNet-34 backbone 仍是 ImageNet 预训练的**：LSS 与 LTC 模块在数据效率上的贡献被预训练权重"稀释"——论文没有给"从零训练 ResNet-34 + LSS + LTC"的对照；
3. **LTC T=4 是硬编码超参**：没有 ablation 不同 T 值对精度 / 推理成本的影响；
4. **缺乏多模态融合**：FUSeg 是 RGB 图像集，没有结合 depth / 热成像 / 临床元数据 (血糖、HbA1c 等) 的多模态拓展；
5. **临床评估缺位**：缺少放射科医生 / 伤口护理专家对 ante-hoc XAI 三通道图的人体感知评估，无法证明"对临床决策真的有用"。

### 6.2 未来展望

- 在更大规模 (如 FUSeg 后续 + 私有 mHealth 数据) 上做 multi-center 验证，并对 LSS 三通道做放射科医师评分；
- 探索 **输入依赖 τ 与不确定性估计** 的结合 (类似 Bayesian LTC)，把边界不确定性作为可解释性的补充维度；
- 与 LFM2 / Liquid Foundation Models 衔接，把 LTC bottleneck 替换为预训练 liquid 编码器，验证在医疗影像上的 scaling 行为；
- **mHealth 部署实证**：论文强调 25.70M 参数对 mHealth 友好，但未给出 Jetson / iPhone / 树莓派等端侧延迟 / 功耗数据；
- 把 BAL 的"梯度对齐"思路推广到其他结构先验 + 连续时间动力学的组合 (例如与 6G 通信中的 Liquid Crystal Antenna 几何先验结合)。
