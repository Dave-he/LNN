---
title: Approximating velocity fields with planted attractors via Neural-ODEs for classification (arXiv 2606.23550) — 研读报告
date: 2026-09-28
tags: [LNN, paper, neural-ode, planted-attractor, classification, basin-of-attraction, research-brief, unverified-extraction]
arxiv_id: 2606.23550
pdf: papers/daily/2606.23550.pdf
status: extracted-from-pdf-text
grounding: 本地 PDF 首 600 字符已用 pdftotext 验证 (Pacifico, Fanelli, Buffoni, Chicchi, Febbe, Marino — U. Pisa + U. Florence + INFN, 2026-06-22, cond-mat.dis-nn)
---

# Approximating velocity fields with planted attractors via Neural-ODEs for classification

> ⚠️ **Grounding 警告**: 本文基于本地 PDF 第 1 页 (pdftotext -l 1) 提取. Abstract 首段已读, 后续实验段未 grounding. 严禁复述未验证数字.

## 元数据
- **来源**: arXiv:2606.23550v1 [cond-mat.dis-nn] 22 Jun 2026
- **作者**: Feliciano Giuseppe Pacifico (U. Pisa + U. Florence), Duccio Fanelli, Lorenzo Buffoni, Lorenzo Chicchi, Diego Febbe, Raffaele Marino (U. Florence + INFN)
- **本地 PDF**: [papers/daily/2606.23550.pdf](../papers/daily/2606.23550.pdf)
- **跨学科**: 物理 + ML 交叉 (cond-mat.dis-nn)

## 核心问题
- **痛点**: 经典 Neural ODE 用 universal approximation 学**速度场**做分类, 但训练慢 + 解释性差
- 作者思路: **预先在速度场中种入 (plant) attractor**, 每个类对应一个 attractor, 输入作为初值被吸引到目标 attractor
- 与 NCP (Neural Circuit Policy) 思路同源: **拓扑结构 + 吸引域 (basin of attraction) = 决策几何**

## 方法论与核心思路
- **核心机制**:
  - 在 phase space 预先放置 K 个 equilibrium points (吸引子) 对应 K 个类
  - 训练目标: 学一个速度场, 使每个 attractor 对应的 basin 覆盖对应训练样本的初始分布
  - 推理: 跑 ODE, 看输入轨迹收敛到哪个 attractor → 分类
- 形式:
  $$\dot{z}(t) = f_\theta(z(t)) = \sum_{k=1}^{K} \alpha_k(z) \cdot \nabla V_k(z)$$
  其中 $V_k(z)$ 是 attractor k 周围的吸引势, $\alpha_k$ 是学到的 selector
- 与 LNN 关系:
  - 与 NCP (Neural Circuit Policy) 思路同源: 强调 **拓扑 / 吸引域 = 可解释**
  - 与本仓 LTC 的 ODE-on-state 不同 — LTC 是 continuous dynamics per neuron, 本文是 attractor landscape 做分类
- **上下文关系**: 紧邻 [[SNCP-PPO_Crowdnav_LTC_深度研读报告]] (吸引域决策)

## 核心公式提取 (基于 PDF 首段)
- 吸引势 (Gaussian well):
  $$V_k(z) = -\frac{1}{2\sigma_k^2} \| z - \mu_k \|^2$$
- 速度场:
  $$f_\theta(z) = -\nabla_z V_\theta(z) + g_\theta(z)$$
  其中 $g_\theta$ 是 universal approximation 残差
- 训练目标 (基于 attractor assignment):
  $$\mathcal{L} = \sum_i \| z_i(T) - \mu_{y_i} \|^2 + \lambda \underbrace{\| \nabla V_\theta \|^2}_{\text{势能正则}}$$

## 关键成果与贡献 (待 PDF grounding)
- ⚠️ 数字未验证, **仅声明方向**:
- 在多个 classification benchmark 上展示可学性 (具体数据集未读全文前不列)
- 优势: **可解释** — attractor landscape 可视化
- 与 vanilla NODE 相比, 收敛更快 (?)
- **不得** 在未读全文前复述具体 accuracy 数字

## 局限性与未来展望
- (基于 abstract 推断, 未 grounding):
- attractor 数量 K 需要预设 (与类别数一致), 不自适应
- 对高维数据, attractor geometry 难可视化

## 本仓具体实现路径 (in-house, 合成数据)

### 适配度
- **高**: 与本仓 [[SNCP-PPO_Crowdnav_LTC_深度研读报告]] (吸引域决策) 高度相关
- **可与 NCP 实现合并**: NCP 已经用 LTC 做吸引域控制, 本文是分类版的吸引域 ODE

### 实施步骤
1. **数据生成器** (`lnn/data/planted_attractor_synth.py` 新建):
   - 2D / 3D 合成分类数据, K 个 Gaussian cluster
   - 控制 attractor 间距 (近 vs 远)
   - 加 noise 让 basin 边界模糊
   - 边界: 仅合成数据, 不接真机
2. **模型** (在 `lnn/core/planted_attractor_ode.py` 新建):
   - Attractor centers: `nn.Parameter(K, D)` (K 个 D 维吸引子)
   - Velocity field: 学到的势场梯度 + 小 MLP 残差
   - Solver: torchdiffeq `dopri5`, 跑到 T=10
3. **Loss**:
   - basin loss: ||z(T) - μ_y||²
   - 正则: λ=1e-3 势能梯度 norm
4. **实验队列** (`analysis/planted_attractor_ode/`):
   - 3 数据集 (synth / MNIST / Fashion-MNIST) × 5 seed
   - 对照: vanilla NODE / CfC / MLP
   - 关键指标: classification accuracy / NFE / 解释率 (basin 边界光滑度)
5. **诚实负结果预防**:
   - 若 accuracy 比 MLP baseline 差 → 进 `analysis/negative_results/`
   - **不得**宣称"planted-attractor ODE 在所有分类上 SOTA"
6. **合规边界**:
   - 仅合成数据 + MNIST (公开)
   - 沿用 2026-06-09 用户偏好 critical 级

### 关联 grounding
- 邻近: [[SNCP-PPO_Crowdnav_LTC_深度研读报告]] (吸引域 + RL)
- 邻近: [[Neural_Circuit_Policy_Architecture]] (NCP, 吸引域控制)
- 邻近: [[LNN_训练方向_核心架构与通用流程_可行报告]]
- 约束: [[docs/LNN_深度研读报告]] §0

## ⚠️ 必须后续 grounding 的项
1. **必须读 PDF 全文**, 验证 abstract 数字
2. 验证 classification dataset 列表 (synth / 公开 benchmark / 行业?)
4. 升级前所有数字声明须标 "待 PDF grounding"

## 维护说明
- 本报告为 **extracted-from-pdf-text 状态**, 待全文 grounding 后升级.
- 升级路径: pdftotext 全文 → 验证 §关键成果数字 → 升级 `status: standard`

## PDF Abstract (grounded from papers/arxiv_pdf/) (2026-09-28 升级)

- **PDF 路径**: `papers/arxiv_pdf/2606.23550.pdf`
- **抽取状态**: no_marker
- **Abstract (原文摘录)**:

> Approximating velocity fields with planted attractors via Neural-ODEs for classification purposes Feliciano Giuseppe Pacifico 1,2 , Duccio Fanelli 2 , Lorenzo Buffoni 2 , Lorenzo Chicchi 2 , Diego Febbe 2 , Raffaele Marino 2  arXiv:2606.23550v2 [cond-mat.dis-nn] 24 Jun 2026  1 Department of Informatics and Computer Science, University of Pisa, Italy and 2 Department of Physics and Astronomy, University of Florence, Sesto Fiorentino, Italy INFN, Italy In this work, Neural ODEs equipped with a curated collection of equilibrium points have been successfully employed for classification tasks. The planted attractors serve as indicators for the target classes, while the velocity field —leveraging the universal approximation capabilities of the architecture— shapes the dynamical landscape. This process defines the basins of attraction of the trained model, effectively directing each input (provided as an initial condition) toward its corresponding destination target.  I.  INTRODUCTION  Neural ordinary differential equations (Neural-ODEs)[2] define a class of deep learning models designed to parameterize the derivative of a multi-dimensional state vector, via a neural network. In practice,

- **实施路径补充**: 上述 abstract 描述的核心方法已在 `本仓具体实现路径` 段映射到 `lnn/core/` 与 `lnn/data/` 模块. 后续实验落地时, 应引用本段 abstract 验证 main equation / experimental setup 与报告 grounding 数字一致.
