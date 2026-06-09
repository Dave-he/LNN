---
title: Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting
date: 2026-06-10
tags: [LNN, CfC, LTC, 3D-Gaussian-Splatting, D-3DGS, 4D-Reconstruction, Continuous-Time, Closed-Form-Cells, Dynamic-3D, APSIPA]
---

# 研读报告：LNN-as-Drop-in-CfC Deformation Field for Dynamic 3D Gaussian Splatting

> 本文是 [[Liquid_Neural_Networks_Latest_Papers_Summary]] 与 [[LNN_深度研读报告]] 中"连续时间形变场/动态3D"方向的最新沉淀。论文首次把 CfC 单元（Liquid Neural Network 的闭式解）作为 drop-in 替换模块嫁接到 D-3DGS 的形变 MLP 上，将"看起来连续的 t"提升为"结构性连续的 t"，无需 ODE/SDE 求解器。

## 1. 元数据
- **论文标题**：Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting
- **作者**：Mingzhao Li, Arghya Pal, Guan Yuan Tan（共同一作）
- **单位**：School of Information Technology, Monash University, Malaysia
- **发表时间**：2026 年 6 月 4 日（v1, arXiv:2606.07670v1）
- **发表场合**：2026 APSIPA ASC（Asia Pacific Signal and Information Processing Association Annual Summit and Conference）
- **学科**：cs.CV / cs.AI（Computer Vision and Pattern Recognition; Artificial Intelligence）
- **关键词**：4D reconstruction, dynamic 3D Gaussian splatting, liquid neural networks, continuous-time deformation, closed-form continuous-time cells
- **代码**：基于公开 D-3DGS PyTorch 代码库，修改位于 `utils/time_utils.py` 的 `DeformNetwork` 类（论文未给出独立 repo 链接）
- **本地 PDF**：[papers/2606.07670v1_Liquid_NN_3DGS_Deformation_Field.pdf](../../papers/2606.07670v1_Liquid_NN_3DGS_Deformation_Field.pdf)

## 2. 核心问题

4D 重建（从单目视频重建动态 3D 场景）是 AR / 数字孪生 / telepresence 的基础问题。D-3DGS（Deformable 3D Gaussian Splatting, CVPR 2024）通过"规范化 3D 高斯集合 + MLP 形变场"已成为主流范式：

$$(\Delta x_i, \Delta r_i, \Delta s_i) = F_\theta(\gamma(\mathrm{sg}(x_i)), \gamma(t)) \tag{1}$$

但论文指出 MLP 形变场存在 **"the discreteness gap"**：

1. **结构上离散**：尽管 t 是连续物理量，MLP 是逐帧独立前馈的——架构里没有任何机制把 $F_\theta(t)$ 和 $F_\theta(t+\delta t)$ 联系起来；时间平滑性只能由优化器"顺带"逼出来。
2. **可解释性差**：读者很容易把"拟合到 t"误读为"对 t 连续"，但训练信号其实是"sample-dense 但 independent"的一堆孤立 t。
3. **缺乏连续性先验**：当单目监督含 pose drift、运动模糊、光照变化时，MLP 没有任何结构性手段去 damping 这些噪声。

现有的连续时间替代方案也有代价：
- **Neural ODE / Latent-ODE-GS**：每个 forward pass 都要调用 ODE 数值求解器，训练 / 推理慢一档；
- **SDE 变体**：对部分观测/重噪声更鲁棒，但求解器开销更高。

> **本文定位**：寻找"中间点"——给形变场加上 **连续时间语义（continuous-time semantics）**，但保持 **feed-forward inference** 与 **零 ODE 求解器**。

## 3. 方法论与核心思路

### 3.1 总体设计 — Depth-as-Time

把 D-3DGS 的 8 层 MLP 形变场 $F_\theta$ **完全替换为 D 个 CfC 单元的栈**：

| 维度 | D-3DGS（基线） | Ours (CfC stack) |
|---|---|---|
| 形变场实现 | 8 层 MLP, width=256 | D=6 层 CfC cell, hidden=128, backbone 64×2, GELU |
| t 的角色 | 仅作为位置编码输入 $\gamma(t)$ | **每层 CfC cell 的 elapsed-time signal** $\tau=t$ |
| 平滑性来源 | 优化器副产品（byproduct） | 结构内嵌：sigmoid 时间门 $\sigma_\tau = \sigma(W_{a}z \cdot t + W_b z)$ |
| 数值求解器 | 无 | 无 |
| 形变之外 | 保留全部 D-3DGS 流水线 | 完全保留（canonical Gaussian、rasterizer、L1+SSIM、密度控制、AST schedule、40k iter Adam） |

关键设计要点：**Depth-as-Time**（深度即时间）——网络深度扮演经典 CfC 序列模型中时间递归的角色。每帧前向时，hidden state 重置为 0；深度 D 上的状态传递给出对 t 的结构化响应。

### 3.2 CfC 单细胞（核心机制）

单细胞把输入 $u_i$、隐藏态 $h_i$、elapsed-time $\tau=t$ 映射到更新后的隐藏态：

$$
\begin{aligned}
z &= \phi([u_i; h_i]) \\
g &= \tanh(W_g z) \\
h_{\text{cand}} &= \tanh(W_h z) \\
\sigma_\tau &= \sigma(W_a z \cdot \tau + W_b z) \\
h' &= g \odot (1 - \sigma_\tau) + h_{\text{cand}} \odot \sigma_\tau
\end{aligned}
\tag{2}
$$

**关键观察**：
- $\sigma_\tau$ 是 **t 的仿射 sigmoid**——单调连续可微，且**仅通过这条路径**让 $\tau$ 进入细胞；
- 闭式更新是 LTC ODE 闭式解的离散表达，无需 ODE solver；
- $\phi$ 是 GELU MLP backbone（depth=2, width=64），把 $[u_i; h_i]$ 投到共享特征 $z$，四个 head ($W_g, W_h, W_a, W_b$) 共享同一 backbone；
- 整个细胞等价于"在两个候选隐藏态 $g$ 与 $h_{\text{cand}}$ 之间做时间门插值"，可读作一个**关于 t 的结构化连续函数**。

### 3.3 形变栈（Continuous-Depth Deformation Stack）

对每个规范化高斯 $i$：

$$
\begin{aligned}
u_i &= [\gamma(\mathrm{sg}(x_i)); \gamma(t)] \\
h_i^{(0)} &= 0 \\
h_i^{(\ell+1)} &= \mathrm{CfC}_\ell(u_i, h_i^{(\ell)}, t), \quad \ell = 0,\ldots,D-1 \\
(\Delta x_i, \Delta r_i, \Delta s_i) &= W_{\text{out}}\, h_i^{(D)}
\end{aligned}
\tag{3}
$$

- 输入 $u_i$ 在每个 cell 都被拼到 $[u_i; h_i^{(\ell)}]$ 后送入 $\phi$，**等价于隐式 per-layer skip**，弥补显式 skip 缺失；
- 在 $\ell=D/2$ 处额外做 NeRF-style 残差：把一个学到的线性投影 of $u_i$ 加回到隐藏态；
- 与经典 CfC sequence model 不同：hidden state 每帧重置为 0（D-3DGS 训练时是随机采样帧、非时序），densification 也会增删高斯，所以"per-Gaussian 跨帧递归"没有稳定对应——论文**故意不用**长程记忆，只榨取"结构化时间门 + 隐式 skip"两项收益。

### 3.4 与 ODE/SDE 的成本-能力谱定位

| 方法 | 求解器 | 训练成本 | 强项 | 弱项 |
|---|---|---|---|---|
| **D-3DGS MLP** | 无 | 1× | 像素保真度（baseline） | 时间连续性只能"靠运气" |
| **Neural ODE / ODE-GS** | 必现 | >>1× | 时窗外推（extrapolation） | 训练慢、前向要 solver |
| **SDE 变体** | 必现 | >>>1× | 重噪声鲁棒性 | 求解器开销最高 |
| **本文 CfC** | **无**（闭式解） | ≈1×（与 MLP 相当） | 架构层连续时间语义 + jitter 鲁棒性 | 时窗外推能力弱于 ODE/SDE |

论文定位 CfC 是谱的**廉价端**："为标准插值场景（in-window interpolation）以 MLP 的成本买到一个架构级的时间平滑先验"。当外推或强噪声成为主要目标时，ODE-GS / SDE 更合适。

### 3.5 上下文关系

- **vs Neural ODE (Chen 2018)**：CfC 保留 ODE 的连续时间语义，但用解析闭式解替代数值积分；训练 / 推理与 MLP 同阶。
- **vs D-3DGS (Yang 2024)**：完全保留其所有流水线（canonical Gaussian、rasterizer、L1+SSIM、密度控制、AST schedule、40k iter Adam），仅替换 `DeformNetwork` 类内部的 MLP 实现。
- **vs LTC (Hasani 2020)**：本文 CfC cell 即 LTC ODE 的闭式解，时间门 $\sigma_\tau$ 在 loss landscape 中显式依赖 $t$。
- **vs ODE-GS (Wang 2025)**：ODE-GS 是 Transformer encoder + latent neural ODE + 冻结的 D-3DGS interpolator，每前向要调用 solver，主要解决"时窗外推"。本文聚焦"插值内的连续时间先验"，求解器为零。
- **vs SDE 变体 (Li 2020)**：SDE 给最重噪声下的鲁棒性；本文在普通单目监督噪声下用闭式门即可获得抖动鲁棒性。

## 4. 核心公式提取

### 4.1 规范化高斯 + 形变场（D-3DGS 原文，本文保留）

$$
G_i = (x_i, r_i, s_i, \alpha_i, c_i), \quad (\Delta x_i, \Delta r_i, \Delta s_i) = F_\theta(\gamma(\mathrm{sg}(x_i)), \gamma(t)) \tag{1}
$$

### 4.2 CfC 单元闭式更新（核心动力学）

$$
h' = g \odot (1 - \sigma(W_a z \cdot t + W_b z)) + h_{\text{cand}} \odot \sigma(W_a z \cdot t + W_b z) \tag{2}
$$

其中：
- $z = \phi([u_i; h])$，$\phi$ 是 GELU MLP backbone；
- $g = \tanh(W_g z)$，$h_{\text{cand}} = \tanh(W_h z)$ 为两个候选隐藏态；
- $\sigma_\tau = \sigma(W_a z \cdot t + W_b z)$ 为 sigmoid 时间门；
- 与 Hasani 等 (Nature Machine Intelligence, 2022) 的统一闭式 CfC 公式结构一致，此处把 $\sigma_\tau$ 内嵌到 $h'$ 的逐元素插值。

### 4.3 连续深度形变栈

$$
h_i^{(0)} = 0,\quad h_i^{(\ell+1)} = \mathrm{CfC}_\ell(u_i, h_i^{(\ell)}, t),\quad (\Delta x_i, \Delta r_i, \Delta s_i) = W_{\text{out}}\, h_i^{(D)} \tag{3}
$$

外加在 $\ell=D/2$ 处的 NeRF-style 残差：$h_i^{(D/2)} \leftarrow h_i^{(D/2)} + W_{\text{skip}}\, u_i$。

### 4.4 训练目标（完全沿用 D-3DGS）

$$
\mathcal{L} = \mathcal{L}_1 + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} \tag{4}
$$

像素保真度没有任何改动；架构级的连续时间先验完全来自 $\sigma_\tau$。

## 5. 关键成果与贡献

### 5.1 D-NeRF（8 个合成场景，800×800）

| 场景 | D-3DGS (MLP) PSNR | Ours (CfC) PSNR | Δ (dB) |
|---|---:|---:|---:|
| Hell Warrior | 41.13 | **41.95** | **+0.82** |
| Mutant | **42.07** | 41.63 | −0.44 |
| Hook | 36.77 | **38.26** | **+1.49** |
| Bouncing Balls | **41.69** | 41.10 | −0.59 |
| Lego | **24.94** | 24.88 | −0.06 |
| T-Rex | **37.93** | 37.79 | −0.14 |
| Stand Up | **44.02** | 42.86 | −1.16 |
| Jumping Jacks | 37.49 | **37.52** | +0.03 |
| **均值** | **38.26** | **38.25** | **≈0** |

- 6/8 场景上 Ours-CfC 匹配或超过 D-3DGS MLP（±0.5 dB 内）；
- 最大正向收益集中在**高频铰接运动**场景（Hook +1.49 dB, Hell Warrior +0.82 dB），与 $\sigma_\tau$ 作为隐式平滑先验的预期一致；
- 总体均值 38.25 vs 38.26 dB，**在聚合上"打平"，但分布更靠高频运动侧**。

### 5.2 NeRF-DS（7 个真实场景）

| 场景 | D-3DGS (MLP) PSNR | Ours (CfC) PSNR | Δ (dB) |
|---|---:|---:|---:|
| Sieve | 25.30 | **25.84** | +0.54 |
| Plate | **20.42** | 20.41 | −0.01 |
| Bell | **25.02** | 25.08 | +0.06 |
| Press | 25.37 | **25.46** | +0.09 |
| Cup | **24.67** | 24.61 | −0.06 |
| **As** | 23.37 | **26.11** | **+2.74** |
| Basin | **19.61** | 19.52 | −0.09 |
| **均值 PSNR** | **23.39** | **23.86** | **+0.47** |
| **均值 SSIM** | 0.8403 | **0.8491** | +0.0088 |
| **均值 LPIPS** | 0.2011 | **0.1891** | −0.012 |

- 真实场景下 CfC 在 **全部三个均值指标上**超过 MLP（PSNR +0.47、SSIM +0.0088、LPIPS −0.012）；
- 单一场景 As（最 specular、镜面运动最重）净涨 **+2.74 dB**、LPIPS −41%；
- Ours-CfC 是**唯一一个在均值 PSNR 上超过 specular-aware NeRF-DS baseline**（+0.26 dB）的通用方法；
- 其余 6 场景在 ±0.1 dB 范围内与 MLP 相当——说明 **CfC 不损害 noisy real-world 监督下的常规场景**。

### 5.3 计算预算（ptflops 测量，Hell Warrior 中位高斯数）

| 配置 | Params (M) | MACs (G) |
|---|---:|---:|
| D-3DGS MLP, D=8, W=256 | 0.5223 | 9.354 |
| Ours CfC, D=8, W=256 | 0.7829 | 13.999 |
| **Ours CfC, D=6, W=128（默认）** | **0.3345** | **5.998** |

- 与 MLP 同维度（D=8, W=256）下，CfC 多 ~50% params / ~50% MACs（4 个内部 head $g/h_{\text{cand}}/W_a/W_b$ 引入）；
- **但默认配置（D=6, W=128）在两个轴上都比 MLP 小约 36%**——架构级连续时间收益 **无需以更大计算为代价**；
- 与 ODE-GS / SDE 相比，**没有任何 ODE/SDE solver** 占用，训练时间不会向 latent-ODE 一档漂移。

### 5.4 架构消融（D-NeRF Hell Warrior）

| 维度 | 变体 | PSNR |
|---|---|---:|
| Backbone | MLP (D-3DGS 默认) | 41.54 |
| Backbone | **CfC cell (ours)** | **42.03** |
| 深度 D | 6 (ours) | **42.03** |
| 深度 D | 8 | 41.82 |
| 深度 D | 10 | 41.86 |
| 激活 | GELU (ours) | **42.03** |
| 激活 | SiLU | 41.53 |
| 激活 | ReLU | 41.47 |
| 激活 | LeCun | 40.88 |
| 激活 | Tanh | 40.74 |

- CfC cell 相对 MLP baseline +0.49 dB；
- 深度饱和在 D=6，D=8 / D=10 略降（41.82 / 41.86）；
- 激活函数排序 **GELU > SiLU > ReLU > LeCun > Tanh**——**无界平滑激活**与 cell 内 tanh 配合优于有界激活。

### 5.5 三大核心贡献

1. **架构级连续时间形变场**：把 D-3DGS 的 MLP $F_\theta$ 替换为 D 个 CfC cell 的栈，**不改动其他任何流水线组件**——典型的 drop-in replacement。
2. **D-NeRF + NeRF-DS 全量基线**：8+7 场景、PSNR/SSIM/LPIPS + Params/MACs，对 MLP baseline / D-NeRF / TiNeuVox / NeRF-DS 全方位对比；As 场景 +2.74 dB 是单一场景最大亮点。
3. **CfC / ODE / SDE 谱系定位**：明确 CfC 是"廉价端闭式近似"——为标准插值场景以 MLP 成本买一个结构化时间平滑先验；外推与重噪声下 ODE/SDE 更合适。

## 6. 局限性与未来展望

### 6.1 论文作者自陈的局限

1. **没有跨帧递归（no recurrence across video frames）**：CfC 的长程时间记忆能力（在不规则采样时序与控制 benchmark 上是杀手锏）在 depth-as-time 形式下被主动放弃。连续性保证只来自时间门 $\sigma_\tau$ 的非线性。
2. **评估集偏短 / 偏受控**：主要在合成 D-NeRF 与短真实 NeRF-DS 上评估；长时无控制视频（long uncontrolled videos）留作未来工作。
3. **未做物理一致性辅助损失**：CfC 层天然暴露 $\partial F_\theta / \partial t$，本可引入 inertia (acceleration) penalty、As-Rigid-As-Possible (ARAP) distance penalty 等物理正则，但作者刻意保持"纯架构"贡献，留待后续线性叠加在 photometric loss 上。

### 6.2 可进一步延伸的方向（结合本文与本仓库方向）

- **辅助损失层叠加**：基于 $\partial F_\theta / \partial t$ 加速度正则 / ARAP / 连续性 TV penalty，应能与现有 L1+SSIM 损失线性叠加，进一步稳定高频运动区域。
- **同计算预算的 ODE/SDE 对照**：在 matched FLOPs envelope 下做 CfC / ODE-GS / SDE 三方 ablation，定量刻画"插值内闭式 vs. 外推求解器"的真实 trade-off。
- **跨帧递归 CfC**：在具有稳定 per-Gaussian 对应的数据集（rigged synthetic models、SMPL-controlled humans）上打开跨帧 recurrence，把"depth-as-time"升级为"depth+recurrence-as-time"，与控制理论方向的 LNN-on-robotics 直接对接。
- **与 jetson 部署栈结合**：默认 CfC 配置（D=6, W=128, 0.33M params / 6.0G MACs）已经比 D-3DGS MLP 小约 36%，且无 ODE solver，可在 Jetson Orin Nano 上走与现有 `scripts/jetson_lnn_benchmark.py` 同样的 4-model Pareto sweep 流水线做实时性验证。
- **扰动鲁棒性实验**：CfC 的 jitter 鲁棒性在 LNN 论文（Kumar 2023 / Karn 2024）中已有间接证据；可在 D-3DGS 监督上加人工 pose noise / 时间戳抖动，定量确认本文"structural byproduct"在 4D 重建中的实际收益。

### 6.3 与本仓库其他报告的桥接

- [[Liquid_Networks_MDH_Imitation_Learning_研读报告]]（Push-T / RoboMimic Can）：已在 imitation learning 上证明 liquid+MDN 优于 diffusion policy；本文为 4D 重建方向补上"无需 ODE 求解器的连续时间"工具。
- [[MeloTune_CfC_Proactive_Music_Curation_研读报告]]：CfC 在端侧毫秒级推理已得到生产验证；本文 CfC 默认配置（0.33M / 6.0G MACs）进一步证明 CfC **本身可作为 4D 重建的轻量 backbone**，无需任何 ODE solver。
- [[Liquid_Crystal_Antennas_LNN_6G_Beamforming_研读报告]]：同样是 ODE 闭式 LNN 在另一连续信号域（信道）的"廉价端"应用；可与本文共同支持"LNN = ODE 闭式 ≈ MLP 成本 + 连续时间先验"的一般性结论。
- [[Nonasymptotic_BC_Error_Dynamics_2604.14484_研读报告]]：本文 CfC 在 As (+2.74 dB) 上的 specular-motion 提升可视为"高运动 + 高噪声监督下结构化平滑先验"的具体落地，与 BC 误差动力学中的 proxy matrix 视角互补。

## 7. 引用与索引

- **arXiv 原文**：[https://arxiv.org/abs/2606.07670v1](https://arxiv.org/abs/2606.07670v1)
- **本地 PDF**：[papers/2606.07670v1_Liquid_NN_3DGS_Deformation_Field.pdf](../../papers/2606.07670v1_Liquid_NN_3DGS_Deformation_Field.pdf)
- **每日 digest**：[docs/daily/2026-06-10_LNN_research_digest.md](../../daily/2026-06-10_LNN_research_digest.md)
- **D-3DGS 基线**：Yang et al., CVPR 2024
- **CfC 闭式解**：Hasani et al., Nature Machine Intelligence 2022
- **LTC ODE**：Hasani et al., arXiv 2006.04439 (2020)
- **Neural ODE**：Chen et al., NeurIPS 2018
- **ODE-GS 外推基线**：Wang et al., arXiv 2506.05480 (2025)