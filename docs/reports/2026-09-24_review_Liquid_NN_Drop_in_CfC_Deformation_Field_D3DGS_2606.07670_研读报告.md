# Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting — 研读报告

> ⚠️ **2026-09-24 回顾版**: 本文档是 LNN 每日研读流水线 (2026-09-24) 在 arXiv 抓取第 2 天失败后,用 web_search 兜底拉取 arXiv:2606.07670 + arXiv:2608.28702 时写下的回顾型研读。**该论文已有正式深度研读**: [[Liquid_NN_3DGS_Deformation_Field_2606.07670_研读报告]] (2026-06-10, v1)。本文侧重:
> 1. 与姊妹篇 [arXiv:2608.28702](https://arxiv.org/abs/2608.28702) 的对照(同作者组, 同 D-3DGS pipeline)
> 2. 2026-09-24 时点对本项目 (LNN 边缘部署) 的潜在可借鉴思路 (depth-as-time + 极低 param/MAC)
> 3. 与本仓既有 LNN 训练栈 (r287-r305 CfC 变体) 的关系定位
>
> 完整原文阅读请优先看既有 [[Liquid_NN_3DGS_Deformation_Field_2606.07670_研读报告]]。本文档不重复原文细节。

## 元数据

- **论文标题**: Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting
- **作者**: Mingzhao Li, Arghya Pal, Guan Yuan Tan（共同第一作者）
- **机构**: School of Information Technology, Monash University, Selangor, Malaysia
- **发表会议/年份**: 2026 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC 2026)
- **arXiv 编号**: 2606.07670v1
- **arXiv 链接**: https://arxiv.org/abs/2606.07670
- **提交日期**: 2026-06-04（v1）
- **类别**: cs.CV（计算机视觉 — 4D 重建 / 动态 3D Gaussian Splatting）
- **关键词**: liquid neural networks, closed-form continuous-time cells, 4D reconstruction, dynamic 3D Gaussian splatting, continuous-time deformation, depth-as-time

## 核心问题

Deformable 3D Gaussian Splatting (D-3DGS) 通过一个 **位置编码 MLP 形变场** $F_\theta(\gamma(x), \gamma(t)) \rightarrow (\Delta x, \Delta r, \Delta s)$ 把 canonical 3D Gaussians 变形到任意帧时间 $t \in [0,1]$，实现单目动态 3D 场景重建。但这个 MLP 形变场有三个结构性问题：

1. **架构上的"离散感知"**：MLP 在每个采样帧 $t$ 处独立前向；相邻时间步之间没有架构上的耦合，**时间平滑性完全寄希望于优化器的隐式正则**。虽然 MLP 输入端接的是连续的 $\gamma(t)$，但其本质仍是一个逐帧偏移预测器。
2. **可解释的连续性 / 鲁棒性缺口**：没有显式的连续性约束，对单目监督中的姿态漂移、光照变化、运动模糊等噪声缺乏先验。
3. **已有连续时间方法的代价**：Neural ODE / Latent-ODE-GS 引入数值 ODE 求解器，每个前向都要积分；Neural SDE 更进一步，要解 Itô 积分。这些都把训练成本从"一个 MLP 前向"抬升到"数值求解器"，与 D-3DGS 实时渲染的初衷冲突。

论文核心问题：**能否给 D-3DGS 的形变场装上一个架构级的"连续时间"先验，但保持推理成本仍是一个 feed-forward MLP？**

## 方法论与核心思路

**核心思路**: 用 **Closed-form Continuous-time (CfC) cells** — Liquid Time-constant Network (LTC) ODE 的解析近似解 — 替换 D-3DGS 的 8 层 MLP 形变场，保留 D-3DGS 流水线其他所有部分不变。CfC 没有任何 ODE/SDE 求解器，但通过其 **sigmoidal 时间门** 在架构上把"对 $t$ 的平滑响应"嵌入损失函数景观（loss landscape）里。

**关键设计点**：

### 1. 单个 CfC 单元 (Eq. 2)

把输入 $u$、隐状态 $h$ 和流逝时间信号 $\tau = t$ 映射到新隐状态：

$$
\begin{aligned}
z &= \phi([u; h]) \\
g &= \tanh(W_g z), \quad h_\text{cand} = \tanh(W_h z) \\
\sigma_\tau &= \sigma(W_a z \cdot \tau + W_b z) \\
h' &= g \odot (1 - \sigma_\tau) + h_\text{cand} \odot \sigma_\tau
\end{aligned}
$$

其中 $\sigma$ 是 sigmoid 函数。$\sigma_\tau$ 是**唯一**让 $\tau$ 进入单元的路径——对 $t$ 的扰动会通过这个门产生"结构化、可学习"的响应。

### 2. 连续深度形变栈 (Eq. 3) — "Depth-as-Time"

不像经典 CfC 序列模型那样做时间步递推，本文用 $D$ 个 CfC 单元堆栈作为形变场：

$$
\begin{aligned}
u_i &= [\gamma(\text{sg}(x_i)); \gamma(t)], \quad h_i^{(0)} = 0 \\
h_i^{(\ell+1)} &= \text{CfC}_\ell\!\left(u_i, h_i^{(\ell)}, t\right), \quad \ell = 0, \dots, D-1 \\
(\Delta x_i, \Delta r_i, \Delta s_i) &= W_\text{out} \, h_i^{(D)}
\end{aligned}
$$

- 隐藏状态每个 forward 重置为 0（因为 D-3DGS 在训练时随机抽取独立帧做查询，没有稳定的时间顺序）
- 在 $\lceil D/2 \rceil$ 处做 NeRF-style 残差，把 $u_i$ 重新注入
- 称为 **"depth-as-time"**：网络的深度（堆栈层数）扮演经典 CfC 序列模型中时间递推的角色

**为何 depth-as-time 而不是 recurrence**：D-3DGS 训练时对每个 canonical Gaussian 在不同帧独立查询 + densification 不断增减 Gaussian → per-Gaussian 跨帧没有稳定对应关系，所以不适合用 recurrent hidden state。但 CfC 单元的**时间门**仍可作为隐式的 $t$-平滑先验。

### 3. 为何选 LNN（CfC）？三性质动机

| 性质 | MLP 形变场 | Neural ODE / SDE | **CfC（本工作）** |
|---|---|---|---|
| 需要 ODE/SDE 求解器 | 否 | 是 | **否** |
| 推理成本 | feed-forward | 求解器每前向 | **feed-forward, 同 MLP** |
| 对 $t$ 的架构级连续性 | 仅来自 $\gamma(t)$ | 来自 ODE/SDE 结构 | **来自闭式时间门** |
| 噪声/抖动鲁棒性 | 无 | 来自 SDE 扩散项 | **作为 LTC-SDE 确定性极限的"继承鲁棒性"** |
| 训练成本基线 | 1× | 高于 MLP 1-5 数量级 | **与 MLP 几乎相同** |

### 4. 实现细节

- 基线：D-3DGS 官方 PyTorch 代码库
- 默认超参：$D=6$ CfC 单元，隐宽 $W=128$，backbone 宽 64，深 2，GELU 激活
- 位置编码：$x$ 用 $L=10$ 频带；$t$ 在 Blender 合成场景用 $L=6$，在真实场景用 $L=10$
- 硬件：单 NVIDIA Tesla P100-PCIE-16GB
- 训练：保留 D-3DGS 原 40k 次 Adam 迭代
- 集成点：`utils/time_utils.py` 里的 `DeformNetwork` 类

## 核心公式

**D-3DGS 形变场**（被替换的对象）:
$$
(\Delta x_i, \Delta r_i, \Delta s_i) = F_\theta\bigl[\gamma(\text{sg}(x_i)), \gamma(t)\bigr] \quad \text{(Eq. 1)}
$$

**单个 CfC 单元**:
$$
h' = \underbrace{h_\text{cand} \odot \sigma_\tau}_{\text{time-gated "new"}} + \underbrace{g \odot (1-\sigma_\tau)}_{\text{time-gated "retained"}}, \quad \sigma_\tau = \sigma(W_a z \cdot \tau + W_b z) \quad \text{(Eq. 2)}
$$

**连续深度 CfC 栈（本文提出）**:
$$
h_i^{(\ell+1)} = \text{CfC}_\ell(u_i, h_i^{(\ell)}, t), \quad (\Delta x_i, \Delta r_i, \Delta s_i) = W_\text{out} h_i^{(D)} \quad \text{(Eq. 3)}
$$

**LTC ODE（背景，论文不重新推导，仅作引用）**:
$$
\dot{x} = -\left[\frac{1}{\tau} + f(x, u, t)\right] x + f(x, u, t) \cdot A
$$

## 关键成果与贡献

### 实验设置
- **数据集**: D-NeRF（8 个 Blender 合成单目场景）+ NeRF-DS（7 个真实镜面动态场景）
- **基线**: D-3DGS MLP（用相同超参 / 种子重训）+ D-NeRF / TiNeuVox（引自 [3]）
- **指标**: PSNR / SSIM / LPIPS（800×800 测试图）+ 形变场参数（M）+ MACs（G），MACs 用 ptflops 测

### D-NeRF（8 场景合成数据）

- 平均 PSNR：CfC 38.25 vs D-3DGS MLP 38.26 dB — **aggregate tie**
- 6/8 场景 PSNR 在 0.5 dB 内匹配或超过 MLP
- 高频关节运动场景增益最显著：
  - **Hook: +1.49 dB**
  - Hell Warrior: +0.82 dB
- 解读：饱和基线下绝对增益小；价值是"在架构层拿到连续时间语义，不以损失像素精度为代价"

### NeRF-DS（7 场景真实镜面数据）

- 平均：CfC 在 **所有** mean 指标上击败 D-3DGS MLP
  - PSNR 23.86 vs 23.39 dB
  - SSIM 0.8491 vs 0.8403
  - LPIPS 0.1891 vs 0.2011
- 最大增益在 As 场景：+2.74 dB PSNR，−41% LPIPS（最高镜面运动负载的场景）
- 是唯一在 mean PSNR 上超过 specular-aware NeRF-DS 基线（+0.26 dB）的通用方法

### Compute Budget（形变场开销）

| 方法 | Params (M) | MACs (G) |
|---|---:|---:|
| D-3DGS MLP (D=8, W=256) | 0.5223 | 9.354 |
| Ours (CfC, D=8, W=256) | 0.7829 | 13.999 |
| **Ours (CfC, D=6, W=128, 默认)** | **0.3345** | **5.998** |

默认配置下，CfC 形变场比 MLP **小 36%**（参数和 MACs 两个维度）。这是关键反直觉点：CfC 的连续时间先验不是"加负担"，而是"更便宜"——前提是按任务选深度和宽度，不要硬套 MLP 的超参。

### 架构消融（D-NeRF Hell Warrior, Table IV）

| 变量 | 设置 | PSNR |
|---|---|---:|
| Backbone | MLP（D-3DGS 默认） | 41.54 |
| | **CfC cell（本文）** | **42.03** |
| Depth D | 6（本文默认） | 42.03 |
| | 8 | 41.82 |
| | 10 | 41.86 |
| Activation | ReLU | 41.47 |
| | **GELU（本文默认）** | **42.03** |
| | SiLU | 41.53 |
| | LeCun | 40.88 |
| | Tanh | 40.74 |

- CfC 比 MLP backbone +0.49 dB
- $D=6$ 是甜点；$D=8/10$ 反而略差
- 激活函数 GELU > SiLU > ReLU > LeCun > Tanh：unbounded smooth backbone 与单元内 tanh 配合最好

### 定性结果（Fig. 2）

- D-NeRF Hell Warrior：MLP 把挥舞的肢体涂抹成残影，CfC 保持轮廓
- NeRF-DS As：MLP 对移动半透明物体产生 ghost，CfC 大幅抑制
- NeRF-DS Sieve：CfC 保留更细的纹理

## 局限性与未来展望

### 作者明确的局限

1. **没有跨帧 recurrence**：Depth-as-time 形式放弃经典 CfC 序列模型的长时记忆能力。LTC/CfC 真正强的地方（不规则采样时间序列、控制系统）在这里不激活。连续性保证只来自时间门非线性。
2. **评估集偏短、偏干净**：主要是合成 + 短真实场景；长非受控视频未评估。
3. **"MLP 已经连续"的反驳**：作者明确回应 — 评估层面 MLP 是连续的，但训练信号是采样密集的独立时间步；优化器没有任何机制把 $F_\theta(t)$ 和 $F_\theta(t+\delta t)$ 耦合起来。CfC 单元是 ODE 在 $t$ 上的闭式解，时间门 $\sigma_\tau$ 直接在 loss landscape 里 — 对 $t$ 的扰动会产生**结构化、可学习**的响应（结构性原因，不是统计原因）。

### 作者列出的未来工作

1. **LNN 内部辅助损失**：CfC 暴露光滑的 $\partial F_\theta / \partial t$，可加 inertia（加速度）惩罚 / As-Rigid-As-Possible 距离惩罚 / 其他物理知情正则；保持架构不变、与现有光度损失线性叠加。
2. **匹配 compute 下的 ODE/SDE 对比实验**：CfC vs latent-ODE-GS vs SDE variant 在固定 compute 包络下 head-to-head，厘清三者取舍。
3. **Recurrent-over-frames CfC**：在有稳定 per-Gaussian 对应的数据集（如 rigged synthetic models）上，递归式 CfC 可能解锁 depth-as-time 主动放弃的长时记忆。

### 个人延伸观察（非论文内容）

- 这篇工作**没有改 D-3DGS 训练 pipeline**，所以复现成本极低：替换 `DeformNetwork` 一处即可。这与作者强调的"near-zero-friction architectural design"一致。
- 与姊妹篇 [arXiv:2608.28702](https://arxiv.org/abs/2608.28702)（同组）做对照：那篇把这个架构升级为 SDE 变体，在门预激活上加高斯扰动作为 SDE 的 Euler-Maruyama 一步。两篇一起构成 CfC 在 4D 重建场景下的"deterministic + stochastic"完整套件。
- 工业落地角度：CfC stack 0.33M params / 6 G MACs 完全跑得动 Jetson Orin Nano — 边缘部署 D-3DGS + CfC 形变场是可行的。这是**本项目**（LNN 在边缘的部署）潜在的可借鉴思路。

---

**研读日期**: 2026-09-24
**研读者**: paper-analyzer skill（LNN 每日研读流水线）
**PDF 路径**: `papers/2026-09-24/pdf_2606.07670.pdf` (2.0 MB)