---
title: Stochastic Liquid Deformation Fields — An SDE Generalisation of Closed-Form Continuous-Time Cells for Dynamic 3D Gaussian Splatting
date: 2026-09-02
tags: [LNN, CfC, LTC, SDE, 3D-Gaussian-Splatting, D-3DGS, 4D-Reconstruction, Euler-Maruyama, APSIPA]
---

# 研读报告：Stochastic Liquid Deformation Fields — SDE 视角下的 CfC 形变场

> 本文是 [[LNN_深度研读报告]] 中 "连续时间形变场 / 动态 3D Gaussian Splatting" 方向的最新沉淀，紧随 [[Liquid_Neural_Networks_3DGS_Deformation_Field_2606.07670_研读报告]]（2606.07670，2026-06-10, v1）。同一作者组（Mingzhao Li / Arghya Pal）在 4 周内把 "deterministic CfC" 推到 "stochastic CfC (SDE)"，并把同一框架在 D-NeRF / NeRF-DS 上做了一次诚实的正负结果对照。

## 1. 元数据
- **论文标题**：Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-Form Continuous-Time Cells for Dynamic 3D Gaussian Splatting
- **作者**：Mingzhao Li, Arghya Pal（共同一作，Monash University Malaysia）
- **单位**：School of Information Technology, Monash University, Selangor, Malaysia
- **发表时间**：2026 年 8 月 27 日（v1, arXiv:2608.28702v1）
- **发表场合**：2026 APSIPA ASC（accepted）
- **学科**：cs.CV
- **关键词**：4D reconstruction, dynamic 3D Gaussian splatting, liquid neural networks, stochastic differential equations, closed-form continuous-time cells
- **代码**：基于公开 D-3DGS PyTorch pipeline（修改位于 cell forward，单行添加 `λε` 到 gate pre-activation；论文未提供独立 repo）
- **本地 PDF**：[papers/daily/2608.28702v1.pdf](../../papers/daily/2608.28702v1.pdf)

## 2. 核心问题

承接 2606.07670 的 "drop-in CfC 形变场" 工作，本文回答一个理论 + 实证问题：

> **CfC 是 LTC ODE 的"闭式（deterministic）极限"——但 LTC 本身是一个噪声驱动的随机过程。闭式解丢弃了扩散项，而这种随机性恰好被普遍认为是 liquid network 抗噪声 / 抗不规则采样的根源。那么：能否在不付出 SDE 求解器代价的前提下，把扩散项"架构性"地放回 CfC？**

具体痛点：

1. **闭式极限丢弃扩散**：CfC 保留了 drift，丢弃了 diffusion；Neural SDE 把两者都保留，但每次 forward 都需要一个随机求解器，对 4D 重建不可接受。
2. **训练鲁棒性 vs 推理成本的矛盾**：liquid family 的鲁棒性归功于随机性；但推理必须是 feed-forward，不能增加任何 ODE/SDE solver 开销。
3. **D-NeRF / NeRF-DS 这类密集监督的插值基准是否真能体现噪声项价值**：本文同时给出 "为什么这上面不一定提升" 的诚实负结果。

## 3. 方法论与核心思路

### 3.1 上下文：2606.07670 的 liquid field

- **场景**：D-3DGS 把 N 个 canonical Gaussians $\{G_i = (x_i, r_i, s_i, \alpha_i, c_i)\}$ 通过一个位置编码 MLP 形变到每帧 $(\Delta x_i, \Delta r_i, \Delta s_i)$。
- **2606.07670 替换**：把这个 MLP $F_\theta$ 替换为 D 个 CfC cells 串联，每帧传入 elapsed-time signal $\tau$。
- **结果**：连续时间 t 的语义进入 loss landscape，但保持 feed-forward 推理，无需 ODE 求解器（区别于 Neural-ODE-based ODE-GS）。

### 3.2 本文核心思想（SDE 视角）

**关键观察**：在 CfC cell 里，elapsed-time signal $\tau$ 进入 cell 的唯一通道是 sigmoidal time gate 的 affine pre-activation：
$$
s = W_a z \tau + W_b z
$$

因此，**只需把 $s$ 视为一个 Itô 过程，加一个 Euler–Maruyama 增量 $\lambda \varepsilon$（$\varepsilon \sim \mathcal{N}(0, I)$），就得到一个 one-step SDE 版本**，扩散项被放回但没有求解器、没有新参数、推理与 deterministic CfC 完全一致。

### 3.3 三条关键性质

1. **Exact reduction**：$\lambda = 0$ 时退化为标准 CfC；与 CfC 的对比是 exactly size-matched 的。
2. **Solver-free, deterministic inference**：noise 仅在训练时采样一次/forward；test-time 关闭，推理与 CfC 完全相同（与 Neural SDE 形成鲜明对比）。
3. **Time-targeting regulariser**：在期望意义下 $\mathbb{E}_\varepsilon[\sigma_\tau]$ 变得更平滑，相当于"针对 $\tau$ 的可学习光滑先验"——这正是 liquid 鲁棒性的来源。

### 3.4 实现细节

- 架构：D=6 cells, hidden width W=128, backbone depth 2, GELU。
- 训练：D-NeRF 40k iter / NeRF-DS 20k iter，单 GPU。
- 噪声标定：$\lambda = 0.05$ 为推荐值；每个 cell 每个 forward pass 独立采样一次。
- 添加位置：gate pre-activation（$W_a z \tau + W_b z$ 之后、sigmoid 之前）。

## 4. 核心公式

### 4.1 Liquid Field Cell（确定性 baseline）

$$
\begin{aligned}
z &= \phi([u; h]) \\
g &= \tanh(W_g z), \quad h_{\text{cand}} = \tanh(W_h z) \\
\sigma_\tau &= \sigma\!\left(W_a z \tau + W_b z\right) \\
h' &= g \odot (1 - \sigma_\tau) + h_{\text{cand}} \odot \sigma_\tau
\end{aligned}
\tag{1}
$$

> Equation (1) 是 LTC ODE 的闭式解；其中 $\sigma_\tau$ 是 cell 的 time gate。

### 4.2 SDE 视角下的 gate pre-activation（**本文核心**）

将 $s = W_a z \tau + W_b z$ 建模为 one-step Itô 过程：

$$
\mathrm{d}s = \underbrace{(W_a z \tau + W_b z)}_{\text{drift}} + \lambda \,\mathrm{d}W_{\text{diffusion}}
\tag{2}
$$

### 4.3 随机 time gate（**Euler–Maruyama 离散**）

$$
\sigma_\tau = \sigma\!\left(W_a z \tau + W_b z + \lambda \varepsilon\right), \quad \varepsilon \sim \mathcal{N}(0, I)
\tag{3}
$$

> 其中常数 $\lambda$ 已吸收 $\sqrt{\Delta t}$。$\lambda = 0$ 时 (3) ≡ (1)（exact reduction）。其他公式（z、g、$h_{\text{cand}}$、$h'$）均保持不变。

### 4.4 平滑性论断

$$
\mathbb{E}_\varepsilon[\sigma_\tau(W_a z \tau + W_b z + \lambda \varepsilon)] = \sigma \ast \mathcal{N}(W_a z \tau + W_b z, \lambda^2)
$$

即期望上的 gate 是原 sigmoid 与 Gaussian kernel 的卷积——$\lambda$ 控制光滑先验的强度。

## 5. 关键成果与贡献

### 5.1 D-NeRF（8 个合成场景，800×800）

| 场景 | D-3DGS MLP PSNR | Ours (SDE-LNN, λ=0.05) PSNR | Δ |
|---|---:|---:|---:|
| Hell Warrior | 41.23 | **41.61** | +0.38 |
| Mutant | 42.05 | 42.08 | +0.03 |
| Hook | 36.94 | **38.18** | **+1.24** |
| Bouncing Balls | **41.37** | 40.22 | −1.15 |
| Lego | 24.92 | 24.94 | +0.02 |
| T-Rex | **37.73** | 37.46 | −0.27 |
| Stand Up | 43.85 | **44.22** | +0.37 |
| Jumping Jacks | 37.40 | **37.55** | +0.15 |
| **Mean** | 38.19 | **38.28** | +0.09 |

- 6/8 场景最佳 PSNR；最高频大幅运动场景（Hook、Stand Up、Hell Warrior）上 liquid 优势最显著——这与 "noise 抹平 $\tau$ 的抖动" 的假设一致。
- 整体差距在 run-to-run 噪声（$\sigma \approx 0.05\,\mathrm{dB}$，3 次重训测量）之内，因此**单次对比属弱显著**；但 Sec. IV-E 的 noise-level sweep 给出单调趋势支持。

### 5.2 NeRF-DS（7 个真实场景）

| 场景 | MLP PSNR | SDE-LNN PSNR | 备注 |
|---|---:|---:|---|
| Sieve | 25.28 | **25.43** | +0.15 |
| Plate | **20.50** | 20.43 | |
| Bell | 25.07 | **25.14** | +0.07 |
| Press | 25.35 | **25.49** | +0.14 |
| Cup | **24.81** | 24.08 | −0.73 |
| As | **26.19** | 25.99 | −0.20 |
| Basin | 19.56 | 19.55 | |
| **Mean** | 23.82 | 23.73 | −0.09 |

- 在 pose noise 主导的真实数据上，**deterministic 极限（λ=0）已经最好**，加 noise 反而掉点。
- 难操作场景（Bell、Press）上 SDE-LNN 拿到最佳；cleaner 场景（Cup、As）上输给 MLP。

### 5.3 计算开销对比

| 形变场 | Params (M) | 推理 |
|---|---:|---|
| D-3DGS MLP (D=8, W=256) | 0.52 | feed-forward |
| Liquid CfC / SDE-LNN — NeRF-DS | **0.32** | feed-forward |
| Liquid CfC / SDE-LNN — D-NeRF | 1.55 | feed-forward |

> Liquid/SDE-LNN 与 MLP 大小相同（同 backbone），**SDE-LNN 与 CfC 共享一行**——因为 λ=0 时退化为 CfC，训练时 noise 不增加参数。NeRF-DS 配置下 liquid 比 MLP **小 39%**。

### 5.4 Noise-level Sweep（λ ∈ {0, 0.05, 0.10, 0.20}）

- **D-NeRF**：λ=0 与 λ=0.05 均值持平（38.28 dB）；λ=0.10 起开始掉（38.06），λ=0.20 更低（37.91）→ noise 抹掉真实运动。
- **NeRF-DS**：λ=0 已是最佳（23.83 dB）；任何正 λ 都略差 → 在真实数据上 noise **净负**。
- **结论**："deterministic CfC 是 sweet spot；λ=0.05 只是在 D-NeRF 上 noise 上界的安全档。"

### 5.5 关键贡献清单

1. **架构性 SDE 视角**：在不引入 solver 的前提下，把 CfC 的 closed-form 形变场解读为一个 one-step SDE；为 liquid 鲁棒性提供清晰公式化解释。
2. **Exact size-matched 对照**：SDE-LNN vs CfC 共享全部参数 / FLOPs / 推理路径，使 noise 项的实证贡献可被干净归因。
3. **诚实负结果**：明确指出 "在 D-NeRF / NeRF-DS 这类密集监督的插值基准上，constant noise 不能提升"——把讨论从 "benchmark 拼分" 拉到 "鲁棒性机制何时真正起作用"。
4. **可调、零推理开销**：λ 是 free knob，未来可作为 curriculum / annealed noise / input-dependent schedule 的基础。

## 6. 局限性与未来展望

### 6.1 论文自陈限制

1. **基准 under-test liquid 鲁棒性**：D-NeRF / NeRF-DS 都是密集插值任务，supervision jitter 太小——noise 项的 "鲁棒性价值" 无法充分体现。**应当的真正压力测试**：jittered timestamps / pose noise / few-view training。
2. **Constant gate-level λ 太朴素**：作者明确提出三种自然改进方向（详见 6.2）。
3. **只做 first-order, single-step, gate-level**：没有在 hidden state 上注入扩散，也没有跑高阶 Itô / Stratonovich 对照。
4. **未见定性的真实场景可视化**：只有 Bell / Press 的 +0.07/+0.14 dB 差异，缺乏 side-by-side 图（论文承认）。
5. **没有 inference-time Monte-Carlo**：不确定性量化（贝叶斯视角下的 SDE）尚未给出。

### 6.2 本文提出的未来方向（作者本人）

1. **Anneal λ → 0**：训练中后期关掉 noise，前期起正则化作用。
2. **Input-dependent λ**：让 noise 强度由输入调制——只在 supervision 抖动的位置注入噪声。
3. **Hidden-state diffusion**：把扩散从 gate 移到隐藏状态，更贴近真实的 LTC-SDE，但仍保持 solver-free / deterministic inference。

### 6.3 我的扩展观察

- 与 **Liquid-S4 / Liquid Structural State-Space 系列** 的结合点：liquid cell + SDE gate 与 SSM 的双重视角互补，**未来可探索在 SSM 的 selective scan 阶段同样做 gate-level SDE**。
- 与 **PLAN / Multi-Rate MoE 训练加速**（2606.12240）的潜在协同：annealed λ 可作为 "训练前期高频正则 + 后期精修" 的天然 curriculum，与 multi-rate 时间尺度的 MoE 路由天然兼容。
- 复现成本极低（1 行代码 + 1 个标量 λ），是 living-domain-research 中 "诚实负结果" 的范例：**贡献不是涨点，而是机制清晰化**。建议作为后续 LNN 鲁棒性研究的标准 baseline。

## 7. 与前作 2606.07670 的对照

| 维度 | 2606.07670 (deterministic CfC) | 2608.28702 (SDE-LNN, 本文) |
|---|---|---|
| 形变模型 | D×CfC stack | D×CfC stack + gate noise |
| 训练 | deterministic | 加 Gaussian noise（训练 only） |
| 推理 | feed-forward | feed-forward（不变） |
| 参数 | 与 MLP 同量级 | 与 CfC **完全相同** |
| 实证贡献 | drop-in 替换 MLP，超越 baseline | 加 noise 在 D-NeRF 上略强、NeRF-DS 上略弱 |
| 理论贡献 | 把 LTC 闭式解嵌入 4D | 把闭式解读作 SDE 极限，机制化解释 noise 项 |
| 立场 | "新 SOTA" | "诚实负结果 + free knob" |

## 8. 复现建议

- **代码量**：约 5–10 行（改写一个 cell forward，单 gate）。
- **算力**：单 GPU 即可，D-NeRF 40k iter + NeRF-DS 20k iter。
- **风险**：λ 过大反而掉点；建议从 λ=0.05 起步，再扫 0 / 0.10 / 0.20。
- **优先复制实验**：D-NeRF Hook（+1.24 dB 单点最佳）与 NeRF-DS Bell/Press（+0.07/+0.14 dB）作为快速 sanity check。
- **不要复现**：完整 noise-level sweep 全 8 场景——性价比低，且主要结论已由 4 个 sweep 点覆盖。

---

> **消化判定**：本文是 LNN 在 4D 视觉方向上 **"机制清晰化" > "benchmark 拼分"** 的一个范例。其 SDE 视角可作为后续 liquid 鲁棒性工作的引用基础；λ 作为 free knob 的设计哲学值得在 PLAN / MR-MoE 等其他 liquid 训练范式中复用。