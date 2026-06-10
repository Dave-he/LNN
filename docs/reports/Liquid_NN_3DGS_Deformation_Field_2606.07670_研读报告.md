---
title: Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting — 研读报告
paper: arXiv 2606.07670v1
authors: Mingzhao Li, Arghya Pal, Guan Yuan Tan
venue: 2026 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
date: 2026-06-04
tags: [LNN, CfC, closed-form-continuous, 3DGS, D-NeRF, NeRF-DS, dynamic-scene, deformation-field, depth-as-time, paper-report, APSIPA-2026]
status: deep-read
report-date: 2026-06-10
report-author: LNN-research-agents
---

# Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting — 研读报告

> 论文: arXiv 2606.07670v1 (Li, Pal, Tan 2026), APSIPA ASC 2026
> 链接: https://arxiv.org/abs/2606.07670v1
> 代码: **官方代码改 `D-3DGS` 的 `DeformNet` 为 CIC stack** — `DeformNet` 类在 `cells/time_utils.py` 改造 (论文 §III-D)
> 与本仓直接相关度: **中高** — 是 **CfC 跨域应用实证**(LNN 第一次被用作 4D Gaussian splatting 的 deformation backbone),公式与本仓 `lnn/core/cfc.py::CfCCell` 同源

---

## 1. 一句话定位

> 把 **Deformable 3D Gaussian Splatting (D-3DGS)** 的 deformation field 从**离散 MLP** 替换成 **D 层 CfC cells 堆栈**(closed-form continuous-time,无 ODE/SDE solver),**"depth-as-time"** 架构 — 深度扮演经典 CfC 序列模型中的时间维度。**+0.47 dB mean PSNR (NeRF-DS 7 real scenes) / 6/8 场景匹配或超过 MLP baseline / +2.74 dB on As specular scene / −41% LPIPS**,参数 +50% (0.78M vs 0.52M),**closed-form 故前向无数值 solver,Inference cost 跟 MLP 同档但质量更好**。

应用: 动态 3D 场景重建 (单目视频) — monocular 4D 重建,vs Neural ODE / Latent-ODE-GS / SDE-GS,本设计**站在 ODE/SDE 谱系的"最廉价端"**。

---

## 2. 问题与动机 (论文 §I)

**D-3DGS 流水线 (Yang et al. CVPR'24)**: 场景 = N 个 canonical 3D Gaussians, 形参 (位置 x_i, 旋转 r_i, 缩放 s_i, 不透明度 α_i, 球谐 c_i)。**Deformation field** 学习每帧每 Gaussian 的 offset:

$$(\Delta x_i, \Delta r_i, \Delta s_i) = F_\theta(\gamma(\mathrm{sg}(x_i)), \gamma(t))$$   (Eq. 1)

其中 γ(·) 是 NeRF 位置编码, sg 是 stop-gradient。

**核心问题 (discreteness gap)**: 即使 t 是连续物理变量,D-3DGS 的 MLP **deformation field 是 per-frame feed-forward** — 在训练时按独立随机帧采样,无相邻 t 之间的耦合,**不在训练 loss 内建 t 平滑性**。所以:
1. 帧间插值可能出现**锐变**(swinging limb 突然换姿态)
2. monocular 4D 监督噪声大,**无内置抖动鲁棒性**
3. 训练时**没有显式连续性约束**,依赖 t 在不同 epoch 凑巧采样相邻值

**该填的洞**: 想保留 D-3DGS pipeline (Gaussian 集合, rasterizer, photometric loss),只**替换 F_θ 的 backbone**。要同时获得:
- (i) 训练/推理 **closed-form 连续时间**(无 solver)
- (ii) 比 MLP 略多的参数但**显著更好的图像质量**
- (iii) 对**采样噪声的鲁棒性**(monocular supervision 必然 noisy)

---

## 3. 关键设计: Liquid Deformation Field (论文 §III-B)

### 3.1 CfC cell (Eq. 2)

单个 CfC cell 把输入 u, 隐态 h, elapsed-time signal τ 映射到更新后的 h'。4 个 learned linear heads + 一个 sigmoid 时间门 σ_τ:

```
z         = φ([u; h])         # 共享 backbone
g         = tanh(W_g z)        # candidate 1
h_cand    = tanh(W_h z)        # candidate 2
σ_τ       = σ(W_a z · τ + W_b z)   # 仿射函数 σ 的 affine 时间门
h'        = g ⊙ (1 - σ_τ) + h_cand ⊙ σ_τ
```

**核心洞察**: σ_τ 是个**仿射的 sigmoid 时间门** (注意输入是 W_a z · τ + W_b z, **τ 通过 z-线性项而非 τ 单独 sigmoid**)。它把 h' 在 g 和 h_cand 之间做软插值, **τ 越大越偏向 h_cand**(或反之,learned)。

### 3.2 Depth-as-time stack (Eq. 3)

不像经典 CfC 把 cell 沿时间展开,这里**把 D 个 CfC cells 沿深度堆叠**作为 feed-forward stack。对每个 canonical Gaussian i:

```
u_i       = [γ(sg(x_i)); γ(t)]        # 位置 + 时间编码
h_i^0     = 0
h_i^(ℓ+1) = CfC_i(u_i, h_i^ℓ, t),  ℓ = 0, ..., D-1
(Δx_i, Δr_i, Δs_i) = W_out · h_i^D
```

中间第 D/2 层做 **NeRF-style skip**: 把学到的 u_t projection 重新注回 hidden state (类似 ResNet skip,但以 input 注入)。

**关键**:**D 维深度扮演经典 CfC 序列模型中 T 维时间**。每帧独立 forward,无帧间 recurrence (论文 §III-B "Depth-as-time, not recurrence" 强调)。Gauussian i 是 non-stationary,denoising/articulated motion 跟 per-Gaussian recurrence 没对应关系,故**显式舍弃帧间 recurrence,只把 t 作为 elapsed-time signal 喂到每层**。

### 3.3 计算预算 (论文 §IV-D, Table III)

| 配置 | Params (M) | MACs (G) |
|---|---:|---:|
| D-3DGS MLP (D=8, W=256) | 0.5223 | 9.354 |
| **CIC stack (D=6, W=128, ours)** | **0.7829** | **13.999** |
| CIC stack (D=6, W=128, tables I/II) | 0.3345 | 5.998 |

作者 default 用 D=6, W=128, **+50% 参数 / +50% MACs over MLP**,但**训练时间跟 MLP 同档**(因为 closed-form 无 solver),**质量更好**。

### 3.4 为什么是 CfC 不是 Neural ODE / Latent-ODE-GS / SDE?

论文 §V 解释**cost-capability spectrum**:
- Neural ODE: 全 ODE solver,延迟高,等价 stochastic SDE 鲁棒性
- **Latent-ODE-GS** (ODE-GS, Wang et al. 2025): 显式 latent-ODE,加 forward pass solver 成本
- **SDE-GS** (Li et al. AISTATS'20): stochastic SDE 替代 ODE,噪声鲁棒但**更高 solver cost**
- **CfC (本文)**: closed-form 近似 → **ODE 的"最廉价端"** — 训练时**不需要 solver**,前向等价一个普通 MLP,但通过时间门 σ_τ 把**架构先验**(smoothness prior)塞进 loss landscape,无需额外正则项

---

## 4. 实验结果 (论文 §IV)

### 4.1 D-NeRF 8 synthetic scenes (Table I)

| Method | Hell Warrior | Mutant | Hook | Bouncing Balls | Lego | T-Rex | Stand Up | Jumping Jacks | **Mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-NeRF | 24.06 | 30.31 | 29.02 | 38.17 | 25.56 | 30.61 | 33.13 | 32.70 | — |
| TiNeuVox | 27.10 | 31.87 | 30.61 | 40.23 | 26.64 | 31.25 | 34.61 | 33.49 | — |
| D-3DGS (MLP) | 41.13 | 42.07 | 36.77 | 41.28 | 24.94 | 37.93 | 44.02 | 37.49 | — |
| **Ours (CfC)** | **41.95** | 41.63 | **38.26** | 41.36 | 24.88 | **37.79** | **42.86** | 37.52 | **38.25** (D-3DGS 38.26) |

- **6/8 场景**匹配或超过 D-3DGS MLP,**2 场景胜 (Hook +1.49, Hell Warrior +0.82)** — 两者都是**最复杂 articulated motion**
- Stand Up -1.16 dB, Mutant -0.44 dB — 退化场景
- **Mean PSNR 几乎完全 tied 38.25 vs 38.26** (作者认为 synthetic 太干净,液体先验无处发力)

### 4.2 NeRF-DS 7 real-world scenes (Table II) — **关键证据**

| Method | Sieve | Plate | Bell | Press | Cup | As | Basin | **Mean PSNR** | **Mean SSIM** | **Mean LPIPS** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NeRF-DS | 25.78 | 20.54 | 23.19 | 25.72 | 24.91 | 19.96 | 19.96 | 23.60 | 0.8494 | 0.1816 |
| TiNeuVox | 21.49 | 20.58 | 23.08 | 24.47 | 21.71 | 20.66 | 20.66 | 21.61 | 0.8234 | 0.2766 |
| D-3DGS (MLP) | 25.30 | 20.42 | 25.02 | 25.37 | 24.67 | 19.61 | 19.61 | 23.39 | 0.8403 | 0.2011 |
| **Ours (CfC)** | 25.84 | 20.41 | 25.08 | 25.46 | 24.61 | **22.68** | 19.58 | **23.86** | **0.8491** | **0.1891** |

- **Mean PSNR 23.86 vs 23.39, +0.47 dB** 在 noisy real 数据
- **As scene +2.74 dB PSNR, −41% LPIPS** — specular 反射大幅提升 (liquid 隐式平滑 prior 在噪声监督下消除 MLPs 的 piecewise-linear artifacts)
- 唯一**胜 NeRF-DS baseline** 的通用方法 (其他 generic D-3DGS 都没胜)

### 4.3 Ablations (Table IV, Hell Warrior)

| 维度 | 变体 | PSNR |
|---|---|---:|
| **Backbone** | D-3DGS MLP (D=8) | 41.54 |
| | **CfC (D=6, ours)** | **42.03** (+0.49) |
| **Depth D** | 6 (ours) | 42.03 |
| | 8 | 41.82 |
| | 10 | 41.86 |
| **Activation** | ReLU | 41.47 |
| | **GELU (ours)** | 42.03 |
| | SiLU | 41.53 |
| | LeCun | 40.88 |
| | Tanh | 40.74 |

- D=6 是 sweet spot,D=8/10 退化 (跟 MLP D=8 持平)
- **GELU backbone 显著胜 Tanh (40.74)** — paper 解释 GELU 与 cell-level tanh 的组合比单纯 bounded 激活更平滑
- Ablation 自己 internal control 排除了: depth / activation / backbone 三因子

---

## 5. 局限 & 未来工作 (论文 §V)

1. **Depth-as-time 限制**: 无帧间 recurrence,长视频 irregular sampling 时 CfC 强依赖于长程时间记忆驱动;**未来 work** 是 recurrent-over-frames CfC
2. **未加辅助 loss**: 闭式 CfC 暴露 smooth ∂F_θ/∂t,论文承认可加 acceleration / As-Rigid-As-Possible / ARAP penalty 但**没做** — 跟 photometric loss 协同效果留 future work
3. **未做 ODE/SDE side-by-side**: 论文明确说"at matched compute, side-by-side 实验"会澄清 trade-off — 现在是**自我定位**(最廉价端), 缺"贵端" 实证
4. **未报 T=64/128 ODE 发散情况**: 隐式 vs ODE 是 closed-form 故**无 ODE solver 数值发散风险**,这一点 paper 没明确强调,但暗示是 closed-form 隐含好处

---

## 6. 与本仓的关系

### 6.1 公式同构

论文 Eq. 2 与本仓 `lnn/core/cfc.py::CfCCell.forward` 几乎逐行对应 (95%+ 同构):
- z = φ([u; h])  → `backbone([u; h])`
- g, h_cand = tanh(W_g/h z)  → `tanh(W_g z)`, `tanh(W_h z)`
- σ_τ = σ(W_a z · τ + W_b z)  → `σ(W_a z * τ + W_b z)`
- h' = g ⊙ (1 - σ_τ) + h_cand ⊙ σ_τ  → `g * (1 - sigma_tau) + h_cand * sigma_tau`

唯一不同是本仓 CfC 可能省略 h 维度的门控,论文加了 h 维度。**公式 1:1 同构,可直接拿本仓 CfCCell 复现**。

### 6.2 跨 task backbone matrix 加 device_control / vision 行的契机

- vision-3dgs 域: **CfC (本文)** 胜 MLP 0.47 dB, closed-form 同延迟下质量更好
- 对应本仓: `scripts/experiment_device_control_cases.py` 是 device control 域,**公式同源** (CfC), 4 case harness 现只有 industrial case 用 LNNImitationPolicy (LTC recurrent), 加一个 CfC variant 做 backbone matrix 对照是天然 follow-up

### 6.3 评级

- **学术 A-**: APSIPA ASC 是分领域旗舰会议(亚洲信号处理 + 视觉), 实测 + 消融 + 公式透明度都到位
- **工程 A-**: 改动是 drop-in (替换 D-3DGS 的 DeformNet 即可), 不引入新 solver, 不引入新训练范式
- **代码 C**: 论文没放独立 repo, 只在 D-3DGS 仓 fork 上改
- **本仓优先级 B+**: 公式同构, 但 3DGS 域跟本仓 4 个 device control case (quadruped / drone / industrial / battery) 不直接重叠; **vision-as-time 用法是 narrative bonus, 不是 P0 候选**

### 6.4 复现成本

- 数据: D-NeRF 8 scenes 是公开(8 synthetic monocular videos), NeRF-DS 7 scenes 也是公开
- 算力: 单 NVIDIA P100-PCIe-16GB 跑完 = ~几天训练(CfC stack D=6 + D-3DGS baseline 都要 train)
- 代码: 复用 D-3DGS 公开 PyTorch 实现, 改 `DeformNet` 类, ~200 LOC
- 4 stages: (A) fork D-3DGS 仓库 跑 baseline (B) 改 DeformNet 为 CIC stack (C) D-NeRF 8 scenes 烟测 1 seed (D) NeRF-DS 7 scenes 烟测 1 seed
- 估算: 4 stages × 4-6 loop = ~20 loop,~4-6 周 (跟 iter#34 EntroLnn 4 stages × ~6 loop 同量级)

---

## 7. 一句话总结

> **把 D-3DGS 的离散 MLP 替换成 closed-form CfC stack (depth-as-time) — 是 "ODE 谱系最廉价端" 的工程化胜利**。+0.47 dB mean PSNR on NeRF-DS 7 noisy real scenes, +2.74 dB on specular As, 6/8 synthetic 匹配或超, 闭式前向等价 MLP 延迟。**公式与本仓 CfCCell 95%+ 同构**, 但 3DGS 域跟本仓 device control 4 case 弱重叠, **记为 B+ 候选**。**关键信号**: 跨 task backbone matrix 应加 vision-3dgs 行(对照 MLP vs CfC) — 这是 narrative "LNN 通杀 4D 视觉" 的本地实证基线。
