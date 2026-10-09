---
title: "ContiLNN: Mitigating Slice Sampling Discontinuity with Liquid Neural Networks for Medical Image Restoration"
date: 2026-10
tags: [LNN, CfC, Bi-CfC, medical-image-restoration, CT-denoising, MRI-super-resolution, PET-restoration, RWKV, Mamba]
---

# 研读报告：ContiLNN — 用液态神经网络沿切片轴建模医学影像修复中的采样间断

## 1. 元数据

- **论文标题**：ContiLNN: Mitigating Slice Sampling Discontinuity with Liquid Neural Networks for Medical Image Restoration
- **作者**：Jialei He†, Enhe Liu†, Sifan Song, Pengfei Jin, Jionglong Su, Hongbin Wang, Zhixiang Lu, Yanhao Huang, Anteng Cai, Zhengyong Jiang, Jiaman Ding, S. Kevin Zhou, Jinfeng Wang (共同一作 †; 共同通讯: Ding / S.K. Zhou / Jinfeng Wang)
- **机构**：昆明理工大学 / 西交利物浦大学 AI 院 / Harvard Medical School (MGH) / 中国科大 MIRACLE 中心
- **发表时间**：2026-10-08 (arXiv: 2610.12337v1, cs.CV)
- **代码**：未在文中明确释放; 训练在 NVIDIA A800 上完成
- **本地 PDF**: [papers/2610.12337/2610.12337.pdf](../../papers/2610.12337/2610.12337.pdf)
- **本地文本**: [papers/2610.12337/2610.12337.txt](../../papers/2610.12337/2610.12337.txt)
- **关联概念**：CfC (Hasani 2022) / Bi-CfC / Liquid Neural Network / 2.5D 修复 / Slice-axis 顺序传播 / Restore-RWKV / DASMamba / Cross-slice consistency / 2D backbone distillation

## 2. 核心问题

医学影像修复 (low-dose CT 去噪 / MRI 超分 / 低计数 PET 重建) 长期由 2D 卷积或纯 in-plane 的 Transformer/SSM (含 Restore-RWKV, DASMamba) 主导。然而, 3D 体数据包含的跨切片 (through-plane) 解剖结构本身就是修复任务的高度互补信号, 单纯 2D 处理无法显式建模这一顺序结构 (论文 Fig. 1A 的 ordered context omission). 已有 2.5D 思路 — 邻切片卷积 (Shan 2018)、三维 MRI SR (Chatterjee 2024)、Spach Transformer (Jang 2024) — 要么把切片数当作上下文度量, 要么把切片均匀重排后再送入序列模型, 都没有区分"原始索引间隔"。

临床上更尖锐的场景: 重建时**只有原始体数据中的稀疏子集切片可用, 且抽样间隔不规则** (regular / sparse / irregular 三种情形). 对 [1,3,5,7] 与 [1,2,3,5,6] 两组索引, 切片数量相同但跨越的解剖范围与信息密度完全不同; 现有 RNN/Transformer 把所有切片当等距节点处理, 丢失了"sampling variation"信号。

ContiLNN 把这些问题重述为: **如何在保留 2D in-plane 修复主干的条件下, 用一个能编码原始索引间隔的沿切片轴算子, 把邻切片上下文传播给每一个被供给的切片**。

## 3. 方法论与核心思路

### 3.1 总体架构

ContiLNN 是一个 **二维 in-plane 主干 + 沿切片轴双向 CfC (Bi-CfC) 模块** 的混合体 (Fig. 2A). 主干优先保留 Restore-RWKV (Yang 2026a) 的 R-RWKV Block (Omni-Shift + Re-WKV), 也可换成 DASMamba (Chan 2026). Bi-CfC 通过两种插入模式作用在选定的特征路径上:

| 插入位 | 集合 | 功能 |
|---|---|---|
| Bottleneck | $S_b$ | 在最深特征上做序列序列化 (serial) 处理 |
| Encoder L2 / Decoder L2 / Skip L2–L3 / Refinement | $S_g$ | 学到的门控残差 (gated residual): $\tilde{A}^l = A^l + \sigma(a^l)(V^l - A^l)$ |

每个 in-plane 像素位置独立拥有一份沿切片轴的 liquid dynamics; 算子 $\Psi^l$ 在所有空间位置上**共享权重**, 见 Eq.(12) 的 $A^l \xrightarrow{R^l} F^l \xrightarrow{\Psi^l} U^l \xrightarrow{(R^l)^{-1}} V^l$ 重排。

### 3.2 Bi-CfC 模块的连续时间导引

沿用 Hasani 2022 的导引思路, 把切片轴上的连续时间状态演化写为:
$$
\frac{dh^l(q)}{dq} = G^l_\theta(h^l(q), e^l(q)) \tag{3}
$$
其中 $q$ 是连续演化坐标, $e^l_t$ 是当前切片的特征. CfC 用闭式更新取代数值积分, 但**保留 interval 输入 $\delta_t$** 让 gate 对采样间隔显式响应 (Eq. 6–9 的核心创新点).

### 3.3 位置与间隔编码 (Eq. 4–5)

对 $K$ 张切片的窗口, 原始索引间隔 $g_t = i_t - i_{t-1}$, 归一化得到:
$$
p_t = \frac{t}{K-1}, \qquad
\delta_t = \begin{cases} g_1 / K, & t = 0, \\ g_t / K, & t = 1, \dots, K-1 \end{cases} \tag{4}
$$
位置编码:
$$
e^l_t = f^l_t + P^l_{\text{pos}}(p_t) \tag{5}
$$

### 3.4 单步 CfC 更新 (Eq. 6–9, 论文核心创新)

```
联合表示    z_t = ρ(W_z [e_t; h_{t-1}] + v_z)        ∈ ℝ^M   (Eq. 6)
候选态    c_t^(1) = tanh(W_1 z_t + v_1)
候选态    c_t^(2) = tanh(W_2 z_t + v_2)             ∈ (-1,1)^C
间隔系数    η_t = W_a z_t + v_a
截距        ξ_t = W_b z_t + v_b
门         α_t = σ(δ_t η_t + ξ_t)                 ∈ (0,1)^C    (Eq. 7)
状态更新    h_t = (1 - α_t) ⊙ c_t^(1) + α_t ⊙ c_t^(2)
                  = c_t^(1) + α_t ⊙ (c_t^(2) - c_t^(1))     (Eq. 8)
```

**关于 $\delta_t$ 的导数 (Eq. 9)** — 论文特别把对间隔的局部灵敏度写出来:
$$
\frac{\partial h_t}{\partial \delta_t} = \alpha_t (1 - \alpha_t) \odot \eta_t \odot (c_t^{(2)} - c_t^{(1)}) \tag{9}
$$
也就是说门和间隔系数共同决定每个通道对间隔的响应强度; 不同通道可正可负, 符合"采样稀疏时不同解剖特征应该有不同的跨切片衰减"。

### 3.5 双向传播与对齐 (Eq. 10–11)

独立的 forward / backward cell (参数不共享) 在同一组 (feature, interval) 上反向遍历; 然后通过 affine $W^l_{\text{bi}}$ 在原顺序位置上对齐:
$$
\overrightarrow{h}^l_t = \Phi^l_{\theta^\to}(e^l_t, \overrightarrow{h}^l_{t-1}, \delta_t), \qquad
\overset{\leftarrow}{h}^l_t = b^l_{K-1-t}, \qquad
u^l_t = P^l_{\text{bi}}([\overrightarrow{h}^l_t; \overset{\leftarrow}{h}^l_t]) \tag{10, 11}
$$

**关键的不变设计**: 反向遍历时, $\delta_t$ 仍按原顺序绑在第 $t$ 个 observation 上, 而不是按反向的 $t$ → 这是"feature–interval pairing"语义保护 (Eq. 10 的最后两行).

### 3.6 训练目标 (Eq. 14–18)

总目标 = 重构 + 跨切片一致性 + 2D 主干蒸馏:
$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \lambda_{\text{KD}} \mathcal{L}_{\text{KD}} \tag{14}
$$

- **$\mathcal{L}_{\text{rec}}$**: L1 到 HQ 切片 (Eq. 15)
- **$\mathcal{L}_{\text{KD}}$**: L1 到冻结的 2D 主干预测, 保留 in-plane 精度
- **$\mathcal{L}_{\text{cons}}$**: 一阶/二阶有限差分与 HQ 的对齐 (Eq. 16–18):
$$
D_1 y_t = y_{t+1} - y_t, \quad D_2 y_t = y_{t+1} - 2 y_t + y_{t-1}
$$
$$
\mathcal{L}^{(1)}_{\text{cons}} = \frac{1}{K-1} \sum_{t=0}^{K-2} \|D_1 \hat{y}_t - D_1 y_t\|_1 \tag{17}
$$
作者明确说: 这一项让重构的**切片间强度变化**对齐 HQ 的解剖边界, 而不是把每张切片都强行拉到一致的强度。

### 3.7 与 LTC / CfC / Bi-GRU 的关系

- 与 **LTC (Hasani 2021)**: LTC 是连续 ODE, 需要数值积分; CfC (Hasani 2022) 给出闭式近似并保留 interval 输入. Bi-CfC 沿用 CfC 的闭式更新但**强调双向**, 且**显式让 $\delta_t$ 调门 (Eq. 7)**, 而不是像标准 CfC 那样只让门静态依赖输入。
- 与 **Bi-GRU**: 在 PET 上参数 / GMAC 几乎一致 (34.91 M vs 34.93 M; 423.5 vs 422.4 GMAC/window), 但 Bi-CfC 把七切片窗口延迟从 380.92 ms 砍到 179.85 ms (-52.8%), 显存峰值从 2,315.80 MB 砍到 995.03 MB (-57.0%) (Table 4); PSNR 在所有采样间隔下都更高 (Table 2).

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 | $F_\theta: X_i \mapsto \hat{Y}_i$, 长度 $K \le T$ 的局部窗口对应输入/输出 | 切片-位置一一对应: 第 $k$ 个输入 $\to$ 第 $k$ 个输出 |
| Eq. 2 | $g_t = i_t - i_{t-1}$, $t=1..K-1$ | 原始索引间隔: contiguous = 1, every-other = 2, mixed = e.g. (1,1,2,1) |
| Eq. 3 | $dh^l(q)/dq = G^l_\theta(h^l(q), e^l(q))$ | 切片轴 ODE 起点 (不可数值积分, 只用于推导) |
| Eq. 4 | $p_t, \delta_t$ 见 §3.3 | 间隔归一化, $K=1$ 时退化为单切片约定 (附录 A) |
| Eq. 5 | $e^l_t = f^l_t + P^l_{\text{pos}}(p_t)$ | 位置嵌入 + 切片特征 → cell 输入 |
| Eq. 6 | $z_t = \rho(W_z [e_t; h_{t-1}] + v_z)$, $c^{(j)}_t = \tanh(W_j z_t + v^{(j)})$ | 双候选分支, $\rho$ 为 scaled tanh (附录 A) |
| Eq. 7 | $\eta_t = Q_a(z_t)$, $\xi_t = Q_b(z_t)$, $\alpha_t = \sigma(\delta_t \eta_t + \xi_t)$ | **间隔-门耦合**: 同一个 affine $z_t$ 给出两条线, 间隔决定混合 |
| Eq. 8 | $h_t = (1-\alpha_t) \odot c^{(1)}_t + \alpha_t \odot c^{(2)}_t$ | 凸组合, $h_t \in (-1,1)^C$ |
| Eq. 9 | $\partial h_t / \partial \delta_t = \alpha_t (1-\alpha_t) \odot \eta_t \odot (c^{(2)}_t - c^{(1)}_t)$ | 每个通道对间隔的独立响应强度 |
| Eq. 10 | $\overrightarrow{h}^l_t, b^l_r, \overset{\leftarrow}{h}^l_t = b^l_{K-1-t}$ | 双向传播, 间隔按原 observation 绑定 |
| Eq. 11 | $u^l_t = W^l_{\text{bi}}[\overrightarrow{h}^l_t; \overset{\leftarrow}{h}^l_t] + v^l_{\text{bi}}$ | 双向融合对齐到原位置 |
| Eq. 12 | $A^l \xrightarrow{R^l} F^l \xrightarrow{\Psi^l} U^l \xrightarrow{(R^l)^{-1}} V^l$ | 重排算子使得 $\Psi^l$ 跨所有 in-plane 位置共享 |
| Eq. 13 | $\tilde{A}^l = V^l$, $l \in S_b$; $A^l + \sigma(a^l)(V^l - A^l)$, $l \in S_g$ | Bottleneck 替换 / 选择路径门控残差 |
| Eq. 17 | $\mathcal{L}^{(1)}_{\text{cons}} = \frac{1}{K-1}\sum \|D_1\hat{y}_t - D_1 y_t\|_1$ | 跨切片一阶差分一致性 (二阶同理) |
| Eq. 19 | $\hat{y}_i = \frac{1}{|W(i)|} \sum_{w \in W(i)} \hat{y}^{(w)}_i$ | 重叠窗口 inference 融合 |

## 5. 关键成果与贡献

### 5.1 主结果 (Table 1, 5 seeds, $\mu \pm s$)

恢复的三个任务 (MRI SR / CT 去噪 / 低计数 PET) 在 Restore-RWKV 基线上同时提升 PSNR + SSIM + 降低 RMSE, 且提升稳定跨 seed:

| 任务 | 指标 | Restore-RWKV (重制) | ContiLNN-RWKV (ours) | Δ |
|---|---|---|---|---|
| MRI SR | PSNR (dB) | 32.0650 | **33.0826 ± 0.0261** | +1.0176 |
| MRI SR | SSIM | 0.9403 | **0.9485 ± 0.0003** | +0.0082 |
| MRI SR | RMSE | 29.0337 | **25.9882 ± 0.0809** | −3.0455 |
| CT 去噪 | PSNR | 33.7642 | **33.9549 ± 0.0026** | +0.1907 |
| CT 去噪 | SSIM | 0.9189 | **0.9208 ± 0.0003** | +0.0019 |
| CT 去噪 | RMSE | 8.3950 | **8.2132 ± 0.0030** | −0.1818 |
| PET | PSNR | 37.2722 | **38.5204 ± 0.0938** | +1.2482 |
| PET | SSIM | 0.9474 | **0.9535 ± 0.0013** | +0.0061 |
| PET | RMSE | 0.0860 | **0.0752 ± 0.0009** | −0.0108 |

跟 DASMamba 主干同样能涨: ContiLNN-DASMamba 在 MRI / PET 上 PSNR +0.36 / +1.13 dB。CT 只有一个测试 patient, 文中明确为"descriptive", seed 间 PSNR SD 仅 0.0026 dB, 说明训练可重复性极好。

### 5.2 算子对比与算力分析

按 Table 2 / Table 4 / Table 5 拼出的核心证据:

- **采样适应性 (Table 2, PET, contiguous training):** Bi-CfC 在 $g=1/2/3$ 以及 irregular 上下文全部高于 Bi-GRU; PSNR 优势从 $g=1$ 的 0.0693 dB 扩大到 $g=3$ 的 0.1357 dB.
- **算力 (Table 4):** 在 128×128, 32-bit, A800, 7-slice profiling 下, Bi-CfC 相对 Bi-GRU, latency 52.8% ↓, peak GPU memory 57.0% ↓, 但参数量与 GMAC 几乎相等 — 等价算力不等价实现成本, 这是 LNN 闭式更新的核心工程收益。
- **架构消融 (Table 5, PET):** Bi-CfC (reconstruction-only) → +1.1444 dB; 加上 $\mathcal{L}_{\text{full}}$ → 累计 +1.3146 dB. 同时与同尺寸 MLP 相比, Bi-CfC 高 1.6185 dB PSNR, 说明涨点来自"ordered propagation"本身而非多出来的参数量.

### 5.3 上下文扰动测试 (Table 6)

保持 checkpoint 不变, 仅修改中心图像的邻切片上下文:

| 上下文 | ΔPSNR (dB) |
|---|---|
| Ordered (default) | 0 |
| Reversed (顺序反向) | −0.2565 |
| Center-repeated (中心复制) | −2.6151 |
| Nonlocal same-case (同 patient 但非局部) | −6.3511 |

这给出一个非常干净的因果链: **去掉邻切片信息, 涨点几乎全部消失**; 仅改顺序仍能保留大部分信息。

### 5.4 贡献清单

1. **Bi-CfC 算子**: 双向闭式 CfC + **间隔调门** (Eq. 7) — 让门同时由当前特征和"原始索引间隔"决定, 不需要 ODE 数值积分, 即可对 sampling variation 显式响应。
2. **2D 主干 + 切片轴混合架构**: 在 Restore-RWKV / DASMamba 上均验证, Bi-CfC 可以 plugin 添加, 不破坏主干已学到的 in-plane 表示。
3. **跨切片一致性 + 2D 蒸馏**: $\mathcal{L}_{\text{cons}}$ 在差分域而非像素域对齐, 适合解剖边界; 蒸馏项防止切片轴信息稀释 in-plane 精度。
4. **系统化验证**: 5-seed 统计、不同采样间隔 (regular/sparse/irregular)、算力 profile、上下文扰动、数据效率 (5%/10% patient) — 共 6 个视角交叉证明涨点稳定。

## 6. 局限性与未来展望

### 6.1 作者自陈局限 (§5.3)
- 数据预处理后丢失了 DICOM 物理切片间距 metadata, 所以 $\delta_t$ 用的是 index gap, 不是真实解剖距离。
- 单张窗口长度与 index gap 共同决定覆盖的解剖范围, 文中没有自适应解。
- CT 测试集只有一个 held-out patient, seed 间 SD 表征的是训练重复性, 不是跨 patient 鲁棒性。
- 没有评估跨采集协议 (different vendors, different protocols) 的泛化。
- Bi-CfC 把 28.37 M → 34.93 M (+23%) 参数量, GMAC/slice 从 37.46 → 60.34 (+61%); latency per slice 从 16.72 ms → 25.69 ms.

### 6.2 隐含局限 (与本仓视角)
- **没有与 SSM/Conv 的严格配对 budget ablation**: 文中比较了 Bi-CfC vs Bi-GRU vs Slice-indep MLP, 但**没有**与 Bi-Conv1D, Bi-Mamba, Bi-Attention 在 7-slice 窗口下做配对 budget 对比; 这留给了一个"模型容量 vs 闭式公式" 的混淆。
- **K=7 是固定的**: 没有任何 ablations 验证更长窗口 (K=11, 15) 是否能进一步涨, 也未给出 K=3 的退化曲线。
- **CT 的 1-patient 测试**: 1.0176 dB / 1.2482 dB 在 MRI/PET 上的数据点很强, 但 0.1907 dB 在 CT 上没有 case-bootstrap CI, 弱统计意义。
- **$\lambda_{\text{cons}}$, $\lambda_{\text{KD}}$, $\beta_2$**: 没有超参敏感度, 也没有自动加权版本 (uncertainty weighting, grad-norm balancing)。

### 6.3 未来方向 (作者建议)
- 用 DICOM 物理切片坐标取代纯索引间隔。
- 自适应窗口长度 / selective propagation (只对低频特征做切片传播)。
- 与切片轴 1D conv / attention 在配对 budget 下对比。
- 跨采集协议 (other vendors, PET tracer species) 评测。

## 7. 对本仓的意义

ContiLNN 是 CfC 在 2025–2026 这一波"通用序列骨干 + 任务特定切片传播" 路线里**最有说服力的应用证据**之一: 几乎每条轴上 (CT/MRI/PET, regular/sparse/irregular, Restore-RWKV/DASMamba, 算力 vs 性能) 都跑过了, 而且把"为什么用 liquid 而不是 GRU" 给出了实测 (latency/memory 等价算力大胜).

- 可作 **`lnn/core/variants.py` 的新变体**: Bi-CfC 的 Eq. 6–8 已经是最干净的 implementation-ready 公式; 在仓内可对照 `lnn_neurons.py::CfCCell` 把 forward/backward 双方向 + interval-gated α 拼出来。
- 可嫁接到 **任何 in-plane 主干**: 论文已经证明在 RWKV 与 Mamba 上都能涨, 本仓现有 `bench_*_cfc.py` 多数已经把 backbone-agnostic 的 cfc 作为单方向算子测过, 现在可以加一组 `bench_bi_cfc_X.py` (X=RWKV, Mamba, GRU baseline) 来验证 "双向 + 间隔" 在本仓 time-series 数据 (Henry Hub / Peristent / Sinusoidal) 上是否有同类收益。
- 提供新的 **ablation axis**: `interval_modulation_strength` (`η_t = 0` 退化成静态门), `directions` (forward-only vs Bi-CfC), `interval_normalization` (1/K vs raw $g_t$ vs log(1+g_t))。这些都应该写成独立的 `bench_*` 文件。
- 可作为 **`scripts/experiment_imitation_lnn.py` 系列方法学更新**: 之前本仓 imitation 在 7-9 维状态空间上做仿生, 而 ContiLNN 给了在更高维 in-plane × sequence 二维网格上 sandwich liquid dynamics 的范式, 可以启发新 imitation 实验把"被模仿系统轨迹"作为切片轴 ground truth。

**Verdict**:
- **TARGET-POSITIVE**: Bi-CfC 是闭式连续时间算子中**目前最有证据的"工业级可落地形态"** — 既有理论 (Eq. 9 的可解释间隔灵敏度), 又有算力 (latency/memory 对 GRU 大胜), 又有三模态实证。强烈建议作为 `lnn/` 的下一个 **release-candidate** default variant.
- **TARGET-DEPENDENT-WITH-NUANCE**: 如果本仓需要部署到 Jetson Orin 这类更紧的资源约束, 那 -57% memory 比 Bi-GRU 的吸引力会被 +23% 参数基本抵消 — 需要在本仓 edge profile 上重做 profiling。
- **TARGET-NEGATIVE-WITH-NUANCE**: 如果要做"超长序列 (K > 100) 的 NAR / state-space 任务", ContiLNN 的双向设计会保持 $O(K)$ 隐状态但**没有显式 long-range attention 备份**, 仍用固定 K=7 ablation, 长上下文场景下与 S4/Mamba 系的竞争力未证, 不要直接外推。
