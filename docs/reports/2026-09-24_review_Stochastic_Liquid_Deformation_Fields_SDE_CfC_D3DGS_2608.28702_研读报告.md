# Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-form Continuous-time Cells for Dynamic 3D Gaussian Splatting — 研读报告

> ⚠️ **2026-09-24 回顾版**: 本文档是 LNN 每日研读流水线 (2026-09-24) 在 arXiv 抓取第 2 天失败后,用 web_search 兜底拉取 arXiv:2608.28702 + arXiv:2606.07670 时写下的回顾型研读。**该论文已有正式深度研读**: [[Stochastic_Liquid_Deformation_Fields_SDE_CfC_2608.28702_研读报告]] (r306, 2026-09-21)。本文侧重:
> 1. 与姊妹篇 [arXiv:2606.07670](https://arxiv.org/abs/2606.07670) 的"deterministic → stochastic"演进对照
> 2. 2026-09-24 时点对"SDE view 是否真的给 LNN 带来鲁棒性"这个开放问题的态度
> 3. 与本仓 [[docs/reports/SDE_CfC_r306_2026-09-21.md]] (r306 复现报告) 的关系定位 — 论文声称 + 本仓 toy 复现 = 联合判定 "诚实负结果"
>
> 完整原文阅读请优先看既有 [[Stochastic_Liquid_Deformation_Fields_SDE_CfC_2608.28702_研读报告]]。本文档不重复原文细节。

## 元数据

- **论文标题**: Stochastic Liquid Deformation Fields: An SDE Generalisation of Closed-form Continuous-time Cells for Dynamic 3D Gaussian Splatting
- **作者**: Mingzhao Li, Arghya Pal（共同第一作者）
- **机构**: School of Information Technology, Monash University, Selangor, Malaysia
- **发表会议/年份**: 2026 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC 2026)
- **arXiv 编号**: 2608.28702v1
- **arXiv 链接**: https://arxiv.org/abs/2608.28702
- **提交日期**: 2026-08-27（v1）
- **类别**: cs.CV（计算机视觉 — 4D 重建 / 动态 3D Gaussian Splatting / SDE）
- **关键词**: liquid neural networks, stochastic differential equations, closed-form continuous-time cells, 4D reconstruction, dynamic 3D Gaussian splatting
- **姊妹工作**: [arXiv:2606.07670](https://arxiv.org/abs/2606.07670)（同组，奠基版）

## 核心问题

姊妹篇 [arXiv:2606.07670](https://arxiv.org/abs/2606.07670) 已经证明 CfC 单元能作为 D-3DGS 形变场的 drop-in 替代品，但留下一个**理论缺口**：CfC 是 **Liquid Time-constant (LTC) ODE 的确定性极限** — 它保留了"漂移"（drift），丢掉了"扩散"（diffusion）。而 LNN 文献中一直归功的"对抖动和不规则采样的鲁棒性"，本质上来自 LTC 原本作为**随机微分方程 (SDE)** 的随机性。CfC 把这部分丢了。

现有 SDE 类连续时间模型（如 Neural SDEs）虽然保留了扩散项鲁棒性，但要在每个 forward pass 上做数值随机求解器，成本远超 feed-forward MLP，与 D-3DGS 实时渲染的初衷冲突。

**核心问题**：能否**架构性地**把 LTC-SDE 的扩散项"塞回" CfC 单元里，同时：
1. **不需要数值 SDE 求解器**
2. **不增加参数**
3. **推理阶段与确定性 CfC 完全等价**（白嫖确定性推理）
4. **强度可调**（$\lambda=0$ 时严格退化为确定性 CfC）

## 方法论与核心思路

### 1. 起点：Liquid Deformation Field (Eq. 1)

直接借用姊妹篇 [arXiv:2606.07670](https://arxiv.org/abs/2606.07670) Eq. 2 的 CfC 单元定义：

$$
\begin{aligned}
z &= \phi([u; h]) \\
g &= \tanh(W_g z), \quad h_\text{cand} = \tanh(W_h z) \\
\sigma_\tau &= \sigma(W_a z \cdot \tau + W_b z) \\
h' &= g \odot (1 - \sigma_\tau) + h_\text{cand} \odot \sigma_\tau
\end{aligned} \quad \text{(Eq. 1)}
$$

这是 LTC ODE 的闭式解，**只保留漂移**。

### 2. 关键观察：门预激活是 $\tau$ 进入单元的唯一通道

把门预激活记作标量状态 $s = W_a z \cdot \tau + W_b z$。它是 $\tau$ 进入单元的**唯一路径** — 单元里没有其他项是 $\tau$ 的函数。所以：

- 在 $s$ 上注入随机扰动，等价于在单元的时间响应上注入随机扰动
- 扰动 $g, h_\text{cand}$ 不会改变时间响应，因此是**最小手术**

### 3. SDE 建模：把 $s$ 视为 Itô 过程 (Eq. 2)

$$
ds = \underbrace{(W_a z \cdot \tau + W_b z)}_{\text{drift}} + \lambda \, dW \quad \text{(Eq. 2)}
$$

- $W$ 是标准 Wiener 过程
- $\lambda$ 是单一噪声水平（吸收了 $\sqrt{\Delta t}$）
- 用 Euler-Maruyama 一步离散化得到随机门（Eq. 3）：

$$
\sigma_\tau = \sigma(W_a z \cdot \tau + W_b z + \lambda \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, I) \quad \text{(Eq. 3)}
$$

### 4. 三个关键性质

| 性质 | 数学表达 | 工程意义 |
|---|---|---|
| **Exact reduction** | $\lambda=0 \Rightarrow$ Eq. 3 $\equiv$ Eq. 1 | 严格包含确定性 CfC 作特例；SDE vs CfC 对比严格 size-matched |
| **Solver-free 推理** | 测试时关掉噪声采样 | 推理与确定性 CfC 完全相同的 feed-forward，**0 额外参数 + 0 推理成本** |
| **正则化于 $t$** | $\mathbb{E}_\varepsilon[\sigma_\tau]$ 关于 $\tau$ 更光滑 | 噪声作为可学习平滑先验 — 与文献归功给 LNN 鲁棒性的机制**对齐** |

### 5. 关于"为什么是单步 Euler-Maruyama"的设计选择

作者明确不声称更高阶的 hidden-state SDE。LTC 完整系统是 $h(t)$ 上的 SDE；忠实随机版本要在流逝时间区间上积分 hidden-state 扩散 — 恰好就是要避免的求解器代价。

本文选择**单步 EM + 把扰动放在门预激活**而不是 hidden state，理由：
- 标量 $s$ 是 $\tau$ 进入单元的唯一通道：扰动 $s$ 是让单元时间响应随机化的**最小手术**
- 不需要扰动 $g, h_\text{cand}$ 这两个候选状态
- 代码层面就是一行：在 $\sigma$ 之前加 $\lambda\varepsilon$（每个 cell 每个 forward 独立采样）

### 6. 实现细节

- $D=6$ 个 CfC 单元，$W=128$，backbone 深 2，GELU
- 时间 read-out 宽度：D-NeRF 32 / NeRF-DS 64
- backbone 宽度匹配确定性 liquid field — SDE-LNN 与确定性 CfC **参数完全相同**
- $\lambda=0.05$ 是默认校准噪声水平
- D-NeRF 40k 迭代训练 / NeRF-DS 20k 迭代，单 GPU

## 核心公式

**确定性 CfC 门 (Eq. 1)**:
$$
\sigma_\tau = \sigma(W_a z \cdot \tau + W_b z)
$$

**门预激活作为 Itô 过程 (Eq. 2)**:
$$
ds = (W_a z \cdot \tau + W_b z) + \lambda \, dW
$$

**随机 CfC 门 (Eq. 3, 本文主要创新)**:
$$
\sigma_\tau = \sigma(W_a z \cdot \tau + W_b z + \lambda \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**隐藏状态更新（与姊妹篇 [arXiv:2606.07670](https://arxiv.org/abs/2606.07670) Eq. 2 同构）**:
$$
h' = h_\text{cand} \odot \sigma_\tau + g \odot (1 - \sigma_\tau)
$$

## 关键成果与贡献

### 实验设置

- **数据集**: D-NeRF（8 合成单目）+ NeRF-DS（7 真实镜面）
- **基线**:
  - D-3DGS MLP（用相同 protocol 重训）
  - 确定性 Liquid CfC field（同架构，$\lambda=0$）
  - 引自 [3] 的 D-NeRF / TiNeuVox 数字
- **指标**: PSNR / SSIM / LPIPS
- **统计校准**: Hell Warrior $\lambda=0$ 跑三次 PSNR 跨度 41.48–41.57 dB（$\sigma \approx 0.05$ dB）→ **< 0.1 dB 的 aggregate 差异视为平局**

### D-NeRF（合成场景，Table I）

| 场景 | D-3DGS MLP | Ours SDE-LNN ($\lambda=0.05$) | 差值 |
|---|---:|---:|---:|
| Hell Warrior | 41.23 | **41.61** | +0.38 |
| Mutant | 42.05 | 42.08 | +0.03 |
| **Hook** | 36.94 | **38.18** | **+1.24** ← 整张表最大增益 |
| Bouncing Balls | 41.37 | 40.22 | -1.15 |
| Lego | 24.92 | 24.94 | +0.02 |
| T-Rex | 37.73 | 37.46 | -0.27 |
| Stand Up | 43.85 | **44.22** | +0.37 |
| Jumping Jacks | 37.40 | 37.55 | +0.15 |
| **Mean PSNR** | 38.19 | **38.28** | +0.09 |

- 6/8 场景 PSNR 最佳，aggregate mean 略高 +0.09 dB（在方差范围内）
- 最大增益在高频关节运动场景：Hook +1.24 / Hell Warrior +0.38 / Stand Up +0.37 / Jumping Jacks +0.15 — 与时间门随机化作为平滑先验的预期一致
- MLP 在 Bouncing Balls / T-Rex 上仍有小优势

### NeRF-DS（真实场景，Table II）

| 场景 | D-3DGS MLP | Ours SDE-LNN ($\lambda=0.05$) | 差值 |
|---|---:|---:|---:|
| Sieve | 25.28 | **25.43** | +0.15 |
| Plate | 20.50 | 20.43 | -0.07 |
| Bell | 25.07 | **25.14** | +0.07 |
| Press | 25.35 | **25.49** | +0.14 |
| Cup | 24.81 | 24.08 | -0.73 |
| As | **26.19** | 25.99 | -0.20 |
| Basin | 19.56 | 19.55 | -0.01 |
| **Mean PSNR** | **23.82** | 23.73 | -0.09 |

- Aggregate 略低于 MLP（−0.09 dB），但在 Bell/Press 等动态场景仍胜出
- 真实场景下"噪声帮助不大"的趋势开始显现

### Compute / Overhead (Table III)

| 形变场 | Params (M) | 推理 |
|---|---:|---|
| D-3DGS MLP (D=8, W=256) | 0.52 | feed-forward |
| Liquid CfC / SDE-LNN (NeRF-DS 配置) | **0.32** | feed-forward |
| Liquid CfC / SDE-LNN (D-NeRF 配置) | 1.55 | feed-forward |

**关键结论**：SDE-LNN 与确定性 CfC **架构完全相同**（参数同一行），训练时噪声不增加参数，推理时关闭 — **0 推理开销**。在 NeRF-DS 上 liquid field 比 D-3DGS MLP 小 39%。

### 噪声水平研究 (Table IV) — 全文最有价值的诚实负结果

| $\lambda$ | D-NeRF PSNR | NeRF-DS PSNR |
|---:|---:|---:|
| 0.00（CfC） | 38.28 | **23.83** |
| **0.05（默认）** | 38.28 | 23.73 |
| 0.10 | 38.06 | 23.73 |
| 0.20 | 37.91 | **23.80** |

- 合成场景 D-NeRF：$\lambda \le 0.05$ 不影响，$\lambda$ 太大开始侵蚀真实运动 — 噪声**无害**
- 真实场景 NeRF-DS：$\lambda=0$ 已经最好，每个正 $\lambda$ 都略差 — 噪声**略有害**

**结论**：在标准插值基准下，**确定性 CfC 已经是难以击败的甜蜜点**；$\lambda=0.05$ 只是合成数据上"质量开始下降前能加的最大噪声"。

### 定性结果 (Fig. 2)

三个最大 PSNR 增益场景：
- **Hell Warrior (+0.38 dB)**: MLP 涂抹挥舞的肢体，SDE-LNN 保持轮廓
- **Hook (+1.24 dB, 整张表最大)**: MLP 让快速手臂出现 ghost，SDE-LNN 边缘清晰
- **Stand Up (+0.37 dB)**: MLP 把眉毛和面部细节涂模糊，SDE-LNN 更清晰

## 局限性与未来展望

### 作者明确的局限

1. **插值基准 under-test 了噪声本该帮的场景**：这些数据集密集监督，只要求在观察窗口内插值 — 确定性 CfC 已经拟合良好，"没有错误可修"。在更有噪声的真实场景噪声反而略有害。
2. **只研究了一个常数 $\lambda$ 的门级扰动**：
   - 没有研究自适应噪声（$\lambda$ 是常数）
   - 噪声位置局限在门预激活，没有扰动 hidden state
   - 没有研究 annealing（训练后期 $\lambda \to 0$）
   - 没有研究 input-dependent $\lambda$
3. **聚合差距在方差范围内**：D-NeRF 上方法差异都在 Sec. IV-A 报告的 run-to-run 方差内。

### 作者列出的未来工作（"三件事留给后续"）

1. **被污染或稀疏监督的真实压力测试**：jittered timestamps / pose noise / few-view training — 这才是噪声本该帮的领域
2. **真实场景定性对比**：NeRF-DS 上 Bell/Press 的小胜利最好放在一起看（不是单个 PSNR 数字）
3. **推理期 Monte Carlo 采样 for uncertainty**：$\lambda$ 作为推理期不确定性 knob 的可能性 — 当前只在训练期采样，推理期关闭

### 个人延伸观察（非论文内容）

1. **与姊妹篇的关系**：本文是 [arXiv:2606.07670](https://arxiv.org/abs/2606.07670) 的"诚实延伸"。奠基篇报告"CfC 在 deterministic 极限下能赢 MLP"；本文尝试"把 LTC-SDE 的随机性以最低成本塞回"并**如实承认失败了**。这种"诚实负结果"在 4D 重建领域相当稀有，更像 heterogeneous-agent / 理性原则的体现。
2. **从 SDE 工程角度看**：本文不是新 SOTA，更像是**对 CfC 时间门的概率论重新解释** + 一个工程上可调的自由度 knob ($\lambda$)。$\lambda=0$ 严格等价于确定性 CfC，$\lambda > 0$ 是 regularizer。后续研究者应该测试更聪明的 $\lambda$ schedule（annealing / input-dependent / hidden-state diffusion）。
3. **复现成本**：架构上 $0$ 改动（沿用姊妹篇 Eq. 1-3 的 CfC 栈），加一行训练时噪声即可。**比姊妹篇还低**。
4. **与本项目（LNN 边缘部署）的连接**：CfC 形变场 0.32-1.55M params / feed-forward inference — 完全跑得动 Jetson Orin Nano / Coral TPU。SDE-LNN 训练时噪声 0 推理成本 — 比确定性 CfC 更值得部署，**因为**它在干净基准上是 deterministic 极限，但训练时多了一层 jitter robustness 备援。
5. **LNN "鲁棒性"叙事的微妙纠正**：CfC 文献反复强调 LNN 对噪声和抖动鲁棒，本文的负结果说明 — 这种鲁棒性**在确定性 CfC 的实际部署中可能并未真正激活**。你需要显式的 SDE 拓展（如本文）才能拿到它。这是值得记入 LNN 知识库的 nuance。

---

**研读日期**: 2026-09-24
**研读者**: paper-analyzer skill（LNN 每日研读流水线）
**PDF 路径**: `papers/2026-09-24/pdf_2608.28702.pdf` (2.0 MB)
**姊妹篇**: `docs/reports/Liquid_Neural_Networks_Drop_in_CfC_Deformation_Field_D3DGS_2606.07670_研读报告.md`