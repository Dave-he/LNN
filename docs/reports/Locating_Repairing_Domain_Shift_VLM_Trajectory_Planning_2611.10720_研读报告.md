---
title: Locating and Repairing Domain Shift in VLM Trajectory Planning
date: 2026-11
tags: [LNN, LFM2, VLM, domain-adaptation, activation-patching, causal-diagnosis, trajectory-planning, layer-partition, MoLaRx]
arxiv_id: 2611.10720
---

# 研读报告：MoLaRx — 把域偏移定位成"模态 × 层"二维张量，只在出问题的位置修复

## 1. 元数据

- **论文标题**：Locating and Repairing Domain Shift in VLM Trajectory Planning
- **作者**：Zhihong Cui, Hengyu Liu*, Michael Alexander Riegler, Guandong Xu, Amir Taherkordi, Tor Skeie —— 奥斯陆大学 / 阿尔堡大学 / Simula 都市数字工程中心 / 香港教育大学 / Simula 研究所（* 通讯）
- **发表时间**：2026-11-07（arXiv:2611.10720v1, cs.CV / cs.RO），**NeurIPS 2026 Main Track 录用**
- **代码**：https://github.com/ZhihongCui/MoLaRx（论文在 comment 中给出）
- **本地 PDF**：[papers/2611.10720/2611.10720.pdf](../../papers/2611.10720/2611.10720.pdf) ／ 文本 [2611.10720.txt](../../papers/2611.10720/2611.10720.txt)
- **arXiv 分类**：cs.CV, cs.RO
- **关联概念**：LFM2-VL-1.6B / Qwen2-VL-2B / PaliGemma2-3B / 激活修补（activation patching）/ 因果重要性评分（CIS）/ 最大均值差异（MMD）/ LoRA / 层级选择性域适应 / 神经流形分析（本仓方法论近亲）
- **入选理由**：本仓**零覆盖**"用**因果定位**方法把域偏移拆成模态 × 层结构，再按结构配修复算子"这一整条方法论路线。更关键的是，本篇是本仓**第一次在 LNN 相关模型（LFM2-VL）上直接检验"逐层结构差异可诊断"** 的公开工作 —— 而本仓自己在 11-13 那轮研读 CT-GRU 时就发现"**逐 τ 变体在数值上可能是一回事的不同写法**"，两者指向同一个待验证问题：**液态模型的层/时间尺度结构到底有多少是真正不同的？**

## 2. 核心问题

VLM 正在成为端到端驾驶规划器（把视觉感知与历史轨迹上下文整合进决策），但它们在**跨城市域偏移**下严重退化：论文给出的代表性数字是，在 Boston 训练的模型部署到 Singapore 时，**轨迹误差增加 +653%**。

现有方法把域偏移当作**一个标量**，用两种方式应对：**统一对齐**（uniform alignment，域对抗 / MMD / 统一 LoRA）或**单点编辑**（single-site editing，如 ROME 只改一个层）。论文对这一视图提出两点反对：

1. **（F1）模态轴**：在 VLM 里，图像 token 与文本 token 的偏移模式**在每个模型里都不同**。图像侧的偏移集中在浅层，文本侧的偏移集中在中深层 —— 把它们加总成一个标量会**同时抹掉两个不同的信号**。
2. **（F2）层轴**：逐层的偏移划分是**架构相关**但**数据集稳定**的。也就是说，偏移在模型内部有**结构**，而这个结构在换数据集时**不变**。

于是问题被重述为：**域偏移在 VLM 规划器内部是否是一个可定位的二维结构（模态 × 层）？如果是，能否用"先定位、后修复"（locate-then-repair）的配方，匹配这个结构地施加修复算子，从而优于所有"统一目标"的基线？**

论文的立场很明确：**"对齐一切"会覆写任务不变计算**（overwrite task-invariant computation），同时**留下模态特异错误**。

## 3. 方法论与核心思路

### 3.1 总体架构

```
                        LOCATE：CIS（激活修补）
  场景匹配对 (x_s, x_t) ──► 前向两路 ──► 在 (模态 m, 层 l) 处把目标路的
                                          模态 m 行替换为源路对应行
                                          ──► 继续前向得到反事实轨迹
                                          ──► 逐 waypoint ℓ2 位移
                                          ──► 蒙特卡洛平均 bΨ[m,l]（N=100 对）
                                              2×L 张量（模态 × 深度）
                                    │
                              阈值 τ_img / τ_txt（排序曲线的肘点）
                                    │
                Φ = (Φ_align, Φ_tune, Φ_freeze)  三段划分
                                    │
                        REPAIR：CISR（分区配算子）
   Φ_align ──► 多尺度 MMD 对齐 + 只挂注意力投影上的 rank-r LoRA
   Φ_tune  ──► 全投影 rank-16 LoRA（α=32），由监督轨迹损失驱动
   Φ_freeze ──► 不挂任何适配器（Δ_l = 0，冻结）
                                    │
                          合成后的适配器 → 跨城市规划器
```

### 3.2 场景匹配对与域偏移间隙（Definitions 3–4）

源数据集 $S_s = \{(x_i^s, y_i)\}_{i=1}^{N_s}$，无标签目标集 $S_t = \{x_j^t\}_{j=1}^{N_t}$。作者构造**成对跨域分布**，把内容偏移与域偏移分离开：

$$
P_{paired} = \mathrm{Uniform}\Big\{(x_s, x_t) : x_t \in S_t,\ x_s = \arg\min_{x\in S_s}\lVert k(x) - k(x_t)\rVert_2^2\Big\}
$$

每个对代表"同一驾驶场景的两个城市"，**从而隔离出域偏移、排除内容偏移**。

域偏移间隙定义为成对预测轨迹的平均 $\ell_2$ 距离：

$$
G(f) = \mathbb{E}_{(x_s,x_t)\sim P_{paired}}\left[\frac{1}{K}\sum_{k=1}^{K}\lVert f(x_t)_k - f(x_s)_k\rVert_2\right]
$$

$G(f)=0$ 意味着模型对每一对场景给出相同轨迹。

### 3.3 模态–层张量 Ψ（Definition 5）—— 本篇的核心定义

$\Psi \in \mathbb{R}^{2\times L}_{\ge 0}$，行是模态 $m \in \{\text{img}, \text{txt}\}$，列是层 $l$。每个条目度量"该 (m,l) 切片对域偏移间隙的贡献"：

$$
\Psi[m,l] = \mathbb{E}_{(x_s,x_t)\sim P_{paired}}\left[\frac{1}{K}\sum_{k=1}^{K}\Big\lVert f_\Theta\Big(\mathrm{do}\big(H^{(l)}_{M_m(x_t)} \leftarrow H^{(l)}_{M_m(x_s)}\big); x_t\Big)_k - f_\Theta(x_t)_k\Big\rVert_2\right]
$$

其中 $\mathrm{do}(\cdot)$ 把第 $l$ 层隐藏态中**模态 $m$ 的那些行**（图像 patch 位置 / 文本 token 位置）替换为源路的对应行，**另一模态的行保持不动**，前向从第 $l+1$ 层继续。

**这就是本篇的方法论内核**：一个**反事实的、逐切片的因果干预**。大条目 = 该切片是域偏移的**因果驱动因素**；近零条目 = **域不变**。

### 3.4 CIS 估计器与统计保证（§4.1）

实证估计量是对 $N=100$ 个场景匹配对做蒙特卡洛平均：

$$
\hat\Psi[m,l] = \frac{1}{N}\sum_{i=1}^{N} d^{(l,m)}_i,\qquad d^{(l,m)}_i = \frac{1}{K}\sum_{k=1}^{K}\Big\lVert \tilde Y^{(l,m)}(x^s_i,x^t_i)_k - f_{\Theta'}(x^t_i)_k\Big\rVert_2
$$

**统计保证**（这是本篇比一般"重要性评分"工作更严谨的地方）：每个 $d^{(l,m)}_i$ 上界为两条预测轨迹间的最大逐 waypoint 距离 $R$，由 Hoeffding 不等式：

$$
\Pr\Big[\big|\hat\Psi[m,l] - \Psi[m,l]\big| > \varepsilon\Big] \le 2\exp\left(\frac{-2N\varepsilon^2}{R^2}\right)
$$

取失败概率 $\eta=0.05$、$N=100$、$R\approx 20\,\mathrm{m}$，反解得**95% 半宽 $\varepsilon \approx 0.6\,\mathrm{m}$**，远小于跨分区的 $0 \to 2.3\,\mathrm{m}$ 展开 ⇒ **分区在统计上可靠**。附录 C.2.4 还有 $N$ 预算的扫描验证。

### 3.5 从张量到分区（Eq. 11–13）

阈值 $\tau_{img}, \tau_{txt}$ 取**排序 CIS 值的肘点**（elbow），然后：

$$
\Phi_{align} = \{l : \hat\Psi[\text{img},l] > \tau_{img}\},\qquad
\Phi_{tune} = \{l : \hat\Psi[\text{txt},l] > \tau_{txt}\} \setminus \Phi_{align},\qquad
\Phi_{freeze} = \{0,\dots,L-1\} \setminus (\Phi_{align}\cup\Phi_{tune})
$$

### 3.6 CISR：按区匹配修复算子

- **$\Phi_{align}$ 上的多尺度 MMD 对齐**（Eq. 14）：对源/目标 image-token 均值池化表示做 MMD，**且只在注意力投影（Q,K,V,O）上挂 rank-$r$ LoRA** —— 作者明确说这是"F1 指向的跨模态混合位点"。
- **$\Phi_{tune}$ 上的选择性 LoRA 重训**（Eq. 15）：rank-$r=16$、$\alpha=32$，挂在**所有线性投影**上，由标准监督轨迹损失驱动。
- **$\Phi_{freeze}$：零梯度**，不挂任何适配器。

总损失（Eq. 16）：

$$
\mathcal{L}_{total} = \mathcal{L}_{task}(\Theta',\Delta) + \lambda_{MMD}\mathcal{L}_{MMD}(\Theta'; B_s, B_t),\qquad \lambda_{MMD}=0.5
$$

**梯度按区路由**；目标域样本**只进入 MMD 的核项**，因此 CISR 是**合法的无监督 DA 方法**（transductive unsupervised adaptation）。当 $\Phi_{tune}=\varnothing$（PaliGemma2 就是这种情况），Eq. 16 自动退化为"MMD-on-$\Phi_{align}$ + freeze-on-$\Phi_{freeze}$"。

### 3.7 优化问题形式（Eq. 6–7）

$$
\Delta_l = \begin{cases} \delta_{\theta_l} & l\in\Phi_{align}\;(\text{rank-}r\text{ LoRA，仅注意力})\\ B_l A_l & l\in\Phi_{tune}\;(\text{rank-}r\text{ LoRA，全投影})\\ 0 & l\in\Phi_{freeze} \end{cases}
\qquad
\Delta^* = \arg\min_\Delta G(f_{\Theta'+\Delta})\;\; \text{s.t.}\;\; \mathbb{E}_s(f_{\Theta'+\Delta}) \le \mathbb{E}_s(f_{\Theta'}) + \epsilon
$$

即：**最小化域偏移间隙，约束源域回归退化不超过 $\epsilon$**。

### 3.8 与 LTC / CfC 的关系（**本篇与本仓主线的关系需要说清楚**）

**MoLaRx 不是液态网络论文。** 它的三个骨干里只有 **LFM2-VL-1.6B** 沾"Liquid"的名字，而本仓已在 `LFM2_Technical_Report_2511.23404_研读报告.md` 核实：**LFM2 是 16 层 hybrid（卷积 + GQA 注意力）骨干，40 层配置中零 ODE / 零 CfC / 零 LTC 层**（NR-05）。所以：

- MoLaRx **不提供**任何关于液态连续时间机制的证据；
- 但它提供了一件本仓**更需要的**东西：**一套"逐层结构差异是可诊断、可归因"的实证方法**，以及一组**逐层 CIS 图谱** —— 正是本仓一直在问的"液态变体之间到底哪里不同"的**可操作代理**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. (3) | $P_{paired}=\mathrm{Uniform}\{(x_s,x_t): x_s=\arg\min_x\lVert k(x)-k(x_t)\rVert_2^2\}$ | 场景匹配对：把内容偏移与域偏移分离 |
| Eq. (4) | $G(f)=\mathbb{E}_{(x_s,x_t)}\big[\frac1K\sum_k\lVert f(x_t)_k-f(x_s)_k\rVert_2\big]$ | 域偏移间隙（要最小化的目标） |
| Eq. (5) | $\Psi[m,l]=\mathbb{E}_{(x_s,x_t)}\big[\frac1K\sum_k \lVert f_\Theta(\mathrm{do}(H^{(l)}_{M_m(x_t)}\leftarrow H^{(l)}_{M_m(x_s)});x_t)_k - f_\Theta(x_t)_k\rVert_2\big]$ | **模态–层张量**：逐 (模态,层) 的因果贡献 |
| Eq. (6) | $\Delta_l \in \{\delta_{\theta_l},\; B_lA_l,\; 0\}$ 按区取 | align / tune / freeze 三区各自的参数更新形式 |
| Eq. (7) | $\Delta^*=\arg\min_\Delta G(f_{\Theta'+\Delta})\ \text{s.t.}\ \mathbb{E}_s(f_{\Theta'+\Delta})\le\mathbb{E}_s(f_{\Theta'})+\epsilon$ | 带源域退化约束的最小化问题 |
| Eq. (8) | $d^{(l,m)}_i=\frac1K\sum_k\lVert \tilde Y^{(l,m)}_k - f_{\Theta'}(x^t_i)_k\rVert_2$ | 单对、单车层的反事实位移 |
| Eq. (9) | $\hat\Psi[m,l]=\frac1N\sum_i d^{(l,m)}_i$ | CIS 的无偏蒙特卡洛估计（$N=100$） |
| Eq. (10) | $\Pr[\lvert\hat\Psi-\Psi\rvert>\varepsilon]\le 2\exp\big(\frac{-2N\varepsilon^2}{R^2}\big)$ | Hoeffding 界；$N=100,R\approx20$m ⇒ 95% 半宽 $\approx0.6$m |
| Eq. (11–13) | $\Phi_{align}/\Phi_{tune}/\Phi_{freeze}$ 定义 | 阈值 $\tau$ 取排序 CIS 曲线的肘点 |
| Eq. (14) | $\mathcal{L}_{MMD}=\frac1{|\Phi_{align}|}\sum_{l\in\Phi_{align}}\mathrm{MMD}^2\big(\bar H^{(l)}_{img}(B_s),\bar H^{(l)}_{img}(B_t);k\big)$ | 多尺度 RBF 核的 MMD 对齐 |
| Eq. (15) | $\mathcal{L}_{task}=-\frac1{|S_s|}\sum_{(x,y)}\sum_{t=1}^{T_y}\log p_{\Theta'+\Delta}(y_t\mid x, y_{<t})$ | 监督轨迹损失（驱动 $\Phi_{tune}$ 的 LoRA） |
| Eq. (16) | $\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda_{MMD}\mathcal{L}_{MMD}$，$\lambda_{MMD}=0.5$ | 总损失，梯度按区路由 |

## 5. 关键成果与贡献

### 5.1 实验设置

- **数据集（跨城市，transductive 无监督适应）**：
  - nuScenes：源 = Boston-Seaport (BS)，目标 = Singapore (SG)；约 26.6k 训练 / 3,569 源测试 / 3,500 目标测试
  - Argoverse 2 Sensor：源 = Pittsburgh (PIT)，目标 = Miami (MIA)；从 57 条验证日志抽 858 训练 / 284 ID-测试 / 1,393 OOD-测试
- **三个骨干**：LFM2-VL-1.6B（16 层 hybrid，因果解码器）/ Qwen2-VL-2B（28 层纯 Transformer，因果解码器）/ PaliGemma2-3B（SigLIP + 26 层 Gemma2 prefix-LM，**双向图像注意力**）
- **指标**：ADE（逐 waypoint $\ell_2$，m）/ FDE（末点误差，m）/ $\Delta$ADE（跨域间隙）/ PE（推理延迟 ms）；**5 个随机种子平均**
- **9 个基线**，分四类：无适应参考（Source-only）/ 统一或标量视图 DA（Full LoRA、Full LoRA+MMD、DANN、MMD-only）/ 层级选择性但非因果（LLRD、Freeze-Last-K）/ locate-then-edit 家族（ROME*、Surgical FT）

### 5.2 主结果（Table 1 节选，ADE / $\Delta$ADE 越低越好）

| 数据集 / 骨干 | 指标 | Source-only | MMD-only（‡） | Full LoRA | **MoLaRx** | 领先幅度 |
|---|---|---|---|---|---|---|
| nuScenes / LFM2-VL | ADE | 4.510 | 9.657 | 4.284 | **4.251** | +0.015 |
| nuScenes / LFM2-VL | $\Delta$ADE | 2.550 | 1.646‡ | 3.328 | **3.150** | — |
| nuScenes / Qwen2-VL | ADE | 5.020 | 10.156 | 4.760 | **4.634** | +0.066 |
| nuScenes / PaliGemma2 | ADE | 4.450 | 10.054 | 4.650 | **4.190** | **+0.130** |
| AV2 / LFM2-VL | ADE | 4.953 | 10.179 | 5.180 | **4.983** | +0.062 |
| AV2 / PaliGemma2 | ADE | 4.715 | 8.349 | 4.730 | **4.507** | +0.053 |

**四条结论（作者自己的归纳）**：

1. **MoLaRx 在全部 6 个主表设定上都是最优**，质量指标领先最强非 MoLaRx 基线 **0.015–0.138 m**，多数单元格的边际**超过 $2\sigma$ 显著性阈值**。
2. **增益随"模态–层结构"的丰富度缩放**：最大在 PaliGemma2/nuScenes（+0.130/+0.105 m），最小在 LFM2（+0.015/+0.040 m）—— 与"层越多，暴露给非因果划分的方差越大"的假设一致。
3. **locate-then-edit 家族（ROME*、Surgical FT）在多数单元上优于统一 DA** ⇒ 即使是单点编辑，**因果诊断也是必要的**。
4. 🔴 **统一对齐在结构化偏移上失败**：**MMD-only 把 ADE 翻倍（LFM2/nuScenes: 4.275 → 9.657 m）**；甚至 **Full LoRA + MMD 在 18 个质量单元中的 17 个都比纯 Full LoRA 更差 0.01–0.09 m** —— "对齐一切"**覆写了 $\Phi_{freeze}$ 里的任务不变计算，同时留下了模态特异错误**。

### 5.3 CIS 图谱与分区（Figure 3，nuScenes）

| 骨干 | $\Phi_{align}$ | $\Phi_{tune}$ | $\Phi_{freeze}$ |
|---|---|---|---|
| LFM2-VL-1.6B (L=16) | {0, 1} | {2:9} | {10:15} |
| Qwen2-VL-2B (L=28) | {0:3} | {4:10} | {11:27} |
| PaliGemma2-3B (L=26) | {0:24} | **∅** | {25} |

**（F1）模态优势随层变化**：LFM2-VL 与 Qwen2-VL 中 $\mathrm{CIS}_{img}$ 在 **0–2 层**达到峰值并迅速衰减，而 $\mathrm{CIS}_{txt}$ 在**中深度上升**并占据一段连续带；**PaliGemma2 则 $\mathrm{CIS}_{img}$ 支配所有层**。作者的解释是：LFM2/Qwen2-VL 的**因果注意力能在早期解决跨模态交互**，形成图像主导层与文本主导层的清晰深度分离；而 prefix-LM 注意力下 PaliGemma2 **持续重新注入域偏移特征**。

**（F2）分区是架构相关的**：三个骨干给出三套完全不同的分区。

### 5.4 分区一致性（Table 2，Partition Consistency Score）

**同架构跨数据集**（nuScenes vs AV2）：

| 比较 | PCS(Φ_align) | PCS(Φ_tune) | PCS(Φ_freeze) | 均值 |
|---|---|---|---|---|
| LFM2: nuScenes vs AV2 | 1.00 | 1.00 | 1.00 | **1.00** |
| Qwen2 vs PaliGemma2 | 0.71 | 0.77 | – | — |

⇒ **同一架构、换数据集，分区几乎完全一致**（LFM2 达到 1.00）—— 这是"分区只诊断一次、跨数据集复用"的依据。

### 5.5 效率

- 推理延迟 **47.4–62.1 ms/样本**，比最强 LoRA 基线**快 8.4–10.9 ms**；
- 两个原因：CIS 分区把 LoRA**集中在 $\Phi_{tune}$**，前向开销小于逐层挂 LoRA 的方法；CIS 分区**按架构诊断一次、跨数据集复用、无需重新标定**。

### 5.6 贡献清单

1. **把域偏移形式化为"模态 × 层"二维张量** $\Psi[m,l]$，并给出**逐切片反事实干预**的估计方法。
2. **给出统计保证**：Hoeffding 界 + $N=100$ ⇒ 每条目 95% 半宽 $\approx 0.6$ m，**低于跨分区展开** ⇒ 分区可靠（这一点比多数"重要性评分"工作严谨）。
3. **实测确认分区是架构相关、数据集稳定的**（同架构跨数据集 PCS = 1.00）。
4. **证伪"统一对齐"**：MMD-only 使 ADE 翻倍；Full LoRA+MMD 在 17/18 个单元上劣于纯 Full LoRA。
5. **结构化修复在 6/6 设定上最优**，同时比最强基线少 50–60% 可训练参数。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

论文的自我定位相当克制，主要限制散落在实验设置中：
- 属 **transductive** 设置（用目标测试域输入做适应），且是**无监督** DA（目标域标签全程不用）；
- CIS 分区依赖**场景匹配对** $P_{paired}$，需要内容向量 $k(\cdot)$ 能可靠地匹配跨城场景；
- 阈值 $\tau$ 取肘点，其敏感性在附录 C.2.1；
- 附录 C.4 报告了一个 **inductive 协议对照**（用于说明结论不只依赖 transductive 设定）。

### 6.2 隐含局限（本仓视角）

1. 🔴 **"CIS 差异"≠"该层的液态性质"**：本篇把 LFM2-VL 当作一个**普通 hybrid 骨干**，它的逐层结构来自 **conv/GQA 混合**，与液态时间常数无关。**不能**把"LFM2-VL 的 $\mathrm{CIS}_{img}$ 在 0–2 层峰值"解读为"液态网络的浅层更液态"。
2. **$N=100$ 的蒙特卡洛 + 95% 半宽 0.6 m vs 跨分区展开 0→2.3 m**：统计保证本身是够的（0.6 m 明显小于 2.3 m），但**分区边界的具体位置**（比如 LFM2-VL 究竟是 {0,1} 还是 {0,1,2}）**仍在 0.6 m 的误差带内摆动**。论文把分区当成**离散且可复用**的决策，但它的**边界不确定性从未被传播到下游**（例如没有"$\Phi_{align}$ 边界不确定时应如何稳健地分配算子"的分析）。
3. **消融缺口**：论文**没有做"去掉 Hoeffding 保证的随机 CIS 分区会怎样"的对照**，也**没有把分区随机化**以证明"结构化分区"本身（而非"任意分区 + 三种算子"）贡献了多少。**"MoLaRx 的增益有多少来自定位、多少来自算子匹配"这一拆分未被回答** —— 这恰恰是全篇最关键的归因问题。
4. **收益绝对量小**：领先最强基线仅 **0.015–0.138 m**，而 ADE 本身是 **4.2–5.1 m** 量级 ⇒ **相对改善约 0.3%–2.6%**。作者用 $2\sigma$ 显著性门槛辩护（并诚实标注了哪些边际未达门槛），但**统计显著 ≠ 实用重要**。相比之下，**MMD-only 让 ADE 翻倍（+126%）**才是真正有工程意义的结果。
5. **PaliGemma2 的 $\Phi_{tune}=\varnothing$**：其最好的结果其实来自"几乎全层 MMD 对齐 + 冻结最后一层"，而 MMD-only 在同骨干上**崩溃**（10.054 vs 4.190）—— 说明**"分区"带来的收益与"LoRA 微调"带来的收益在 PaliGemma2 上无法区分**（因为 tune 区为空，方法退化为纯 MMD + freeze）。这是一个**削弱"定位优于统一"论证**的结构性证据，论文未讨论。
6. **不涉及任何液态/连续时间机制**（见 6.2.1），因此对本仓主线的**直接证据价值为零**，只有方法论价值。

### 6.3 未来方向

- 把 CIS 从"模态 × 层"扩展到更细的粒度（头、通道、token）；
- 分区边界不确定性的稳健处理（软分配而非硬三分）；
- 跨架构迁移分区（本文只证明了同架构跨数据集稳定）。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的什么变体**：**不可**。但它给出一个本仓**一直没有的诊断工具**：把模型切成"模态/位置/层"的切片，**逐切片做反事实干预**，度量每个切片对某个宏观量的因果贡献。**本仓可以直接把这个方法迁移到液态 cell 的内部** —— 例如把一个 LTC cell 的内部按 $\tau$ 通道 / 门控分支 / 输入通道切片，用 CIS 式的反事实替换（把某通道的 $\tau$ 换成源域/另一批次的 $\tau$）测出每个通道对输出的因果贡献。**本仓至今的 $\tau$ 变体比较全是端到端 MSE 差**，这正是本篇所批评的"标量视图"。

**可嫁接到哪个 bench 脚本**：本仓 `bench_mdl_cfc.py` / toy benchmark 系列目前只做端到端指标。MoLaRx 的做法可以直接改造成一个**"分层/分通道归因"分析脚本**：对 `lnn/core/variants.py` 的每个变体，输出逐层/逐 $\tau$ 通道的贡献剖面。**这恰好可以回答本仓 11-13 轮 CT-GRU 研读留下的悬案** —— "若数值接近则几十个 $\tau$ 变体可能是一回事的不同写法"：如果逐通道 CIS 剖面显示不同 $\tau$ 变体的剖面**几乎重合**，那就证实了"它们是一回事"；如果剖面**显著不同**，那就说明端到端 MSE 差**掩盖了**内部结构差。

**提供哪个新 ablation axis**：① **反事实切片归因**（本篇核心，可迁移）；② **"对齐一切"的反面消融** —— 本篇证明统一 MMD 对齐在结构化偏移下有害，本仓可以问："**把本仓所有 $\tau$ 变体统一对齐到一个基准 $\tau$ 再比较，是否比各自最优更差？**"这是一个全新的、且与本仓"变体库"直接相关的轴；③ **统计严谨性标准** —— 本篇用 Hoeffding 界证明"分区可靠"，本仓的多变体比较**从未做过任何显著性检验**，这本身就是一个应当补上的**方法论债**。

**Verdict**：
- **对"液态网络在跨域/域偏移任务上更优"**：**TARGET-DEPENDENT-WITH-NUANCE** —— 本篇的骨干是 LFM2（**非液态**），方法是对抗域偏移而非液态机制；**不能**用来支持或否定液态在域偏移上的表现。但它**间接提示**：VLM 规划器的域偏移**结构极强**且**统一对齐有害**，若液态网络要用于跨域场景，**"对齐一切"很可能同样有害** —— 这是一个值得本仓记住的风险提示。
- **对"本仓的多变体比较方法论是否足够"**：**TARGET-NEGATIVE** —— 本篇用 Hoeffding 界 + $2\sigma$ 门槛 + 反事实归因，把"分层定位"的可靠性**量化**；而本仓的多变体比较至今是**裸 MSE 差、无显著性检验、无内部归因**。这是本仓方法论上的一笔**明确负债**，应在下一轮优先偿还。
- **对"逐层/逐结构差异是否可诊断"这一元命题**：**TARGET-POSITIVE** —— 实测证据强（三个架构、两数据集、统计保证、分区跨数据集 PCS=1.00），且**与液态无关地成立**。对本仓最有价值的不是这篇论文的结论，而是它的**方法**。
