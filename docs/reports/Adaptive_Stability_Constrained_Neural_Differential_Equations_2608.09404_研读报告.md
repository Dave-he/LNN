---
title: "Adaptive Stability-Constrained Neural Differential Equations for Controlled Dynamical Systems with Unknown Inputs"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-ODE, Contraction-Analysis, Riemannian-Metric, Stability-Certificate, Incremental-ISS, System-Identification, Preprint]
arxiv_id: 2608.09404
---

# 研读报告：AS-NDE — 把"收缩"写成一个**带输入依赖度量的可计算证书**：以及一篇**把所有数字都标成占位符**的诚实草稿

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描，第七天）捞出来的。本仓**零覆盖**（`2608.09404` 在 180 篇研读报告中零命中）。
> 🔴 **本篇对本仓的意义有三层**：(a) **一个本仓至今零测量的可算量** —— **收缩残差 $S_\Theta=\dot M+A_\theta^\top M+MA_\theta+2\lambda M$ 的 $\lambda_{\max}(S)$ 分布**（配合 11-22 `2611.21030` 的 "$L_fV_\theta<0$ 符号统计" 与 11-23 `2602.11738` 的 "梯度敏感性 $R^2$"，本仓的"证书轴"三件套至此齐了）；(b) **一条本仓实现里已经违反但没人写下来的约束** —— 论文明确指出 $\dot M$ 里**必须保留 $\partial M/\partial u\cdot\dot u$ 这一项**，*"a term that is easily omitted in heuristic stability regularizers"*，而**本仓所有 $\tau$ / 度量型变体都是输入依赖的**（`ltc.py` 的 $\tau_{\text{eff}}=|\tau_0|+f_{\text{out}}+0.01$），**这意味着本仓现有的 Lyapunov / ISS 证书若真要做成收缩证书，缺一项**；(c) **一个本仓必须原样抄进去的"诚实报告纪律"** —— 论文自陈*所有数值都是占位符、实验尚未执行*，并把"若 AS-NDE 降低了残差违规率但没有改善留出 rollout 误差，则结论应是**证书正则化了敏感度而没有展示预测收益**"写成明文规则 ⇒ **本仓 `bench_lyapunov_stable_cfc.py` / `bench_iss_stable_cfc.py` 现有的 H1/H2/H3 假设结构与本文的判定口径完全同构**，可以直接升级为引用本文的"证书 ≠ 预测收益"纪律。

## 1. 元数据

- **论文标题**：Adaptive Stability-Constrained Neural Differential Equations for Controlled Dynamical Systems with Unknown Inputs
- **作者**：Syed Pouladi（单作者）
- **机构**：College of Engineering and Physical Sciences, Khalifa University, Abu Dhabi, UAE
- **发表时间**：2026-08-10 (v1)（arXiv:2608.09404v1 [eess.SY]）
- **发表渠道**：**Preprint / 草稿** —— 论文自陈 *"Because no measured data or executed training runs accompany this draft, all numerical curves and tables are clearly identified as illustrative synthetic placeholders"* ⇒ **未评审、无实验数据**
- **来源**：arXiv:2608.09404v1 [eess.SY]
- **代码**：**无**（论文未给出仓库地址）
- **本地 PDF**：[papers/2608.09404/2608.09404.pdf](../../papers/2608.09404/2608.09404.pdf) ／ 文本 [2608.09404.txt](../../papers/2608.09404/2608.09404.txt)
- **页数**：6 页
- **关联概念**：Neural ODE / **Controlled Differential Equation（控制量显式进入状态方程）** / **Contraction Analysis（Lohmiller–Slotine 1998；Forni–Sepulchre 2014）** / **Control Contraction Metrics** / **Input-to-State Stability** / **Incremental Stability** / **Riemannian 度量下的微分 Lyapunov 证书** / **度量总导数 $\dot M$** / **Newton–Raphson / 幂迭代估计 $\lambda_{\max}$** / 强制 Duffing 振荡器 / PMSM（永磁同步电机 dq 框架）/ **SINDy** / **ICODE** / ControlSynth Neural ODE
- **入选理由**：本仓**零覆盖**"把连续时间稳定性做成**可计算的、与输入耦合的证书**"这一整族；而本仓自 11-22 起连续三轮在索引里写"本仓至今全部是 MSE 证据，缺一份稳定性/证书类证据"，**本文恰好是那份证据的方法论模板**——但**它的所有数字都是占位符**，因此它的价值在**协议与纪律**，不在结果。

## 2. 核心问题

**任务**：从采样轨迹 $\mathcal{D}=\{(t_k,y_k,u_k)\}$ 学习一个**受控**连续时间模型

$$\dot x = f_\star(x,u) + Ew,\qquad y = Cx + v \tag{Eq. 1}$$

其中 $u$ 是**被测量的物理控制输入**，$w\in\mathbb{R}^r$ 是**未知的本质有界扰动**，$v$ 是测量噪声，$E$ 是已知的（或保守取的）扰动注入矩阵。目标是 (i) 辨识标称流 $\dot{\hat x}=f_\theta(\hat x,u)$（Eq. 2），(ii) **限制 rollout 对初值、输入、扰动失配的敏感度**。

论文指出的结构性缺陷有四层：

1. **"自主系统"是建模幻觉**。真实工程系统（电机驱动、电网、化工过程）**状态依赖指令、负载与环境信号**；把 $u$ 吸收进隐藏参数或潜变量会让"预测误差"变成**不可辨识的混合**。论文援引 ICODE 把同时刻的外生输入显式注入动力学作为正确对照。
2. **一步误差在分布外 rollout 中会迅速放大** —— 这是本篇的题眼（*"a small one-step error can grow rapidly when a learned vector field is rolled out under inputs that differ from those used for training"*）。**注意这句话同时是本仓 §0 的核心张力的另一种表述**：本仓关心的"连续时间是否更抗扰"，在这个框架里被拆成"**学习到的模型在输入失配下敏感度多大**"。
3. **收缩分析比 Lyapunov 更适合"预测"** —— 它比较的是**相邻轨迹**而非"轨迹与某个固定平衡点"，这与"长程 rollout"的目标天然对齐。
4. **启发式稳定正则会漏项**。论文点名 $\dot M$ 里 **$\partial M/\partial u\cdot\dot u$ 这一项**，*"is easily omitted in heuristic stability regularizers"* —— 漏掉它会**高估**收缩性（因为少了沿输入变化方向的度量增长）。

于是问题被重述为：**能否构造一个"输入条件化的黎曼度量 $M_\phi(x,u)$"，其正定性由构造成立，其收缩性由一个**采样得到的残差**惩罚，从而给出一条"两条轨迹在控制与扰动不同时如何分离"的**增量 input-to-state 界**——并且明确区分"学到的模型的证书"与"真实物理系统的证书"？**

## 3. 方法论与核心思路

### 3.1 总体架构

```
数据 (y_k, u_k) ──► 插值 u(t) 与 u̇(t) ──► f_θ(x,u) + 度量网络 L_φ(x,u)
                                                    │
                            ┌───────────────────────┴────────────────────┐
                            ▼                                            ▼
              预测损失 (Eq.8)                          收缩残差 S_Θ (Eq.6) ──► L_ctr (Eq.7)
                            │                                            │
                            └──────────► 总损失 L = 预测 + ρ_c·L_ctr + ρ_M·L_metric + ρ_J·L_Jac
```

**关键设计选择**：度量网络**独立于**向量场网络（向量场 4×64 tanh，度量因子网络 3×48 tanh，输出 $n(n+1)/2$ 个下三角元素）⇒ 论文明确写 *"This separation keeps the predictor capacity identical and prevents an apparent benefit caused merely by adding parameters to the vector field."* —— **这条"参数量匹配"纪律本仓的 bench 尚未普遍执行**（11-22 `2611.25274` 与 11-13 `2512.12116` 都提过同一问题）。

### 3.2 核心模块 1：度量参数化与正定性的**构造性**保证

令 $L_\phi(x,u)$ 下三角，对角取 $\mathrm{softplus}(\ell_{ii})+\epsilon_M$，严格下三角无约束：

$$M_\phi(x,u) = L_\phi(x,u)\,L_\phi(x,u)^\top + \epsilon_M I \tag{Eq. 4}$$

⇒ **正定性由构造成立**，不是靠惩罚项逼出来的。论文再叠加谱惩罚把 $mI\preceq M\preceq\bar m I$（Eq. 3 的假设 1）压实。

### 3.3 核心模块 2：度量总导数 $\dot M$（本篇最可移植的一段）

记 $A_\theta=\partial f_\theta/\partial x$，沿标称轨迹的**全导数**：

$$\dot M = \sum_i \frac{\partial M}{\partial x_i}f_{\theta,i}(x,u) + \sum_j \frac{\partial M}{\partial u_j}\dot u_j \tag{Eq. 5}$$

**这一行是本篇对本仓最有价值的技术点**：
- 第二项 $\partial M/\partial u\cdot\dot u$ **要求训练样本含 $\dot u$**（论文：*"training samples must include $\dot u$ or a differentiable interpolation of $u$"*）⇒ 对分段常值指令，论文建议**逐区间处理并在跳变处单独检查度量跳变**（混合系统处理）。
- 对**输入依赖的度量**（本仓的 $\tau$ 机制正是输入依赖）**漏掉该项 = 系统性地高估收缩性**。

### 3.4 核心模块 3：收缩残差与惩罚

$$S_\Theta = \dot M + A_\theta^\top M + MA_\theta + 2\lambda M,\qquad \Theta=(\theta,\phi) \tag{Eq. 6}$$

$$L_{\text{ctr}} = \frac{1}{N_c}\sum_{i=1}^{N_c}\mathrm{softplus}\!\left(\frac{\lambda_{\max}(S_{\Theta,i})}{\tau}\right) \tag{Eq. 7}$$

- 目标是 $S_\Theta$ **半负定**；用 $\lambda_{\max}$ 的 softplus 惩罚（而不是 $\lambda_{\max}^2$，更稳）。
- 大 $n$ 时用**幂迭代**估计最大特征值，不做完整特征分解。
- **配点（collocation）必须是混合的**：观测状态 + rollout 状态 + 局部抖动状态各占一部分（论文 Table II：50% data / 25% rollout / 25% jitter）—— 理由是 *"checking only observed points cannot certify the region between trajectories"*。

### 3.5 核心模块 4：训练目标与收缩权重 warm-up

$$\mathcal{L} = \frac{1}{NH}\sum_{q,k}\big\|W_y(\hat y_{q,k}-y_{q,k})\big\|_2^2 + \rho_c L_{\text{ctr}} + \rho_M L_{\text{metric}} + \rho_J L_{\text{Jac}} \tag{Eq. 8}$$

$$L_{\text{metric}} = \frac{1}{N_c}\sum_i\Big[\max(0,m-\lambda_{\min}(M_i))^2 + \max(0,\lambda_{\max}(M_i)-\bar m)^2\Big] \tag{Eq. 9}$$

**warm-up 是必需的**：论文 Algorithm 1 第 7 步 *"ramp $\rho_c$ from zero after $N_w$"*，理由是 *"otherwise the randomly initialized metric can dominate before the vector field has acquired the direction of the flow"* —— **度量网络先于向量场学坏，会把后续梯度带偏**。

### 3.6 增量 input-to-state 界（定理 1）

**变分动力学**：$\delta\dot x = A_\theta\delta x + B_\theta\delta u + E\delta w$，$B_\theta=\partial f_\theta/\partial u$（Eq. 10）。

**定理 1**：设假设 1 成立且在某个前向不变域上 $\dot M + A_\theta^\top M + MA_\theta\preceq -2\lambda M$。令

$$b_u=\sup\big\|M^{1/2}B_\theta\big\|,\qquad b_w=\sup\big\|M^{1/2}E\big\| \tag{Eq. 11}$$

则两条轨迹的黎曼距离满足

$$d_M(t)\le e^{-\lambda t}d_M(0)+\int_0^t e^{-\lambda(t-\tau)}\big[b_u\|u_1-u_2\|+b_w\|w_1-w_2\|\big]d\tau \tag{Eq. 12}$$

配合假设 1 的谱等价性，退化到欧氏范数：

$$\|x_1(t)-x_2(t)\| \le \sqrt{\frac{\bar m}{m}}\,e^{-\lambda t}\|x_1(0)-x_2(0)\| + \frac{b_u\|u_1-u_2\|_{\infty,[0,t]}+b_w\|w_1-w_2\|_{\infty,[0,t]}}{\lambda\sqrt m} \tag{Eq. 13}$$

**证明骨架**（可复算）：取 $V=\delta x^\top M\delta x$，沿连接路径求导并代入矩阵不等式得 $\dot V\le -2\lambda V + 2\delta x^\top M(B_\theta\delta u+E\delta w)$；在度量下用 Cauchy–Schwarz 得 $D^+\sqrt V\le-\lambda\sqrt V+b_u\|\delta u\|+b_w\|\delta w\|$；积分后再对连接路径取极小得 Eq. 12；最后用谱等价性得 Eq. 13。

**Remark 1（本篇最该被抄进本仓索引的一句）**：

> *"The result certifies the learned model on the domain where the inequality holds; it is not automatically a certificate for the unknown plant. A plant-level claim requires a verified bound on $f_\star-f_\theta$ and its differential, or an independent validation argument. **Sampled residual penalties are empirical regularizers unless completed by formal verification.**"*

### 3.7 与 LTC (Hasani 2021) / CfC / GRU 的关系

- **与 LTC**：LTC 是 $\dot h=-h/\tau_{\text{eff}}+f(\cdot)A$ 的**状态方程**；AS-NDE 是**在状态方程之外再加一层度量与残差约束**。二者可以叠加（把 LTC 的 $\dot h$ 当 $f_\theta$），但论文**没有做**这个组合。
- **与 CfC**：CfC 是**闭式一步更新**（本仓 `cfc.py`：`decay = sigmoid(-f·τ·dt)`，`h' = decay·g + (1-decay)·h_out`），**不经过数值积分器** ⇒ 论文的 $A_\theta=\partial f_\theta/\partial x$ 与"通过求解器反传"的整条分析链**在 CfC 上不直接成立**（不涉及 $R(z)$ 与刚性阻尼），这既是机会（**CfC 可能天然避开本篇所诊断的梯度抑制，见本日另两篇之一 `2508.01519`**）也是障碍（**收缩证书不能照搬到 CfC 上，因为 CfC 不是 ODE 积分**）。
- **与 GRU**：GRU 无度量、无证书；本篇的 $b_u$（输入增益上界）正是 GRU 完全没有的量。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 3 | $mI\preceq M_\phi(x,u)\preceq\bar m I$ | 度量的谱有界性（假设 1）；同时约束条件数 $\kappa(M)\le\bar m/m$ |
| Eq. 4 | $M_\phi=L_\phi L_\phi^\top+\epsilon_M I$ | **正定性由构造成立**（Cholesky 型因子化），不靠惩罚项 |
| Eq. 5 | $\dot M=\sum_i\frac{\partial M}{\partial x_i}f_{\theta,i}+\sum_j\frac{\partial M}{\partial u_j}\dot u_j$ | **度量总导数**；第二项（输入方向）常被漏掉 ⇒ 高估收缩性 |
| Eq. 6 | $S_\Theta=\dot M+A_\theta^\top M+MA_\theta+2\lambda M$ | 收缩残差矩阵；$S_\Theta\preceq0$ 即为收缩条件 |
| Eq. 7 | $L_{\text{ctr}}=\frac1{N_c}\sum_i\mathrm{softplus}(\lambda_{\max}(S_{\Theta,i})/\tau)$ | 残差惩罚；用 $\lambda_{\max}$ 而非平方；大维用幂迭代 |
| Eq. 8 | $\mathcal{L}=\text{预测}+\rho_c L_{\text{ctr}}+\rho_M L_{\text{metric}}+\rho_J L_{\text{Jac}}$ | 总损失；$\rho_c$ 需 warm-up（Algorithm 1 第 7 步） |
| Eq. 11 | $b_u=\sup\|M^{1/2}B_\theta\|,\ b_w=\sup\|M^{1/2}E\|$ | 输入增益与扰动增益的度量范数上界 |
| Eq. 12 | $d_M(t)\le e^{-\lambda t}d_M(0)+\int_0^t e^{-\lambda(t-\tau)}[b_u\|\Delta u\|+b_w\|\Delta w\|]d\tau$ | **增量 input-to-state 收缩界**（定理 1 主体） |
| Eq. 13 | $\|x_1-x_2\|\le\sqrt{\bar m/m}e^{-\lambda t}\|\cdot\|+\frac{b_u\|\Delta u\|_\infty+b_w\|\Delta w\|_\infty}{\lambda\sqrt m}$ | 欧氏形式；显式含 $\lambda$ 与增益比 |
| Eq. 17 | $\mathrm{NRMSE}=\sqrt{\sum\|e\|^2/\sum\|y-\bar y\|^2}$ | 归一化 RMSE；论文明确说**不要只用 test-set range 归一化**（单点离群值敏感） |

## 5. 关键成果与贡献

### 5.1 主结果

⚠️ **本节的所有数字都是论文自标的占位符（illustrative synthetic placeholders），没有任何一行来自已执行的实验。**

论文 Table I 的标题就是 *"ILLUSTRATIVE SYNTHETRIC DRAFT RESULTS—REPLACE BEFORE SUBMISSION"*：

| 方法 | Duffing NRMSE (20-s rollout) | PMSM $i_d$ RMSE (A) | Violation (%) $\lambda_{\max}(S)>0$ |
|---|---|---|---|
| NODE | 0.184 | 0.42 | 61.0 |
| Input-conditioned NODE | 0.112 | 0.31 | 48.0 |
| AS-NDE, fixed $M$ | 0.078 | 0.25 | 9.0 |
| AS-NDE, adaptive $M$ | **0.049** | **0.19** | **2.0** |

**⇒ 这张表不能被引用为任何形式的证据**（包括"AS-NDE 更好"）。它的价值在于**表结构本身**：它把"精度"与"证书违规率"分成两列并行报告 —— 这正是本仓 bench 至今缺的"双列"结构。

论文 Fig. 1/2 的曲线同样标注 *"Coordinates are placeholders, not outputs of an executed experiment."*

### 5.2 方法论贡献（这才是本篇的真实贡献）

1. **一个"证书审计"独立于"预测评估"的协议**（§VI.E）：在识别域的凸包 + 一层薄外壳上抽 $(\text{state},\text{input},\dot{\text{input}})$ 三元组，记录 $r=\lambda_{\max}(S_\Theta)$、$M$ 的最小/最大特征值、条件数，报告 **最大值 + 分位数 + $r>0$ 的比例**；并明说 *"A zero violation rate on finite samples is not a proof."*（若要真正的区域证书，需要区间界传播 / 混合整数验证 / SOS 松弛，且所选技术与其近似误差**成为定理的前提**）。
2. **五个主消融**：去掉 $L_{\text{ctr}}$ / 把 $M_\phi(x,u)$ 换成 $I$ / 用 $M_\phi(x)$（去掉输入依赖）/ **漏掉 $\partial M/\partial u$ 项** / 变 $\lambda$。
3. **四条有效性威胁**（§VI.F）：段泄漏（切分必须先于开窗）、求解器容差偏向更平滑的模型（**精度与算力必须同时报**）、配点密度本身是一种额外监督（基线正则与算力预算必须匹配）、仿真 PMSM 数据可能让解析 dq 结构" unrealistically easy"。
4. **一条明文的诚实负结果规则**：*"If AS-NDE reduces residual violations but does not improve held-out rollout error, the conclusion should be that the selected certificate regularizes sensitivity without demonstrated predictive benefit."*
5. **四条实测建议**（§VI.C）：报 $\lambda_{\max}(S)$ 的最大值与 95 分位、度量条件数、求解器函数求值次数、墙钟成本。

### 5.3 贡献清单

- 🔴 **一条可算量**：$S_\Theta$ 的 $\lambda_{\max}$ 分布 —— 本仓第一次有了"连续时间模型在输入失配下敏感度"的**可计算证书代理量**，且与本仓 11-22 的 $L_fV_\theta$ 诊断、11-23 的 $R^2$ 构成互补三角。
- 🔴 **一条实现纪律**：$\dot M$ 必须保留 $\partial M/\partial u\cdot\dot u$；对输入依赖度量漏掉它 = 高估收缩性。
- 🔴 **一条引用纪律**（Remark 1）：采样残差惩罚是**经验正则**，不是证书；对"学到的模型"的证书**不等于**对"真实对象"的证书。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- **无任何已执行实验**（全篇占位符）；作者的定位是 *"a technically consistent starting point, not as evidence of empirical superiority before the prescribed experiments are run."*
- **度量增加导数与条件数风险**：自适应度量可以比欧氏证书更不保守，但代价是导数与条件数；*"disturbance robustness in (13) depends on $b_w/\lambda$; a small sampled residual is not meaningful if the learned metric is poorly conditioned."*
- **过大的 $\lambda$ 会压掉有用动态或诱发刚性**。
- **结构限制**：*"A globally contracting model cannot represent systems with multiple isolated attractors under a common fixed input."*（⚠️ 见 §7 与本仓多吸引子变体的张力）
- **高维域上的认证仍困难**；分段指令在跳变处需混合系统处理。
- **PMSM 硬件数据缺失**，无法区分"稳定约束带来的改进"与"普通正则化 / 有利仿真假设"。

### 6.2 隐含局限（本仓视角）

- **与 CfC 不兼容的部分**：$A_\theta$ 与残差 $S_\Theta$ 的推导假设"通过数值积分器得到轨迹"，而 CfC 是闭式一步 ⇒ 收缩证书要在 CfC 上重新定义（可以定义，但要小心 $\partial/\partial x$ 与 $\partial/\partial u$ 的链式项）。
- **单一固定 $\lambda$**：论文的 $\lambda$ 是全局常数，而本仓的 $\tau$ 是**逐时间步、输入依赖**的 ⇒ 天然对应一个**状态依赖的 $\lambda(x,u)$**，论文未讨论（这本身是一个可做的扩展）。
- **配点是额外监督**：本仓的 bench 全部用同源数据，一旦引入配点，**"本仓变体 A vs B" 的比较就变成了"谁有更多监督"** —— 论文自己承认这点（威胁 3），本仓若照搬必须同步匹配基线。
- **"占位符论文"本身的方法论风险**：这篇论文是**协议模板**，但它的 Table I 数字（0.184 / 0.049 / 2.0% 等）一旦被下游引用，就是伪造证据的现成模板。**本仓索引必须显式标注为占位符。**

### 6.3 未来方向（作者给出的）

- 在**已释放的轨迹切分**上跑齐基线（plain NODE / input-conditioned NODE / SINDy / AS-NDE），替换全部占位符坐标，报不确定性与残差覆盖率后再谈经验结论。
- 硬件 PMSM 数据；带死区、饱和、参数漂移、传感器噪声、离散控制的仿真器。
- 若需形式化区域证书：区间界传播 / 混合整数验证 / SOS 松弛。

## 7. 对本仓的意义

**（a）可作为 `lnn/core/` 的什么变体？** —— **`ContractionCertificateCfCCell`**：在 `lnn/core/cfc.py` 的 `CfCCell` 之上叠加一个**度量因子网络**（下三角 + softplus 对角 + $\epsilon_M I$），并把 $S_\Theta$ 的 $\lambda_{\max}$ 惩罚加进 loss。**但必须先解决一个前置问题**：CfC 的状态更新是 $h'=d\cdot g+(1-d)h_{\text{out}}$（$d=\sigma(-f\tau\Delta t)$），其"流"不是 ODE ⇒ **度量总导数 $\dot M$ 在 CfC 上没有直接对应物**。⇒ 更诚实的落地路径是：先在 **`lnn/core/ltc.py`（ODE 路线）**上做，因为 `ltc.py` 本来就用 `odeint(rk4)` 积分，`A_\theta=\partial f_\theta/\partial x` 有定义。

**（b）可嫁接到哪个 bench 脚本？** —— `scripts/bench_lyapunov_stable_cfc.py`（现有的 `LyapunovStableCfCCell` 用固定 Lyapunov 矩阵 $P$）与 `scripts/bench_iss_stable_cfc.py`（现有的 ISS 证书）：两者都只报"任务损失 + 证书残差"，**都缺"审计"**。⇒ 建议新增 **`scripts/bench_contraction_certificate_cfc.py`**，按论文 §VI.E 的审计协议报 $\lambda_{\max}(S)$ 的 max / 95 分位 / $r>0$ 比例 + 度量条件数 + **求解器 NFE + 墙钟**（11-22 `2611.25274` 已要求 NFE 与 wall time 单独成列）。

**（c）提供哪个新 ablation axis？** —— 🔴 **"证书强度 vs 预测收益"的分离轴**（本仓此前所有变体比较都是"任务损失"单轴）：

| 变体 | 证书残差 | 预测收益 |
|---|---|---|
| 无证书 | — | 基线 |
| 固定欧氏度量 $M=I$ | 论文消融 (ii) | ? |
| 输入依赖度量 $M(x,u)$，**保留** $\partial M/\partial u\dot u$ | 论文消融 (v) | ? |
| 输入依赖度量，**漏掉** $\partial M/\partial u\dot u$ | 论文消融 (iv) | **本仓现有实现的近似** |

⇒ 第四行是**本仓当前状态**：本仓所有度量/证书型变体的度量都是**状态依赖但通过 $\tau_{\text{eff}}$ 隐式依赖输入**，而 $\partial M/\partial u\dot u$ 项**从未被显式检查**。这一行的结果会直接告诉本仓"现有证书残差是不是被高估了"。

**（d）一条必须进索引的引用纪律**（本文 Remark 1 + §VI.F 合起来）：

> **凡本仓报告"稳定性 / ISS / Lyapunov 证书"，必须同时标注三件事**：(i) 证书只覆盖**学到的模型**在**残差成立的域**上，**不覆盖真实对象**；(ii) 采样残差是**经验正则**，**不是形式化证书**（零违规率 ≠ 证明）；(iii) 若**只降低违规率而未改善留出预测误差**，则结论只能写成"**证书正则化了敏感度，未展示预测收益**"。

⇒ 这条纪律与本仓 11-22 的 "$L_fV_\theta<0$ 只是诊断、不构成保证"**同源且互相加强** —— 这是**连续第二天**出现同一类"证书 ≠ 预测收益"的独立外部证据。

**Verdict**：**"连续时间稳定性可以通过输入依赖度量做成可计算的证书代理量" → TARGET-POSITIVE（方法论层面）** —— 但**证据等级为"协议模板"而非"实证"**（全篇占位符、无代码、未评审），**本仓只能引用其协议与纪律，绝不可引用其数字**。
