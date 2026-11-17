---
title: Graph Neural Rough Differential Equations for Traffic Forecasting
date: 2026-11
tags: [LNN, Continuous-Time, Neural-RDE, Rough-Path, Log-Signature, Spatio-Temporal, Graph-Convolution, Traffic-Forecasting, Irregular-Sampling]
---

# 研读报告：STG-NRDE — 用 log-signature 把交通流预测的时序与空间两个维度都搬进粗糙路径

## 1. 元数据

- **论文标题**：Graph Neural Rough Differential Equations for Traffic Forecasting
- **作者**：Jeongwhan Choi, Noseong Park (Yonsei University, South Korea)
- **发表时间**：2023-03-20 (v1) / 2023-06-06 (v2)，ACM 期刊格式 (J. ACM 37, 4, Article 111)
- **来源**：arXiv:2303.10909v2 [cs.LG]
- **代码**：https://github.com/jeongwhanchoi/STG-NRDE
- **本地 PDF**：[papers/2303.10909/2303.10909.pdf](../../papers/2303.10909/2303.10909.pdf) ／ 文本 [papers/2303.10909/2303.10909.txt](../../papers/2303.10909/2303.10909.txt)
- **数据集**：PeMSD3 / PeMSD4 / PeMSD7 / PeMSD8 / PeMSD7(M) / PeMSD7(L)，共 6 个基准、27 个基线
- **关联概念**：Neural RDE / 粗糙路径理论 / log-signature / Neural ODE / Neural CDE / 图卷积 / 自适应邻接矩阵 / LTC / CfC

## 2. 核心问题

**问题一（本文的主线）**：时空预测的主流做法是把**图卷积（空间）**与 **RNN（时间）**拼接——DCRNN、STGCN、GraphWaveNet、AGCRN、STGODE、STG2Seq 等等，无一例外。但这种拼接有一个未被言明的结构性代价：**RNN 的时间处理是离散的**，而真实交通流的时间结构由采样与传播的自然物理决定。为了在离散 RNN 里塞进连续性，常见做法是**分箱/插补/加 $\Delta t$ 通道**——这正是本仓 11-13 轮 CT-GRU 研读里被独立证伪过的那条路（"从 GRU 移除时间戳仅掉 ≤5%"）。

**问题二（本文解决的空白）**：作者自己指出——**此前没有任何研究把 NRDE（Neural Rough Differential Equation）与图卷积结合**。NRDE 是 Neural ODE 在粗糙路径框架下的连续类比（"NRDEs, which are considered as a continuous analogue to recurrent neural networks"），但**一直只被当作纯时间序列模块使用**，空间维度始终由 GCN 承担。

**问题三（鲁棒性）**：现实中的交通传感器会损坏，某些区域在一段时间内采不到数据。**Table 3 的 27 个基线里，除 STG-NCDE 外全部无法做不规则预测**——这意味着"不规则采样下的时空预测"在文献中几乎是空白。

本文的解决方案是：把**时间维度**（经由 log-signature 变换的粗糙路径）和**空间维度**（经由自适应图卷积）**统一到同一个 NRDE 框架**里，用一个增广 ODE 同时求解。

## 3. 方法论与核心思路

### 3.1 总体架构（两个 NRDE 级联）

```
每个节点 v 的时间序列 {F_i(v)} ──插值──> 连续路径 X(v)
        │
        ├─(1) 时间 NRDE:  H(v)(T) = H(v)(0) + ∫ f(H(v)(t);θ_f) · [LogSig(X)]/(r_{i+1}-r_i) dt
        │        │  堆叠成矩阵 H(t) ∈ R^{|V|×dim(h)}
        │        ▼
        └─(2) 空间 NRDE:  Z(T) = Z(0) + ∫ g(Z(t);θ_g) · [LogSig(H)]/(r_{i+1}-r_i) dt
                 │  g 内部含自适应图卷积（行混合）
                 ▼
              输出层 → Ŷ(v) ∈ R^{S×M}
```

**关键设计**：两个 NRDE **不是串联实现的**，而是合成**一个增广 ODE**（Eq. 14）一次求解。空间 NRDE 的**驱动源不是原始数据，而是时间 NRDE 产生的隐状态轨迹 $H(t)$**——即"时间处理的结果控制空间处理"。

### 3.2 签名与 log-signature（Eq. 4）

对区间 $[r_i, r_{i+1}]$ 内的路径 $X$，其 $k$ 阶签名（signature）为

$$
S^{i_1,\dots,i_k}_{r_i,r_{i+1}}(X) = \int_{r_i<t_1<\dots<t_k<r_{i+1}} \prod_{j=1}^{k}\frac{dX^{i_j}}{dt}(t_j)\,dt_j
$$

全体 $1$ 到 $D$ 阶签名构成 $\mathrm{Sig}^D_{r_i,r_{i+1}}(X)$。但签名**有冗余**：

$$
S^{1,2}_{a,b}(X) + S^{2,1}_{a,b}(X) = S^{1}_{a}(X)S^{2}_{b}(X)
$$

（知道三个就能推出第四个）⇒ 去掉冗余得 **log-signature** $\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)$。

**为什么 log-signature 是关键**：论文引 Morrill et al. 的结果——**对有界变差路径，log-signature 是唯一的**，而深度学习里的绝大多数时间序列正是有界变差的。因此 log-signature 可以当作**路径的唯一特征**。这是本文相对"直接用 $X_s$ 的值"（即 Neural CDE/GRU-ODE 路线）的核心增量：**它把路径的"形状"（含交叉项 $X^iX^j$ 的时序结构）编码进驱动信号**。

### 3.3 Neural RDE（Eq. 5）

$$
Z(T) = Z(0) + \int_0^T f(Z(t);\theta_f)\,\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)}{r_{i+1}-r_i}\,dt,\qquad t\in[r_i,r_{i+1})
$$

其中 $\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)}{r_{i+1}-r_i}$ 是 log-signature 在短区间 $[r_i,r_{i+1})$ 上**时间导数的分段近似**；$D$ 为 log-signature 深度，$P=r_{i+1}-r_i$ 为子路径长度。论文指出：**由于连续读取 log-signature 的时间导数，NRDE 可被视为 RNN 的连续类比**——初始值 $Z(0)$ 与 log-signature 导数序列共同决定 $Z(T)$。同样可用 **adjoint 敏感度方法**训练。

### 3.4 时间处理 NRDE（Eq. 6 / Eq. 7）

逐节点：

$$
H^{(v)}(T) = H^{(v)}(0) + \int_0^T f(H^{(v)}(t);\theta_f)\,\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)}{r_{i+1}-r_i}\,dt
$$

堆叠为矩阵 $H(t)\in\mathbb{R}^{|V|\times\dim(h^{(v)})}$ 得

$$
H(T) = H(0) + \int_0^T f(H(t);\theta_f)\,\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(F)}{r_{i+1}-r_i}\,dt
\tag{Eq. 7}
$$

**RDE 函数 $f$（Eq. 10）**：纯全连接层堆叠（$K$ 层），逐行独立处理：

$$
f(H(t);\theta_f) = \psi\!\left(\mathrm{FC}^{(K)}\big(\boldsymbol{A}_K\big)\right),\quad
\boldsymbol{A}_1 = \sigma\!\left(\mathrm{FC}^{(1)}(H(t))\right),\quad
\boldsymbol{A}_0 = \sigma\!\left(\mathrm{FC}^{(0)}(H(t))\right)
$$

（$\sigma$ = ReLU，$\psi$ = tanh）——**不需要是 RNN**，用全连接层即可，于是 Eq. 7 就是一个"连续版 RNN"。

### 3.5 空间处理 NRDE（Eq. 8 / Eq. 9）

$$
Z(T) = Z(0) + \int_0^T g(Z(t);\theta_g)\,\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(H)}{r_{i+1}-r_i}\,dt
\tag{Eq. 8}
$$

两者合并为单一方程：

$$
Z(T) = Z(0) + \int_0^T g(Z(t);\theta_g)\,f(H(t);\theta_f)\,\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(F)}{r_{i+1}-r_i}\,dt
\tag{Eq. 9}
$$

**RDE 函数 $g$（Eq. 11–13）**：这是图卷积被塞进 NRDE 的地方。设 $E\in\mathbb{R}^{|V|\times C}$ 为可训练节点嵌入矩阵，

$$
g(Z(t);\theta_g) = \psi\!\left(\mathrm{FC}^{(1)}(B_1)\right)
\tag{Eq. 11}
$$

$$
B_1 = \big(I + \phi(\sigma(E\cdot E^\top))\big) B_0 W_{\text{spatial}}
\tag{Eq. 12}
$$

$$
B_0 = \sigma\!\left(\mathrm{FC}^{(0)}(Z(t))\right)
\tag{Eq. 13}
$$

其中 $\phi$ = softmax，$W_{\text{spatial}}$ 可训练。论文明确指出：$\phi(\sigma(E E^\top))$ **概念上对应归一化自适应邻接矩阵** $D^{-1/2}AD^{-1/2}$（$A=\sigma(EE^\top)$，softmax 起归一化作用），且 **Eq. 12 等价于图卷积运算的一阶 Chebyshev 多项式展开**。**Eq. 11 与 13 不混合行；只有 Eq. 12 混合行**——这就是"空间处理"在数学上的确切位置。

### 3.6 训练：增广 ODE（Eq. 14）

$$
\frac{d}{dt}\begin{pmatrix} Z(t)\\ H(t)\end{pmatrix} =
\begin{pmatrix}
g(Z(t);\theta_g)\,f(H(t);\theta_f)\,\dfrac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)}{r_{i+1}-r_i}\\[1ex]
f(H(t);\theta_f)\,\dfrac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)}{r_{i+1}-r_i}
\end{pmatrix}
$$

初值：$H(0)=\mathrm{FC}_{D\to\dim(h)}(F_{t_0})$，$Z(0)=\mathrm{FC}_{\dim(h)\to\dim(z)}(H(0))$。输出层（Eq. 15）：$\hat y^{(v)} = Z^{(v)}(T)W_{\text{output}} + b_{\text{output}}$。损失（Eq. 16）为 $L_1$：

$$
L = \frac{1}{|V|\times|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\sum_{v\in V}\big\|y(\tau,v)-\hat y(\tau,v)\big\|_1
$$

### 3.7 与 LTC / CfC / NCP 的关系（本仓视角）

- **与 LTC / CfC 的关系 = "数据驱动"这一维度上的严格增强**。LTC/CfC 的向量场是**自治**的（只依赖 $x$ 与输入 $I$），$\Delta t$ 仅作积分长度；STG-NRDE 的驱动信号是 $\frac{\mathrm{LogSig}^D(X)}{r_{i+1}-r_i}$，**依赖路径形状而非路径取值**——这与 Neural CDE（2005.08926）的 $dX_s$ 属同一族机制。
- **与 NCP 的关系 = 家族不同**。NCP 是离散稀疏接线的固定时间常数模型；STG-NRDE 是连续受控流 + 图结构。
- **一个本仓必须记录的区分**：STG-NRDE 的 log-signature 变换**比 Neural CDE 的 $dX_s$ 更进一步**——$dX_s$ 只保留一阶增量，而 log-signature 保留了路径的**二阶及以上交叉项**（$X^iX^j$ 的时序结构）。**这是"路径形状编码"的两个层级**，本仓若要评估"连续时间机制的收益来自哪一层"，这篇提供了一个可比的第三档位。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1 (NCDE) | $Z(T)=Z(0)+\int_0^T f(Z(t);\theta_f)\,dX(t)$ | 受控微分方程：隐藏态被路径 $X$ 驱动 |
| Eq. 2 | $Z(T)=Z(0)+\int_0^T f(Z(t);\theta_f)\frac{dX(t)}{dt}dt$ | 用路径导数改写 ⇒ 可用标准 ODE 求解器 |
| Eq. 4 | $S^{i_1,\dots,i_k}_{r_i,r_{i+1}}(X)=\int_{r_i<t_1<\dots<t_k<r_{i+1}}\prod_{j=1}^k\frac{dX^{i_j}}{dt}(t_j)dt_j$；$\mathrm{LogSig}^D$ 为去冗余后的签名 | **签名 / log-signature**：路径形状的唯一特征 |
| Eq. 5 | $Z(T)=Z(0)+\int_0^T f(Z(t);\theta_f)\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(X)}{r_{i+1}-r_i}dt$ | **Neural RDE**：驱动源是 log-signature 的导数 |
| Eq. 7 | $H(T)=H(0)+\int_0^T f(H(t);\theta_f)\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(F)}{r_{i+1}-r_i}dt$ | 时间处理 NRDE（矩阵形式，逐行独立） |
| Eq. 9 | $Z(T)=Z(0)+\int_0^T g(Z(t);\theta_g)f(H(t);\theta_f)\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(F)}{r_{i+1}-r_i}dt$ | **时空统一 NRDE** |
| Eq. 10 | $f(H(t);\theta_f)=\psi(\mathrm{FC}^{(K)}(\boldsymbol{A}_K))$，$\boldsymbol{A}_1=\sigma(\mathrm{FC}^{(1)}(H(t)))$ | 时间侧 RDE 函数（纯 FC） |
| Eq. 12 | $B_1=\big(I+\phi(\sigma(E\cdot E^\top))\big)B_0W_{\text{spatial}}$ | **空间侧 RDE 函数**：自适应邻接矩阵 + 一阶 Chebyshev 图卷积 |
| Eq. 14 | $\frac{d}{dt}\binom{Z(t)}{H(t)}=\binom{g(Z;\theta_g)f(H;\theta_f)\Lambda}{f(H;\theta_f)\Lambda}$，$\Lambda=\frac{\mathrm{LogSig}^D(X)}{r_{i+1}-r_i}$ | **增广 ODE**：一次求解两个 NRDE |
| Eq. 16 | $L=\frac{1}{|V|\times|\mathcal{T}|}\sum_\tau\sum_v\|y(\tau,v)-\hat y(\tau,v)\|_1$ | $L_1$ 训练目标 |
| Eq. 20 | $Z(T)=Z(0)+\int_0^T g(Z(t);\theta_g)\frac{\mathrm{LogSig}^D_{r_i,r_{i+1}}(F)}{r_{i+1}-r_i}dt$ | "Only spatial" 消融（直接用数据驱动、跳过时间 NRDE） |

## 5. 关键成果与贡献

### 5.1 主结果：6 数据集 × 27 基线（Table 1 汇总 + Table 3/4 明细）

STG-NRDE 在**全部 6 个数据集**上取得最优。摘录 PeMSD3（MAE / RMSE / MAPE）：

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| STGCN | 17.55 | 30.42 | 17.34% |
| DCRNN | 17.99 | 30.31 | 18.34% |
| GraphWaveNet | 19.12 | 32.77 | 18.89% |
| AGCRN | 15.98 | 28.25 | 15.23% |
| TrafficTransformer | 16.39 | 27.87 | 15.84% |
| **STG-NRDE** | **15.50** | **27.06** | **14.90%** |
| STG-NCDE（前作） | 15.57 | 27.09 | 15.06% |

**关键读法（诚实边界）**：**相对最强基线 AGCRN，STG-NRDE 在 PeMSD3 上的 MAE 改善仅 0.48（15.98 → 15.50，约 3.0%）**；相对自己的前作 STG-NCDE 仅 0.07。**相对 STGODE 等前人的 ODE 路线基线优势明显**，但相对 SOTA 的 GNN 基线**增益是"非平凡但不巨大"**。论文摘要用词 "outperforming all those 27 baselines by non-trivial margins" —— 在 PeMSD3 上 3% 的 MAE 差距称 "non-trivial" 偏强，引用时宜按**小幅领先**表述。

### 5.2 主结果：不规则采样鲁棒性（Table 6 / Table 7，本文最有价值的部分）

随机丢弃 10% / 30% / 50% 的传感值（每节点独立）：

| Missing rate | Model | PeMSD3 MAE | PeMSD4 MAE | PeMSD7 MAE | PeMSD8 MAE |
|---|---|---|---|---|---|
| 10% | STG-NCDE | 15.89 | 19.36 | 20.65 | 15.68 |
| 10% | **STG-NRDE** | **15.62** | **19.10** | **20.61** | **15.57** |
| 30% | STG-NCDE | 16.08 | 19.40 | 20.76 | 16.21 |
| 30% | **STG-NRDE** | **15.64** | **19.39** | **20.74** | **15.47** |
| 50% | STG-NCDE | 16.50 | 19.98 | 21.51 | 16.68 |
| 50% | **STG-NRDE** | **16.14** | **19.54** | **21.25** | **15.95** |

**关键读法**：STG-NRDE 在 10%–50% 缺失下**退化幅度明显小于 STG-NCDE**（例如 PeMSD8 从 15.68→16.68 涨 1.00，而 STG-NRDE 从 15.57→15.95 涨 0.38；相对完整数据基线 15.32，STG-NRDE 在 50% 缺失下仅退化 4.1%）。**但绝对差距很小**（MAE 差 0.15–0.73）。论文原话：*"our model's performance is not significantly degraded"*（相对完整数据情形），这个说法**只对 STG-NRDE 自身成立**，相对 STG-NCDE 的优势则很小。

### 5.3 消融：log-signature 的贡献（Table 5）

"Only spatial" 模型加/不加 log-signature：

| Model | PeMSD3 MAE/RMSE | PeMSD4 MAE/RMSE | PeMSD7 MAE/RMSE | PeMSD7(M) MAE |
|---|---|---|---|---|
| w/o LS | 15.92 / 27.17 | 19.86 / 31.92 | 21.72 / 34.73 | 2.77 |
| w/ LS | 15.79 / 26.88 | 19.46 / 31.41 | 21.01 / 34.09 | 2.67 |

**关键读法**：log-signature 带来**一致但很小**的改善（MAE 0.13–0.71）。**注意一个例外**：PeMSD3 的 MAPE 在 w/o LS 时是 15.14%，w/ LS 反而是 15.45%（**变差**）；PeMSD8 的 MAPE 也从 11.27% 变到 11.37%。论文原文承认："performance improves when log-signature is used in all traffic flow benchmark datasets **except MAPE for PeMSD3 and PeMSD8**"。

### 5.4 消融：时间 vs 空间（Table 6/7 的 "Only Temporal" / "Only Spatial"）

"Only Temporal" 在不规则数据上**崩溃**（PeMSD3 10% 缺失时 MAE 58.41 vs 完整模型 15.62，**3.7× 恶化**），而 "Only Spatial" 仅轻微退化（16.62）。论文结论：**空间处理贡献更大**，但两者都需要。

### 5.5 敏感性（§4.6）

- **节点嵌入维度 $C$**：PeMSD4 / PeMSD7(M) / PeMSD7(L) 在 $C=6$ 后稳定，$C=10$ 最优；PeMSD8 在 $C=3$ 最优。
- **log-signature 深度 $D$**：多数数据集 $D=2$ 最优，PeMSD7(L) 例外（$D=4$ 更好）。
- **GNN 层数**：**层数增加性能反而下降**；基线与 STG-NRDE 都在 **2 层**时最好。论文解释：局部范围优于大范围邻域信息。

### 5.6 贡献清单

1. **首次把 NRDE 与图卷积结合**（作者自称 "to our knowledge, the first proposing NRDE-based spatio-temporal models"），并给出一个统一的增广 ODE 表述（Eq. 14）。
2. **把 log-signature 作为路径形状的唯一特征引入时空预测**，并给出 $D$（深度）与 $C$（嵌入维度）的敏感性分析。
3. **在不规则采样下给出实证鲁棒性**（10%–50% 缺失），且**保持模型设计不变**——这是 Table 3 中 27 个基线里除 STG-NCDE 外都做不到的。
4. **自适应邻接矩阵 + 一阶 Chebyshev 图卷积嵌入 ODE 向量场**（Eq. 12），不依赖预定义邻接矩阵。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

1. **不处理动态图**：论文明确声明 "our study does not focus on time evolution graphs"，尤其不处理节点数变化的动态图。
2. **27 个基线里多数不支持不规则预测**，故 Table 6/7 的对比只在 STG-NCDE 与自身消融之间进行——**没有第三方不规则预测基线可对照**。
3. **ACM 期刊模板的元数据错误**：论文 PDF 的 ACM Reference Format 写 "Jeongwhan Choi and Noseong Park. 2018. ... J. ACM 37, 4, Article 111 (August 2018)"，而 arXiv 版本是 2023 年 —— **模板占位符未改**（含 "https://doi.org/XXXXXXX.XXXXXXX"）。引用时以 arXiv 日期为准。

### 6.2 隐含局限（本仓视角）

1. **没有与 LTC / CfC 做对照**：全部 27 个基线里**没有一个是液态家族成员**。因此本文**不能**被引用为"NRDE 优于 LTC/CfC"或"LNN 家族落后于 NRDE"的证据。
2. **等间隔数据上的增益有限**：在**完全等间隔**（5 分钟粒度）的标准基准上，相对 AGCRN 的 MAE 优势仅约 3%。本文真正的差异化在**不规则场景**，而这恰恰与本仓的关切重合。
3. **log-signature 变换的代价未被充分量化**：计算 log-signature（尤其深度 $D\ge2$）的开销在论文中**没有报告**（无参数量、无训练时长、无 wall-clock 对比）。对一个声称"高效"的连续时间方法，这是**关键的缺失指标**。
4. **消融的归因不干净**：log-signature 的改善（MAE 0.13–0.71）与整个 NRDE 框架的改善（vs AGCRN 约 3%）**数量级相当甚至更大**。也就是说，"NRDE 框架的收益"与"log-signature 的收益"**无法干净分离**——STG-NCDE（前作、无 LS）已经拿到了几乎全部的增益。这与本仓 11-16 轮记录的"归因不干净"是同一类问题。
5. **"Only Temporal 崩溃"这一现象的解读需谨慎**：它可能反映的是**空间信息缺失**（交通流本质是强空间相关），而不是"时间连续性无用"。把它读成"时间建模无用"是错误的。
6. **$P$（子路径长度）与采样间隔的关系未被讨论**：论文取 $P=r_{i+1}-r_i$（即子路径 = 相邻观测间隔），这在 5 分钟等间隔数据上意味着 $P$ 固定。**在不规则场景下 $P$ 是否随间隔变化、是否需要重采样对齐**，论文没有交代。

### 6.3 未来方向

- 扩展到**动态图**（节点数变化、时间演化拓扑）——作者自陈的开放方向。
- 把 NRDE 与**更多空间模块**结合（当前只有自适应图卷积）。
- 量化 log-signature 深度 $D$ 与计算成本的关系。

## 7. 对本仓的意义

**可作为 `lnn/core/variants.py` 的什么变体**：新增一个 **LogSigDrivenCfCCell**。本仓现有 `cfc.py` 的驱动源是 $\Delta t$（仅"积分了多久"），`ltc.py` 是 $\Delta t$ 积分的 ODE 版本。本篇给出的第三档驱动源是 $\frac{\mathrm{LogSig}^D(X)}{r_{i+1}-r_i}$。落地方式：把每步输入 $x_t$ 与时间戳 $t$ 构成二维路径，展开到深度 $D=2$ 的 log-signature（$1$ 阶项 + 2 阶交叉项 $x^ix^j$），作为 CfC 的 $f$ 分支的额外输入。**注意 $D=2$ 正是本文的实测最优点**。

**可嫁接到哪个 bench 脚本**：本仓 toy benchmark 目前只有等间隔采样，本篇提供**两套可直接照搬的协议**：
1. **不规则采样协议**（§5.2）：按节点独立随机丢弃 10%/30%/50% 观测，报告退化曲线（**而不是只报一个点的 MSE**）——这正好补上本仓索引 11-16 轮自己点名的缺口（"本仓 toy benchmark 全是干净信号 + MSE，至今没有一条扰动-恢复曲线"）。
2. **$D$ 与 $C$ 敏感性协议**（§5.5）：$D=2$ 最优、$C$ 在 6–10 稳定 —— 这两个超参本仓的液态变体库里**完全没有对应旋钮**。

**提供哪个新 ablation axis**：🔴 **"驱动信号层级轴"** —— 这是本仓至今**从未区分过**的三档：

$$
\text{轴：}\quad
\{\ \text{① }\Delta t\text{ 仅作积分长度（本仓现状）},\quad
\text{② }\Delta t\text{ 作标量输入通道（LTC/CfC/GRU-}\Delta t\text{）},\quad
\text{③ 路径导数驱动（Neural CDE 的 }dX_s\text{）},\quad
\text{④ log-signature 驱动（本文，保留二阶交叉项）}\ \}
$$

本仓 11-13 轮的 CT-GRU 研读已经证明"②档相对①档无优势（甚至更差）"，但**③④档从未被本仓任何 bench 测过**。本文与 2005.08926 恰好提供了③④两档的完整实现细节与对照协议，**这是当前本仓最具体、最可执行的一条实验债**。

**Verdict**：
- **TARGET-POSITIVE（对"连续时间机制在不规则采样场景有实质价值"这一命题）**：在**真正不规则**的设置下（10%–50% 缺失、每节点独立），连续路径方法的优势是**结构性的**：Table 3 的 27 个基线中 26 个**根本做不了**不规则预测；本文与 STG-NCDE 都能做，且退化幅度显著更小。这与本仓 11-06/11-07/11-11/11-13 四轮独立研读得到的共同结论（"收益来自资源效率/工程可用性，而非连续时间机制本身"）**方向相反**，值得重视——**但差别在于：本文处理的是真·不等间隔 + 缺失，而本仓此前研读的多数论文处理的是"名义不规则、实际等间隔"**。这条差异本身就是本仓今后选论文时必须先核对的第一项。
- **TARGET-NEGATIVE-WITH-NUANCE（对"log-signature / 粗糙路径是必需的"）**：log-signature 的独立贡献（Table 5）**一致但很小**（MAE 0.13–0.71），且在 PeMSD3/PeMSD8 的 MAPE 上**变差**；更重要的是，**去掉 LS 的 STG-NCDE（前作）已经拿到了几乎全部的增益** ⇒ **"log-signature 是关键"的叙事被数据削弱**，真正起作用的是"把路径导数放进向量场"这件事本身（这正是 2005.08926 的 Neural CDE 所证明的）。引用本文时**不得**把增益归因于 log-signature 的二阶交叉项。

**引用纪律**：① 必须带"**相对最强基线 AGCRN 的 MAE 增益仅约 3%**"这一限定（"non-trivial margins" 的原始措辞偏强）；② 必须带"**27 个基线中无一是液态家族成员**"；③ 必须带"**log-signature 的独立贡献很小且部分指标变差**"；④ 论文的 ACM 模板日期为占位符错误（写 2018），**引用日期以 arXiv 2023-03-20 为准**。
