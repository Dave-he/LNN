---
title: "Parameter-Efficient Neural CDEs via Implicit Function Jacobians"
date: 2026-11
tags: [LNN, Continuous-Time, Neural-CDE, Parameter-Efficiency, Implicit-Function-Theorem, Jacobian, Preprint]
---

# 研读报告：Jacobian NCDE — 把"连续 RNN"写成**隐函数方程**，用隐函数定理 + Taylor 展开把 $d\times u\times v$ 的参数量砍掉一半

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描，第八天）捞出来的。本仓**零覆盖**（`2512.20625` 在 183 篇报告中零命中）。
> ⚠️ **本篇篇幅极短（3 页 / 7286 字符），是本轮三篇里最弱的一篇**，但**它落在一个本仓完全空白、且本仓正在推进的交叉点上**：(a) 11-19 研读的 `2607.05280` 证明了**闭式解与连续时间无绑定**（线性化向量场 ⇒ 矩阵指数 + parallel scan，**相对 NCDE 约 1500× 加速**），本篇给出**另一条独立的降参路径**（不改求解器、改参数化）；(b) 本篇把 **Neural CDE 与"真正的连续 RNN"的关系说清楚了** —— 这是本仓索引里一直含糊的一个词（"Continuous RNN"）；(c) 🔴 **本篇的 Jacobian 计算方式对本仓有一条直接的、此前没写下来的实现约束**：它显式区分 $J_x=\partial f/\partial X$（**输入方向**）与 $J_h=\partial f/\partial h$（**隐状态方向**），而本仓 `lnn/core/controllability_cfc.py` 的 `input_jacobian_norm` **只算了 $J_x$** ⇒ 本仓至今**从未测量过 $J_h$**，而 $J_h$ 正是本篇降参的关键（也是 11-24 强调的"度量总导数必须保留 $\frac{\partial M}{\partial u}\dot u$"同类问题的另一面）。

## 1. 元数据

- **论文标题**：Parameter-Efficient Neural CDEs via Implicit Function Jacobians
- **作者**：Ilya Kuleshov, Alexey Zaytsev
- **机构**：Applied AI Institute（两人同属）
- **发表时间**：2025-11-25 (v1)（arXiv:2512.20625v1 [cs.LG]）
- **发表渠道**：**Preprint**（正文首页自标 *"Preprint. Under review."*）⇒ **未评审**
- **代码**：**论文未给出仓库链接**（仅引用 `patrick-kidger/torchcde` 的示例文件与 `timeseriesclassification.com`）⇒ **不可复现**
- **本地 PDF**：[papers/2512.20625/2512.20625.pdf](../../papers/2512.20625/2512.20625.pdf) ｜ 文本 [papers/2512.20625/2512.20625.txt](../../papers/2512.20625/2512.20625.txt)（3 页，7286 字符）
- **关联概念**：连续时间机制层 / Neural CDE / 参数效率 / 隐函数定理 / Jacobian 向量积

## 2. 核心问题

Neural CDE（Kidger et al. 2020）是处理不规则时间序列的主力模型，其隐状态轨迹由下述 Cauchy 问题定义：

$$\dot h_t=f_\theta(X_t,h_t)\,\dot X_t \tag{Eq.1}$$

其中 $X_t$ 是输入插值，$f_\theta:\mathbb{R}^u\times\mathbb{R}^v\to\mathbb{R}^{u\times v}$ 是神经网络。

论文指出这个定义有一个**立即可见的结构性缺陷**：因为 $\dot h_t$ 必须是**向量**，所以 $f_\theta\dot X_t$ 也必须是向量 ⇒ **$f_\theta$ 必须是一个矩阵** ⇒ 权重维度 $\sim d\times u\times v$（$d$ = 隐层维度）⇒ **参数量被最后一层矩阵主导，且随通道维度呈三次增长**。

论文的第二个观察是一个**概念澄清**：现有的"连续 RNN"并不真的连续 —— Neural CDE **不以 RNN 为骨干**，GRU-ODE 构造的是**差分方程**而非微分方程。**没有任何工作真正把 RNN 的差分步直接取连续极限 $\Delta t\to 0$**。

⇒ 问题被重新表述为：(i) 真正的连续极限 $h_t=\mathrm{RNN}(X_t,h_t)$ 该怎么高效地数值求解？(ii) 顺带地，能否因此省掉那个三次方增长的末层矩阵？

## 3. 方法论与核心思路

### 3.1 总体架构

```
离散 RNN 步:   h_i = RNN(x_i, h_{i-1})          (Δt = 1 的差分方程)
                     │ 替换 Δt、索引 i→t、x_i→插值 X_t
                     ▼
连续极限:       h_t = RNN(X_t, h_t)              ← 隐式方程 (Eq.2)
                     │ 对 t 求导
                     ▼
隐式方程组:     (I − J_h) ħ_t = J_x Ẋ_t        (J_h = ∂f/∂h, J_x = ∂f/∂X)
                     │ Taylor 展开
                     ▼
可计算形式:     ħ_t ≈ J_x Ẋ_t + J_h J_x Ẋ_t + O(‖J_h‖²)   (Eq.3)
                     │
                     ▼
Algorithm 1:  gx ← J_x Ẋ_t  (JVP)
              gxh ← J_h gx   (JVP)
              RETURN gx + gxh
```

### 3.2 模块一：真正的连续 RNN（隐式方程）

把离散 RNN 步写成连续极限：

$$h_t=\mathrm{RNN}(X_t,h_t) \tag{Eq.2}$$

**注意这个方程是隐式的** —— $h_t$ 出现在右边。论文的全部技术都在处理这个隐式性。

### 3.3 模块二：隐函数定理 —— 求解 $\dot h$

对 Eq.(2) 关于 $t$ 求导：

$$\dot h_t=\frac{df}{dX}\dot X_t+\frac{df}{\partial h}\dot h_t$$

记 $J_x\coloneqq\frac{\partial f}{\partial X}$，$J_h\coloneqq\frac{\partial f}{\partial h}$，整理得

$$(I-J_h)\dot h_t=J_x\dot X_t\quad\Longrightarrow\quad \dot h_t=(I-J_h)^{-1}J_x\dot X_t$$

**问题**：$(I-J_h)^{-1}$ 极其昂贵 ⇒ **无法用于数值求解器**。

### 3.4 模块三：Taylor 展开 —— 关键的一步

对 $(I-J_h)^{-1}$ 做 Taylor 展开：

$$(I-J_h)^{-1}=I+J_h+J_h^2+\cdots$$

代入并只保留一阶项：

$$\boxed{\;\dot h_t=J_x\dot X_t+J_hJ_x\dot X_t+O\big(\|J_h\|^2\big)\;} \tag{Eq.3}$$

⇒ **只需要两个 Jacobian 向量积（JVP），完全不需要矩阵求逆**：
- $g_x=J_x\dot X_t$ —— 对 $\dot X_t$ 求 JVP；
- $g_{xh}=J_hg_x$ —— 再对 $g_x$ 求一次 JVP。

**Algorithm 1**（论文原文）：

```
INPUT  t, h_t
       X_t, Ẋ_t ← Interpolation(t|S)     ▷ 同时求值与导数
       gx  ← J_x Ẋ_t                     ▷ JVP
       gxh ← J_h gx                      ▷ JVP
RETURN  gx + gxh
```

**注意**：这**不是**一个离散递推式，而是**给 ODE 求解器用的向量场** —— 论文的 Eq.(3) 定义的是 $\dot h_t$，因此整个机制仍走标准的 ODE 求解路径。

### 3.5 模块四：手工 Jacobian

论文采用 Elman 式 RNN 结构：

$$f_\theta(X_t,h_t)=\tanh\big(W_2\,\mathrm{ReLU}(W_hh_t+W_xX_t+b_1)+b_2\big)$$

两个 Jacobian 可**手工写出**（无需 autodiff 求全 Jacobian）：

$$J_x=\tanh'\cdot W_2\cdot\mathrm{ReLU}'\cdot W_x,\qquad J_h=\tanh'\cdot W_2\cdot\mathrm{ReLU}'\cdot W_h$$

**⚠️ 关键实现细节**：$\mathrm{ReLU}$ 的二阶导为零，会让反向传播"卡住"，论文采用 SNN 文献中的 **surrogate gradient** 技巧 —— **把 ReLU 的导数替换成 sigmoid**。

**参数量对比**（这是本篇标题的由来）：原矩阵式 NCDE 的末层权重是 $d\times u\times v$；本篇的 $W_x$、$W_h$ 是**两个独立的矩阵**（而非一个大矩阵）⇒ 参数量减半。

### 3.6 与 LTC / CfC / NCP 的关系

- **与 LTC（Hasani 2021）**：LTC 的 ODE 是**显式**的（$\dot x=-[1/\tau+f]\odot x+f$），**没有隐式性** ⇒ 本篇的隐式/Jacobian 机制**与 LTC 无关**。⚠️ 反过来：**LTC 的显式性正是它相对 NCDE 的优势之一**（NCDE 要解 CDE，LTC 只要解 ODE）。
- **与 CfC（Hasani 2022）**：CfC 是闭式解，**无求解器、无隐式性** ⇒ 本篇机制**完全不适用**。
- **与 NCP**：无关。
- **与本仓 `lnn/core/controllability_cfc.py`**：🔴 **本篇最直接的一条命中** —— 本仓的 `input_jacobian_norm` 只算 $\partial h_{t+1}/\partial x_t$（$J_x$），**从未算 $\partial h_{t+1}/\partial h_t$（$J_h$）**；而本篇证明 $J_h$ 才是决定"隐式性/参数量/稳定性"的关键量（$I-J_h$ 可逆性直接决定 $\dot h$ 是否爆炸）。⇒ 本仓应当补测 $J_h$ 的谱半径。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq.(1) | $\dot h_t=f_\theta(X_t,h_t)\dot X_t$ | 标准 Neural CDE（论文指出 $f_\theta$ 必须是矩阵 ⇒ 参数量 $\sim d\times u\times v$） |
| Eq.(2) | $h_t=\mathrm{RNN}(X_t,h_t)$ | **真正的连续 RNN**（隐式方程） |
| — | $(I-J_h)\dot h_t=J_x\dot X_t$ | 隐函数定理的直接推论（$J_x=\partial f/\partial X$，$J_h=\partial f/\partial h$） |
| — | $(I-J_h)^{-1}=I+J_h+J_h^2+\cdots$ | Taylor 展开 |
| Eq.(3) | $\dot h_t=J_x\dot X_t+J_hJ_x\dot X_t+O(\|J_h\|^2)$ | **核心公式**：一阶近似，只需 2 次 JVP |
| Algorithm 1 | $g_x\leftarrow J_x\dot X_t$；$g_{xh}\leftarrow J_hg_x$；返回 $g_x+g_{xh}$ | 每个时间步的向量场计算（2 次 JVP） |
| — | $J_x=\tanh'\cdot W_2\cdot\mathrm{ReLU}'\cdot W_x$，$J_h=\tanh'\cdot W_2\cdot\mathrm{ReLU}'\cdot W_h$ | 手工 Jacobian |
| — | $\mathrm{ReLU}'\ \text{替换为}\ \sigma'$ | surrogate gradient 技巧（避免 ReLU 二阶导为 0） |

## 5. 关键成果与贡献

### 5.1 主结果（作者自报，**唯一一个表**）

**Table 1 —— CharacterTrajectories 数据集，4 个种子取平均 ± 标准差**：

| Name | Matrix Neural CDE | Jacobian Neural CDE |
|---|---|---|
| Metric | 0.93 ± 0.02 | **0.94 ± 0.01** |
| # Params | **66K** | **33K** |

⇒ **准确率持平（0.93 vs 0.94，差值远小于标准差），参数量减半。**

### 5.2 机制分析

- **参数量的来源被显式定位**：论文指出 NCDE 的参数量"**由最后那个权重矩阵主导，随通道维度呈三次增长**"，并指出其根本原因是 $\dot h$ 必须是向量 ⇒ $f_\theta$ 必须是矩阵 ⇒ 把 $W_x$ 与 $W_h$ **拆成两个矩阵**即得 $\sim 2\times$ 节省。
- **与 torchcde 官方实现对齐**：作者明确说本方法的架构"与 torchcde 的 Matrix-based Neural CDE 一致"，并把 baseline 取自 `torchcde/example/time_series_classification.py` ⇒ 对照是公平的（**同架构族**）。

### 5.3 贡献清单

1. **"连续 RNN"这个概念的正确形式化**（Eq.2）：把 RNN 的差分步直接取 $\Delta t\to 0$，并指出既有工作（NCDE、GRU-ODE）**都不是**真正的连续 RNN。
2. **隐式向量场的可计算形式**（Eq.3）：用隐函数定理 + 一阶 Taylor 展开，把 $(I-J_h)^{-1}$ 变成 2 次 JVP。
3. **一个 2× 的参数效率结论**（Table 1，1 个数据集）。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

- 论文**自陈实验是"最小集"**（*"We conducted a minimum set of experiments to verify that our method works"*）⇒ **只有 1 个数据集、1 个任务、4 个种子、1 张表**。
- **未评审**（preprint, under review）。
- **无代码** ⇒ 不可复现。
- **没有讨论 Taylor 截断误差**：$O(\|J_h\|^2)$ 这一项在 $\|J_h\|$ 不小时可能不可忽略，论文**未给任何误差界或数值监控**。
- **未讨论 surrogate gradient 的代价**：把 ReLU 导数换成 sigmoid 改变了函数的真实梯度，论文**未评估这是否损害训练**（只在 4 种子上测了最终准确率）。

### 6.2 🔴 隐含局限（本仓视角，逐条）

- 🔴 **"$\|J_h\|$ 小"这个前提从未被检查** —— Eq.(3) 的一阶截断要求 $\|J_h\|<1$（否则级数不收敛）。论文**既没报告 $J_h$ 的谱半径，也没做扫描**。本仓补测：把 $\|J_h\|_2$ 扫过 $\{0.1, 0.5, 0.9, 1.0, 1.5, 2.0\}$，看 Eq.(3) 的相对误差如何随 $\|J_h\|$ 发散 ⇒ **这是把本篇方法用到本仓前必须先做的第一步**。
- 🔴 **一阶 Taylor 展开没有收敛性保证，且与 ODE 求解器误差叠加** —— 本篇的向量场本身就有 $O(\|J_h\|^2)$ 的偏差，**再经数值积分器** ⇒ 总误差是两者之和，论文**没有做误差预算**。
- 🔴 **只在"矩阵式 NCDE"这一个基线上验证** ⇒ **参数量减半的收益有多少来自隐式化、多少来自"把 $W_x$/$W_h$ 拆开"这个纯参数化选择，未被分离**。🔴 事实上，**光把一个大矩阵拆成两个小矩阵**就会减半参数量（$d\times u\times v$ vs $d\times u+d\times v$，当 $u,v\approx d$ 时是 $d^3$ vs $2d^2$）⇒ **本篇标题里的"parameter-efficient"有相当部分可能只是这个显然的参数化收益，而非隐式化的功劳**。本仓认为这是一个**必须分离的消融**：(i) 矩阵式 + 拆权重（不提参数量）、(ii) 矩阵式原样、(iii) 隐式式 + 拆权重 —— 只有 (iii) vs (i) 的差才归功于隐式化。
- 🔴 **"真正的连续 RNN"的主张与本篇的实现有距离**：Eq.(2) 的 RNN 是 **Elman 式**（$W_hh_t+W_xX_t$），而论文在 §3.1 用它当"一般 RNN"的代表；Elman 式 RNN **没有门控**，不是 LSTM/GRU ⇒ **"连续 RNN"这个标签在本篇里指的是"连续极限的 Elman RNN"**，不是"连续极限的 GRU"。引用时不能泛化。
- 🔴 **对 CfC/LTC 完全不适用** —— 本篇的整个价值在"NCDE 贵"，而**本仓的主力路线是 CfC/LTC**（CfC 闭式、LTC 显式 ODE）⇒ **本篇方法对本仓现有主线的直接价值低**。
- **数据量极小**（单数据集、单任务）⇒ **不能作为"参数效率不损失精度"的一般性证据**。

### 6.3 未来方向

论文未单列未来工作节。

## 7. 对本仓的意义

**(1) 🔴 一条本仓至今零测量的量：$J_h=\partial h_{t+1}/\partial h_t$ 的谱半径** —— 本仓 `lnn/core/controllability_cfc.py` 的 `input_jacobian_norm` **只算输入方向** $J_x$（$\partial h_{t+1}/\partial x_t$），用途是"审计输入是否还有作用"（防止饱和）。而本篇指出 **状态方向 $J_h$ 才是决定动态行为的关键**：
- $(I-J_h)$ 的可逆性直接决定 $\dot h$ 是否爆炸（论文的整个推导都围绕它）；
- $\|J_h\|>1$ 意味着隐状态自我放大 ⇒ 与 11-24 强调的"度量总导数必须保留 $\frac{\partial M}{\partial u}\dot u$"是**同一族问题**（度量沿流的导数包含状态方向）。
⇒ **本仓应当补测**：对每个变体报 $\rho(J_h)$（谱半径）、$\|J_h\|_2$（谱范数）、以及 $J_h$ 的实部谱分布 ⇒ 这将是**第一条能直接回答"本仓哪个变体的隐状态在自我放大"**的诊断（现有的 `smoothness_metrics.py` 只有输入侧光滑度）。

**(2) 一条新消融轴（成本低）**：🔴 **"末层参数化形状"轴** —— 在同一 CfC/LTC 架构下比较：(i) 大矩阵式（$d\times u\times v$）、(ii) 拆成 $W_x$/$W_h$ 两个矩阵、(iii) 低秩分解（$W=AB$，$A\in\mathbb{R}^{d\times r}$, $B\in\mathbb{R}^{r\times u\times v}$）⇒ 报参数量 / 精度 / 训练时间三元组。**注意本篇的 2× 收益很可能主要来自 (ii)**（拆矩阵本身就减半）⇒ 该轴**同时是对本篇"parameter-efficient"归因的检验**。

**(3) 一条对 11-19 结论的交叉确认**：11-19 研读的 `2607.05280` 证明**闭式解与连续时间无绑定**（线性化 ⇒ 矩阵指数 + parallel scan，**相对 NCDE 约 1500× 加速**）。本篇给出**另一条独立的降本路径**（改参数化，不改求解器）⇒ 两条路线的**共同结论**是：**NCDE 的成本不在"连续时间"本身，而在"通用非线性向量场 + 隐式控制路径"** ⇒ 这**强化**了 11-19 的核心口径修正（"闭式不是液态的护城河，真正稀缺的是路径级驱动"）。⇒ **本仓引用时必须并置这两篇**，否则会误以为"降 NCDE 成本 = 让液态更便宜"。

**(4) 一条引用纪律**：本篇标题含 "Neural CDE"，**极易被当成"连续时间有效"的证据** —— 但它**不做任何连续时间 vs 离散的对照**，其全部结论是"如何用更少参数实现连续 RNN" ⇒ **不可引用为"连续时间机制更优/更省"**（延续 11-23/11-24 纪律）。

**Verdict**：
- **"连续 RNN 应写成隐式方程 $h_t=\mathrm{RNN}(X_t,h_t)$" → TARGET-POSITIVE**（概念澄清级：本仓索引里"Continuous RNN"这个词一直没有精确定义，本篇给出了正确形式）
- **"隐式化可以把 NCDE 参数量减半且精度不变" → TARGET-DEPENDENT-WITH-NUANCE**（1 数据集 / 4 种子 / 未评审 / 无代码；且 **2× 收益很可能主要来自"拆矩阵"而非隐式化** ⇒ 归因未分离）
- **"对本仓主线（CfC / LTC）有直接价值" → TARGET-NEGATIVE-WITH-NUANCE**（🔴 CfC 闭式、LTC 显式，**两者都不需要隐式化** ⇒ 本篇方法对本仓主力路线**不适用**；其价值主要在**概念澄清**与**$J_h$ 诊断量的引入**）
