---
title: The Shape of a Program: Path Signatures for Trace-to-Program Induction
date: 2026-11
tags: [LNN, Signature-Method, Log-Signature, Program-Induction, Interpretability, Rough-Path, Tensor-Algebra, Bootstrap-Contractivity]
---

# 研读报告：路径签名做程序归纳 —— "程序有形状"，而 log-signature 能把形状读出来

> **本篇不在 digest 候选池里** —— 标题不含 LNN/LTC/CfC/NCP 任一强关键词，是靠 AGENTS.md 约束 7 的独立覆盖度自检（机制层谱系扫描）捞出来的。本仓**零覆盖**（`2611.13005` 在 171 篇报告中零命中）。
> 🔴 **本篇对本仓的意义是"把路径几何从时序建模搬到序列建模"** —— 它证明 **log-signature 不只适用于连续时间动力学，也可以作为通用序列编码器**，且在**数据稀缺时优势最大**。这是本仓至今没测过的方向。

## 1. 元数据

- **论文标题**：The Shape of a Program: Path Signatures for Trace-to-Program Induction
- **作者**：Mohamed Ghanem, Bernd Finkbeiner
- **机构**：CISPA Helmholtz Center for Information Security / Technical University of Munich
- **发表时间**：2026-11-08 (v1)
- **发表渠道**：**NeurIPS 2026**（40th Conference on Neural Information Processing Systems）；**已评审**
- **来源**：arXiv:2611.13005v1 [cs.LG]
- **代码**：**https://oschart.github.io/program-signature**（项目页）；使用 **PyTorch 原生重实现的 SIGNATORY** 做可微 log-signature 计算
- **本地 PDF**：[papers/2611.13005/2611.13005v1.pdf](../../papers/2611.13005/2611.13005v1.pdf) ／ 文本 [papers/2611.13005/2611.13005v1.txt](../../papers/2611.13005/2611.13005v1.txt)
- **页数**：16 页
- **关联概念**：path signature / log-signature / tensor algebra / free Lie algebra / **rough path** / Chen 恒等式 / tree-like equivalence / 程序归纳（program induction）/ Karel / List / RASP / 语义准确率 / **contractivity** / 自举（bootstrapping）/ t-SNE 潜空间几何

## 2. 核心问题

**程序归纳（program induction）**：从**执行轨迹**（execution trace）反推生成该轨迹的程序。与 I/O 对合成不同，轨迹揭示的是"**怎么算的**"而非"算出什么" —— 这对**逆向工程、恶意软件分析、算法可解释性**（把不透明 agent 蒸馏成可读代码）是刚需。

主流做法把轨迹当作**离散 token 序列**，用 LSTM / Transformer 编码。论文提出一个**根本性的视角转换**：

> "Rather than viewing traces merely as sequences of tokens, we conceptualize them as high-dimensional rough paths and encode them using the path signature transform."

论文的论证分三层：

1. **【理论】为什么签名与程序同构**：
   - 程序 $\rho$ 在行为等价意义下由它诱导的**轨迹分布** $\mu_\rho$ 决定；Chevyrev & Lyons 证明 **$\mathbb{E}_{X\sim\mu_\rho}[S(X)]$ 唯一刻画 $\mu_\rho$**（有界长度轨迹下收敛半径无限）⇒ **聚合签名原则上可识别生成程序**。
   - **Chen 恒等式** $S(X*Y)=S(X)\otimes S(Y)$ ⇒ 共享连续子段的轨迹（同一个循环体、同一个子程序被调用）**共享对应的签名因子** ⇒ **结构相似的执行落在结构相似的嵌入区域**。
   - ⚠️ **关键对照**：**三个基线都没有这种"部分—整体"的闭式代数律** —— LSTM 的嵌入依赖固定递归；Transformer 编码器**只能靠位置编码打破自注意力的置换不变性**来恢复顺序；截断 Fourier 谱**无法从各部分的谱还原出拼接的谱**。
2. **【方法】签名 vs 序列模型的本质差异**：签名作用在**几何（geometry）**上，LSTM/Transformer 作用在**内容（content）**上。论文的核心主张是"**程序行为本质上是几何的**"—— 控制流把执行组织成子程序、循环、条件分支，这些**在轨迹中表现为几何特征**。
3. **【实践】** 三条发现：log-signature 编码器在几乎所有设定下更准（**数据稀缺时优势最大**）；trace-to-program 流水线**天然是压缩的（contractive）**；log-signature 的**潜空间几何更可解释**。

## 3. 方法论与核心思路

### 3.1 总体架构

**三段式 encoder–decoder**：

1. **状态编码器 $\phi$**：把环境状态 $s_t\in\mathcal{S}$ 映射为稠密嵌入 $v_{k,t}\in\mathbb{R}^d$。
   - Karel（网格视觉）⇒ **CNN**；
   - List（符号列表）⇒ **逐元素小嵌入器**；
   - RASP ⇒ **线性映射 + 残差 MLP**。
2. **轨迹编码器（本文核心）**：把 $V_k=(v_{k,1},\ldots,v_{k,T_k})$ 编码为**单一固定尺寸**嵌入 $h_k\in\mathbb{R}^d$。
   - 🔴 **唯一可训练组件是一个线性投影** $W_{\mathrm{proj}}\in\mathbb{R}^{d\times D_{\logsig}(d,M)}$ ⇒ **签名表示本身无参数**。
3. **程序解码器**：**Transformer decoder**，通过 cross-attention 条件化于轨迹嵌入集合 $H=\{h_1,\ldots,h_K\}$；并**屏蔽语法非法的 token** ⇒ 保证输出永远是语法合法的程序。

### 3.2 核心模块 1：签名与 log-signature（§3）

**签名**（Eq. 1–2）：路径 $X:[0,T]\to\mathbb{R}^d$ 的签名是所有迭代积分：

$$S(X)_{0,T}=\Big(1,\ X^{(1)}_{0,T},\ X^{(2)}_{0,T},\ldots\Big),\qquad X^{(\ell)}_{0,T}=\int_{0<t_1<\cdots<t_\ell<T}dX_{t_1}\otimes\cdots\otimes dX_{t_\ell}$$

**离散路径的高效计算**（Eq. 3，分段线性插值 + Chen 恒等式）：

$$S(x)=\bigotimes_{i=1}^{N-1}\exp(x_{i+1}-x_i)$$

**log-signature** 把张量代数映到自由 Lie 代数，去掉 shuffle 乘积产生的冗余项；第 $\ell$ 层维度由 **Witt 公式**给出，**远小于** $d^\ell$。论文给的例子：$\mathbb{R}^2$ 路径截断到 $M=4$ 时，**签名有 30 项，log-signature 只有 8 项**。

### 3.3 核心模块 2：轨迹编码的四个关键处理（§4.3）

论文的编码器不是"直接算个 log-signature"，而是**四个精心设计的处理**，每一个都对应一个具体问题：

**① 位置增广（positional augmentation）—— 解决 tree-like equivalence**

标准签名满足 **tree-like 等价**：路径的"回退"（backtracking）会被抹掉。例：`turnRight, turnLeft` 回到原状态，若状态编码相同则签名**可能为零增量** ⇒ **丢失执行历史**。

$$\tilde v_{k,t}=v_{k,t}+PE(t),\qquad PE(t)\in\mathbb{R}^d\ \text{为固定正弦位置编码}$$

因为 $PE(t)$ 每步不同，"回到原状态"的动作序列会到达**增广路径上的不同点** ⇒ 不再构成被忽略的回退段。
🔴 **口径**：论文明确说"**在很多标准签名应用中，time 是作为额外通道加入的，这彻底排除了 tree-like 等价**"；**本文选择加法式位置编码**是因为它在 Karel 与 List 上更好；**RASP 上则改为拼接 $t$ 的 one-hot**（因为 RASP 每条轨迹的步骤 $t$ 都会把被扫过的 token 设为 $t$）。

**② 单位步长归一化 —— 解决"大步主导"**

$$\hat v_{k,1}=v_{k,1},\qquad \hat v_{k,t+1}=\hat v_{k,t}+\frac{\Delta_{k,t}}{\sqrt{\|\Delta_{k,t}\|_2^2+\epsilon^2}}$$

**理由**：签名**没有自己的权重**去平衡不同大小的步；它的第 $\ell$ 层项是 $\ell$ 个增量的乘积之和 ⇒ **几个大步会主导整个签名**。论文的具体观察：List 域中 `pushZero`/`pop` 一次移动多个元素，其嵌入步长**超过 `incElem`/`decElem` 的十倍** ⇒ 原地编辑对 log-signature 的贡献被压没。归一化后，**每个状态变化贡献相同权重**，log-signature 记录"**发生了哪些变化、顺序如何**"而非"移动了多远"。

🔴 **这是一条本仓极可能需要的口径**：本仓的 CfC/LTC 驱动也是连续输入，其"步长大小"由数据决定 ⇒ **若本仓做路径几何相关分析，必须知道不同通道的步长是否可比**。

**③ 尾部重复填充（repeat-padding）—— 解决变长 batch**

重复最后一个状态填充：$\tilde v_{k,t}=\tilde v_{k,T_k},\ \forall t>T_k$。**每个重复点贡献零增量**，签名因子为 $\exp(0)=1$ ⇒ **签名不变**（这是重参数化不变性的一个实例）。论文明确指出这**优于零填充**（后者会在路径末端引入**虚假增量**）。

**④ log-signature 截断到 $M=2$ + 可学习线性投影**

$$h_k=W_{\mathrm{proj}}\cdot\log\big(S^M(\tilde V_k)\big)+b_{\mathrm{proj}}\in\mathbb{R}^d$$

$M=2$ 的选择理由：**保留嵌入路径坐标间的成对交互项**的最小截断阶；附录 A.1 的 $M\in\{1,2,3\}$ 消融显示 **超过 $M=2$ 没有一致收益**，而维度代价显著。

### 3.4 核心模块 3：压缩性与自举（§4.5）

**观察**：trace-to-program 流水线**天然是压缩的（contractive）** —— 在相当比例的样例上，模型产出的程序 $\hat\rho$ **比真值 $\rho^*$ 更短**，但**仍能复现全部监督轨迹**。

**自举流程**：从 $D_0=\{(T_i,\rho^*_i)\}$ 出发，每 **5 个 epoch**（5 epoch 热身之后）做一次**重标注 pass**：当前模型对每个训练样例生成预测 $\hat\rho_i$，若 $\hat\rho_i$ **有效**（在 $T_i$ 的每个初始状态上执行能复现对应轨迹）**且更短**，则替换标签：

$$\rho_i^{(t+1)}=\begin{cases}\hat\rho_i,& \hat\rho_i \text{ 在 } T_i \text{ 上有效且 } |\hat\rho_i|<|\rho_i^{(t)}|\\ \rho_i^{(t)},& \text{否则}\end{cases}$$

**论文对这一方法的口径极为克制（必须照抄）**：

> "We frame this as a measurement rather than a self-improvement claim: ... both quantities are encoder-derived **lower bounds** on how compressible the supervision was."

即：**只替换"已验证更短且有效"的标签**，因此累计缩短量与被缩短样本比例都是"监督可压缩程度"的**下界**，**不是**自我改进的证明。

### 3.5 与 LTC / CfC / NCP / GRU 的关系

本篇**与液态路线无直接关系**（是程序归纳，不是时序建模），但与本仓的关系在于**通用性**：

| 维度 | 本篇 log-signature 编码器 | LSTM | Transformer | Fourier |
|---|---|---|---|---|
| 作用对象 | **几何**（路径形状） | 内容（递归状态） | 内容（注意力池化） | 内容（频谱） |
| 聚合方式 | 张量代数乘积（可结合 ⇒ 可并行） | 递归 | 池化 + 位置编码 | 截断谱 |
| 复杂度 | **线性**于序列长度 | 线性 | **二次**（注意力） | 线性 |
| 长度不变 | **是**（固定尺寸嵌入） | 否 | 否 | 否 |
| 部分—整体代数律 | **有**（Chen 恒等式） | **无** | **无**（只有位置编码近似） | **无** |
| 表达力 | **通用非线性**（任意路径的连续泛函可由签名的线性泛函逼近） | 有限 | 有限 | **不满足**（论文明确点出 Fourier 不具备该性质） |
| 可训练参数 | **仅一个线性投影** | 全部 | 全部 | 全部 |

🔴 **对液态路线最重要的一条推论**：**log-signature 是一个"零参数的、线性复杂度的、长度不变的、通用非线性"的序列编码器**。这与本仓"液态网络的核心是**连续时间归纳偏置**"的叙事形成了一个有趣的对照：**如果一个纯几何编码器（不建模时间流、不做状态递推）在某些任务上就足够好，那么"连续时间动力学"的必要性问题就需要被重新提问**。⚠️ 但**这是跨域推论**（程序归纳 ≠ 时序预测），**不可直接套用到本仓的时序任务上**。

## 4. 核心公式提取

| 编号 | 公式 | 含义 |
|---|---|---|
| Eq. 1–2 | $$S(X)_{0,T}=\big(1,X^{(1)}_{0,T},X^{(2)}_{0,T},\ldots\big),\quad X^{(\ell)}_{0,T}=\int_{0<t_1<\cdots<t_\ell<T}dX_{t_1}\otimes\cdots\otimes dX_{t_\ell}$$ | **签名的定义**：所有迭代积分的层级序列 |
| Eq. 3 | $$S(x)=\bigotimes_{i=1}^{N-1}\exp(x_{i+1}-x_i)$$ | 离散路径的签名 = 各段指数的**张量积**（Chen 恒等式的算法形式） |
| Eq. 5 | $$S(X_{[0,T]})=S(X_{[0,t]})\otimes S(X_{[t,T]})\quad\forall t\in[0,T]$$ | **Chen 恒等式**：签名在任意时间切分下可分解 ⇒ 结构相似执行落在相似区域 |
| Eq. 6 | $$\tilde v_{k,t}=v_{k,t}+PE(t)$$ | **位置增广**：打破 tree-like 等价，保留回退前的执行历史 |
| 归一化 | $$\hat v_{k,t+1}=\hat v_{k,t}+\frac{\Delta_{k,t}}{\sqrt{\|\Delta_{k,t}\|_2^2+\epsilon^2}}$$ | **单位步长**：防止大步长主导签名（List 域中 push/pop 步长 > inc/dec 十倍） |
| Eq. 7 | $$h_k=W_{\mathrm{proj}}\cdot\log(S^M(\tilde V_k))+b_{\mathrm{proj}}\in\mathbb{R}^d$$ | **可学习线性投影**（编码器唯一可训练部分） |
| Eq. 8 | $$P(\hat\rho\mid H)=\prod_{j=1}^{|\hat\rho|}P_\theta(y_j\mid y_{<j},H)$$ | **程序解码**：Transformer decoder + cross-attention 于 $H$ |
| Eq. 9 | $$\mathcal L(\theta)=-\sum_{j=1}^{L}\log P_\theta(y^*_j\mid y^*_{<j},H)$$ | 交叉熵损失（对真值程序 token） |
| Eq. 10 | $$\rho_i^{(t+1)}=\begin{cases}\hat\rho_i,&\hat\rho_i\text{ 有效且 }|\hat\rho_i|<|\rho_i^{(t)}|\\ \rho_i^{(t)},&\text{否则}\end{cases}$$ | **自举重标注规则**（只替换"已验证更短"的标签） |
| Eq. 4 | $$\tau=E(\rho,s_0)=(s_1,\ldots,s_T)$$ | 轨迹定义：程序 + 初始状态 ⇒ 状态序列 |
| — | $$D_{\log sig}(d,M)\text{ 由 Witt 公式给出}$$ | 第 $\ell$ 层 log-signature 维度（$\mathbb{R}^2$, $M=4$: 签名 30 项 vs log-signature 8 项） |

## 5. 关键成果与贡献

### 5.1 主结果：语义准确率随训练数据量的变化（Finding #1, Figure 3）

**设置**：Karel / List 各 100K 训练 + 2500 测试（每个程序采样 5/6 条轨迹训练，测试用全部 6 条）；RASP 5,816 训练程序 + 2,500 测试程序（每个程序 8 次 sweep，训练采样 4 次）；**3 个随机种子**；训练 100 epoch，batch 32，embedding size 256（RASP 的 log-signature 编码器 $d=32$）；Adam，lr $10^{-4}$，梯度范数裁剪 0.5；单张 A100。

**准确率定义（semantic accuracy）**：解码出的程序**复现全部给定轨迹**（Karel/List）或**全部 sweep 输出**（RASP）即算正确，**不要求与真值逐 token 相同**。

**核心结论**：
- **log-signature 编码器在 12 个（域 × 数据量）设定中的 11 个最准**；唯一例外是 **Karel 全数据量**下 Transformer 与 LSTM 领先（且领先幅度**超过种子间散布**）。
- 🔴 **最大优势在低数据区**：**1/4 训练数据时，log-signature 在 Karel 上把最强基线的准确率"几乎翻三倍"**（near-triples）；在 List 与 RASP 上明显领先，但**优势随数据量增加而收窄**。
- **参数化编码器最需要数据**：1/4 数据时 **LSTM 与 Transformer 在 Karel 上几乎学不会**，在 RASP 上**每个数据量下都远落后于 log-signature**。
- **Fourier** 在 Karel 低数据量与 RASP 全数据量上击败 LSTM/Transformer，但在 Karel 全数据量上落后，且**在 List 上每个数据量都是最弱编码器**。

**两条"不是表示尺寸带来的"的消融**：
- **宽度消融**：50% 数据量下改变状态嵌入宽度 $d$，**即使最窄的 log-signature 编码器在 List 与 RASP 上也比任何宽度的基线更准**，Karel 上从 $d=64$ 起领先。
- **log vs full signature**：log-signature 与完整签名**表现几乎完全一致**，但**特征数只有一半**，在 Karel 与 List 上**参数量少约三分之一**。

**收敛速度**：log-signature 编码器达到其最佳准确率的 90% **所需 epoch 数少于 LSTM 与 Transformer**（在二者能学会的设定上）；Fourier 在多数设定下也成立。

### 5.2 主结果：自举暴露监督冗余（Finding #2, Table 1）

**数据准备（Karel+/List+/RASP+）**：原数据几乎无冗余（Karel 与 List 测试集上贪心搜索可缩短约 1/4 的程序，但**只去掉 4.5% / 5.2% 的 token**）⇒ 论文**人为注入冗余**：一半程序被改写为行为等价但结构更长的形式（Karel/List：$S\to\texttt{REPEAT 1}(S)$、$S\to\texttt{IFELSE c}(S)\texttt{ELSE}(S)$；RASP：$\texttt{CMPC EQ c}\to\texttt{COMBINE min}(\texttt{CMPC GEQ c})(\texttt{CMPC LEQ c})$），注入 token 约占总量 **20%**。

| Encoder | Karel+ Tokens removed (%) | Karel+ Programs shortened (%) | List+ Tokens removed (%) | List+ Programs shortened (%) | RASP+ Tokens removed (%) | RASP+ Programs shortened (%) |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 21.11 | 58.66 | 22.37 | 59.42 | 13.41 | 32.33 |
| Transformer | 20.92 | 57.81 | 22.58 | 59.82 | 18.26 | 44.08 |
| Fourier | 18.43 | 52.89 | 18.19 | 49.62 | 18.64 | 44.98 |
| **Log-Signature** | 21.07 | **58.59** | **22.70** | **59.88** | **19.84** | **47.92** |

**读数**：
- log-signature 在 **RASP+ 上两项都最好**（19.84% token 移除、47.92% 程序被缩短）。
- 在 **Karel+ / List+ 上，LSTM / Transformer 与 log-signature 基本持平**（21.11 vs 21.07；22.70 vs 22.37）⇒ **log-signature 不是唯一能触发压缩的编码器**。
- 🔴 **论文自己的结论**：**"每一个编码器（包括参数化的 LSTM/Transformer 与谱方法的 Fourier）都产生非平凡的缩短值，所以压缩性应被理解为 trace-to-program 流水线的性质，而非任何单一编码器的独特属性。"** ⇒ **不能引用为"log-signature 独有压缩性"**。

### 5.3 主结果：潜空间几何的可解释性（Finding #3, Figure 4）

用 t-SNE 投影各编码器读到的**状态嵌入**（Karel 域，5 个测试程序 × 每个 6 条轨迹，每条轨迹一个颜色的链）。

**结果**：**log-signature 模型的嵌入在空间上明显更有结构** —— 同一执行轨迹的状态**保持聚拢并形成各自独立的链**，**不会互相穿越**；而基线的链会交叉。论文的解读："程序有内在形状，在其轨迹中显现；直接编码几何就能与执行结构对齐，并绕过内容级噪声。"

### 5.4 贡献清单

1. **【方法】** 把 **rough-path 视角引入程序归纳**：用 log-signature 作为轨迹编码器，在 3 个域（Karel / List / RASP）的 12 个设定中 11 个最准，**且数据稀缺时优势最大**。
2. **【洞见】** 发现 trace-to-program 流水线**天然压缩**，并把它变成一个**无需外部 oracle** 的监督冗余测量仪器（自举）。
3. **【洞见】** log-signature 编码器诱导出**明显更可解释的潜空间几何**（轨迹内最近邻落在自己的轨迹上）。
4. **【工程】** **零参数签名表示 + 单个线性投影**；log-signature 与完整签名表现相当但特征数减半、参数量少约 1/3。

## 6. 局限性与未来展望

### 6.1 作者自陈局限

> "The truncated log-signature provides a compact representation for moderately complex traces, but the dimensionality of the feature space $D_{\log sig}$ still **grows exponentially with the truncation level $M$**. We keep $M$ small in practice, but **very-long-horizon programs that require fine-grained higher-order interactions across the full trace remain a challenging case**."

### 6.2 隐含局限（本仓视角）

- 🔴 **跨域有效性未被检验**：全部实验在**程序归纳**域上。本仓的时序任务（工业传感器、EEG、金融序列）**没有对应证据** ⇒ **不可引用为"log-signature 编码器在时序任务上更好"**。
- **"通用非线性"的论证有边界**：论文说"任何路径的连续泛函可由签名的线性泛函逼近"，但这是**逼近**（需足够多特征/合适的线性泛函）**不等于"训练更省"**。低数据区的优势虽然显著，但论文没有给出**样本复杂度层面的解释**，只给了经验现象。
- **单位步长归一化是一个强假设**：它**丢掉了"移动多远"这一信息**。对于**幅值本身有语义**的任务（如金额、速度、浓度），这一步可能**主动丢掉重要信息** ⇒ **对本仓的时序任务，直接照搬归一化是有风险的**。论文在 List 域上的理由（push/pop 步长差十倍）是**结构性**的，但**并非所有时序数据都有这个性质**。
- **位置增广是必需的"补丁"**：标准签名满足 tree-like 等价 ⇒ **必须额外注入位置信息**才能保留顺序信息 ⇒ **这削弱了"签名天然对齐程序结构"的说服力**（至少在"顺序"这一维度上，它需要外挂）。论文自己也很诚实地点出了 Fourier 的对应缺陷（"Fourier 不满足通用非线性"），但**没有把位置增广对 log-signature 的同类缺陷放在同等位置讨论** ⇒ 🔴 **这是一处对本仓引用时必须补的口径**。
- **只比较了三种基线**（LSTM / Transformer / Fourier）—— **没有与 Mamba / S4 / xLSTM 等现代序列模型对比** ⇒ "log-signature 优于序列模型"的**证据范围仅限 2023 年前后的经典架构**。
- **语义准确率"几乎每设定都赢"但**未做显著性检验**（3 个种子，只报了"最好 epoch 的准确率"）⇒ **"Karel 全数据量下基线领先超过种子散布"这一判断是定性的**。

### 6.3 未来方向（作者 + 本仓视角）

- 作者：把 rough-path 方法推广到更长的程序、更深的签名阶数、更多域。
- 本仓视角：把 log-signature 作为**通用序列编码器**接入本仓的**变体库**（见 §7），检验"**几何编码器能否替代一部分状态递推**"。

## 7. 对本仓的意义

**① 可作 `lnn/core/variants.py` 的什么变体？**
🔴 **本篇最有价值的可落地物不是"新的 cell"，而是一个"**编码器替换实验**"** —— 本仓至今所有的时序模型都是**状态递推式**（GRU / LTC / CfC / SSM）。本篇提供了一个**零参数、线性复杂度、长度不变**的替代：
- **`LogSignatureEncoder`**：给定一串状态/特征嵌入 $v_1,\ldots,v_T$，输出固定尺寸的 $h\in\mathbb{R}^d$（按 §4.3 的四步：位置增广 → 单位步长 → 重复填充 → $M=2$ log-signature → 线性投影）。
- 🔴 **但必须注意：它与 LTC/CfC 是"编码器 vs 动力学"的替换关系，不是互补关系** ⇒ 正确用法是做一个 **"纯几何编码器 + 下游分类头"** 的基线，**与 "LTC/CfC + 分类头"** 对比 ⇒ **若纯几何编码器在某些本仓任务上就能打平液态模型，那"连续时间动力学"的必要性就真的成了问题**。
- **可选变体**：`LogSigFeatureTokenizer`（只取 log-signature 作为固定特征，不训练投影）⇒ **与 11-19 研读的 `2610.05051`（LogSig-SSM）思路完全一致** ⇒ 本仓可以把两篇**串成一条线**（几何 token 化 → 喂 SSM），**这是本仓第一条把"路径几何"完整接进"序列骨干"的路线**。

**② 可嫁接到哪个 bench 脚本？**
- **新建 `bench_logsig_vs_liquid.py`** —— 核心对照：{GRU, LSTM, Transformer, S4/Mamba, LTC, CfC, LogSig-Encoder} × {本仓 3–4 个标准时序任务} ⇒ **本仓至今缺"非递推几何基线"**。
- **`bench_cfc_temporal_smoothness.py`** —— log-signature 的**能量分布（各阶占比）**是一个**现成的"路径几何复杂度"度量**，可直接与本仓的平滑度度量并列 ⇒ **给 11-19 建立的"驱动路径粗糙度"纪律再添一维**。
- **`bench_cfc_input_dropout.py` / `bench_cfc_temporal_dropout.py`** —— 本篇的 $M\in\{1,2,3\}$ 消融给出一个现成的**阶数扫描**协议。

**③ 提供哪个新 ablation axis？**
🔴 **轴 A（最有价值）："几何 vs 递推"轴。** 本仓所有结论都建立在"连续时间状态递推"这一族模型上。log-signature 编码器提供了一个**几何但非递推**的对照 ⇒ **这条轴一旦补上，本仓对"液态机制必要性"的整个论述都需要重审**（这与 11-13 CT-GRU、11-19 `2607.05280` 的方向一致，但更激进）。

🔴 **轴 B："步长归一化"轴。** 论文的 List 观察（push/pop 步长 > inc/dec 十倍 ⇒ 大步主导签名）**在本仓的连续驱动上完全适用** ⇒ 本仓的 CfC/LTC 驱动也是连续信号，其**通道间幅值尺度差异**可能同样造成"某些通道主导" ⇒ **一条本仓从未测过的消融：把本仓驱动做单位步长归一化 vs 不做，看结果差多少**（若差异大，说明本仓既有结论里有一部分来自幅值尺度而非时间结构）。

🔴 **轴 C："阶数 $M$"轴。** $M=1$ vs $M=2$ vs $M=3$ ⇒ 这与 11-19 的 "depth-1 vs depth-2 log-sig" 轴**是同一条**，但本篇给出了**第三个独立数据点**（$M>2$ 无一致收益）⇒ **可以合并成本仓的一条常设消融**。

🔴 **轴 D："数据稀缺"轴。** 本篇最有说服力的结果是**低数据区优势最大** ⇒ 本仓的多 $\tau$ 变体扫描（几十个变体）本质上就是"数据稀缺"场景 ⇒ **本篇暗示：在这个场景下，一个零参数的线性投影可能比几十个可学习变体更稳** ⇒ **本仓的变体扫描结论需要补一个"零参数几何基线"对照**。

**④ 对本仓既有结论的直接修正要求**：
- 本仓索引中"**连续时间归纳偏置**是液态网络的核心优势"这一表述，**必须加一条限定**：**若一个零参数、几何的、无状态递推的编码器在低数据区就能达到可比性能，那么"归纳偏置"的来源就未必是"时间"** ⇒ 这是**本篇对本仓最有杀伤力的一条**，虽然证据来自跨域（程序归纳），但**足以要求本仓补做轴 A 的对照**。
- 本仓若在文档里写"log-signature 是连续时间模型的专属工具"，**必须修正** —— 本篇证明它是**通用的序列编码器**（程序归纳域），与 `2602.02157`（本仓已研读，11-20）的用法**完全不同**（那里是驱动路径，这里是特征聚合）⇒ **两者应作为两条不同的用法分别记录**。

**Verdict**：
- **"log-signature 是更好的轨迹编码器，在数据稀缺时优势最大" → TARGET-POSITIVE-WITH-NUANCE**：12 个设定中 11 个领先、1/4 数据下 Karel 准确率"几乎三倍"、宽度消融与 log-vs-full 消融排除了"表示尺寸"解释 ⇒ **证据扎实**；**但须标注**：**仅限程序归纳域**、仅 3 个种子、**只比较 LSTM/Transformer/Fourier 三种基线**（无 Mamba/S4/xLSTM）、**必须外挂位置增广才能保住顺序信息**、**单位步长归一化主动丢弃幅值语义**。
- **"trace-to-program 流水线天然压缩，可作监督冗余测量仪" → TARGET-POSITIVE（作为测量工具）**：机制清晰（只替换已验证更短的标签 ⇒ 下界性质）；**且论文自己明确指出压缩性是流水线性质而非编码器独有** ⇒ **不得引用为 log-signature 的独有能力**。
- **"log-signature 的潜空间更可解释" → TARGET-DEPENDENT-WITH-NUANCE**：t-SNE 证据（轨迹成链、不交叉）直观有力，但**是二维投影的可视化证据**，**无定量指标** ⇒ **应作为"定性观察"引用，不可当结论**。
- **"签名是通用非线性，可替代序列模型" → TARGET-DEPENDENT-WITH-NUANCE**：理论上成立（任意路径连续泛函可由签名线性泛函逼近，**Fourier 不具备**），但**这是逼近性陈述**（不等于可训练性/样本效率）⇒ **在跨域外推到时序任务前必须本仓自己验证**。
- **对液态路线（LtC/CfC）本身 → 不适用（族外，但有强方法论含义）**：本篇**不是**液态家族的证据；**但它给本仓一条必须补做的对照**（轴 A：几何 vs 递推）—— 在补上这条之前，本仓"连续时间归纳偏置带来优势"的表述**缺少一个必要的非递推基线**。
