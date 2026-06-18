---
title: Liquid Random Feature Methods for Time-Dependent Partial Differential Equations 研读报告
date: 2026-06-19
tags: [LNN, LTC, liquid-time-constant, PDE-surrogate, random-feature-method, residual-collocation, mesh-free, L-RFM, math]
arxiv_id: 2606.15571v1
authors: Jiale Linghu, Yangshuai Wang
institutions:
  - School of Mathematics and Statistics, Xidian University, Xi'an, China
  - Department of Mathematics, National University of Singapore, Singapore
submitted: 2026-06-14
source: https://arxiv.org/abs/2606.15571v1
local_pdf: papers/arxiv/2606.15571.pdf
selected_reason: |
  该论文是 2026-06-19 digest 中唯一尚未被研读、且摘要显式包含
  "closed-form liquid time-constant responses" 关键词的候选 (select_papers_for_report
  因 digest summary 列被截断而给出 score=0, 但完整 abstract 明确命中强关键词
  "liquid time-constant" 与 "closed-form continuous-time"). 距今 5 天 (远 < 30 天),
  主题与本仓 LNN 持续研究方向高度对齐: 把 LTC/CfC 的液体时间常数响应作为
  frozen-feature primitive, 嵌入 mesh-free 偏微分方程残差最小二乘求解器.
---

# Liquid Random Feature Methods for Time-Dependent Partial Differential Equations (L-RFM) 研读报告

## 0. 元数据

| 字段 | 内容 |
|---|---|
| 论文标题 | Liquid Random Feature Methods for Time-Dependent Partial Differential Equations |
| 作者 | Jiale Linghu (Xidian University), Yangshuai Wang (NUS, corresponding author) |
| arXiv 编号 | [2606.15571v1](https://arxiv.org/abs/2606.15571v1) |
| 学科分类 | physics.comp-ph (cs.LG, math.NA 交叉) |
| 提交日期 | 2026-06-14 (v1) |
| PDF 长度 | 27 页 (含两个附录) |
| 关键词 | random feature method, **liquid time-constant networks**, time-dependent PDEs, least squares, partition of unity |
| 致谢 | Jiale Linghu 在 NUS 访问期间完成, Weizhu Bao 教授提供指导 |

> 注: 本文用 "liquid time-constant (LTC) model" 明确指向 Hasani 等 (2021) 的 Liquid Time-Constant Networks (LTC), 复用其 closed-form ODE 响应思想, 但**不**学习网络参数 — 把 LTC 当 frozen feature primitive, 再做线性 LS readout. 这是 LTC 从序列模型向**算子/PDE 代理**的一次概念迁移.

## 1. 核心问题

时间依赖偏微分方程 (PDE) 的无网格 (mesh-free) 残差最小二乘求解, 长期被一个根本问题卡住:

- **静态 frozen activation (e.g. random Fourier features / ridge functions)** 把空间-时间基函数全部冻死, 然后只拟合线性读出层. 算法简单, 代数核保持线性 LS.
- 但**时间维度没有任何"松弛尺度"机制**: 快瞬态、慢弛豫、多尺度相互作用只能靠采样几何**间接**表示. 在 stiff / dispersive / multi-scale 场景下, 单纯增加采样点数或改善线性代数无法弥补 trial space 缺时间尺度这一**有限维瓶颈**.

具体痛点:
1. PINN 需要训练所有网络参数, 优化非凸, 对架构/采样/损失权重重 (Wang 2022, Wu 2023).
2. DeepONet / FNO 学习"问题→解"的算子映射, 与一次性 forward 求解互补, 但学的是"映射"而非"当前 PDE 的解".
3. 现有 RFM/ELM 类方法 (PIELM, XTFC, ST-RFM, PoU-RFM) 保持 LS 简单性, 但其时间成分仍然是**静态 space-time ridge**, 没有显式的弛豫尺度.

**核心研究问题**: 能否把 "liquid time-constant" (LTC) 的指数松弛响应**作为 frozen feature primitive**, 让 trial space 在冻死前就已经携带一组采样得到的松弛尺度, 同时仍保留线性 LS readout, 从而在 stiff / dispersive / multi-scale 时间依赖 PDE 上以更少的特征数达到更高精度?

## 2. 方法论与核心思路

### 2.1 整体设计: 共享 liquid primitive + 空间支持规则

论文的关键设计是**把"时间响应"与"空间支持"解耦**:

- **共享闭式液体原语 (shared closed-form liquid primitive)**: 对一个采样参数元组 $\theta=(\tau,A,w,b,w_0,b_0)$, 在空间坐标 $\zeta(x)$ 上定义两个 ridge 形状
  $$g_\theta(x)=\tanh(w^\top\zeta(x)+b),\qquad h_\theta^0(x)=\tanh(w_0^\top\zeta(x)+b_0),$$
  以及一个**随空间变化的速率**
  $$\alpha_\theta(x)=\tau^{-1}+g_\theta(x).$$
  液体原语是该 ODE 的闭式解
  $$\partial_t\phi_\theta(x,t)=-\alpha_\theta(x)\phi_\theta(x,t)+g_\theta(x)A,\quad \phi_\theta(x,0)=h_\theta^0(x). \tag{4}$$
  数值稳定形式 (Duhamel):
  $$\phi_\theta(x,t)=h_\theta^0(x)e^{-\alpha_\theta(x)t}+g_\theta(x)A\,\eta_0(\alpha_\theta(x),t),\quad \eta_0(\alpha,t)=\tfrac{1-e^{-\alpha t}}{\alpha},\;\eta_0(0,t)=t. \tag{5}$$

- **空间支持规则**: 这是 L-RFM-Local 与 L-RFM-Global 唯一的差异.
  - **L-RFM-Global**: 单一全局仿射坐标 $z(x)$, 采样 $P$ 个独立 liquid 特征, trial space 是 $\Phi_q(x,t)=\phi_q(x,t)$ (无空间局部化).
  - **L-RFM-Local**: 用 partition-of-unity (PoU) 把 $\Omega$ 分成 $K$ 个重叠 patch, 每个 patch 上采样 $M$ 个 local liquid 特征, 最终 trial space 是
    $$\Phi_{q}(x,t)=\Phi^{\text{loc}}_{i,k}(x,t)=\psi_k(x)\phi_{i,k}(x,t),\quad q=(k,i),\;P=KM. \tag{9}$$

这种分离让作者可以在固定 readout 维度 $P$ 下, 单独考察"局部 vs 全局"对精度的影响.

### 2.2 解析闭式导数 (Analytic Closed-form Derivatives)

由于 Duhamel 形式 (5) 在 $\alpha=0$ 处有 removable limit, 一阶、二阶 (甚至三阶, KdV 需要) 空间导数都能**直接用代数公式写出来**, 不需要 ODE solver 也不需要 auto-diff. 关键一阶公式:

$$\partial_{x_j}\phi_{i,k}=h_j E - h\alpha_j tE + A\!\left[g_j\eta_0+g\alpha_j\eta_1\right],\quad E=e^{-\alpha t},\;\alpha_j=g_j. \tag{13}$$

二阶:

$$\partial^2_{x_j}\phi_{i,k}=h_{jj}E-2h_j\alpha_j tE+h(\alpha_j^2 t^2-\alpha_{jj}t)E+A\!\left[g_{jj}\eta_0+2g_j\alpha_j\eta_1+g(\alpha_{jj}\eta_1+\alpha_j^2\eta_2)\right]. \tag{14}$$

Leibniz 给出 PoU 局部化的乘积规则. 这些解析导数是 L-RFM 与"通用黑盒 PINN"的关键区别: 残差矩阵元素可以**直接计算**, 避免通过 ODE integrator 反向传播.

### 2.3 加权残差最小二乘 + Picard 线性化 + Block Marching

求解 $u\approx u_{L\text{-}RFM}=\sum_{q=1}^{P}c_q\Phi_q$ 的步骤:

1. 在当前解窗口 $T_{\text{win}}$ 上选**内部 + 初始 + 边界** collocation 集 ($N_{\text{int}}, N_{IC}, N_{BC}$), 拼成**行加权**过定系统
   $$Ac\approx b,\qquad A=[\lambda_{\text{int}}A_{\text{int}};\;\lambda_{IC}A_{IC};\;\lambda_{BC}A_{BC}], \tag{17}$$
   块权重 $\lambda_b=\sqrt{N_{\text{int}}/N_b}$ 用于平衡各块贡献.

2. **线性 PDE**: 一次性求解 $c=\arg\min_d\|Ad-b\|^2$ (truncated SVD, 阈值 $\sigma_{\text{cut}}=10^{-12}$).

3. **非线性 PDE** (Burgers / Allen-Cahn / KdV / NLS): Picard 迭代冻结上一个 iter 的非线性部分, 在线性化残差上重复 LS 求解, 直到相对更新 $< \delta$ 或达到最大步数.

4. **长时窗口**: 分 $B$ 段 block-march, 每段重新采样 liquid 特征, 用上一段终值作下一段初值.

### 2.4 算法复杂度

- 特征评估 + 矩阵组装: $O(NP)$ 每 Picard 步.
- LS 求解 (overdetermined, $N\gtrsim P$): $O(NP^2+P^3)$ 每步.
- Local 变体还带来**patch 块稀疏**结构的实现级常数缩减.

## 3. 核心公式

### 3.1 液体原语 (Eq. 4, 5)

$$\boxed{\phi_\theta(x,t)=h_\theta^0(x)\,e^{-\alpha_\theta(x)\,t}+g_\theta(x)\,A\,\eta_0\!\left(\alpha_\theta(x),t\right),\quad \eta_0(\alpha,t)=\frac{1-e^{-\alpha t}}{\alpha},\;\eta_0(0,t)=t.}$$

其中 $\alpha_\theta(x)=\tau^{-1}+g_\theta(x)$, $g_\theta=\tanh(w^\top\zeta+b)$, $h_\theta^0=\tanh(w_0^\top\zeta+b_0)$. 论文 Eq. (12) 给出等价的稳态形式:

$$\phi_{i,k}(x,t)=s_{i,k}(x)+\left[h^0_{i,k}(x)-s_{i,k}(x)\right]E_{i,k}(x,t),\quad s_{i,k}=\frac{g_{i,k}A_{i,k}}{\alpha_{i,k}},\;E_{i,k}=e^{-\alpha_{i,k}t}.$$

这个分解的物理含义: $s_{i,k}(x)$ 是 $t\to\infty$ 时的**渐近态** (空间慢变量), $h^0_{i,k}(x)$ 是**初始 trace**, $\alpha_{i,k}$ 决定**弛豫速率** (随空间变, 因为 $\tau^{-1}+g(x)$).

### 3.2 参数采样律 (Eq. 6)

$$\log_{10}\tau\sim\mathcal{U}\!\left[\log_{10}\tau_{\min},\log_{10}\tau_{\max}\right],\quad w,w_0\sim\mathcal{U}[-R_w,R_w]^d,\;b,b_0\sim\mathcal{U}[-R_b,R_b],\;A\sim\mathcal{U}[-1,1],$$

其中 $\tau_{\min}=c_{\min}T_{\text{win}}$, $\tau_{\max}=c_{\max}T_{\text{win}}$. **关键设计**: $\tau$ 用 log-uniform 采样, 跨越多个时间尺度, 这是 L-RFM 区别于单尺度 frozen activation 的核心.

### 3.3 PoU bump (Eq. 8)

$$\tilde{b}(r)=\begin{cases}\exp\!\left(-\dfrac{1}{1-r^2}\right),&|r|<1,\\0,&|r|\geq 1.\end{cases}$$

配合 $z_k(x)=D_k^{-1}(x-x_k^c)$, 给出 $\tilde\psi_k(x)=\prod_j b(z_{k,j}(x))$, 再归一化为 $\psi_k=\tilde\psi_k/\sum_\ell\tilde\psi_\ell$, 确保 $\sum_k\psi_k\equiv 1$ 且 $\psi_k$ 全局 $C^\infty$.

### 3.4 Picard 线性化残差 (Eq. 19)

$$R^{(\ell)}(c;x,t)=\partial_t u^{(\ell)}(x,t)+N_{\text{lin}}\!\left[u^{(\ell)};u^{(\ell-1)}\right](x,t)-f(x,t).$$

例如 Burgers $u_t+uu_x=\nu u_{xx}$ 中, 把 $N[u]=u\,\partial_x u$ 冻结一个因子: $N_{\text{lin}}^{(\ell)}=u^{(\ell-1)}\partial_x u^{(\ell)}$.

### 3.5 时序秩 (Temporal Rank) 命题 (Prop. 1, App. A.2)

设 $V\in\mathbb{R}^{N\times J}$ 的列为 $v_{\lambda_j}=(e^{-\lambda_j t_1},\dots,e^{-\lambda_j t_N})^\top$ (N 个时间网格, J 个不同速率). 当 $\lambda_j$ 两两不同且 $N\geq J$ 时, $V$ 列满秩 (Chebyshev 系统性质). 这是为什么 "采样多个 $\tau$ 能扩展时序表示" 的**严格代数依据**.

### 3.6 密度定理 (Thm. 1, App. A.1)

两个引理组合:
- **Lemma 1 (时序密度)**: $\text{span}\{e^{-\alpha t}:\alpha\in[\alpha_0,\alpha_1]\}$ 在 $\mathcal{C}([0,T])$ 中稠密 (Hahn-Banach + 整函数恒等定理).
- **ridge 函数密度** (Cybenko 1989, Leshno 1991): $\text{span}\{x\mapsto\tanh(a^\top x+b)\}$ 在紧集 $X\subset\mathbb{R}^d$ 上 $\mathcal{C}(X)$ 稠密.

张量积 + Stone-Weierstrass 得到 $\mathcal{A}_{\text{loc}}$ 与 $\mathcal{A}_{\text{glob}}$ 均在 $\mathcal{C}(X\times[0,T])$ 中稠密. 这是**理论保证**: L-RFM trial space 是 PDE 解空间的 dense 子集.

## 4. 关键成果与贡献

### 4.1 验证层 (Section 4.2)

- **1-D heat equation** $u^\star=e^{-\pi^2 t}\sin(\pi x)$: 5 seed, $K=4,M=100$ 时相对 $L_2$ 误差 $(5.99\pm 2.84)\times 10^{-7}$; $K=8,M=100$ 时 $(2.01\pm 1.75)\times 10^{-5}$ (over-parameterization 撞上秩瓶颈).
- **2-D heat equation** $u^\star=e^{-2\pi^2 t}\sin(\pi x_1)\sin(\pi x_2)$, $K=4$ ($2\times 2$ 网格), $M=60$: 相对 $L_2$ 误差降到 $(2.57\pm 0.20)\times 10^{-3}$, 验证张量积 PoU + Laplacian 的多维一致性.

### 4.2 匹配容量主表 (Table 4, $P$ 固定, 1-D 代表 PDE)

| 问题 | ST-RFM-SoV | ST-RFM-STC | PIELM | L-RFM-Local | L-RFM-Global |
|---|---:|---:|---:|---:|---:|
| Allen-Cahn ($P{=}400$) | $4.09\!\times\!10^{-6}$ | $1.04\!\times\!10^{-5}$ | $3.22\!\times\!10^{-6}$ | **$5.98\!\times\!10^{-8}$** | $3.78\!\times\!10^{-7}$ |
| Burgers ($P{=}480$) | $7.39\!\times\!10^{-6}$ | $4.56\!\times\!10^{-6}$ | $1.09\!\times\!10^{-5}$ | $3.30\!\times\!10^{-5}$ | **$1.81\!\times\!10^{-6}$** |
| KdV ($P{=}480$) | $4.63\!\times\!10^{-3}$ | $6.16\!\times\!10^{-3}$ | $1.60\!\times\!10^{-2}$ | **$1.61\!\times\!10^{-3}$** | $3.57\!\times\!10^{-3}$ |
| NLS ($P{=}960$) | $8.93\!\times\!10^{-4}$ | $2.30\!\times\!10^{-3}$ | $1.82\!\times\!10^{-2}$ | **$8.64\!\times\!10^{-5}$** | $2.25\!\times\!10^{-4}$ |

> **关键观察**: 同一 $P$ 下, liquid trial space 在 4 个 PDE 上**都拿到最低 mean error**. L-RFM-Local 适合有局部结构的问题 (Allen-Cahn 界面, KdV 三阶, NLS 孤子); L-RFM-Global 在 Burgers 这种时空光滑的问题上更高效. 这与**"局部化与时序响应是两个独立的设计维度"**的设计哲学一致.

### 4.3 时序多尺度测试 (Section 4.4.1, Eq. 24)

人造多尺度目标 $u^\star(x,t)=\sin(\pi x)\sum_{j=1}^{3}\tfrac{1}{3}e^{-t/\tau_j}$, $\tau=(1,10^{-2},10^{-4})$:
- L-RFM-Global @ $P=640$: **$4.22\!\times\!10^{-4}$**.
- ST-RFM-SoV: $6.33\!\times\!10^{0}$ (差 4 个数量级).

这是**最干净**的时序尺度证据: 没有非线性界面干扰, 直接证伪 "静态 ridge 也可以" 的假设.

### 4.4 刚性扩散尺度扫描 (Table 5, Allen-Cahn @ $P=400$)

| $\epsilon$ | ST-RFM-SoV | ST-RFM-STC | PIELM | L-RFM-Local | L-RFM-Global |
|---|---:|---:|---:|---:|---:|
| $10^{-1}$ | $1.28\!\times\!10^{-5}$ | $1.95\!\times\!10^{-5}$ | $1.57\!\times\!10^{-6}$ | $2.92\!\times\!10^{-7}$ | **$2.89\!\times\!10^{-7}$** |
| $10^{-4}$ | $1.11\!\times\!10^{-5}$ | $1.14\!\times\!10^{-5}$ | $2.22\!\times\!10^{-5}$ | **$3.09\!\times\!10^{-7}$** | $8.67\!\times\!10^{-7}$ |

随着 $\epsilon$ 缩小 (过渡区变锐), L-RFM-Local 的相对优势**扩大**: L-RFM-Local 在 $\epsilon=10^{-4}$ 上比 best static local row 小 ~36 倍. raw LS condition number 也小 2-5 个数量级 (Table 6).

### 4.5 消融 (Section 4.4.3, Figure 7)

- **去掉 ODE 时序响应 (用 $t=0$ 静态替换)**: Heat 1D 误差从 $3.28\!\times\!10^{-6}$ 暴增到 $9.13\!\times\!10^{-1}$; Allen-Cahn 从 $9.66\!\times\!10^{-8}$ → $1.84\!\times\!10^{-1}$; KdV 从 $1.63\!\times\!10^{-3}$ → $3.66\!\times\!10^{-1}$. 全部问题**退化 5-8 个数量级**.
- **固定 $\tau$ (丢掉 log-uniform 多尺度采样)**: Allen-Cahn 上 $\tau/T=0.05$ 时退化到 $2.49\!\times\!10^{-1}$; $\tau/T=20$ 时反而能到 $5.20\!\times\!10^{-8}$ (与 log-uniform 的 $7.55\!\times\!10^{-8}$ 接近). 结论: **ODE 时序响应是绝对主导, log-uniform 主要提升对"未知时序尺度"的鲁棒性**.

### 4.6 条件数 (Table 6, Appendix B Table B.10)

| 问题 | $\kappa(A_w)$ L-RFM-Local | $\kappa(A_w)$ ST-RFM-Local |
|---|---:|---:|
| Heat 1D | $5.55\!\times\!10^{13}$ | $1.31\!\times\!10^{16}$ |
| Allen-Cahn 1D | $2.75\!\times\!10^{13}$ | $6.13\!\times\!10^{16}$ |
| NLS 1D | **$1.02\!\times\!10^{11}$** | $2.53\!\times\!10^{16}$ |

L-RFM-Local 的 row-weighted LS 矩阵条件数在 tested difficult rows 上**比静态基低 2-5 个数量级**. 这是"liquid time-constant + 局部 patch"组合的**几何红利** — 多尺度 $\tau$ 提供有意义的时序方向, 局部 PoU 抑制空间列对齐.

### 4.7 计算开销 (Table 7)

Allen-Cahn 1D 在相对 $L_2=10^{-4}$ 目标下:
- L-RFM-Global: 误差 $1.40\!\times\!10^{-6}$, 时间 $4.86$ s
- L-RFM-Local: 误差 $3.31\!\times\!10^{-6}$, 时间 $18.5$ s (PoU 与多 patch 装配更贵)
- PIELM: 误差 $4.77\!\times\!10^{-5}$, 时间 $0.96$ s

**精度高 1-2 个数量级, 时间高 1.5-19 倍**: 在 stiff PDE 上, L-RFM 用几秒到十几秒换来了无法用静态基达到的精度. 当前实现还**没有**做特征评估向量化 / 稀疏块装配 / Picard LS 分解复用, 论文明确指出这些是加速方向.

## 5. 局限性与未来展望

### 5.1 论文作者承认的局限性

1. **多维度经验覆盖不够**: 文中 2-D / 3-D 仅做了已知解验证 (heat, 3-D 时序尺度测试), 缺少多维非线性 benchmark (Allen-Cahn 2-D, Burgers 2-D).
2. **理论缺口**: 仅有 density theorem (Thm. 1) + 时序秩命题 (Prop. 1) + 经验条件数; 缺**有限 $M$** 的 PDE 误差估计, 缺 $\kappa(A_w)$ 的理论刻画, 缺 Picard 收敛与 block-marching 误差传播的严格界. 作者明确点名这些都是"自然的下一步".
3. **计算成本**: 当前 dense LS $O(NP^2+P^3)$ 在大 $P$ 时开销可观; 论文未提供稀疏 block 装配的实测加速比 (只预告了方向).
4. **Picard 适用性边界**: 当 $N[u]$ 是**强非线性算子** (e.g. 乘法依赖当前 iter 的高阶导数) 时, Picard 收敛速度可能很慢甚至不收敛, 文中没有给出 Picard 失败的 case study.
5. **采样律依赖超参**: $c_{\min}, c_{\max}, R_w, R_b$ 需要"按 benchmark 配置"; 没有给出**自适应的** $\tau$ 采样策略 (例如基于 PDE 特征值尺度自动设定).

### 5.2 与 LNN/CfC/LTC 主线的桥接 (本仓视角)

- **LTC → frozen feature**: L-RFM 把 Hasani 等的 LTC 神经元响应**抽出来当 frozen feature**, 不再依赖任何可学习参数, 转而用线性 LS 拟合 readout. 这是 "LTC 思想离开序列模型, 进入算子代理 / PDE 求解"的一次有意义迁移.
- **与本仓 `CfCCell` 的关系**: 本仓 `lnn/core/liquid_tad.py` 中的 `PLRCfCCell` 把 PLR (ODE-1 闭式解) 与 CfC (closed-form continuous-time) 串联, 用做序列建模. L-RFM 的核心方程 $\phi=h_0 e^{-\alpha t}+gA\eta_0$ 与本仓 PLR / CfC 数学同源, 区别只在**"是否冻结参数"**:
  - LNN 主流 (含本仓): 参数**学**, 序列建模, 边缘部署.
  - L-RFM: 参数**冻结**, PDE 代理, 网格无关, 不需要 GPU 训练.
- **对本仓的可能启发**: 本仓 round 系列目前聚焦序列建模 + 边缘部署 + 1-D 时间算子. 若未来扩展到 2-D 时空 PDE 代理 (e.g. 连续介质建模、扩散过程), L-RFM-Local 的"PoU + frozen liquid + LS readout"是一个**有理论保证的轻量起点**, 不需要引入 PINN 的训练复杂度.

### 5.3 未来研究方向 (作者列出 + 本仓补充)

| 方向 | 来源 | 与本仓的潜在关联 |
|---|---|---|
| 多维非线性 benchmark (2-D Allen-Cahn, 2-D NLS) | 作者 | 若本仓要做 LNN PDE 代理 benchmark, 直接对照 |
| 有限 $M$ 误差估计 + 条件数理论 | 作者 | 给本仓 frozen-feature LTC 提供"何时胜出"的判据 |
| 自适应 $\tau$ 采样 (e.g. 基于 PDE 谱) | 作者 | 本仓若做"动态 LTC" 时, 自动设定 $\tau$ 分布 |
| 向量化特征评估 + 稀疏 block 装配 + Picard 分解复用 | 作者 (Section 5) | 实际加速 10×-100× 的工程路径 |
| 复数域 / 量子 PDE (Schrödinger, BEC) 拓展 | 本仓视角 (L-RFM 已支持 NLS) | 量子动力学方向可借鉴 |
| 与 operator learning 混合: L-RFM 做特征层, 后接 FNO / DeepONet 输出层 | 本仓视角 | "frozen liquid features + learned readout" 是天然中间形态 |

## 6. Verdict

- **学术新颖度**: HIGH. 首次把 LTC 闭式响应**作为 frozen feature primitive** 嵌入 mesh-free PDE LS 求解器, 同时给出密度定理 + 经验条件数 + 6 个 benchmark 的系统验证.
- **与 LNN 主线契合度**: HIGH. 数学同源于 CfC/LTC 闭式路径, 但把"训练"换成"冻结+线性 LS", 扩展到 PDE 算子代理. 对本仓"两轴 (PLR+CfC)"思路是**有意义的概念变体** — 把学到的两轴换成先验的两轴.
- **可复现性**: HIGH. 27 页 PDF 含完整算法 (Algorithm 1) + 协议附录 (App. B) + benchmark 参数表 (Table B.8) + 五 seed 报告. 唯一缺开源代码 (Elsevier 投稿风格, 论文未承诺 release).
- **本仓 1-D 序列视角下的可复现性**: MEDIUM. 本仓目前没有 frozen-feature PDE 代理入口; 若要做对照, 需要新增 `lnn/core/l_rfm_cell.py` (frozen liquid feature) + `lnn/core/l_rfm_solver.py` (LS 装配 + Picard), 跑 Burgers / Allen-Cahn 这两条 1-D 路径. 估时: 1 天内可完成 toy 版, 与本仓 `CfCCell` / `LiquidTAD` 并行不冲突.
- **本仓 round 系列定位建议**: 不进入 round 立即落地, 但**加入"算子代理方向"作为未来 round 候选**. 论文给出的关键 insight ("多尺度 log-uniform $\tau$ 采样 + 局部 PoU 抑制列对齐") 对本仓"两轴 + 多尺度 + 局部化"研究路径具有概念启发价值.

## 7. 一句话总结

> L-RFM 把 "liquid time-constant" 响应从序列神经元搬进偏微分方程求解器, 用**冻结的多尺度 $\tau$ 采样 + 解析闭式导数 + 加权残差最小二乘**, 在 Allen-Cahn / Burgers / KdV / NLS 这 4 个代表性 1-D 时间依赖 PDE 上, 以匹配的 readout 维度取得 1-4 个数量级的精度优势, 同时 LS 条件数低 2-5 个数量级; 它是 LTC 思想在 PDE 代理方向的一次干净落地, 与本仓序列建模路线数学同源但应用域不同.