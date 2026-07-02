# LNN 研究报告 — 2026-07-03 (Round 278, /loop session #2)

## 1. 文献检索总结(液态神经网络 2026 非平稳性主线)

本轮聚焦"input-dependent τ 在非平稳序列上的响应性 vs 稳定性"这一
张力,学术检索命中若干直接相关工作:

- **Liquid NN for Natural Gas Spot Price** (arXiv:2604.24788, 2026)
  —— 关键句 *"limit responsiveness when market regimes shift
  rapidly"*。明确把 LNN 预测框定为**非平稳时序**问题,并指出
  naive 的高响应性在 regime 快速切换时反而有害。用了 Strict CfC /
  Hybrid CfC / CT 多个变体。
- **Urban-flood CfC proxy** (Liu et al., Water Research 2026) ——
  用 CfC 动态把 LNN 应用于城市洪水模拟,强调 *"discrete-time
  networks are difficult to adapt to the non-stationary
  characteristics"*。
- **SCTP-Net** (2026) —— 多阶段 continuous-time propagation,用于
  非平稳混凝土排量异常识别,结论 *"highly dependent on gating
  dynamics"*。
- **PM2.5 长期预测**(Remote Sensing 2025)—— gating CfC 用于长程
  时空图预测。
- **LTC 原文**(AAAI 2021)复核 canonical 公式
  `τ_sys = 1 + f(x(t),I(t),t,θ)` —— τ 本质就是**输入依赖**的,
  且"single hidden-state elements identify specialized dynamical
  systems for input features arriving at each time-point"。

### 共同主题
> **响应性(responsiveness)必须选择性:对结构化 regime 切换要响
> 应,对不可预测噪声要抑制。** 这正是 r277 遗留问题的理论出处。

## 2. 承接 r277 的研究缺口

r277 引入 input-dependent(liquid)τ,得到 **target-dependent
正面**:toy_sin -59% / structured -12%(赢),但 **random +106%**
(τ 追噪声,over-adapt)。r277 报告已提出修复方向:**用信号可预测
性 gate liquid 强度**(类比 r99 reliability gate)。arXiv:2604.24788
恰好给出理论框架。

## 3. 新思路 — Round 278:Predictability-Gated Liquid τ

**parameter-free 可预测性门控**:
```
vol_t = EMA_0.5(mean_c |x_t − x_{t-1}|)   # 因果输入波动率
g_t   = exp(−4·vol_t) ∈ (0,1]             # 平滑→1, 噪声→0
τ_i(t)= tau_min+(tau_max−tau_min)·σ(bias_i + g_t·s·(W_τ·[x_t,h])_i)
```
门无可学习参数 ⇒ **结构上无法学会追噪声**,从根上禁掉 r277 失败
模式。beta=0 ⇒ 严格等价 r277(超集)。

## 4. 验证结果(27 cell,100 epoch)

| 数据集 | static | liquid(r277) | gated(r278) | gate |
|---|---|---|---|---|
| toy_sin | 0.000031 | **0.000013(-59%)** | 0.000044(+41%) | 0.79 |
| structured | 0.000171 | 0.000150(-12%) | 0.000167(**-2.5%**) | 0.84 |
| random | 1.002469 | **2.066662(+106%)** | 1.005347(**+0.3%**) | 0.06 |

**H2 确认(头条)**:门控把 random 回归从 +106% 修到 **+0.3%**,
gate 塌到 0.06,τ 几乎冻结(tau_tstd 0.03 vs liquid 最高 0.29)。

**诚实权衡**:因为 EMA(|Δx|) 对干净正弦波也非零,门在 toy_sin 上
偏保守(gate=0.79),牺牲了 r277 的 -59% 头条胜绩(gated 变
+41%)。所以:
- liquid(r277)= 上限高、下限灾难(平滑最好、噪声最差);
- gated(r278)= 稳健、最坏情形有界(+41% 是其最差,不再 +106%)。

**判定**:**honest positive with tradeoff** —— 一个稳健性机制,
不是严格 Pareto 改进。部署未知/混合分布时选 gated(最坏有界);
已知平滑/结构化时选 ungated liquid(上限高)。

## 5. 下一步思路(r279 候选)

用 **volatility-relative gate**(按序列 z-score 归一化波动率),
让干净周期信号读作"可预测"(gate→1)而非"中等波动",以恢复
toy_sin 胜绩而不重新引入噪声爆炸。或让 gate sharpness(beta)可
调度。

---
*Generated in /loop 1h session #2, 2026-07-03. Sources: dataPro
academic_search — arXiv:2604.24788, urban-flood CfC (Water Research
2026), SCTP-Net 2026, PM2.5 gating-CfC (Remote Sensing 2025),
LTC AAAI 2021.*
