# LNN 研究报告 — 2026-07-03（/loop 1h session）

## 1. 文献检索总结（液态神经网络最新进展）

本轮通过学术检索（dataPro academic_search）聚焦 2026 年液态神经
网络（LNN / LTC / CfC）的最新论文，命中若干高相关工作：

### 1.1 LTC + 自适应动态（Adaptive Dynamics）
- **Efficient Semantic Segmentation via Liquid Time-constant
  Networks with Adaptive Dynamics** (A. Al, 2026)
  - 将 LTC 网络与自适应动态结合用于语义分割，强调把
    **CfC 动态**从"通用通道门控（generic channel gating）"中
    **隔离出来**做消融，证明 liquid dynamics 本身带来收益。
  - 结论：liquid 动态在保持竞争性能的同时**大幅提升效率**。
  - 关键词：*adaptive dynamics*, *isolate CfC dynamics from
    generic gating*。

- **Liquid Time-Constant Networks for Water-Level Forecasting in
  Urban Drainage** (R. Buczyński, 2026, SSRN)
  - 城市排水水位预测，核心是一个**门控机制计算自适应连续时间
    动态**（"A gating mechanism computes ... adaptive
    continuous-time dynamics for hydrological processes"）。
  - 再次印证：**输入依赖的时间常数**（input-dependent τ）是
    LTC 在非平稳序列上的关键。

### 1.2 LNN 用于金融/能源时序
- **Liquid Neural Network Models for Natural Gas Spot Price
  Time-Series Forecasting** (Y. Liu, J. Niu, A. Kelleher,
  S. Das, arXiv:2604.24788, 2026)
  - Henry Hub 天然气现货价短期预测；用 **Hybrid CfC 与 LTC**。
  - 论点：LNN 通过**动态内部状态更新持续适应演化的时间模式**，
    特别适合**非平稳（nonstationary）**价格行为与频繁的
    regime change。

### 1.3 Wi-Fi / 信道质量预测
- **Wi-Fi Channel Quality Prediction with Liquid Neural
  Networks** (S. Scanzio et al., 2026, IEEE)
  - **诚实负面细节**：作者报告"对 CfC 未观察到改进（for CFC we
    did not notice any improvements），仅 LTC 模型受益于 early
    stopping"。提示：CfC 的收益是**任务/数据依赖**的——与本仓
    91-276 轮反复得到的 "target-dependent" 结论一致。

### 1.4 综合观察
横跨全部 2026 命中论文的**共同主题**是：
> **自适应 / 输入依赖的时间常数（adaptive, input-dependent time
> constant）是 LTC/CfC 的核心增益来源，尤其在非平稳序列上。**

## 2. 本仓现状与研究缺口

本仓 263-276 轮走的是 **STE（straight-through estimator）稀疏化**
主线，基于 `NeuronWiseCfCCell`：

- **前向**：hard top-k 二值 mask（真稀疏）
- **反向**：soft sigmoid mask（梯度可流）
- 已完成 (τ, λ, hidden, T, d_in, density) 超参 sweep（267-275）。

**关键缺口（gap）**：本仓的 `NeuronWiseCfCCell` 使用的是
**静态、可学习的 per-neuron τ**（`tau_per_neuron` 是一个
`nn.Parameter`，训练后固定，不随输入变化）：

```python
# lnn/core/neuron_wise_cfc.py:203-204
raw = torch.sigmoid(self.tau_per_neuron)      # (d_h,), 与 t 无关
return self.tau_min + (self.tau_max - self.tau_min) * raw
```

这**恰恰丢失了 LTC 定义性的"liquid"特性**：Hasani 2021 的 LTC
以及上述所有 2026 论文都强调 τ 应当是**输入依赖**的
（`τ = τ(x_t, h_{t-1})`）。我们的 STE 主线是在一个"半液态"
（static-τ）基座上做稀疏，从未测过恢复真正的 input-dependent τ
是否有增益。

## 3. 新研究思路 — Round 277：Liquid（输入依赖）τ

### 3.1 假设
在 STE 稀疏 CfC 上，把 per-neuron τ 从**静态参数**升级为
**输入依赖的门控**：

```
τ_i(t) = τ_min + (τ_max − τ_min) · sigmoid( a_i + W_τ · [x_t, h_{t-1}] )_i
```

其中 `a_i` 是保留的 per-neuron 偏置（→ λ=0 时退化为 r265 静态 τ，
构成 strict superset），`W_τ` 是新的小门控层。

### 3.2 待验证问题
- **H1**: liquid τ 在 ≥1 数据集上优于 static τ（尤其 structured
  这类有 regime/分段结构的数据——对应论文的 nonstationary 主张）。
- **H2**: liquid τ 在 toy_sin（平滑单频）上**不劣于**静态 τ
  （平滑数据不需要自适应，预期打平或轻微负——延续本仓
  "平滑数据上花哨机制常是税" 的模式）。
- **H3**: gate 初始化为 0 → 训练初期严格等价 static τ（superset
  性质），确保不引入训练不稳定。
- **H4**: liquid τ 的 τ 分布随时间的方差 > 0（真正在"流动"，
  而非塌回常数）。

### 3.3 预期结果分类
- 若 structured 显著改善且 toy_sin 打平 → **target-dependent
  正面**（与 2026 论文主张一致：自适应 τ 帮助非平稳数据）。
- 若全面打平 → **诚实负面**：在 1D toy regime 下，static
  per-neuron τ 已足够（本仓 toy 数据的 regime 变化太弱）。
- 两种结果都有信息量，都记录。

## 4. 本轮附带完成的遗留工作

- **Round 276（STE × Batch Size sweep）** 上一 session 仅跑完
  1/45 cell。本轮补齐全部 45 cell（5 batch × 3 dataset × 3 seed）
  并修复了 bench 汇总打印在 <3 数据集时的 `IndexError`（改为按
  `args.datasets` 动态生成表头/列）。

---
*Generated in /loop 1h session, 2026-07-03. Sources: dataPro
academic_search (LTC adaptive dynamics 2026, arXiv:2604.24788,
Wi-Fi LNN 2026, urban-drainage LTC 2026).*
