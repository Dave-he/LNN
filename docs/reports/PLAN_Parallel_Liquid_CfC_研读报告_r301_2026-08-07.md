# PLAN 研读报告 — arXiv:2608.03041v1 (Kannan et al. 2026)

> Round 301 (2026-08-07) — 研读 + PLAN-CfC 复现 + toy_sin 5-seed 验证

## 1. 论文元信息

- **标题**: PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling
- **作者**: Dhivya Dharshini Kannan, Wei Zhang, Jieyi Bi, Yingpeng Du, Tianjun Wei, Jie Zhang, Zuming Liu, Anupam Trivedi
- **arXiv**: [2608.03041v1](https://arxiv.org/abs/2608.03041v1) (2026-08-04)
- **任务域**: 柔性作业车间调度 (FJSP) 的深度强化学习 — DRRL state-encoder
- **核心思路**: 把顺序液态神经网络的"sequential liquid-state dynamics"重写为可并行的离散形式,解耦 state evolution 与 context aggregation

## 2. 关键贡献 (paper §3 改写)

1. **Discretized parallel liquid dynamics**: 在单次 batched matmul 内评估 W 步 liquid-state 更新,丢弃中间 h_t 的串行依赖(以 h_0 为 anchor)
2. **Decoupled context aggregation**: 引入一个轻量级 context aggregator 提供全局补充信息,与 state evolution 路径解耦
3. **Plug-and-play backbone**: PLAN 可作为 FJSP DRL 的 state-encoder 替换 heterogeneous graph transformer,也可与紧凑 stochastic module 配对做随机 FJSP

## 3. 数学与算法

### 3.1 传统 LNN 顺序更新 (Hasani 2021 / Lechner 2022)

```
h_{t+1} = liquid_step(h_t, x_t; θ)
```

需 O(T) 顺序 forward,GPU 无法利用 batch 维并行。

### 3.2 PLAN 的并行化

对窗口 W = {t, t+1, …, t+W-1}:

```
对每个 s ∈ {0, 1, …, W-1}:
  f_s = σ(Wf · [x_{t+s} ; h_anchor] + bf)     # h_anchor = h_t
  g_s = tanh(Wg · [x_{t+s} ; h_anchor] + bg)
  hp_s = tanh(Wh · [x_{t+s} ; h_anchor] + bh)
  h̃_s = σ(-f_s · τ · dt) * g_s + (1 - σ(-f_s · τ · dt)) * hp_s
  h̃_{t+W} = h̃_{W-1}   # 末态作为下一窗口 anchor
```

**核心近似**: 在窗口 W 内假设 h_t ≡ h_anchor,丢弃中间 h 的串行修正。

### 3.3 与 LNN/CfC 的对应

PLAN 的"discretized liquid state"在数学上等价于 **CfC 闭式更新在固定 h_anchor 下的 batched evaluation**:
- PLAN 论文没有用 "CfC" 这一名字,但其方程组(一个 sigmoid-gated tanh-blend closed-form update)与 Lechner 2022 的 CfC 几乎完全同构
- 区别在于 CfC 顺序依赖 h_{t-1},而 PLAN 在 W 内固定 anchor

## 4. 论文实验结果 (paper §5)

| 基准 | makespan 改进 | 推理延迟降低 | 参数占比 |
|---|---:|---:|---:|
| Deterministic FJSP | -1.2% | -13.2% | 47% |
| Stochastic FJSP | -1.4% | -31.7% | 30% |
| Multi-faceted FJSP | -2.3% | -26.9% | 22% |
| (max) | -10.2% | -69.2% | — |

**诚实说明 (paper §6.3)**: PLAN 在 *inter-step state transitions* 剧烈的任务上退化 — 与我们在 r152 tdsa_cfc (self-attention 在 T=32 不足) 的发现一致。

## 5. LNN 复现: ParallelCfC

### 5.1 实现

文件:
- `lnn/core/parallel_cfc.py` (172 行) — `ParallelCfCCell` + `ParallelCfCNetwork`
- `tests/test_parallel_cfc.py` (21 tests, 全部通过)
- `scripts/bench_parallel_cfc.py` (toy_sin 5-seed 协议)

关键设计:
- `window=1` 路径退化为 vanilla CfC 行为(无 anchor approximation)
- `window>1` 路径在窗口内使用 h_0 anchor 并行评估 CfC 闭式更新
- 跨窗口顺序: 上一窗口的 h_W 作为下一窗口的 h_0
- 窗口 W 必须整除序列长度 T(显式 assert)

### 5.2 toy_sin 协议

复现我们仓库 r155-r200 系列 toy_sin 协议 (平滑混合正弦):
- `y(t) = sin(2π·1.5·t) + 0.3·cos(2π·4.7·t)`, `t ∈ [0, 1]`
- 256 train / 64 test, T=64, 5 seeds
- 3-layer 不必要(单层即可), h=64, Adam(lr=2e-3), 100 epochs

### 5.3 结果 (5-seed mean ± std)

| 模型 | MSE | Δ vs vanilla | 推理延迟 (10 pass) | Δ latency |
|---|---:|---:|---:|---:|
| vanilla_cfc | 0.11372 ± 0.00467 | — | 14.30 ms | — |
| parallel_w2 | 0.11225 ± 0.00059 | **-1.3%** | 9.96 ms | **-30%** |
| parallel_w4 | 0.10733 ± 0.00107 | **-5.6%** | 7.66 ms | **-46%** |
| parallel_w8 | 0.10564 ± 0.00225 | **-7.1%** | 5.74 ms | **-60%** |

### 5.4 解读 (含 honest finding)

1. **STRONG POSITIVE — Pareto 改进**: W=8 同时实现 -7.1% MSE 和 -60% 推理延迟。这与 PLAN 论文"13-69% latency reduction, 22-47% params" 的方向一致,且我们额外观察到 **MSE 也改善**。
2. **方差塌缩**: vanilla_cfc σ=0.0047, parallel_w2 σ=0.0006 — W=2 的 std 降低 **7.8×**。anchor 假设可能起到了 *implicit regularization* 作用,减少跨种子的可变性。
3. **HONEST CAVEAT — anchor 假设的代价**: 我们在 toy_sin (周期平滑) 上验证;对"sharp inter-step transitions" 类任务(论文 §6.3 自承)未测试。W=8 的 MSE 仅比 W=4 改善 1.6%,而延迟再降 25% — 边际收益递减,提示 anchor 假设在更长的窗口开始失效。
4. **未测**: 长序列 (T>128) / 不规则 Δt / 分类 / 多变量回归 — 后续 rounds 需要。

## 6. 与 LNN 既有研究的相关性

- **r244-r256 Basin-Lyapunov**: PLAN 的 anchor 假设可视为"anchor basin" — 窗口内 h_0 是吸引子,parallel step 是该 basin 内的离散轨。值得在 r257+ inter-basin-distance 框架中验证:window 内是否有 *basin switching*?
- **r265-r272 STE Neuron-Wise**: 离散 neuron-wise 路由 + PLAN 的并行 CfC 可组成"router-parallel-cell" 协同;STE 的 L1 折叠 (r266) 与 PLAN 的 anchor 假设同属"显式简化 ODE" 的不同路径。
- **r299 TopologicalCfC**: TopologicalCfC 的 sparse graph-mix 是 *inter-neuron* simplification;PLAN 是 *inter-timestep* simplification — 两者正交,可串联。
- **LFM2.5 边缘部署**: PLAN 的 22-47% 参数占比对 Jetson Orin Nano 的 memory budget 直接友好,值得在 lfm2 目录下加 PLAN-CfC 替换 nn.LSTM 的 demo。

## 7. 后续工作

- **r302**: 在 N-MNIST / EMMA rover regression / Long-Sequence Arena 上验证 PLAN-CfC,测 sharp-transition 退化
- **r303**: 联合 STE routing + PLAN-CfC,看离散路由能否补偿 anchor 误差
- **r304**: 把 PLAN-CfC 接入 LFM2.5 推理 demo,测 TTFT/TPOT 影响
- **r305**: 探索 non-anchor parallel scan(真正的 parallel prefix-sum 形式),需对 CfC 做线性化近似

## 8. 数据源

- 论文: https://arxiv.org/abs/2608.03041v1
- 复现代码: `lnn/core/parallel_cfc.py`
- 测试: `tests/test_parallel_cfc.py` (21/21 通过)
- Bench 结果: `bench_parallel_cfc_results.json`
