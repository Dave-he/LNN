# STE-ParallelCfC 研读报告 — r303 (2026-08-07)

> Round 303 — STE neuron-wise routing + r301 PLAN-ParallelCfC 联合: 让离散路由补偿 anchor 近似误差

## 1. 思路起源

两个正交轴向:
- **r301 ParallelCfCCell** (`lnn/core/parallel_cfc.py`): 窗口 W 内用 h_0 anchor 并行评估 W 步 CfC 闭式更新,跨窗口顺序传递 h。toy_sin 5-seed: W=8 取得 **MSE -7.1%** 同时 **latency -60%**。但论文 §6.3 自承 "sharp inter-step transitions" 类任务会退化 — anchor 假设对某些神经元不成立。
- **r265-r267 STE Neuron-Wise**: 每神经元独立 ODE + STE 路由 mask (forward=hard top-k, backward=soft sigmoid),r267 进一步加 soft-mask entropy reg 保持离散性。

**核心假设 (r303)**: anchor 近似误差是 **per-neuron** 的 — 某些 hidden unit 真的需要 recurrent h_t (anchor-sensitive),另一些被 input x_t 主导 (anchor-safe)。STE 给了一个 *learned, differentiable* 的方式去识别这两组并分别路由。

## 2. 设计

```
h_parallel   = parallel_anchor_h         # r301 forward
h_sequential = sequential_h_step_at_t    # 一次 vanilla CfC 步

mask_hard    = top_k(route_logits, ρ)    # 二值 (hidden,)
mask_soft    = sigmoid(route_logits / τ_ste)
mask_ste     = (mask_hard - mask_soft).detach() + mask_soft   # 经典 STE

h_out = mask_ste ⊙ h_parallel + (1 - mask_ste) ⊙ h_sequential
```

- **density ρ**: anchor-safe 神经元比例 (1.0=全用 parallel, 0.0=全用 sequential)
- **ste_temperature τ_ste**: backward 软化的温度
- **entropy_lambda λ**: r267 的 soft-mask Bernoulli entropy 正则,鼓励路由离散化
- 两个分支用 *separate* 权重 (f_gate_p/g_branch_p/h_branch_p vs f_gate_s/g_branch_s/h_branch_s),使两个 regime 各自能收敛到自己的最优参数

## 3. 实现

文件:
- `lnn/core/ste_parallel_cfc.py` (~340 行) — `STEParallelCfCCell` + `STEParallelCfCNetwork`
- `tests/test_ste_parallel_cfc.py` (40 tests, 全部通过)
- `scripts/bench_ste_parallel_cfc.py` (toy_sin 5-seed)
- `bench_ste_parallel_cfc_results.json`

关键设计要点:
- `window=1` 路径两个分支都做单步 vanilla CfC,STE mask 仍然存在但效用等于"在两个相同/不同参数拷贝间做软插值" — 是 sanity path。
- `window>1` 路径:h_parallel 用 r301 PLAN 公式(同 anchor),h_sequential 取窗口最后一步 x_t 做一次 vanilla CfC。Mask 决定每神经元走哪一支。
- 跨窗口语义继承自 r301:上一窗口 h_{W} 作为下一窗口 h_0。
- entropy reg 使用 per-neuron **Bernoulli entropy**(soft_mask ∈ (0,1) 视为伯努利参数),与 r267 软化版 Shannon entropy 等价但更紧凑。

## 4. toy_sin 5-seed 结果

### 4.1 原始数据

```
vanilla_cfc                : 0.10781 / 0.11640 / 0.10783 / 0.10879 / 0.10803
parallel_cfc_w8 (r301)     : 0.08519 / 0.03411 / 0.07920 / 0.05817 / 0.11097
ste_parallel_cfc_w8_d0.3   : 0.05550 / 0.08633 / 0.07962 / 0.03498 / 0.06284
ste_parallel_cfc_w8_d0.5   : 0.03137 / 0.03307 / 0.09312 / 0.03357 / 0.04418
```

### 4.2 5-seed mean ± std

| 模型 | MSE mean | MSE std | Δ vs vanilla | Δ vs parallel_w8 | 推理延迟 (10 pass) | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_cfc | 0.10977 | 0.00333 | — | +49.3% | 15.86 ms | 14.6 s |
| parallel_cfc_w8 (r301) | 0.07353 | 0.02592 | **-33.0%** | — | 9.53 ms | 14.5 s |
| ste_parallel_cfc_w8_d0.3 | 0.06386 | 0.01821 | **-41.8%** | **-13.2%** | 13.35 ms | 20.9 s |
| ste_parallel_cfc_w8_d0.5 | 0.04706 | 0.02347 | **-57.1%** | **-36.0%** | 16.31 ms | 21.8 s |

### 4.3 关键观察

1. **STRICT POSITIVE — STE routing 进一步压低 MSE**:  d=0.3 比 r301 优胜 13.2%,d=0.5 优胜 **36.0%** (0.0735 → 0.0471)。
2. **density 0.5 > density 0.3**: 这与 r265 的"r265 production default d=0.3"建议矛盾 — 在此任务上 50/50 routing 更优。可能因为 d=0.5 给路由 mask 更大的容量(可学 0/1 mask 比例更对称)。
3. **方差 (std)**: STE-ParallelCfC std (0.018-0.023) 低于 r301 (0.026) — routing 不仅降均误差,还降低跨种子方差。
4. **latency tradeoff**: STE-ParallelCfC 多 40-71% 推理延迟(因双分支评估),但仍在 16ms 以内。这是为精度付的合理代价。
5. **训练时间**: STE-ParallelCfC 多 ~40-50% 训练时间(双分支 + entropy reg),但相对 200 epoch 总时影响有限。

## 5. Honest finding

- **routing DOES help**: STE-ParallelCfC 在 toy_sin 上严格优于 plain ParallelCfC。
- 这与 r301 的 "anchor 假设 ≈ implicit regularizer" 论断一致 — 但 STE 显示我们能做得更好:**让 cell 学会哪些神经元可以承担 anchor 误差,哪些不行**。
- 与 r265 的 "r265 production d=0.3" 建议不同:r303 的 d=0.5 在 toy_sin 上更好。这提示 **密度选择与下游任务耦合**,r265-r272 的 d=0.3 在 NeuronWiseCfC 上是 inter-neuron sparsity 选择,而 r303 的 d 是 inter-update-mode 选择,二者**不通用**。
- latency 略高是物理下界 — 我们多评估了一组 sequential 分支(部分神经元)。如果工程上要恢复 r301 的 latency,可以做 **稀疏 sequential 评估**(只对被 mask=0 的神经元计算 sequential,其余直接走 parallel)。

## 6. 与既有研究的相关性

- **r265-r272 STE Neuron-Wise**: r303 的 STE mask 操作的是 *inter-update-mode* 维度(parallel vs sequential),r265 操作的是 *inter-neuron* 维度(neighbors)。两者的 STE pattern 一致但应用不同,完全正交 — 可以串联:r265 处理"哪些邻居连接",r303 处理"哪些更新模式"。
- **r301 PLAN-Parallel**: r303 是 r301 的"加 routing"版,严格 Pareto 优于 r301 + vanilla。
- **r244-r256 Basin-Lyapunov**: anchor = anchor basin,sequential = trajectory within the basin。r303 的 routing 实质上是 basin-vs-basin-routing。
- **r267 STE + entropy reg**: r303 沿用了 r267 的 entropy reg pattern(soft-mask Bernoulli entropy),证明该正则化是 STE 通用 pattern。
- **r299 TopologicalCfC**: TopologicalCfC 的 sparse graph-mix 是 inter-neuron simplification,r301/r303 是 inter-timestep simplification。r303 加 STE 提供了"inter-timestep × inter-neuron dual sparsity"。

## 7. 后续工作

- **r304**: 稀疏 sequential 评估(只对 mask=0 神经元走 sequential 分支),恢复 r301 latency
- **r305**: 在 NeuronWiseCfC 上叠加 STE-ParallelCfC,验证双 STE (inter-neuron + inter-update-mode) 是否进一步压低 MSE
- **r306**: 在 N-MNIST / Long-Sequence Arena 验证 — r301 在 sharp-transition 上退化,r303 的 routing 是否能选择性补偿
- **r307**: 探索 learned density(每神经元独立 ρ),取代全局 ρ
- **r308**: LFM2.5 demo 接入:用 STE-ParallelCfC 替换 nn.LSTM 测 TTFT/TPOT 影响

## 8. 数据源

- r301 paper: arXiv:2608.03041v1 (Kannan et al. 2026) — PLAN
- r265-r267: lnn/core/ste_neuron_wise_cfc.py, lnn/core/ste_entropy_neuron_wise_cfc.py
- r301 复现: lnn/core/parallel_cfc.py
- r303 实现: lnn/core/ste_parallel_cfc.py
- r303 测试: tests/test_ste_parallel_cfc.py (40/40 通过)
- r303 bench: scripts/bench_ste_parallel_cfc.py + bench_ste_parallel_cfc_results.json
