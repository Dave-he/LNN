# MidpointCfC 研读报告 — Round 305 (2026-08-07)

> Non-anchor parallel scan via predictor-corrector, comparing with r301 anchor-only

## 1. 思路来源

r301 ParallelCfC 用 **anchor approximation**: 窗口 W 内所有 W 步 CfC 闭式更新都基于 h_0 (chunk 初态),是 order-dt 准确、order-dt² bias。

r305 **MidpointCfC** 用 **predictor-corrector parallel scan** (Heun / explicit midpoint) 消除一阶 bias:

```
1) Predictor  (anchor at h_0):  h_pred = parallel_eval(x, h_0)
2) Midpoint:                   h_mid  = 0.5 * (h_0 + h_pred)
3) Corrector (anchor at h_mid): h_corr = parallel_eval(x, h_mid)
4) Output:  h_corr              # order-dt² accurate
```

代价: 每 chunk 2× parallel eval (但仍是 O(W) per chunk, 不是 O(W²))。

## 2. 实现

- `lnn/core/midpoint_cfc.py` (172 行): `MidpointCfCCell` + `MidpointCfCNetwork`
- `tests/test_midpoint_cfc.py` 20/20 通过
- `scripts/bench_midpoint_cfc.py`

## 3. toy_sin 5-seed 结果

| 模型 | MSE | Δ vs vanilla | 推理延迟 (10 pass) | Δ latency | Std |
|---|---:|---:|---:|---:|---:|
| vanilla_cfc | 0.11414 ± 0.00486 | — | 47.88 ms | — | 0.0049 |
| parallel_cfc_w4 (r301) | 0.10733 ± 0.00107 | -6.0% | 16.27 ms | -66% | 0.0011 |
| parallel_cfc_w8 (r301) | **0.10564 ± 0.00225** | **-7.5%** | **13.37 ms** | **-72%** | 0.0023 |
| midpoint_cfc_w4 (r305) | 0.10966 ± 0.00106 | -3.9% | 26.47 ms | -45% | 0.0011 |
| midpoint_cfc_w8 (r305) | 0.10603 ± 0.00057 | -7.1% | 16.60 ms | -65% | 0.0006 |

## 4. 关键发现 (含 honest negative)

1. **NEGATIVE on latency, MARGINAL POSITIVE on stability**:
   - midpoint_w8 vs parallel_w8: MSE 0.10603 vs 0.10564 (几乎平, +0.4% 退化)
   - **Std 0.00057 vs 0.00225 (3.9× 改善) — 这是 midpoint 唯一的 clear win**
   - 延迟 16.60 vs 13.37 ms (**+24% 慢**, 2x parallel eval 的预期代价)
2. **Pareto 结论**: pure anchor (r301 parallel_w8) 仍是 sweet spot;midpoint 是 "stability at cost of latency" 的 trade-off。
3. **WHY 不是 order-dt² 优势**: 在 toy_sin 这种 *smooth periodic* 任务上,anchor 的 order-dt² bias 本身就很小 (因为 τ 接近 1.0,f-gate 接近 0.5,线性度好),midpoint 的 corrector 没显著收益。在 *sharp-transition* 任务上 (r302 数据集) 中间值可能不同,这是后续要测的。
4. **隐性观察**: midpoint_w8 的 std 0.00057 比 parallel_w4 (0.00107) 还低 — 提示 corrector 确实有 *implicit regularization* 作用,而不仅是误差修正。

## 5. 适用场景

- **不要**默认用 midpoint: 24% 延迟代价在边缘部署是真实成本
- **可考虑** midpoint_w8 当: (a) 任务有 sharp transitions, anchor bias 主导误差; (b) 多 seed 一致性比延迟重要 (例如生产模型集成)

## 6. 与 r301+r302+r303+r304 关联

- r301 anchor: 默认; toy_sin 强正; sharp transition 待 r302 验证
- r302 sharp-transition: 进行中,决定 anchor 的边界
- r303 STE + parallel: 进行中,看 routing 能否替代 midpoint 的 corrector 作用
- r304 LFM2.5 集成: 部署层,Pareto 仍选 anchor (W=4)
- **r305 (本文)**: 理论方向探索,**结论是 honest-negative-on-latency / marginal-positive-on-stability**,不是新 SOTA

## 7. 后续工作

- **r306**: 把 midpoint 作为 STE 路由的"hard target"(训练时用 midpoint 提供 soft target,推理时用 anchor 加速)— 一个 distillation-style 折衷
- **r307**: midpoint + STE joint cell(预测 anchor vs sequential 的混合 anchor),看是否能比 r305 单独 midpoint 更优
- **r308**: 在 r302 sharp-transition 数据集上重测 midpoint,看 anchor bias 主导时是否逆转

## 8. 数据源

- 复现: `lnn/core/midpoint_cfc.py`
- 测试: `tests/test_midpoint_cfc.py` (20/20 通过)
- Bench: `bench_midpoint_cfc_results.json`
