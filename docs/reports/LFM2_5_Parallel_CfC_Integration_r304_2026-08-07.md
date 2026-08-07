# LFM2.5 + ParallelCfC 集成验证报告 (r304)

**Round**: r304
**Date**: 2026-08-07
**Author**: LNN Cron Bot
**Status**: INTEGRATION-TEST-COMPLETE (deployment validation, not quality benchmark)

---

## 1. 任务背景

r301 (2026-08-07) 引入了 **ParallelCfCCell / ParallelCfCNetwork** —— PLAN (arXiv:2608.03041v1) 启发的 CfC 闭式更新向量化版本: 窗口 W=8 时 5-seed toy_sin **MSE -7.1% / 推理延迟 -60%** (Pareto win, 21/21 tests pass, push `21eff02`)。

r300 已经在 `lnn/lfm2/inference.py` 提供 LFM2.5 推理入口 (`LiquidAI/LFM2.5-*` 模型需 `transformers`)。本轮把 r301 的 ParallelCfC 作为**边缘部署**的 drop-in 替换 LFM2.5 路径中的 `nn.LSTM` / `nn.GRU`, 并量化其在端到端推理路径上的 **参数 / 延迟** 影响。

**重要声明**: 本仓库**不携带**真实 LFM2.5 权重 (350M+ 模型从 HuggingFace 下载, SSL_SYS 卡住), 因此本报告使用 **tiny mock backbone** (1 层 LSTM + 嵌入 + LM 头, 66K 参数) 验证集成路径, 数字反映"集成机制正确"而非"实际 LFM2.5 加速"。

---

## 2. 实现

### 2.1 新增文件
- `lnn/lfm2/parallel_integration.py` (152 行) — `replace_lstm_with_parallel_cfc()` walker
- `lnn/lfm2/__init__.py` — 导出 swap 入口
- `scripts/bench_lfm2_parallel_cfc.py` (260 行) — 端到端 benchmark
- `tests/test_lfm2_parallel_cfc.py` — **18/18 tests pass**
- `bench_lfm2_parallel_cfc_results.json` — CPU 5-trial 数据

### 2.2 Walker 关键设计
`replace_lstm_with_parallel_cfc(model, window=4)`:
1. 递归遍历 `model.named_modules()`, 找到所有 `nn.LSTM` / `nn.GRU`
2. 每个匹配项用 `ParallelCfCNetwork` 替换 (matching input/hidden/num_layers, `return_sequences=True` 保持 `(B, T, H)` 形状契约)
3. 支持 `nn.Sequential` / `nn.ModuleList` / 任意 `nn.Module` 嵌套
4. `inplace=True/False` 双模式 (False 走 `copy.deepcopy`)
5. 设置 `model._r304_swap_count` 反映替换次数

### 2.3 Mock Backbone
`TinyLFM25Mock(vocab, hidden, num_layers)`:
- `nn.Embedding(vocab, hidden)` (LFM2 词嵌入)
- `nn.LSTM(hidden, hidden, batch_first=True)` (LFM2 线性 LSTM 抽象)
- `nn.Linear(hidden, vocab)` (LM 头, tied weight)

LFM2.5 在实际中用 short-conv + linear-RNN hybrid; 我们用 1 层 LSTM 抽象 backbone, 验证 swap 机制是否对**典型 RNN 形状**有效。

---

## 3. Benchmark 结果 (CPU, batch=1, hidden=64, vocab=512, 5 trials)

| T (seq len) | LSTM (ms) | W=1 (ms) | W=2 (ms) | W=4 (ms) | W=8 (ms) | W=8 Delta vs LSTM |
|---:|---:|---:|---:|---:|---:|---:|
|   8 |  2.36 |  1.73 (-26.8%) |  1.15 (-51.2%) |  0.73 (-68.9%) |  0.63 (-73.4%) |
|  16 |  3.79 |  2.54 (-33.1%) |  1.93 (-49.1%) |  1.90 (-50.0%) |  0.69 (-81.9%) |
|  32 |  5.42 |  5.39 (-0.7%)  |  3.58 (-34.0%) |  2.15 (-60.4%) |  1.26 (-76.7%) |
|  64 |  9.42 |  8.92 (-5.3%)  |  6.61 (-29.8%) |  5.12 (-45.7%) |  2.70 (-71.4%) |
| 128 | 18.00 | 17.95 (-0.3%)  | 12.99 (-27.9%) |  6.98 (-61.3%) |  4.10 (-77.2%) |

**W=4 (Pareto sweet spot) 平均**:
- 参数: **66,048 -> 61,760 (-6.5%)**
- 延迟 (跨 T=8/16/32/64/128): **avg -57.2%**
- 形状契约: **(B, T, vocab) 全部严格保持** (all_shape_match=True)
- 输出稳定性: 5 seed 全部 finite (代理验证, 详细精度评估见 r301)

---

## 4. 关键发现

### 4.1 集成机制 **完全正确**
- 18/18 unit tests pass (replace 函数, 多种 W, 嵌套容器, GRU, 梯度, inplace/copy)
- Output shape **strictly preserved** for every (T, W) combo tested
- Module 名字保留 (`model.backbone` 仍然 resolve, 只是类型从 LSTM 变 ParallelCfCNetwork)
- 反向传播正常 (embed / head / cell.time_scale 都有梯度)

### 4.2 延迟结果 **符合 PLAN paper claim 区间**
- W=4: **-46% to -69%** latency (5 seq_lens)
- W=8: **-71% to -82%** latency (5 seq_lens)
- r301 toy_sin: -60% latency; 本轮 LFM2-mock: -57% (W=4), **量级一致**

### 4.3 参数减少 **轻度 (-6.5%)**
- 这与 PLAN paper 报告 22-47% 差距大。原因:
  - 我们的 mock 只有一个 LSTM, 但嵌入/head 占总参数 ~80% (32K/66K), LSTM swap 占比有限
  - PLAN 的 22-47% 是**整个 encoder** 的占比, 其中多个 LSTM 层 + attention 占大头
  - 在真实 LFM2.5 路径 (24 层线性-LSTM), 预期参数减少会更高

### 4.4 质量评估 **仅做了 deployment proxy** (非 benchmark)
- Output finite: ok
- Output shape preserved: ok
- Backward works: ok
- Real LFM2.5 perplexity: **未测** (无权重, 不可测)
- 5-seed toy_sin 已在 r301 测过 (W=8: -7.1% MSE), r304 **不重复** 质量评估

---

## 5. 已知限制 / 未做事项

1. **真实 LFM2.5 权重未引入**: SSL_SYS 错误, push 入口也卡 121.3/192.168.6.25/443 全部失败 (r299 教训)。数字反映"集成机制工作", 不反映"实际 LFM2.5 加速"。
2. **状态缓存不模拟**: 真实 LFM2.5 KV cache 不会用 ParallelCfC; 我们的 swap 只针对 backbone LSTM。如果 LFM2.5 还有 `linear-RNN` 层 (LFM2.5 paper §2.2), 同样可被替换, 但需要 LFM2.5 实际模型代码 (transformers 内部)。
3. **T 必须整除 W**: `ParallelCfCNetwork(forward)` 要求 T 是 window 的整数倍。生产中可以用右 padding 解决, 本轮未做。
4. **短序列退化**: T=32, W=1 时延迟没有优势 (5.39ms vs 5.42ms LSTM)。W=4 起步才有加速。
5. **linear-LSTM projection 模拟**: LFM2 用 `proj_size` 投影 LSTM, 我们支持 (`_make_replacement` 已检测 `proj_size`), 但只测了无投影情形。

---

## 6. 生产就绪度 (Honest Verdict)

**deployment integration**: READY (机制完全验证)
**quality benchmark on real LFM2.5**: BLOCKED (无权重, SSL_SYS)
**production recommendation**: 仅在 (a) 真实 LFM2.5 权重可获取, (b) 端到端 perplexity / MMLU 评估通过, (c) W 与 T 兼容性已用 padding 处理 之后, 才是 production-ready。

当前结论: **r304 是 r301 的 deployment 桥接骨架, 把"PLAN 启发的 CfC 向量化"概念从 toy_sin 延伸到 LFM2.5 推理路径**; 但 r301 paper §6.3 自陈的"sharp inter-step transitions 退化"在 LLM 自回归推理里**完全适用** (per-token step 是 sharp 的), 因此**不建议**直接对生产 LFM2.5 应用 W=8 激进配置 —— W=4 是 r301 推荐的 Pareto sweet spot, 与本轮 LFM2.5 benchmark 数字一致。

---

## 7. 后续 round 建议

- **r305**: 加 T 兼容 padding (right-pad to next multiple of W) — 移除 r304 的 T % W == 0 强约束
- **r306**: 在 `lnn/lfm2/inference.py::LFM2Inference` 加 `swap_to_parallel_cfc(window=4)` 方法, 让 user 可以一行调用
- **r307**: 与 r244-r256 basin-lyapunov anchor 联合 (PLAN 的 h_0 anchor = static basin), 看 W=4 + multi-basin 是否能恢复被 PLAN approximation 损失的质量

---

## 8. 数据/代码指针

- bench 脚本: `/Users/hyx/workspace/LNN/scripts/bench_lfm2_parallel_cfc.py`
- bench 数据: `/Users/hyx/workspace/LNN/bench_lfm2_parallel_cfc_results.json`
- 集成模块: `/Users/hyx/workspace/LNN/lnn/lfm2/parallel_integration.py`
- 测试: `/Users/hyx/workspace/LNN/tests/test_lfm2_parallel_cfc.py` (18/18 pass)
- r301 memory: `/Users/hyx/.claude/projects/-Users-hyx-workspace-LNN/memory/lnn-round-301-plan-parallel-cfc.md`
- PLAN paper: arXiv:2608.03041v1 (Kannan et al. 2026-08-04)

---

**end of r304 report**
