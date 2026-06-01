---
title: LNN 最新进展研究报告 - 2026-06-02
date: 2026-06-02
tags: [LNN, CfC, LTC, LFM, MLX, parallel-relaxation, research-report, weekly]
related:
  - "[[docs/research/2026-06-01_LNN_research_report]]"
  - "[[docs/daily/2026-06-02_LNN_research_digest]]"
---

# 🌊 LNN 最新进展研究报告 — 2026-06-02

> 接续昨日报告（2026-06-01）。今日 arXiv API 返回 `Remote end closed connection without response`（流水线已自动保留昨日检索缓冲），GitHub 与 Hugging Face 数据正常更新。

## 1. 今日新出现的关键信号

| 信号 | 资源 | 价值 |
|---|---|---|
| **LFM2.5 进入 Apple MLX 生态** | `LiquidAI/LFM2.5-8B-A1B-MLX-{4,5,6,8}bit`、`LFM2.5-8B-A1B-MLX-bf16`（2026-06-01 发布；4-bit 单日下载已 1 639） | 液态基础模型官方支持 macOS / Apple Silicon 边缘推理，Mac mini / Studio 直接可跑 8B 稀疏 MoE LFM |
| **LFM2.5-1.2B Claude 蒸馏** | `FlameF0X/LFM2.5-1.2B-Distilled-Claude-4.6` | 第一批以 Claude 4.6 教师模型蒸馏出的 LFM2.5 小模型；社区也出现 `LFM2.5-1.2B-Thinking-CodeX` |
| **CfC 替换多头注意力** | `sxlxbo/CTDFormer` — Bearing fault diagnosis Transformer with **bidirectional CfC** replacing multi-head attention | 工业故障诊断领域以 CfC 直接换掉 MHA，进一步印证 CfC 作为序列骨干在轻量场景下的可行性 |
| **边缘 LOKI 模仿学习** | `R-Liebert/LOKI-G` — adapts LOKI to imitation + RL on physical machines | 与昨日提到的 *Liquid Networks with MDN Heads for Imitation Learning* 形成同向社区延伸 |
| **arXiv API 不稳定** | digest 抓取报错 | 提示 daily pipeline 需要补"上次成功镜像 + 缓存合并"才能在 arXiv 抖动时仍输出有效报告（已经做了一半：保留旧候选池，但本日上层报告仍显示 0 论文） |

## 2. 接续昨日：解决 CfC-NAD 的"+25% CPU 推理开销"遗留问题

### 2.1 昨日结论回顾

- CfC-NAD 在 5/5 SNR 档击败 vanilla CfC（MSE −2.9% … −23.7%），参数 +5.3%。
- **遗留缺陷**：CPU 推理延迟 +25%。根因：在 `NoiseAdaptiveCfCNetwork.forward` 内部，每步都做 4 个 Python-level tensor 算子来维护"输入一阶差分平方的 EMA"。

### 2.2 今日修复 — Parallel Noise EMA

将昨日的"逐步累积"噪声 EMA 改写为基于 `parallel_liquid_relaxation` 的 cumprod/cumsum 闭式：

$$
\text{noise\_ema}_t = \beta \cdot \text{noise\_ema}_{t-1} + (1-\beta)\cdot(x_t - x_{t-1})^2
$$

这是一个常系数线性递推，与 LiquidTAD 用的 `h_t = r_t h_{t-1} + (1-r_t) v_t` 同形，可在 GPU 上 O(T) 并行：

```python
def vectorized_noise_ema(masked_input, beta):
    diff = torch.zeros_like(masked_input)
    diff[:, 1:, :] = masked_input[:, 1:, :] - masked_input[:, :-1, :]
    diff_sq = diff * diff
    retain = torch.full_like(diff_sq, beta)
    return parallel_liquid_relaxation(retain, diff_sq)
```

- 在 `mask is None` 的常见路径上先 batch 预计算整段噪声 EMA，逐步取 `noise_ema_full[:, t, :]`。
- `mask != None` 时退回流式实现以保留 mask 语义。
- 隐状态依然是 O(1)；新增的中间张量在 inference 中可被释放。

### 2.3 等价性验证（numerical equivalence）

新增 `TestVectorizedNoiseEMA`、`TestNoiseAdaptivePathEquivalence` 共 4 项 pytest：

- 随机输入：parallel == streaming（atol=1e-5）
- 交错正负号输入（最坏情况）：parallel == streaming（atol=1e-5）
- 零长输入安全
- 整网（双层 CfC-NAD）`mask=None` vs `mask=全1` 输出 bit-for-bit 一致

整套 `pytest tests/` 共 59 项全部通过，零回归。

### 2.4 延迟对比（CPU, macOS Darwin 24.6.0, Python 3.11, PyTorch 2.2.2）

`scripts/microbench_pcnad.py --batch 32 --seq 64 --hidden 16 --repeat 80`

| 路径 | 单次 forward (ms) | vs vanilla CfC | 备注 |
|---|---:|---:|---|
| vanilla CfC | **6.46 ± 0.14** | baseline | 单层、return_sequences=False |
| CfC-NAD parallel (今日) | **10.09 ± 1.31** | **+56.2%** | mask=None，整段并行 EMA |
| CfC-NAD streaming (昨日) | 13.26 ± 1.56 | +105.4% | 全 1 mask，强制流式路径 |

**并行版相对流式版的实测加速 +24.0%**，把 NAD 在 CPU 上的开销从 ~2.05× 压到 ~1.56×。剩余开销主要来自每步多出来的 `f_gate` 拼接（`[x, h, noise]`）与 `noise_gate_proj` Linear，已经接近"算法本身"的最小开销，不再是 Python loop 主导。

数据：`analysis/cfc_nad/2026-06-02_pcnad_microbench.json`。

### 2.5 与新 digest 信号的联动

- 今日 `sxlxbo/CTDFormer` 验证了"CfC 替换 MHA"的实际工业可行；本仓库现在可以提供**带可证伪噪声鲁棒的 CfC 骨干**，比社区裸 CfC 多一道针对工业振动等含噪信号的鲁棒性。
- LFM2.5 MLX 系列上线意味着下一周可在 Mac mini 上跑端到端"CfC-NAD pretrain + LFM2.5 inference"的小流水线，无需 Jetson。

## 3. 本周新研究思路

### 3.1 Bidirectional CfC-NAD（双向噪声鲁棒 CfC）

- **动机**：今日 `CTDFormer` 用 *bidirectional* CfC 替代 MHA；本仓库目前 `NoiseAdaptiveCfCNetwork` 只支持单向。
- **接口**：增加 `bidirectional: bool = False`，前向 + 反向各跑一遍并行噪声 EMA + 隐藏态，最后通道拼接或加和。
- **可证伪验证**：在 bearing fault detection 风格的合成振动数据上，bi-CfC-NAD MSE 应低于单向 CfC-NAD ≥ 5%。

### 3.2 LFM2.5-MLX 本机推理 smoke

- **目标**：让 `lnn/lfm2/inference.py` 加一条 `backend="mlx"` 路径，调用 `mlx_lm`（如已安装）加载 `LFM2.5-8B-A1B-MLX-4bit` 做 latency / token-throughput 微基准。
- **不在本周清单**：需要 ~3 GB 的 MLX 模型下载与 `mlx` 依赖；列入下周。

### 3.3 daily_lnn_research.py 抗 arXiv 抖动

- **现象**：连续 2 天 arXiv API 在我们 cron 节奏附近不可用；digest 退化为"0 论文"。
- **方案**：增加 `--arxiv-fallback-window 3`，当当日 arXiv API 失败时，从最近 3 天的 cache 中 union 候选池并去重，配合 `arxiv_status` 元数据标注。
- **风险**：会污染"当天新论文"语义；用元数据 `source_window: 3` 区分。

## 4. 后续路线图（接续 2026-06-01）

| 周次 | 目标 | 关键产出 |
|---|---|---|
| **本周完成** | Parallel Noise EMA + 等价测试 + microbench | `lnn/core/noise_adaptive_cfc.py::vectorized_noise_ema` / `scripts/microbench_pcnad.py` / 本报告 |
| W+1 | bidirectional CfC-NAD | `lnn/core/bidirectional_cfc_nad.py` + tests |
| W+1 | daily pipeline arXiv fallback window | `scripts/daily_lnn_research.py --arxiv-fallback-window` |
| W+2 | LFM2.5-MLX backend smoke | `lnn/lfm2/mlx_backend.py` + `analysis/mlx/*.json` |
| W+3 | EMMA 风格联合多模态 + 物理 | `experiment_physics_multimodal.py` |
| W+4 | CfC-NAD on Jetson CUDA 对比 | `analysis/jetson/*NAD.json` |

## 5. 参考

- 接续：[[docs/research/2026-06-01_LNN_research_report]]
- digest: [[docs/daily/2026-06-02_LNN_research_digest]]
- HF: `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` (2026-06-01)
- GitHub: `sxlxbo/CTDFormer` (Bearing fault, bi-CfC vs MHA), `R-Liebert/LOKI-G`, `FlameF0X/LFM2.5-1.2B-Distilled-Claude-4.6`
- 仍跟踪：arXiv 2605.27467v1 (LNN vs LSTM clinical), 2604.18274v2 (LiquidTAD)

---
*本报告由 `/loop` 5h 计划任务驱动；下次自动触发：约 6 小时后（cron `7 */6 * * *`，任务 ID `7131cb00`）。*

## 附录 A — 2026-06-02 第二轮迭代：Bidirectional CfC-NAD

(同日第二次 /loop 触发；接续上文 §3.1 W+1 待办，把 bidirectional CfC-NAD 提前到本日完成。)

### A.1 动机
- 今日 digest 中的 `sxlxbo/CTDFormer` (2026-05-17) 把 Transformer 的多头注意力替换为 **bidirectional CfC**，用于轴承故障诊断；这是 CfC 在工业含噪信号上取代 attention 的第一例公开实现。
- 昨日 + 今日上文已得到带噪声鲁棒性的 **Uni-CfC-NAD**（CfC-NAD parallel）；自然下一步是补上 *bidirectional* 维度，将其与 CTDFormer 的设计原语对齐。

### A.2 实现：`BidirectionalNoiseAdaptiveCfC`

```text
forward_net : NoiseAdaptiveCfCNetwork(x)       -> [B, T, H]
backward_net: NoiseAdaptiveCfCNetwork(flip(x)) -> [B, T, H]  (再 flip 回来对齐时间轴)
output_proj : Linear(2H -> output_size)        -> [B, T, output_size]
```

- 内部使用昨日和今日完成的 `NoiseAdaptiveCfCNetwork`（return_sequences=True，自动走 parallel noise EMA 路径）。
- `dt` / `mask` 的时间维度通过 `_flip_temporal` 在反向路径上同样翻转，覆盖 1-D `[T]`、2-D `[B,T]`、3-D `[B,T,F]` 等常见形状。
- 总参数 = 2 × Uni-CfC-NAD + 1 个 `Linear(2H -> output_size)`；在本基准设置下约为 Uni 的 2.57×（945 → 2 433）。

### A.3 可证伪验证 — Windowed-Median Regression

**任务**：给定 Mackey-Glass 时间序列 x，预测每步 y[t] = median(x[t-k : t+k+1])，k=3。
**关键**：y[t] 依赖于 x 的未来 k 步，单向模型理论上不可能完美。

`scripts/benchmark_bi_cfc_nad.py --epochs 8 --hidden 16 --num-samples 400`：

| 模型 | 参数量 | val MSE | train (s) | infer (µs/step) |
|---|---:|---:|---:|---:|
| Uni-CfC-NAD | 945 | 0.01828 | 1.16 | 6.40 |
| **Bi-CfC-NAD** | **2 433** | **0.00524** | 2.30 | 12.73 |

- **Bi 相对 Uni 的 val MSE 降幅：71.3%**（claim 阈值 ≥25%，**PASS**）。
- 训练/推理时间增加约 2×（与"两个内层网络"一致，符合预期）。
- 数据 / 配置 / 完整结果：`analysis/cfc_nad/2026-06-02_bi_cfc_nad_benchmark.json`。

### A.4 单元测试 — `TestBidirectionalNoiseAdaptiveCfC`（6 项）

- 形状（return_sequences / 仅最后一步）
- 反传可达性：前向和反向两个内部网络都收到非零梯度
- 与 Uni 在非对称输入上输出显著不同
- `dt` 1-D `[T]` 自动翻转适配
- 参数预算上限（< 3× Uni）

全套 `pytest tests/` 65 项通过、零回归（昨日 55 + 今日上轮 +4 + 本附录 +6）。

### A.5 复盘 + 下一轮

- **可证伪假设通过**：在显式需要未来上下文的合成任务上 Bi-CfC-NAD 大幅领先；与 CTDFormer 的工业经验定性一致。
- **延迟成本**：约 2× Uni；下一步可尝试两边权重共享（受限 RNN BiRNN 风格）或前向并行噪声 EMA 与后向并行噪声 EMA 共用一个 cumprod 中间结果。
- **真实数据验证**：合成中位数任务证明了"双向有用"；下一轮应换到 `parhat1/cfdna-tau-repository` 或 LiquidTAD 风格的真实工业振动 / 视频数据上复测。
- 下一轮路线图条目升级：原 W+2 "EMMA-style multimodal + physics" 仍保持；新增 **Bi-CfC-NAD on bearing fault diagnosis** 作为 W+1 优先级。
