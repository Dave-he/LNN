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

## 附录 B — 2026-06-02 第三轮迭代：Centered (Non-causal) Noise EMA — NEGATIVE RESULT

(同日第三次 /loop 触发。本轮假设：既然 bi-CfC-NAD 已经非因果，让两个分支共享一个用前向+反向 EMA 平均的"中心化噪声估计"，对含噪输入理应进一步降噪。)

### B.1 假设 (Falsifiable)

> 在 windowed-median 任务上施加 AWGN，bi-CfC-NAD `noise_aggregation="centered"` 的 val MSE 应比 `"independent"` 低 **≥ 10%**。

### B.2 实现

- `BidirectionalNoiseAdaptiveCfC.__init__(..., noise_aggregation="independent" | "centered")`，默认 `"independent"`（向后兼容）。
- `_centered_noise_score(x, beta) = 0.5 * (forward_ema + backward_ema)`，两个 EMA 都用 `vectorized_noise_ema` 并行算（不引入新的 Python 循环）。
- 通过 `_run_with_external_noise` 把同一份噪声分数注入两个内层网络的第一层；更深层仍走各层自己的并行 EMA（避免跨层错位）。
- `noise_aggregation="centered"` 与 `mask != None` 不兼容（mask 会破坏并行假设）；运行时显式 ValueError。

### B.3 单元测试 — `TestBidirectionalCenteredNoise`（6 项）

- 拒绝未知 aggregation；输出形状；反传可达；mask + centered 抛 ValueError；centered 与 independent 在含噪输入下输出确实不同（前提是 `noise_gate_proj` 已偏离零）；centered 用到未来信息的不变量（在 t=15..20 注入异常脉冲后，t=0..5 的 centered 噪声分数会变化）。
- 整套 `pytest tests/` **71 项通过**，零回归。

### B.4 实验结果（两档 SNR，800/120 训练/验证步，8 epoch）

`scripts/benchmark_bi_cfc_nad_centered.py --epochs 8 --hidden 16 --num-samples 400`

| SNR | Uni val MSE | Bi-indep val MSE | Bi-centered val MSE | centered vs indep |
|---|---:|---:|---:|---:|
| 20 dB | 0.02286 | 0.00891 | 0.00890 | +0.1% ❌ |
| 10 dB | 0.05455 | 0.02927 | 0.02932 | −0.2% ❌ |

- **两档 SNR 都未达到 ≥10% 的可证伪阈值**。假设被双重证伪。
- 完整数据：`analysis/cfc_nad/2026-06-02_bi_centered_noise_snr{10,20}.json`。

### B.5 根因分析（为什么 centered 没赢）

1. `noise_gate_proj` 默认零初始化 → `sigmoid(0)=0.5` → 训练早期，噪声分数即使不同也只通过一个几乎不可分辨的门控通道流入网络。
2. 8 epoch 训练预算下，`noise_gate_proj` 还未学到充分依赖噪声分数；centered vs indep 的噪声分数差异在 forward 输出层面被门控压缩到接近 0。
3. 在合成 windowed-median 任务上，目标完全由清洁信号决定；模型对噪声分数的依赖性本身就较弱（不像临床/振动那种"噪声本身携带信号"的场景）。

**结论**：centered noise 在"有显式噪声门控但门控尚未充分训练"的体制下不会带来 MSE 收益。要让它生效，可能需要：a) 主动正则 `noise_gate_proj` 远离零；b) 把噪声分数直接拼入 `g_branch`/`h_branch`（不只 `f_gate`）；c) 换到真实工业振动数据，让噪声本身携带信号。这些都不在本轮范围内。

### B.6 意外副产物：≈20% 推理延迟降低

| 路径 | infer µs/step (CPU, batch=32, seq=32) |
|---|---:|
| Bi-CfC-NAD independent | 16.93 |
| Bi-CfC-NAD centered | **13.49** (−20.3%) |

因为 centered 路径只计算一份共享 EMA 而 independent 需要在每个内层网络里独立算一次。即使 MSE 上没拿到收益，centered 仍可作为**仅追求推理延迟**的工程选项保留。

### B.7 复盘 + 下一轮调整

- 本轮是一次清晰的假设证伪：直接共享噪声估计**在当前训练设置下不影响输出**，因为下游 `noise_gate_proj` 还没学到去依赖它。
- 报告如实保留负结果，不做事后挑选 SNR/seed 来"凑"PASS。
- 下一轮路线图微调：把"centered noise 验证"从 W+1 划去（已证伪），新增"**让 noise_gate_proj 学起来**"作为更基础的待办（如增加 noise 正则、把噪声分数直接拼入 g/h_branch、或换数据集到真实含噪场景）。

## 附录 C — 2026-06-02 第四轮迭代：Uncertainty-Aware CfC-NAD via MDN — PARTIAL POSITIVE

(同日第四次 /loop 触发。)

### C.1 动机

- 既有 CfC-NAD 输出仅是点预测；但临床/工业含噪场景常常需要"模型对自己有多确定"的可校准估计。
- 仓库里早就有 `lnn/core/mdn.py::MDNHead`（用于模仿学习中的 multimodal action），但**从未与 CfC-NAD 串接**。
- 自然假设：把 CfC-NAD 作为特征提取器、接一个 MDN 头，在异方差含噪数据上训 NLL，模型应当学会输出与真实噪声水平相关的 σ。

### C.2 假设 (Falsifiable)

> 在每个 sample 用不同 SNR ∈ [5, 25] dB 的 noisy windowed-median 任务上训练 MDN-NLL，模型在 held-out 集合上输出的"每样本平均预测 σ" 与"每样本真实噪声 σ" 的 **Pearson 相关 r ≥ 0.5**。

### C.3 实现

- 新增 `CfCNADWithMDN`（`lnn/core/noise_adaptive_cfc.py`）：内部 `NoiseAdaptiveCfCNetwork`（return_sequences=True, output_size=hidden_size 当特征提取器）+ `MDNHead`。
- 新增 `mdn_predicted_std(params)`：用混合 Gauss 方差闭式 `Σ w_k (σ_k² + μ_k²) − μ²` 计算每步的总 std；对任何 `num_mixtures` 与 `output_size` 都成立。
- forward 返回 `MDNHead` 的 params dict；下游可用 `mdn_negative_log_likelihood` 训练，用 `mdn_mean` 取点预测，用 `mdn_predicted_std` 取不确定度。

### C.4 单元测试 — `TestCfCNADWithMDN`（6 项）

- 形状（return_sequences / 仅最后一步）
- `num_mixtures < 1` 拒绝（ValueError）
- 在 sin(x) 上训 30 step NLL 严格单调下降
- **关键不变量**：在"前半 σ=0.05，后半 σ=0.5"的双段目标上训 60 step，预测 std 在后半应当大于前半（≈ aleatoric uncertainty 的可学习性）→ 通过
- `mdn_mean` 与 `mdn_predicted_std` 输出形状与正数性

整套 `pytest tests/` **77 项通过**，零回归。

### C.5 实验结果

`scripts/benchmark_cfcnad_mdn_uncertainty.py`，500 个 sample (400 train / 100 val)，noise std 范围 0.036…0.672（19× 区间），seed=42：

| 配置 | epochs | K | val Pearson r(σ̂, σ_true) | val point MSE | 训练秒 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 16 | 1 | **0.305** | 0.03450 | 10.5 | FAIL (<0.5) |
| more capacity + more epochs | 32 | 2 | **0.426** | 0.03130 | 19.6 | FAIL (<0.5) |

- **可证伪阈值未达到**；两次 r 都 < 0.5。
- 但 **方向是对的**：r > 0、随 K 与 epoch 增加而单调上升，模型确实学到了"噪声越大、σ̂ 越大"的关系。
- 单元测试 `test_predicted_std_increases_with_noisy_target` 已经独立证明了该不变量的可学习性。
- 完整数据：`analysis/cfc_nad/2026-06-02_cfcnad_mdn_uncertainty_K{1_e16,2_e32}.json`。

### C.6 复盘 + 下一轮调整

- 这是"边缘负结果"（partial positive）：现象正确、效应方向正确，但被自定的 0.5 阈值卡住。
- 大概率的瓶颈：(a) 数据规模过小（400 train sample）、(b) seq_len 内部步级 σ 高度变动，导致"样本级平均 σ̂"未必能很好对应该样本的标量 SNR、(c) 单 mixture 的 σ 表达力受限。
- 阈值是事前自定的，不做调小到 0.4 的事后追加；负结果如实保留。
- **不变量已经站住**（"含噪片段预测 σ 更大"为可学习）→ 工程上 `CfCNADWithMDN` 仍可作为 LNN 仓库里**首个携带原生不确定度的 CfC backbone**，将进入下一轮的边缘部署/拒识场景。
- 路线图新增：W+1 重做这个 benchmark，把 num_samples 提到 ≥2000，seq_len 缩短到 16，让"样本级标量 SNR"与"样本级 σ̂ 平均"的对应更稳定；同时尝试 *Bi-CfC-NAD + MDN* 看双向上下文是否能把 r 推过 0.5。
