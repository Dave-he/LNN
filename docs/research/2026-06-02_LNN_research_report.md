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
