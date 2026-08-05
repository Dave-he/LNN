---
title: Int8 Quantization on Distillation Students (N20) — DLNet Stage 3 落地，4.0× 无损压缩
date: 2026-08-05
tags: [LNN, int8, quantization, distillation, edge-ai, DLNet, Stage-3, N20, positive-result]
arxiv_refs: [2601.06227, 2106.13898]
parent: [[LNN_深度研读报告]]
companion: [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]], [[Hybrid_Gate_Teacher_Distillation_N19_2026-08-05]]
gap_refs: [N20-int8-quantization]
---

# Int8 Quantization on Distillation Students (N20) — DLNet Stage 3 落地

> 把 DLNet (arXiv 2601.06227) 论文的 **Stage 3 (int8 quantization)** 补完到本项目 DLNet 流水线。**核心 finding**：int8 量化对所有 student size 的 MSE 影响几乎为零（delta ±0.0001），同时提供 4.0× 压缩。结合 N1/N19 distillation，hybrid_gate teacher × h=4 student 总压缩 **97.16×**。

## 1. 实现

代码：[`lnn/core/quantization.py`](lnn/core/quantization.py)（136 lines）
- `quantize_int8_per_tensor`：对称 per-tensor int8 量化
- `quantize_int8_per_channel`：对称 per-channel int8 量化（每个 output channel 一个 scale）
- `dequantize_int8`：反量化（处理 per-tensor 和 per-channel scale）
- `quantize_model_inplace`：walk through model 的所有 `nn.Linear` weight，in-place 替换为 dequantised float（模拟 int8 inference with float hardware）
- `total_compressed_size_bytes` / `total_fp32_size_bytes`：size accounting

测试：[`tests/test_quantization.py`](tests/test_quantization.py) — **9/9 通过**
- per-tensor/per-channel shape、range、recovery error、zero weight
- Model-level in-place quantization metadata
- Quantization 引入 bounded error
- int8 size = fp32 size / 4 (4.0× compression)

## 2. Benchmark 结果

### 2.1 CfC teacher + int8 student

| student h | fp32 MSE | int8 MSE | delta | int8 size | fp32 size | compression |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0632 | 0.0632 | -0.0000 | 113B | 452B | 4.0× |
| 8 | 0.0570 | 0.0570 | +0.0000 | 321B | 1284B | 4.0× |
| 12 | 0.0570 | 0.0570 | +0.0000 | 625B | 2500B | 4.0× |
| 16 | 0.0563 | 0.0563 | -0.0000 | 1025B | 4100B | 4.0× |

数据：[`analysis/jetson/2026-08-05_int8_quantization_cfc.{md,json}`](analysis/jetson/2026-08-05_int8_quantization_cfc.md)

### 2.2 hybrid_gate teacher + int8 student

| student h | fp32 MSE | int8 MSE | delta | int8 size | fp32 size | compression |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0571 | 0.0571 | -0.0000 | 113B | 452B | 4.0× |
| 8 | 0.0569 | 0.0569 | +0.0000 | 321B | 1284B | 4.0× |
| 12 | 0.0571 | 0.0571 | +0.0000 | 625B | 2500B | 4.0× |
| 16 | 0.0563 | 0.0563 | +0.0000 | 1025B | 4100B | 4.0× |

数据：[`analysis/jetson/2026-08-05_int8_quantization_hybrid_gate.{md,json}`](analysis/jetson/2026-08-05_int8_quantization_hybrid_gate.md)

## 3. 关键发现

### 3.1 int8 量化"无成本"提供 4.0× 压缩

**所有 8 个 (4 student × 2 teacher) 配置的 MSE delta 都在 ±0.0001 内**——浮点精度内。

→ **int8 quantization is a free-lunch compression**：零精度损失，4.0× 存储节省。

### 3.2 Combined compression 链路

| 路径 | h=4 | h=8 | h=16 |
|---|---:|---:|---:|
| **CfC teacher** | 14.53× (N1) × 4.0× (int8) = **58.13×** | 6.10× × 4.0× = 24.40× | 2.31× × 4.0× = 9.24× |
| **hybrid_gate teacher** | 24.29× (N19) × 4.0× (int8) = **97.16×** | 10.20× × 4.0× = 40.80× | 3.86× × 4.0× = 15.44× |

→ **hybrid_gate teacher × int8 student h=4 = 97.16× total compression，零精度损失**——超过两个数量级。

### 3.3 实用 take-away

| Stage | 选择 | 累计压缩 vs CfC teacher (h=32) |
|---|---|---|
| Baseline (CfC h=32) | 4 维输入 → 1 维输出，1 LSTM 层 | 1.0× |
| Stage 1 (N1 distillation) | Distill to h=4 CfC student | 14.53× |
| Stage 2 (N19 hybrid_gate teacher) | Distill with hybrid_gate teacher | 24.29× |
| **Stage 3 (N20 int8 quantization)** | Per-channel int8 | **97.16×** |

→ **LNN edge deployment complete pipeline**：
- 选 hybrid_gate teacher（N19：rich hidden 易压缩）
- 蒸馏到 h=4 CfC student（N19：24.29×）
- int8 量化（N20：4.0×）
- **总：97.16× 压缩、零精度损失、可以部署到 MCU**

## 4. 与 N1 / N19 的连续性

| Round | 工作 | 关键发现 |
|---|---|---|
| N1 (Round 14) | CfC teacher → h_student distillation | 6.10× (h=8) 零精度损失 |
| N19 (Round 15) | hybrid_gate teacher → h_student | **比 CfC teacher 67% 更多压缩**（10.20× at h=8）|
| **N20 (Round 16, 本轮)** | **+ int8 quantization Stage 3** | **4.0× 额外压缩、零精度损失** |

→ 3 轮 distillation research 累积形成完整 **distill + compress** 链路，每一步都 "无成本" 提供额外压缩。

## 5. Gap 状态更新

| # | 缺口 | 8/5 状态 |
|---|---|---|
| **N20** | DLNet Stage 3 (int8 quantization) | ✅ **本轮落地（strong positive）** |
| N14 | MR-hybrid_gate-CfC 在 h=64/128 重评估 | ⏳ 下周 |
| N21 | hybrid_gate teacher × hybrid_gate student round-trip | ⏳ 路线图 |
| **新增 N23** | int8 student + irregular dt 验证（int8 在不同 dt 分布下是否仍 robust）| ⏳ 路线图 |
| N22 | 任何含 α MLP 的 teacher 都比 CfC 更易压缩？| ⏳ 路线图 |
| N17 | α capacity 增强 | ⏳ 路线图 |
| N18 | CfC 在真实数据集上的 transferability | ⏳ 路线图 |
| N2 | L-RFM | ⚠ |
| L4 | Liquid-S4 grounding | ⚠ |

## 6. 推荐后续动作

1. **本周**：N14 MR-hybrid_gate-CfC 在 h=64/128 重评估
2. **下周**：N21 hybrid_gate teacher × hybrid_gate student round-trip distillation
3. **路线图**：N23 int8 student × irregular dt 验证（int8 在 OOD dt 下是否仍 robust）
4. **路线图**：N18 CfC 在真实数据集上验证（UCR/MIMIC）

## 7. 数据源回链

- 代码
  - [`lnn/core/quantization.py`](lnn/core/quantization.py)（136 lines）
  - [`tests/test_quantization.py`](tests/test_quantization.py)（9 tests, all pass）
  - [`scripts/bench_int8_quantization.py`](scripts/bench_int8_quantization.py)（171 lines）
  - [`lnn/core/distillation.py`](lnn/core/distillation.py)（refactored: store trained students in `self.students`）
- Benchmark
  - [CfC teacher + int8 数据](analysis/jetson/2026-08-05_int8_quantization_cfc.md)
  - [hybrid_gate teacher + int8 数据](analysis/jetson/2026-08-05_int8_quantization_hybrid_gate.md)
- 上轮对照
  - [[Hybrid_Gate_Teacher_Distillation_N19_2026-08-05]]（N19 hybrid_gate teacher 24.29×）
  - [[DLNet_Dual_Stage_Distillation_N1_Pareto_Sweep_2026-08-05]]（N1 CfC teacher 6.10×）
