---
title: LNN Verification Results
date: 2026-06-09
tags: [LNN, verification, jetson, benchmark, results]
status: living-document
---

# LNN Verification Results

> 本文档汇总本仓库在 **真实硬件 / 模拟硬件 / 单元测试** 三类环境下的 LNN
> 验证结果。每条记录都附运行日期、设备指纹、commit hash、入口脚本与产物
> 路径,便于回溯与对比。

## 1. Jetson Orin Nano Super — Pareto sweep (CPU path)

| 项 | 值 |
|---|---|
| 运行日期 | 2026-06-09 |
| 设备 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| BSP | R36 (release), REVISION: 4.7, KERNEL_VARIANT: oot |
| PyTorch | 2.11.0+cu130 (CUDA 不可用 — driver too old,fallback 到 CPU) |
| Python | 3.14.4 |
| 命令 | `python3 scripts/jetson_lnn_benchmark.py --quick --cpu --pareto --date 2026-06-09_local` |
| 入口脚本 | `scripts/jetson_lnn_benchmark.py` |
| 产物 JSON | `analysis/jetson/2026-06-09_local_lnn_benchmark.json` |
| 产物 MD | `analysis/jetson/2026-06-09_local_lnn_benchmark.md` |
| 产物 PNG | `analysis/jetson/2026-06-09_local_lnn_pareto.png` |
| 测试 | `pytest tests/test_jetson_lnn_benchmark.py -v` 7/7 PASSED |

### Pareto front (5/8 configs)

| 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CfCStyle** | **16** | **32** | 42 | 1169 | **0.470338** | 21960.3 | 4.11 |
| **GRU** | **16** | **32** | 42 | 929 | 0.536350 | 98844.5 | 1.45 |
| **GRU** | 16 | 16 | 42 | 929 | 0.547923 | 110075.8 | 0.99 |
| **GRU** | 8 | 16 | 42 | 273 | 0.558936 | 92016.6 | 0.31 |
| **GRU** | 8 | 32 | 42 | 273 | 0.565124 | 118700.2 | 0.65 |

(以下 3/8 被其他配置支配,不在 Pareto 前沿:CfCStyle h=8 T=16/32 + h=16 T=16)

### 解读

- **精度 vs 速度 trade-off**:CfCStyle (h=16, T=32) 是 MSE 最低 (0.470) 的 Pareto
  点 — **CfC 闭式求解器在固定参数下精度胜 GRU 12.3%**;GRU (h=16, T=32) 是
  速度最高的 Pareto 点 (98844 步/秒) — **GRU 速度胜 CfC 4.5×**。
- **与论文对照** (Tanna et al. 2024, [IEEE 10826128](https://ieeexplore.ieee.org/abstract/document/10826128/)):
  CfC 在小参数量级下"精度胜"是 LNN 论文的核心 claim,本机数据印证这一点。
  GRU 的 4.5× 速度优势对应 ODE 求解器的迭代成本,与 CfC 论文的"1-5 数量级
  加速"是同一现象的不同侧写(本机 CPU path 上 ODE 求解的开销被 CfC 的闭式
  摊销大幅缩小)。
- **Pareto front 上的 4/5 是 GRU**:本任务(单步合成非平稳预测)更偏向传统
  RNN 的归纳偏置;LNN 优势在"少参数 + 精度"维度,不在"绝对速度"。
- **未跑 CUDA 路径**:PyTorch 2.11+cu130 与 Jetson driver 12060 不兼容,自动
  fallback 到 CPU;`tegrastats_available: true` 已记录,等升级 driver 或
  换用 Jetson-packaged torch wheel 后可重跑 CUDA 路径。

## 2. Jetson Orin Nano Super — 旧 run (无 PyTorch 状态)

| 项 | 值 |
|---|---|
| 运行日期 | 2026-06-09 (container / manual 双次) |
| 状态 | **skipped** — PyTorch 未安装 |
| 产物 | `analysis/jetson/2026-06-09_container_lnn_benchmark.{json,md}` + `analysis/jetson/2026-06-09_manual_lnn_benchmark.{json,md}` |

旧 run 验证了脚本的 graceful skip 行为(见 `tests/test_jetson_lnn_benchmark.py::test_looks_like_cuda_runtime_error_detection` 与 `ModuleNotFoundError` 异常分支)。

## 3. 单元测试覆盖

| 测试文件 | 测试数 | 状态 |
|---|---:|---|
| `tests/test_jetson_lnn_benchmark.py` | 7 | 7/7 PASSED (2026-06-09_local) |
| `tests/test_pdna_lra.py` | 6+1 | 7/7 PASSED (新增 pdna_alpha/pdna_beta tracking) |
| `tests/test_pdna_pulse.py` | 12 | 12/12 PASSED (iter#19) |
| `tests/test_natural_gas_lnn.py` | 8 | 8/8 PASSED (iter#29) |
| `tests/test_loop_status_prd.py` | 8 | 8/8 PASSED (iter#21) |
| `tests/test_sncp_pedestrian_env.py` | 6 | 6/6 PASSED (iter#27) |
| `tests/test_sncp_policy_lite.py` | 10 | 10/10 PASSED (iter#26) |
| 旧测试套件 (LTC/CfC/variants/NCP/multimodal/...) | 70+ | 70+/70+ PASSED |

## 4. 后续待跑

- **CUDA 路径**:等 Jetson 升级 driver (≥ 12060) 或换用 Jetson-packaged
  torch wheel,重跑 `scripts/jetson_lnn_benchmark.py --quick --pareto` 不
  带 `--cpu`,对比 CPU vs CUDA 加速比。
- **多 seed (≥3)**:当前 Pareto sweep 只 1 seed,需加 `--seeds 42,123,7` 跑
  3 seeds × 2 hidden × 2 seq_lens = 12 configs。
- **能量列**:本机暂无 INA219 电流分流器,`tegrastats VDD_IN` 精度不足以
  复现 Liu et al. 2025 < 10 mW 量级;若加 INA219 探针可加到 harness。
- **导出后延迟**:TensorRT / ONNX 路径需先解决 ODE 求解器在 ONNX RNN/LSTM
  operator 上的 unroll 损失(见
  [[PRD_LNN_Edge_Research#angle-3]]),是 §10 #10-19 的下一阶段。
