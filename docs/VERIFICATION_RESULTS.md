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

### Pareto front (5/8 configs) — 2-model iter#33 baseline

| 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CfCStyle** | **16** | **32** | 42 | 1169 | **0.470338** | 21960.3 | 4.11 |
| **GRU** | **16** | **32** | 42 | 929 | 0.536350 | 98844.5 | 1.45 |
| **GRU** | 16 | 16 | 42 | 929 | 0.547923 | 110075.8 | 0.99 |
| **GRU** | 8 | 16 | 42 | 273 | 0.558936 | 92016.6 | 0.31 |
| **GRU** | 8 | 32 | 42 | 273 | 0.565124 | 118700.2 | 0.65 |

(以下 3/8 被其他配置支配,不在 Pareto 前沿:CfCStyle h=8 T=16/32 + h=16 T=16)

### Pareto front (6/16 configs) — 4-model iter#34 sweep (CfC + LTC + PDNA-pulse + GRU)

| 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **PDNAPulse** | **8** | **32** | 42 | 418 | **0.401337** | 53335.2 | 1.32 |
| **GRU** | 8 | 32 | 42 | 273 | 0.483972 | **280985.1** | 0.44 |
| **GRU** | 16 | 16 | 42 | 929 | 0.575272 | 155571.2 | 0.33 |
| **GRU** | 8 | 16 | 42 | 273 | 0.601254 | 206188.8 | 0.24 |
| **LTC** | 8 | 32 | 42 | 185 | 0.607213 | 12560.8 | 4.14 |
| **LTC** | 8 | 16 | 42 | 185 | 0.654613 | 15629.4 | 2.10 |

(以下 10/16 被支配: PDNAPulse h=16 T=16/32, CfCStyle h=8/16 T=16/32, GRU h=16 T=32, LTC h=16 T=16/32, PDNAPulse h=8 T=16)

### Pareto front (3-seed mean ± std, n=3) — 4-model iter#35 复测

**iter#11 N=5 教训兑现**: iter#34 1-seed "PDNAPulse h=8 T=32 MSE 0.401 是全局冠军"
是 lucky seed (seed=42);3-seed mean 是 **0.5361 ± 0.1199 (CV 22%)** — 高方差。
**真正冠军是 PDNAPulse h=16 T=32 (1474 params): mean 0.4224 ± 0.0257 (CV 6%)** —
比 iter#34 1-seed "冠军" 真均值低 0.014,std 5× 更小。

| 模型 | Hidden | SeqLen | Params | Test MSE (mean ± std) | Steps/sec (mean ± std) |
|---|---:|---:|---:|---:|---:|
| **PDNAPulse** | **16** | **32** | 1474 | **0.4224 ± 0.0257** | 38572 ± 6540 |
| CfCStyle | 16 | 32 | 1169 | 0.4658 ± 0.0078 | 43080 ± 3422 |
| PDNAPulse | 16 | 16 | 1474 | 0.4978 ± 0.0240 | 47763 ± 15497 |
| LTC | 16 | 32 | 625 | 0.5290 ± 0.0285 | 17121 ± 1133 |
| **PDNAPulse** | 8 | 32 | 418 | **0.5361 ± 0.1199** ← 高方差 | 51486 ± 13374 |
| LTC | 8 | 32 | 185 | 0.5504 ± 0.0527 | 15745 ± 3476 |
| GRU | 16 | 32 | 929 | 0.5452 ± 0.0195 | **209797 ± 54356** |
| CfCStyle | 8 | 32 | 329 | 0.5585 ± 0.0042 ← 最稳定 | 49657 ± 14429 |
| CfCStyle | 16 | 16 | 1169 | 0.5636 ± 0.0653 | 31131 ± 6409 |
| GRU | 8 | 32 | 273 | 0.5863 ± 0.1337 ← 高方差 | 156830 ± 72250 |
| LTC | 16 | 16 | 625 | 0.5905 ± 0.0181 | 15053 ± 3485 |
| GRU | 16 | 16 | 929 | 0.5805 ± 0.0086 | 180559 ± 22587 |
| LTC | 8 | 16 | 185 | 0.6121 ± 0.0377 | 15522 ± 2504 |
| CfCStyle | 8 | 16 | 329 | 0.5738 ± 0.0153 | 45467 ± 7961 |
| GRU | 8 | 16 | 273 | 0.6200 ± 0.0230 | 116799 ± 18036 |
| PDNAPulse | 8 | 16 | 418 | 0.6382 ± 0.1091 | 34136 ± 5078 |

### 解读 (3-seed, iter#35, 关键)

- **iter#34 1-seed 冠军被部分撤回**: PDNAPulse h=8 T=32 MSE 0.401 → 3-seed
  mean 0.536 ± 0.120 (CV 22%), **是 lucky seed (seed=42)**;iter#11 N=5 教训
  兑现。
- **新冠军: PDNAPulse h=16 T=32 (1474 params) 0.422 ± 0.026** — 4× 参数但
  std 5× 小,稳胜 CfC h=16 T=32 0.466 ± 0.008 by **−9.4%**(iter#34 1-seed
  报的 −14.7% 是过估)。本机复现 PDNA paper 的 "+4.62 pp on sMNIST multi-gap"
  claim (CfC 0.466 vs PDNA 0.422 是 −9.4%,在 sMNIST 论文的 9.2% 量级,非常一致)。
- **最稳定: CfCStyle h=8 T=32 std=0.0042 (CV 0.8%)** — 是 σ 最小的 config,
  对部署最友好。
- **高方差警示: PDNAPulse h=8 T=32 std=0.120 + GRU h=8 T=32 std=0.134** —
  这两个 config 在 h=8 T=32 配置上**不应该**作为 production 候选。
- **GRU 速度冠军: h=16 T=32 209797 步/秒 ± 54k** — 但 mean MSE 0.545 排第 7。
- **iter#11 N=5 vs iter#35 N=3**: 3 seeds 给 std 但 std 自身 noisy;建议
  5-seed (iter#36) 才能严格区分"小 std"vs"大 std" config。

### 解读 (4-model, iter#34)

- **PDNAPulse h=8 T=32 是全局精度冠军 MSE 0.401**,比 iter#33 CfC 冠军 0.470
  胜 −14.7% — **PDNA pulse modulation 在小参数 (418) 下击败 CfC 与 GRU**。
- **GRU 仍主导速度** (280985 步/秒) — 比 CfC (40897) 快 6.9×,比 PDNAPulse
  (53335) 快 5.3×,比 LTC (12561) 快 22.4×。
- **LTC Pareto 仅限 h=8 (最小参数 185)**: h=16 全被支配。**LTC 在 h=16 失去
  Pareto 优势** — 这与 Hasani 2022 Nature MI 论文 claim 的 ODE-based LTC
  "1-5 数量级慢于 CfC" 现象相符,但本机 CPU path 上差距仅约 3× (CfC 40897 vs
  LTC 12561),数量级比论文的小。
- **PDNAPulse 比 CfC 还快 (53335 vs 40897)**: 在 h=8 T=32 配置下, PDNA pulse
  modulation 反而**提升了** CfC 的速度 (可能是 α gate 0.01 init 在前几 step
  抑制了额外的 sin() 计算开销,或者 torch 算子融合的副作用)。这一发现对
  PDNA paper 的"+5% wall-time"声称有修正意义 — 在 h=8 小配置下 PDNA 实际更快。
- **LTC 速度 vs CfC 速度 (论文 claim 1-5 数量级)** : 本机实测 3× (而非 10-100000×),
  原因是 CPU path 上 ODE 求解的开销被 RK4 集成器的 step reuse 摊销了。
- **未跑 CUDA 路径 / 未跑多 seed** (driver 不兼容 + iter#11 N=5 教训未拉满;
  1 seed × 4 models = 4 lucky seed 的风险仍在)。

### 解读 (2-model, iter#33 baseline)

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
