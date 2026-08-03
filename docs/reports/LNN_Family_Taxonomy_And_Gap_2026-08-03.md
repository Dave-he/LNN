---
title: LNN 全家桶谱系与缺口清单 - 2026-08-03
date: 2026-08-03
tags: [LNN, LTC, CfC, NCP, LFM2, taxonomy, gap-analysis, jetson, orin-nano]
---

# LNN 全家桶谱系与缺口清单

> 本机 = **NVIDIA Jetson Orin Nano Super Engineering Reference Developer Kit** (Ampere SM 8.7, 1024-core GPU, 8/16 GB shared),Host PyTorch `2.11.0+cu130` 与 JetPack CUDA 12.6 不匹配,所以 cuda 路径需要走 docker 容器(实测 `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` + torch 2.10.0 + cu126 已跑通)。

## 1. 谱系分类

### 1.1 单元级 (Cell)

| 模型 | 文件 | 重点 |
|---|---|---|
| `LiquidNeuron` | `lnn/core/liquid_neuron.py` | 输入相关 τ 的最简 ODE 单元 |
| `LTCCell` | `lnn/core/ltc.py` | ODE-based,走 `torchdiffeq.odeint`(rk4/euler/dopri5) |
| `CfCCell` | `lnn/core/cfc.py` | 闭式连续时间 — sigma(-fτt)*g + (1-sigma(-fτt))*h;**已支持 `n_tau` 多时间常数** (PRD #10-29, 2026-06-14) |
| `MultiRateMoECfC` ⭐ NEW | `lnn/core/multirate_moe_cfc.py` | 多速率 K-τ + EC routing (per-step top-K branches),合 arxiv 2606.12240 |

### 1.2 网络级 (Network / Layer)

| 模块 | 说明 |
|---|---|
| `LTCNetwork` | 序列→输出,可堆叠 LTC |
| `CfCNetwork` | 序列→输出,可堆叠 CfC |
| `LiquidLayer` / `LiquidNN` | 经典 Hasani 风格 ODE 池 |
| `LiquidS4Block` / `LongSequenceLiquidClassifier` | 长序列场景的 liquid+S4 混合结构 |
| `LiquidTADHead` | 时序动作检测 head |
| `GraphSnapshotEncoder` / `GraphLNNPredictor` | 图快照编码→液体图传播预测 |
| `MultiRateMoECfCNetwork` ⭐ NEW | 多速率 MoE 的序列级 wrapper |

### 1.3 变体与正则 (lnn/core/ 共 202 个 .py)

关键子家族(共 ~28 个 CfC 变体):

- **β 学习类**: `learned_beta_ps_init_cfc` / `learned_beta_ps_ln_cfc` / `learned_beta_ps_ln_khlfft_*_cfc` / `learned_beta_ps_ic_cfc` / `learned_beta_xh_cfc`
- **分支 / MoE**: `expert_choice` / `prob_moe` / `gumbel_moe` / `anchored_moe` / `combined_per_branch_per_step_aux_cfc` / `multirate_moe_cfc` ⭐ NEW
- **门控 / 注意力**: `binary_gated_pulse_cfc` / `gated_linear_unit_cfc` / `soft_neuron_attention_cfc`
- **频域 / 谱**: `freq_experts` / `khlfft_attn_cfc` / `khlfft_ssm_cfc`
- **抗噪 / 时间**: `noise_adaptive_cfc` / `temporal_dropout_cfc` / `decorrelation_loss`
- **结构化**: `bidirectional_cfc` / `grud_cfc` / `controllability_cfc` / `clockwork_cfc`
- **概率 / 不确定**: `cfcnad_mdn_uncertainty`
- **Lyapunov / Frozen**: `frozen_multibasin_lyapunov_cfc`

### 1.4 头部分层

| 头 | 文件 | 用途 |
|---|---|---|
| `MDNHead` | `lnn/core/mdn.py` | 不确定性 + 概率采样 |
| `LiquidTADHead` | `lnn/core/liquid_tad.py` | 时序动作检测 |
| `PDNAPulseHead` | `lnn/core/cfc.py` | 用 arxiv 2603.00153 pulse 调制 + recurrent self-attend |

### 1.5 适配层

- `lnn/ncps_integration/ncps_models.py`: `NCPSCfC` / `NCPSLTC` / `NCPSAutoNCP`,可选依赖 ncps
- `lnn/core/liquid_neuron.py` 系列 — 与 LiquidAI 官方 LNN 行为一致

### 1.6 LFM2 路径

- `lnn/lfm2/inference.py`: `LFM2Inference`, `AVAILABLE_MODELS = {LFM2-350M/700M/1.2B/2.6B-Exp/24B-A2B}`(LiquidAI 官方别名表)
- 推理后端: transformers + accelerate + sentencepiece

### 1.7 数据生成器(全 in-house 合成)

`lnn/data/` 17 个生成器: 时序 / LRA 类 / 回归 / 通用。

> 安全边界: **所有"设备操控 LNN"主题都走合成数据, 0 真传感器 / 0 adb / 0 设备驱动**,沿用 2026-06-09 critical 级偏好。

## 2. Orin Nano Super 上的现状

| 维度 | 当前 | 备注 |
|---|---|---|
| Host CUDA 兼容 | ❌ torch 2.11+cu130 vs l4t CUDA 12.6 | 见 [[Orin_Nano_Super_LNN_Deployment_v2_2026-08-03]] |
| 容器路径 | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` (21.3 GB cached) | **本次实测 GPU 路径走这条** |
| 既有产物 | `analysis/jetson/2026-08-03-{cpu,gpu}-pareto_*` | 2026-08-03 CPU + GPU 双侧 Pareto |
| GPU Pareto top | PDNAPulse h=24 seq=32 seed=43 MSE 0.279 | vs CPU 同配置 0.412 |

### 2.1 Orin Nano Super vs 旧 Jetson Nano (Maxwell SM 5.3)

| 维度 | 旧 Nano(4GB) | Orin Nano Super(8GB) | 影响 LNN 部署的关键差 |
|---|---|---|---|
| SM arch | Maxwell 5.3 (sm_53) | Ampere 8.7 (sm_87) | Tensor cores → fp16/int8 通路打开 |
| CUDA cores | 128 | 1024 | CfC forward 吞吐潜在 8× |
| Shared mem / SM | 64 KB | 228 KB | 大 hidden_size 的 CfC 可整列驻留 |
| Mem BW (LPDDR) | 25.6 GB/s | 102.4 GB/s (LPDDR5) | stateful hidden 状态读写显著加速 |
| PyTorch wheel | sm_53 cu11 | sm_87 cu12.6 / cu13 | torch 2.5 + cu126 或 cu130 在容器中已可走 |

## 3. 缺口清单 (与 2026-08-03 digest 高价值候选对照)

### 3.1 高优先级 — 立即补

| # | 缺口 | 候选触发源 | 状态 |
|---|---|---|---|
| G1 | Orin Nano Super GPU 真 CUDA benchmark | (空) | ✅ 2026-08-03-gpu-pareto 落地 |
| G2 | Multi-Rate MoE for LNN (arxiv 2606.12240) | arxiv:2606.12240 | ✅ `MultiRateMoECfC` 落地(2026-08-03) |
| G3 | Liquid-3DGS (arxiv 2606.07670) | arxiv:2606.07670 | 待做 |
| G4 | LiquidTAD streaming stateful checkpoint | arxiv:2604.18274 | 待做 |
| G5 | LFM2.5-350M Jetson 部署 prototype | HF 2026-08-02 LFM2.5 GGUF | 离线 README 落地,需重启后实测 |

### 3.2 中优先级 — 2026-09 月内

| # | 缺口 | 触发源 | 建议 |
|---|---|---|---|
| M1 | COGENT (arxiv 2606.11162) 多时间常数无对应"上下文门控"实现 | arxiv:2606.11162 | 在 `CfCCell(n_tau=K, gate_kind="context")` 加 `gate_kind` 参数 |
| M2 | 仓库 GRUD-CfC 没有 NaN/inf 兜底 unit test | (内部) | `tests/test_grud_cfc_numerical.py` |
| M3 | Liquid-SSM (KHLFFT SSM) 与 CfC 的 ablation 缺乏 Pareto 报告 | (内部缺失) | `scripts/bench_khlfft_ssm_cfc_pareto.py` |
| M4 | AutoNCP 的演化搜索**只在 ncps 端实现,仓库里没有自训版** | 调研 | `examples/auto_ncp_evolution_lite.py` |
| M5 | iOS 导出 (`scripts/export_lnn_for_ios.py`) 路径未覆盖 Orin Nano 的 Triton inference | (空) | `projects/lnn_onnx_ios_orin/` |

### 3.3 低优先级 — 路线图

| # | 缺口 |
|---|---|
| L1 | Liquid Random Feature PDEs (arxiv 2606.15571) |
| L2 | FlowFake audio deepfake 用 CfC (arxiv 2606.19579) |
| L3 | GazeLNN human attention (arxiv 2606.20491) |
| L4 | TFP-Memory-Fusion Policies (arxiv 2607.08283 v2) |
| L5 | LTC fall detection 双 LTC edge (arxiv 2607.12909 v1) |

## 4. 推荐落地顺序

1. ✅ G1 跑今天 2026-08-03 的 jetson benchmark (CPU + GPU 全套,包括 lnn-jetson-orin 容器)
2. ✅ G2 multi-rate MoE (MultiRateMoECfC + Network + smoke)
3. G3 G4 选一补代码层变体
4. G5 LFM2.5-350M 端到端 8GB-smoke(在重启后)
5. L1-L5 各落 1 行进入下一份 digest 的"研读队列候选"

## 5. 数据源回链

- [[LNN_每日研究追踪 - 2026-08-03]]
- [[LNN_深度研读报告]]
- [[每日自动化任务与Jetson验证]]
- [[Liquid_Neural_Networks_Latest_Papers_Summary]]
- [[Orin_Nano_Super_LNN_Deployment_v2_2026-08-03]]
- `analysis/jetson/2026-08-03-cpu-pareto_*` / `analysis/jetson/2026-08-03-gpu-pareto_*`
