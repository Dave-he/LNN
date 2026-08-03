---
title: Jetson Orin Nano Super + LNN 部署可行性 v2 - 2026-08-03
date: 2026-08-03
tags: [LNN, Jetson, Orin-Nano-Super, Ampere, sm_87, TensorCore, quantization, deploy]
---

# Jetson Orin Nano Super + LNN 部署可行性 v2

> 配合 [[LNN_Family_Taxonomy_And_Gap_2026-08-03]] 一起读。本文档是结论 + 决策清单。
> 设备 = **NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super**(Ampere SM 8.7, 1024-core GPU, 8/16 GB shared)。

## 1. 硬件事实

| 维度 | 数值 | 对 LNN 部署的含义 |
|---|---|---|
| SM arch | Ampere 8.7 (sm_87) | 支持 TF32 / BF16 / FP16 / INT8 / INT4 (Tensor Core);SM_87 不支持 FP8 |
| CUDA cores | 1024 | CfC forward (per-frame 4 small matmuls): 顺序可达 100K+ frames/s |
| Tensor cores | 32 (3rd gen) | matmul throughput 30+ TFLOPS INT8 / 15 TFLOPS BF16 |
| Memory | 8/16 GB LPDDR5, 102.4 GB/s BW (shared with GPU/CPU) | LFM2.5-350M fp16 ~700MB,KV cache 吃剩余 |
| Power | 7-15W (Super mode) | 边界量化阈值 |
| Jetson BSP | R36.4.7 (本机) | JetPack 6.2.1 = CUDA 12.6 |

> 实测容器内 GPU 报告:**sm_87, 8 SM, 7.6 GB**(Orin Nano Super 是 8 SM × 128 cores 配置,CUDA 7.6 GB shared)。

## 2. Host PyTorch 不兼容 (本次发现)

```text
torch version: 2.11.0+cu130
CUDA available: False
Driver / CUDA runtime: 12060 (JetPack 12.6) < 13.0 expected by torch 2.11
```

Host PyTorch 2.11+cu130 与 JetPack 12.6 driver 不匹配。本 session 选了**沙箱内 l4t 容器**(不动 host):

```bash
docker pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin  # 已 cache 21.3 GB
docker build -f projects/lfm25_orin_nano_smoke/Dockerfile.lnn -t lnn-jetson-orin .
docker run --rm --runtime nvidia --network host \
  -v "$PWD":/workspace/LNN -w /workspace/LNN \
  lnn-jetson-orin \
  python3 scripts/jetson_lnn_benchmark.py --no-cpu-fallback
```

实测成功:`torch 2.10.0`, `cuda available = True`, `device = Orin`, `arch = (8, 7)`, `multiprocessors = 8`。

## 3. 量化与 kernel 后端

### 3.1 OK(生产可用)

| 路径 | 推荐度 | 备注 |
|---|---|---|
| TF32 matmul | ⭐⭐⭐⭐⭐ | 默认开,精度损失 < 0.1%,速度 +10-30% |
| FP16 (half) | ⭐⭐⭐⭐⭐ | sm_87 native,速度 +50-100%;CfC/LTC 数值稳定 |
| BF16 | ⭐⭐⭐⭐ | 推荐 LFM2.5 + CfC head,sm_87 native |
| INT8 静态量化 (PyTorch Eager) | ⭐⭐⭐⭐ | weight-only 量化简单,activation calibration 复杂 |
| INT8 TensorRT | ⭐⭐⭐⭐⭐ | 最优,sm_87 + cu126 容器已可走 |
| TorchScript 导出 → iOS | ⭐⭐⭐⭐ | 仓库已有 `scripts/export_lnn_for_ios.py` |

### 3.2 NOT OK(本硬件/版本不支持)

| 路径 | 不支持原因 | 替代 |
|---|---|---|
| FP8 (E4M3/E5M2) | sm_87 不支持 | 用 INT8 或 BF16 |
| Tensor Core TF32x3 sparse | 不在 sm_87 公开 instruction set | 用 INT8 dense |

### 3.3 不确定(要看实现)

| 路径 | 待评估 |
|---|---|
| Triton kernel for CfC forward | sm_87 支持,需要重写多步循环 |
| torch.compile (inductor) | cu126 应该 OK,本机 torch 2.11 未经 inductor 测试 |
| CUTLASS fused linear+gating | sm_87 支持 CUTLASS 2.x,实现门槛高 |

## 4. CPU smoke + GPU 实测 (2026-08-03)

### 4.1 GPU Pareto front 头部 (4 模型 × 3 hidden × 2 seq × 2 seed = 48 条)

| 模型 | hidden | seq_len | seed | MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|
| **CfCStyle** | 24 | 32 | 43 | **0.249** | 32 119 | 3.61 |
| PDNAPulse | 16 | 32 | 43 | 0.267 | 36 020 | 3.24 |
| PDNAPulse | 24 | 32 | 43 | 0.279 | 37 089 | 3.17 |
| CfCStyle | 16 | 32 | 43 | 0.289 | 34 150 | 3.55 |
| PDNAPulse | 8 | 32 | 43 | 0.307 | 34 633 | 3.27 |
| GRU | 24 | 32 | 42 | 0.352 | **1 229 079** | 0.16 |
| LTC | 8 | 32 | 43 | 0.438 | 6 386 | 13.28 |

### 4.2 双侧加速比 — 校正之前的预测

| 模型 | CPU 实测 steps/s | GPU 实测 steps/s | 加速比 | 估算 (本报告 v1) |
|---|---:|---:|---:|---:|
| CfCStyle h=24 | 18 201 | 34 444 | **1.9×** | ✗ 8-10×(预测高估) |
| GRU h=24 | 69 905 | 1 229 079 | **17.6×** | ✓ 8-10× |
| LTC h=8 seq=32 | 10 575 | 6 469 | **0.61×** | ✗ 8×(实际更慢) |
| PDNAPulse h=24 | 23 593 | 36 723 | **1.6×** | ✗ 8× |

**校正结论**:

1. **估算是错的**。CfC / PDNA 是 inner-loop sequential,只能在 GEMM-bound 的部分拿到 GPU 加速,实际加速比 1.6-1.9×(host sync 锁住)。
2. **LTC 必须换 backend**(`torchdiffeq.odeint_adjoint` / `torchode`)否则在 GPU 上亏。
3. **预测修订**:sm_87 上 hidden=24-256 CfC forward 在 GPU 实际可达 **30-50K steps/s**;`torch.compile(reduce-overhead)` + hidden=128+ 理论可推到 **80-120K steps/s**;GRU 类 full-GEMM 实测 **1.2M steps/s** (h=24)。
4. 数据来源:`analysis/jetson/2026-08-03-cpu-pareto_*` vs `2026-08-03-gpu-pareto_*`。

## 5. CfC 的 sm_87 部署策略

### 5.1 Hidden size 与 TensorCore 映射

| hidden | TensorCore 利用率估计 |
|---:|---|
| 16-32 | < 50%(浪费大) |
| 64-128 | 70-85% |
| 256-512 | 85-95% |
| 1024 | 90+ |

CfC 默认 hidden=8(smoke)是低效的;hidden≥64 才能打 TensorCore。

### 5.2 Precision trade-off

| 模式 | 速率倍 | CfC 数值风险 |
|---|---|---|
| FP32 | 1× | 0 |
| TF32 | 1.1-1.3× | 极低 |
| FP16 | 2-3× | 低 |
| BF16 | 2-3× | 极低 |
| INT8 | 4-6× | 中(calibration 后稳定)|

### 5.3 SMEM 约束

`CfCCell(n_tau=K, hidden=H)` 每步 `combined = cat([x, h])` → `O((in+H)·B) bytes` SMEM。Smoke 设置 H=8, B=192 时,SMEM < 4 KB/branch。

**瓶颈不是 SMEM,而是 inner loop 是 sequential** — `for t in seq_len` 不可向量化。

### 5.4 seq-loop 解向量化

| 方法 | 复杂度 | 推荐 |
|---|---|---|
| `torch.compile(model, mode="reduce-overhead")` | 低 | 优先试用 |
| Triton 自定义 `cfc_step_kernel` | 中 | hidden=64+ 才划算 |
| CUDA-graph capture | 低 | 适合 seq_len 固定 inference |

## 6. LFM2.5 在 Orin Nano Super 的部署图

> 详见 `projects/lfm25_orin_nano_smoke/README.md`。

```
LFM2.5-350M (fp16 ~700MB)   ← HuggingFace
   ↓ transformers + accelerate
bf16 model
   ↓ onnx export
ONNX graph
   ↓ tensorrt --fp16 --workspace=4096
TensorRT engine
   ↓ CUDA stream on Orin Nano GPU
expected ~12-18 tokens/s  (350M, batch=1, ctx=512)
```

CPU fallback 期望 ~2-3 tokens/s。

## 7. 决策清单

1. ✅ 沙箱内 l4t 容器已跑通(`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` + `Dockerfile.lnn`)
2. `torch.compile` 在 CfCStyle hidden=64/128 上做 mode="max-autotune",期望推到 80-120K steps/s
3. 把 `lnn/core/ltc.py` 的 `odeint` 切到 `torchdiffeq.odeint_adjoint` 或 `torchode`,否则 GPU 慢
4. LFM2.5-350M onnx export + tensorrt fp16 engine(容器内)
5. 把 PDNAPulseHead 的 amplitude/phase 模块单独 fp16 化,验证精度损失 < 1%

## 8. 关联文档

- [[LNN_Family_Taxonomy_And_Gap_2026-08-03]]
- [[LNN_每日研究追踪 - 2026-08-03]]
- `analysis/jetson/2026-08-03-gpu-pareto_lnn_benchmark.{md,json,png}`
- `analysis/jetson/2026-08-03-cpu-pareto_lnn_benchmark.{md,json,png}`
- `projects/lfm25_orin_nano_smoke/README.md`
- `examples/multirate_moe_cfc_smoke.py`
- [[每日自动化任务与Jetson验证]]
