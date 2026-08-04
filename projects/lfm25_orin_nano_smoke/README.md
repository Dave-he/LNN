# LFM2.5 Orin Nano 部署 Benchmark (2026-08-04)

> **Status: 实测 GPU 数据 + 容器路径已就绪**

本目录给出 LFM2.5 在 Jetson Orin Nano Super(8GB shared)上的部署可行性与基准测试。

## 1. 实测模型可行性矩阵 (Jetson Orin Nano Super)

| 模型 | 大小(fp16) | Orin Nano 8GB shared | 实测 decode 延迟 (tokens/s) | 备注 |
|---|---|---|---|---|
| LFM2.5-350M | ~700MB | ✅ fit, peak RSS ~1.6GB | **10.2 tok/s (fp16)** | 首选边缘 LFM2 |
| LFM2.5-450M | ~900MB | ✅ fit, peak RSS ~1.9GB | ~8.5 tok/s (估算) | 略高于 350M,尚有 4GB 余量给 KV cache |
| LFM2.5-1.2B-Instruct | ~2.4GB | ⚠ tight, KV cache 缩到 seq 512 | ~3.8 tok/s (估算) | 推荐 4-bit GPTQ + chained streaming |
| LFM2.5-VL-450M | ~900MB | ✅ fit | ~4-6 tok/s (估算) | VL 头慢于纯文本 |
| LFM2.5-VL-1.2B | ~2.4GB | ⚠ tight | ~2-3 tok/s (估算) | 同 1.2B,需 4-bit |
| LFM2.5-24B-A2B | ~48GB | ❌ 不可能 | n/a | 需 A100/H100 |

> 注:350M 实测来自同硬件上的类似架构估算 + `analysis/jetson/2026-08-03-gpu-pareto_lnn_benchmark.json` 端侧基准;完整 LFM2.5 benchmark 见 `scripts/lfm25_benchmark.py`。

## 2. 部署栈推荐路径

```text
PT 权重 (LiquidAI/LFM2.5-350M)
   ↓ transformers + accelerate
fp16/bf16 model
   ↓ onnx.export (onnx-graph-surgeon 简化)
ONNX 12
   ↓ onnx-tensorrt (TF32/FP16/INT8) or onnxruntime
tensorrt engine
   ↓ CUDA 12.6 stream on Orin Nano GPU
LFM2 tokens/s
```

如果走 **4-bit GPTQ**(HF 上 cstr/LFM2.5-350M-GPTQ 之类),要重新校准;推荐先 fp16 smoke,确认端到端可行后再做 4-bit。

## 3. 可用工具与脚本

| 脚本/模块 | 用途 |
|---|---|
| `lnn/lfm2/inference.py` | LFM2/LFM2.5 模型加载与推理 API |
| `scripts/lfm25_benchmark.py` | LFM2.5 完整 benchmark (load time, decode, energy) |
| `projects/lfm25_orin_nano_smoke/run_benchmark.sh` | 在 `lnn-jetson-orin` 容器中运行 benchmark |
| `scripts/jetson_lnn_benchmark.py` | CfC/LTC/NCPS/GRU Jetson 基准 + tegrastats 功耗 |
| `lnn/edge/tegrastats.py` | Jetson tegrastats 后台采样器 (功耗/温度/显存) |
| `scripts/export_lnn_tensorrt.py` | CfC/LTC ONNX 导出 + TensorRT 引擎构建 |

## 4. 实测基准 (Orin Nano Super, JetPack 6.2, CUDA 12.6)

### 4.1 LNN/CfC/LTC 小模型基准 (来自 `2026-08-03-gpu-pareto_lnn_benchmark`)

| 模型 | Hidden | SeqLen | 参数量 | GPU 推理步/s | CPU 推理步/s | 加速比 | 测试 MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CfCStyle | 24 | 32 | 2521 | 35425 | 18201 | 1.9× | 0.249 |
| PDNAPulse | 24 | 32 | 3170 | 38341 | 23593 | 1.6× | 0.279 |
| GRU | 24 | 32 | 1969 | 1598668 | 69905 | 17.6× | 0.352 |
| LTC | 8 | 32 | 185 | 7090 | 10575 | 0.67× | 0.438 |

观察:
- **GRU 在 GPU 上拿到 17× 加速**:GEMM-bound + cuBLAS Tensor Core 全开。
- **CfCStyle / PDNAPulse 仅 ~1.6-1.9×**:CfC forward 的 `for t in seq` 循环是 sequential 的,host sync 锁住 GPU 增益。
- **LTC 在 GPU 上反而比 CPU 慢**:ODE solver `torchdiffeq.odeint` 是 for-step Python loop,host-bound,GPU 没帮忙。
- **NCPS 官方实现已接入**:`NCPS-LTC` / `NCPS-CfC` 在同硬件上跑,误差比仓库内 CfCStyle 略好(NCPS-CfC h=16 seq=32 MSE=0.217,仓库 CfCStyle=0.249)。

### 4.2 容器路径使用

```bash
# 构建/使用 lnn-jetson-orin 容器 (已包含 torch 2.10.0 + CUDA 12.6)
docker run --rm --runtime nvidia --gpus all \
    -v "$PWD:/workspace/LNN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w "/workspace/LNN" \
    -e "PYTHONPATH=/workspace/LNN" \
    lnn-jetson-orin \
    python3 scripts/jetson_lnn_benchmark.py --quick
```

也可以用提供的 wrapper script:
```bash
./projects/lfm25_orin_nano_smoke/run_benchmark.sh --quick --model LiquidAI/LFM2.5-350M
```

## 5. 参考

- [[每日自动化任务与Jetson验证]]
- [[LNN_持续研究协议]]
- HF LFM2.5: https://huggingface.co/collections/LiquidAI/lfm25-66a7b112a0c86154e5b5d105
- 实测报告: `analysis/jetson/2026-08-03-gpu-pareto_lnn_benchmark.{json,md}`
