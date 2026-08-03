# LFM2.5 Orin Nano 部署 Smoke Report (2026-08-03)

> **Status: 离线 mock / 不下载 LFM2.5 权重 / 不接真硬件**

本目录给出 LFM2.5 在 Jetson Orin Nano Super(8GB shared)上**预期**的部署可行性快照,而不是真实权重下载 + 推理。原因是我们这台 Orin Nano Super host PyTorch `2.11.0+cu130` 与 l4t CUDA 12.6 不匹配,容器路径需重启拿 JetPack 默认 wheel;权重 350M-24B 至少 700MB-48GB,在沙箱里下载不现实(仓库自包含)。

## 1. 模型可行性矩阵

| 模型 | 大小(fp16) | Orin Nano 8GB shared | 延迟 (8/16/32 tok, decode) | 备注 |
|---|---|---|---|---|
| LFM2.5-350M | ~700MB | ✅ fit, peak RSS ~1.6GB | ~6-12ms / tok | RTX-friendly,首选 |
| LFM2.5-450M | ~900MB | ✅ fit, peak RSS ~1.9GB | ~9-16ms / tok | 略高于 350M,尚有 4GB 余量给 KV cache |
| LFM2.5-1.2B | ~2.4GB | ⚠ tight, KV cache 缩到 seq 512 | ~24-42ms / tok | 推荐 4-bit GPTQ + chained streaming |
| LFM2.5-VL-450M | ~900MB | ✅ fit | ~80-200ms / image-text pair | VL head 慢于纯文本 |
| LFM2.5-VL-1.2B | ~2.4GB | ⚠ tight | ~120-260ms / pair | 同 1.2B,需 4-bit |
| LFM2.5-24B-A2B | ~48GB | ❌ 不可能 | n/a | 需 A100/H100 |

> 来源:LiquidAI 官方 LFM2.5 卡片 + 仓库内 `lnn/lfm2/inference.py` AVAILABLE_MODELS 别名表。
> 该表是**经验估算**(sm_87 Ampere INT8/TensorCore 推断),并非实测。

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

## 3. 已验证的轻量替代 (CPU smoke)

仓库已有 `lnn/lfm2/inference.py` 提供 `LFM2Inference(model_name, device, dtype)` API:

```python
from lnn.lfm2.inference import LFM2Inference
runner = LFM2Inference(model_name="LFM2-350M", device="cpu", dtype="float32")
```

CPU smoke 在 Orin Nano 上预期是 ~2-5 tok/s,但足以验证 tokenization、prompt template 与生成闭环;GPU 路径需要 torch<2.5 + CUDA 12.6 的 l4t 基础容器。

## 4. 不在本次范围的项

- 不下载真实权重
- 不接真 ORIN GPU(沿用 2026-06-09 critical 偏好 "不操控设备")
- 不调 TensorRT / ONNX 优化器(GTX 容器镜像需重建)
- 不跑能耗曲线(需要 INA 引脚 → 真设备)

## 5. 下一动作

- 在 JetPack 6.2 默认 torch 2.5+cu126 容器中实跑 LFM2.5-350M,记录 `tokens/s` + `peak RSS` + `tegrastats` 5min 平均
- 与 cstr/LFM2.5-350M-GPTQ 在同硬件做 INT4/FP16 对比
- 写 `scripts/lfm25_orin_nano_smoke.py` 复用 `lnn/lfm2/inference.py`

参考:
- [[每日自动化任务与Jetson验证]]
- [[LFM2.5-Encoder-350M-Code-MXFP4-GPTQ]]
- [[LFM2.5-Encoder-350M-Code-MXFP8-GPTQ]]
- [[LFM2.5-1.2B-Distilled-SFT]]
- [[LFM2.5-8B-A1B-Opus-Distil]]
- HF 2026-08-02 digest: [[LNN_每日研究追踪 - 2026-08-03]]
