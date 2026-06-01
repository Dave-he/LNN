---
title: LFM2.5 本地推理与 1.2B DPO 量化验证
date: 2026-06-01
tags: [LFM2.5, GGUF, DPO, llama.cpp, quantization, Jetson]
---

# LFM2.5 本地推理与 1.2B DPO 量化验证

## 结论

- **官方 GGUF 路径可用**：`LiquidAI/LFM2.5-1.2B-Instruct-GGUF` 的 `Q4_0` 文件已在本机 `llama.cpp` CPU 路径完成加载、生成和 `llama-server` OpenAI-compatible 接口调用。
- **1.2B DPO Transformers 路径可用**：`LiberteEPFL/lfm25-1.2b-dpo-bigchat-v2-s1` 的 full-model `model.safetensors` 已在本机 CPU 路径完成短生成。
- **1.2B DPO 量化路径可用**：DPO safetensors 已成功转换为 F16 GGUF，并通过 `llama-quantize` 量化为 Q4_0；新生成的 DPO Q4_0 GGUF 也能被 `llama-cli` 加载并生成。
- **CUDA Docker 路径可用**：参考 jetson-containers 的方式，使用 `ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin` 通过 `--runtime nvidia` 成功执行 CUDA 版 `llama-cli`，官方 Q4 与 DPO Q4 均完成 GPU layer offload smoke test。
- **主机 PyTorch CUDA 暂不可用**：原因是本机 NVIDIA driver 版本低于当前 PyTorch CUDA 13.0 build 需求；PyTorch 侧建议后续改用匹配 L4T/CUDA 12.6 的 Jetson 容器或 wheel。

## 环境

- 设备：`NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super` / Linux `5.15.148-tegra` / `aarch64`
- L4T：`R36.4.7`
- 系统 CUDA：`12.6.11`
- Python：`3.14.4`
- PyTorch：`2.11.0+cu130`
- CUDA：`torch.cuda.is_available() == False`
- `llama.cpp`：commit `a511424`，CPU-only build
- CUDA Docker 镜像：`ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`
- 容器内 `llama.cpp`：`version 8638 (5803c8d11)`，`ggml_cuda_init` 检测到 `Orin, compute capability 8.7`
- 已构建工具：
  - `projects/llama.cpp/build/bin/llama-cli`
  - `projects/llama.cpp/build/bin/llama-server`
  - `projects/llama.cpp/build/bin/llama-quantize`

## 模型与文件

| 路径 | 来源 | 大小 | SHA-256 |
|---|---|---:|---|
| `models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf` | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | 695,751,488 bytes | `2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f` |
| `models/lfm25-dpo-s1/model.safetensors` | `LiberteEPFL/lfm25-1.2b-dpo-bigchat-v2-s1` | 2,340,697,936 bytes | `22a288f17e8ced897bbd2bd35ceea5528fc0c0b4c2805e780afde5675c927fe6` |
| `models/lfm25-dpo-s1/LFM25-DPO-F16.gguf` | 本地转换 | 2,343,325,600 bytes | `0f799994672277549327f03db61db7bf399c435f134803789746e36b7850245c` |
| `models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf` | 本地量化 | 695,750,560 bytes | `d5897de9fb77ed025f8a9a05c565b72f0394de369eda9fcae7d24baf7db1bd8b` |

## 验证命令

### 官方 GGUF 推理

```bash
projects/llama.cpp/build/bin/llama-cli \
  -m models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf \
  -p 'Write one short sentence about liquid neural networks.' \
  -n 24 -c 512 -t 4 --temp 0.1 --top-k 40 \
  --repeat-penalty 1.05 --no-display-prompt --no-warmup \
  --single-turn --simple-io
```

结果：

- 输出：`Liquid neural networks are adaptive AI models that dynamically adjust their structure in response to data.`
- 速度：Prompt `58.8 t/s`，Generation `19.0 t/s`
- 机器可读记录：`analysis/lfm25/2026-06-01_lfm25_local_validation.json`

### `llama-server` OpenAI-compatible 接口

```bash
projects/llama.cpp/build/bin/llama-server \
  -m models/lfm25/LFM2.5-1.2B-Instruct-Q4_0.gguf \
  -c 512 -t 4 --host 127.0.0.1 --port 18080 --no-webui
```

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-1.2b-instruct-q4_0","messages":[{"role":"user","content":"Reply with exactly three words about LNNs."}],"temperature":0.1,"max_tokens":12}'
```

结果：

- HTTP 正常返回 `chat.completion`
- `system_fingerprint`: `b1-a511424`
- `prompt_per_second`: `45.05`
- `predicted_per_second`: `16.81`

### DPO Transformers 推理

```bash
python scripts/validate_lfm25_local.py \
  --max-new-tokens 24 \
  --hash-dpo \
  --timeout 300 \
  --output analysis/lfm25/2026-06-01_lfm25_local_validation.json
```

结果：

- DPO 状态：`ok`
- DPO 输出：`Liquid neural networks are a type of artificial neural network that uses a continuous, flowing representation of data, allowing for`
- 加载耗时：`1.863s`
- 生成耗时：`13.716s`
- 输入 / 输出 token：`19 / 24`
- 注：PyTorch 在 ARM CPU 上触发 BF16 `mkldnn_matmul` fallback 到 BLAS GEMM，但生成成功。

### DPO 转 GGUF

```bash
python projects/llama.cpp/convert_hf_to_gguf.py \
  models/lfm25-dpo-s1 \
  --outfile models/lfm25-dpo-s1/LFM25-DPO-F16.gguf \
  --outtype f16
```

结果：

- 识别架构：`Lfm2ForCausalLM`
- GGUF 架构：`lfm2`
- 张量数：`148`
- 输出文件：`models/lfm25-dpo-s1/LFM25-DPO-F16.gguf`
- 转换成功。

### DPO GGUF Q4_0 量化

```bash
projects/llama.cpp/build/bin/llama-quantize \
  models/lfm25-dpo-s1/LFM25-DPO-F16.gguf \
  models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf \
  Q4_0
```

结果：

- F16 model size：`2232.50 MiB`
- Q4_0 quant size：`661.25 MiB`
- 量化耗时：`8026.68 ms`
- 输出文件：`models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf`

### DPO Q4_0 GGUF 推理

```bash
projects/llama.cpp/build/bin/llama-cli \
  -m models/lfm25-dpo-s1/LFM25-DPO-Q4_0.gguf \
  -p 'Write one short sentence about liquid neural networks.' \
  -n 24 -c 512 -t 4 --temp 0.1 --top-k 40 \
  --repeat-penalty 1.05 --no-display-prompt --no-warmup \
  --single-turn --simple-io
```

结果：

- 输出：`Liquid neural networks are a type of artificial intelligence that uses liquid-like materials to process information, allowing for faster and`
- 速度：Prompt `77.4 t/s`，Generation `25.7 t/s`
- 状态：量化产物可加载、可生成。

## CUDA Docker / Jetson 验证

新增 CUDA Docker 验证脚本：

```bash
python scripts/validate_lfm25_cuda_docker.py
```

脚本行为：

- 使用 `docker run --rm --runtime nvidia --network host --ipc host` 启动 `ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`。
- 挂载当前仓库到容器内 `/workspace/LNN`。
- 先运行容器探针，确认 `nvidia-smi` 与 CUDA 版 `llama-cli` 可用。
- 对候选 `--n-gpu-layers` / `ctx` 自动降级尝试，避免 Jetson 统一内存压力导致一次 OOM 后直接误判失败。
- 输出机器可读记录到 `analysis/lfm25/2026-06-01_lfm25_cuda_docker_validation.json`。

本次容器探针结果：

- 镜像：`ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`
- GPU：`Orin (nvgpu)`
- Driver：`540.4.0`
- CUDA：`12.6`
- 容器内 `llama-cli`：`version 8638 (5803c8d11)`，`built with GNU 11.4.0 for Linux aarch64`
- CUDA 初始化：`ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7619 MiB)`

本次 CUDA smoke test 结果：

| 模型 | 成功配置 | 输出 | Prompt | Generation | CUDA 内存摘要 |
|---|---|---|---:|---:|---|
| 官方 `LFM2.5-1.2B-Instruct-Q4_0.gguf` | `--n-gpu-layers 1 -c 256 -n 8` | `Yes, **CUDA works**` | `23.7 t/s` | `15.6 t/s` | `CUDA0 self 177 MiB = model 105 MiB + compute 72 MiB` |
| DPO `LFM25-DPO-Q4_0.gguf` | `--n-gpu-layers 1 -c 128 -n 8` | `Ah, I see what you mean,` | `23.2 t/s` | `10.1 t/s` | `CUDA0 self 141 MiB = model 105 MiB + compute 36 MiB` |

约束与建议：

- 当前主机内存快照：`7.4 GiB` RAM 中 `4.0 GiB` used，`218 MiB` free，`3.1 GiB` available；swap 已使用 `4.1 GiB`。
- `--n-gpu-layers 4/3/2` 在当前内存压力下会触发 `NvMapMemAllocInternalTagged` / `cudaMalloc failed: out of memory`，其中部分失败路径会伴随 llama.cpp 进程 segfault。
- 这不是模型格式问题；`--n-gpu-layers 1` 已验证 CUDA offload 可用。若要提高 offload 层数，建议先停止无关常驻容器或重启释放统一内存，再重新运行 `python scripts/validate_lfm25_cuda_docker.py --official-gpu-layers 4,3,2,1`。
- Jetson 上优先使用该 Docker CUDA 路径；主机 Python 的 `torch==2.11.0+cu130` 与当前 driver/CUDA 组合不匹配，不建议继续在 host venv 里调 PyTorch CUDA。

## 可复用脚本

新增脚本：

```bash
python scripts/validate_lfm25_local.py --help
python scripts/validate_lfm25_cuda_docker.py --help
```

用途：

- 校验 GGUF 文件大小与 SHA-256。
- 调用本地 `llama-cli` 做 GGUF smoke inference。
- 调用 Transformers 做 DPO safetensors smoke inference。
- 调用 Jetson CUDA Docker 镜像做 GGUF GPU offload smoke inference。
- 输出 JSON 到 `analysis/lfm25/`，便于后续自动化任务和 Jetson 复验。

## 来源链接

- 官方 GGUF：<https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF>
- DPO 变体：<https://huggingface.co/LiberteEPFL/lfm25-1.2b-dpo-bigchat-v2-s1>
- llama.cpp：<https://github.com/ggml-org/llama.cpp>
