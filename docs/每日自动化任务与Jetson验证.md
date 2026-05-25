---
title: 每日自动化任务与 Jetson 验证流程
date: 2026-05-25
tags: [LNN, automation, Jetson, benchmark, GitHub]
---

# 每日自动化任务与 Jetson 验证流程

本文档记录本项目当前已落地的每日资料追踪、Jetson 本地验证和 GitHub 推送流程。

## 1. 每日资料追踪

核心脚本：`scripts/daily_lnn_research.py`

输出位置：
- `docs/daily/YYYY-MM-DD_LNN_research_digest.md`：每日研究摘要。
- `papers/daily/YYYY-MM-DD_lnn_research.json`：arXiv、GitHub、Hugging Face 原始结构化数据。
- `analysis/repo_watchlist/YYYY-MM-DD_lnn_open_source_watchlist.md`：开源仓库与模型观察清单。
- `docs/Liquid_Neural_Networks_Latest_Papers_Summary.md` 与 `docs/LNN_深度研读报告.md`：自动追加每日追踪索引。

手动运行：

```bash
python3 scripts/daily_lnn_research.py --max-results 25 --per-query 8
```

如需同时归档 arXiv PDF：

```bash
python3 scripts/daily_lnn_research.py --download-pdfs --max-pdf-downloads 5
```

## 2. 本机每日任务

核心入口：`scripts/run_daily_lnn_task.sh`

该脚本会依次执行：
1. 拉取 LNN / LTC / CfC / NCP / LFM 相关资料。
2. 如果当前机器是 Jetson，则自动运行 benchmark。若本机已有 `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` 容器，会优先通过该容器使用 CUDA 12.6；否则回退到本机 Python。
3. 将 `docs/`、`papers/`、`analysis/` 的变化提交并推送到 `origin`。

本地安装 user systemd timer：

```bash
./scripts/install_daily_lnn_timer.sh
```

默认运行时间为本机时区每天 `06:30`，可通过环境变量覆盖：

```bash
ON_CALENDAR="*-*-* 08:00:00" ./scripts/install_daily_lnn_timer.sh
```

只预演、不提交推送：

```bash
COMMIT_AND_PUSH=0 ./scripts/run_daily_lnn_task.sh
```

## 3. GitHub Actions 每日任务

已新增 `.github/workflows/daily-lnn-research.yml`：
- `schedule`：每天 `22:30 UTC` 运行，对应北京时间 `06:30`。
- `workflow_dispatch`：支持手动触发，并可选择是否下载 PDF。
- 默认 job 会自动提交并推送每日研究摘要。
- Jetson benchmark job 只在手动触发且选择 `run_jetson_benchmark=true` 时运行，需要自托管 runner 标签：`self-hosted`, `linux`, `ARM64`, `jetson`。

## 4. Jetson LNN 验证

核心脚本：`scripts/jetson_lnn_benchmark.py`

当前 smoke benchmark 使用合成非平稳时间序列，对比：
- `CfCStyle`：轻量闭式连续时间风格模型，用于近似验证 LNN/CfC 类动态门控。
- `GRU`：传统循环网络基线。

手动运行：

```bash
python3 scripts/jetson_lnn_benchmark.py --quick
```

使用本机已有 Jetson Orin CUDA 容器运行：

```bash
docker run --rm --runtime nvidia --gpus all \
  -v "$PWD":/workspace/LNN \
  -w /workspace/LNN \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  bash -lc 'python3 scripts/jetson_lnn_benchmark.py --samples 64 --seq-len 16 --hidden-size 8 --epochs 1 --batch-size 8 --inference-repeats 2'
```

本次已生成：
- [[analysis/jetson/2026-05-25_lnn_benchmark.md]]
- `analysis/jetson/2026-05-25_lnn_benchmark.json`

### 2026-05-25 结果快照

检测到的设备：
- 宿主机：`NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super`
- CUDA 容器设备名：`Orin`
- Jetson BSP：R36.4.7
- 宿主机 PyTorch：`2.11.0+cu130`，CUDA 不可用
- CUDA 容器 PyTorch：`2.10.0`，CUDA 12.6 可用

quick benchmark 结果：

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| CfCStyle | 329 | 0.691654 | 10048.1 | 0.81 |
| GRU | 273 | 0.671285 | 242200.0 | 0.10 |

结论：
- 本次 CUDA smoke test 已确认 Jetson Orin GPU 路径可用，`analysis/jetson/2026-05-25_lnn_benchmark.json` 中 `cuda_available=true`，峰值显存约 25.39 MB。
- 宿主机默认 Python 仍是 pyenv Python 3.14 + `torch 2.11.0+cu130`，与 JetPack CUDA 12.6 不匹配；每日任务已优先使用 Jetson 容器规避该问题。
- 当前系统内存碎片较高时，完整 quick 配置可能触发 cuBLAS 分配失败；正式 benchmark 前建议重启或释放长期运行容器，再提高样本数和隐藏维度。

参考：
- NVIDIA JetPack 6.2.1 Release Notes：https://docs.nvidia.com/jetson/jetpack/release-notes/index.html
- NVIDIA PyTorch for Jetson Platform：https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- NVIDIA PyTorch for Jetson 兼容矩阵：https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html

## 5. 后续实验队列

建议按以下优先级推进：
1. 修复 Jetson CUDA/PyTorch 版本匹配，让 benchmark 使用 GPU/NPU 可用路径。
2. 将 `ncps` 的 LTC/CfC 官方实现纳入 `projects/` 或实验依赖，替换当前 smoke benchmark 的近似模型。
3. 对 `docs/daily/` 中高相关论文生成独立研读报告，并追加到 [[LNN_深度研读报告]]。
4. 对 Hugging Face 中 LFM2.5-350M、LFM2.5-VL-450M 等边缘模型做量化推理验证。
