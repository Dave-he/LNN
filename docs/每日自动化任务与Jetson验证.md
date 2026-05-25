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
2. 如果当前机器是 Jetson，则自动运行 quick benchmark。
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

本次已生成：
- [[analysis/jetson/2026-05-25_lnn_benchmark.md]]
- `analysis/jetson/2026-05-25_lnn_benchmark.json`

### 2026-05-25 结果快照

检测到的设备：
- `NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super`
- Jetson BSP：R36.4.7
- PyTorch：`2.11.0+cu130`
- CUDA 可用状态：`False`

quick benchmark 结果：

| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312890 | 39303.1 | 3.41 |
| GRU | 1969 | 0.435523 | 136813.0 | 1.32 |

结论：
- 在本次合成非平稳序列 smoke test 中，`CfCStyle` 的误差低于同隐藏维度 `GRU`，但 CPU 推理吞吐低于 GRU。
- 当前 PyTorch/CUDA 组合未启用 CUDA；终端曾提示系统 NVIDIA driver 与 `torch 2.11.0+cu130` 不匹配。若要利用 60 TOPS 级别算力，应优先安装与 JetPack / L4T / CUDA 版本匹配的 PyTorch 轮子，或升级 JetPack 后再运行 full benchmark。

## 5. 后续实验队列

建议按以下优先级推进：
1. 修复 Jetson CUDA/PyTorch 版本匹配，让 benchmark 使用 GPU/NPU 可用路径。
2. 将 `ncps` 的 LTC/CfC 官方实现纳入 `projects/` 或实验依赖，替换当前 smoke benchmark 的近似模型。
3. 对 `docs/daily/` 中高相关论文生成独立研读报告，并追加到 [[LNN_深度研读报告]]。
4. 对 Hugging Face 中 LFM2.5-350M、LFM2.5-VL-450M 等边缘模型做量化推理验证。
