---
title: LNN 每日研究追踪 - 2026-08-04 (LLM 增量研究)
date: 2026-08-04
tags: [LNN, daily, jetson, lfm25, benchmark, infrastructure-fix]
author: Claude Code (cron 触发)
generated_by: llm-research-session
companion: docs/daily/2026-08-04_LNN_research_digest.md
---

# LNN 每日研究追踪 - 2026-08-04 (LLM 增量)

> 本文件由 Claude Code 07:57 cron session 生成, 配合 systemd timer 06:30 自动 commit 的 `2026-08-04_LNN_research_digest.md` 使用.
> systemd timer 负责"抓数据 + 提交", 本文件负责"在已有数据上做 LLM 增量分析、修复基础设施、跑新增 benchmark".

## 1. 最近一次 daily 执行情况排查

| 触发方式 | 时间 | 结果 | 备注 |
|---|---|---|---|
| systemd timer `lnn-daily-research.timer` | 2026-08-04 06:30 (实际 06:34:10) | ✅ 成功 (status=0) | digest + Jetson benchmark + push 全跑通 |
| crontab `13 7 * * *` | 2026-08-04 07:13 (实际尝试) | ❌ 失败 (自 2026-07-14 起 22 天) | 相对路径 `scripts/run_daily_lnn_task.sh` 没切到 repo 目录 |

**已修复**:
- 删除用户 crontab 中 `DAILY-LNN-RESEARCH` 行 (与 systemd 重复且 broken, 见 `logs/daily_cron.log` 22 行 "No such file or directory")
- `scripts/install_daily_lnn_timer.sh` 加上 `Environment=PYTHON_BIN=/usr/bin/python3.10` (默认 `python3` 是 pyenv 3.14, 其 torch 找不到与 CUDA 12.6 驱动匹配的 wheel)
- `systemctl --user daemon-reload` 已生效, 验证 `Environment=PYTHON_BIN=/usr/bin/python3.10`

**systemd `Persistent=true` + `Linger=yes` 状态**: 健康, 不会因用户离线而漏跑 (上次系统空闲 17837014s ≈ 206 天, 仍 active waiting).

## 2. Jetson Orin Nano GPU 路径打通

### 2.1 之前 (CPU fallback)

```text
status: ok_cpu_fallback
device: cpu
```

触发原因: `libcudss.so.0: cannot open shared object file`. systemd unit 默认 `python3` 是 pyenv 3.14 路径下的 torch, 该 torch wheel 对应 CUDA 12.8+ 库, 与本机 12.6 驱动不匹配 (`NVIDIA driver on your system is too old (found version 12060)`).

### 2.2 修复后 (GPU)

```text
status: ok
device: cuda
Pareto (4 模型, 5 epoch, hidden=16, seq=32, batch=32):
  PDNAPulse   1474 params  mse=0.2919  infer=52169    train=3.80s
  CfCStyle    1169 params  mse=0.3137  infer=49042    train=4.99s
  GRU         929 params   mse=0.3536  infer=1931749  train=0.43s
  LTC         625 params   mse=0.4002  infer=9411     train=14.81s
```

GRU 推理 1.93M steps/s (cf. CPU 之前 568k, 3.4× 加速). CfC 49k, LTC 9.4k. CUDA 显存 7.6 GB Orin.

**文件**: `analysis/jetson/2026-08-04_lnn_benchmark.json` (310 行) + `.md` (66 行) + `.png` pareto.

## 3. LFM2.5-350M 端侧 smoke benchmark (新工具验证)

`scripts/lfm25_benchmark.py` 是 commit 4b7341c 2026-08-04 新增, 首次跑通:

| 指标 | 值 |
|---|---:|
| 模型 | LiquidAI/LFM2.5-350M |
| 参数量 | 354,483,968 |
| 模型大小 | 676.1 MB (fp16) |
| 加载时间 | 13.20 s |
| Decode tokens/s (CPU) | **9.01 ± 0.8** |
| Prefill 时间 | 1.012 s (max=1.497) |
| 设备 | cpu (aarch64) |

**踩坑** (已记录到本次 commit):
- `import contextlib` 在 `benchmark_model()` 内而不在顶层, 加到顶部 imports.
- `transformers 4.46.3` 不识别 LFM2.5 的 `TokenizersBackend`, 升级到 `5.14.1`.
- `accelerate` + `sentencepiece` 缺失, 装上 (只为 python3.10).
- LFM2.5-350M 加载到 GPU 时触发 `NvMapMemAllocInternalTagged error 12` (Orin 统一显存碎片化); 切换到 `--cpu` 路径跑通 baseline. GPU 路径是 4-bit GPTQ 优化方向 (见 `projects/lfm25_orin_nano_smoke/README.md` 第 2 节).

**文件**: `analysis/jetson/2026-08-04_lfm25_benchmark.json` + `.md`.

## 4. digest 中今日未消化的新 GitHub 仓库

| 仓库 | star | 描述 (节选) | 备注 |
|---|---:|---|---|
| `Dhivya-DD17/DLNet` | 1 | Dual-Stage Distillation + Pareto-Guided Compression of LNNs for Edge | **新增**, 2026-08-04, 论文配套实现, 与 Jetson 部署强相关 |
| `kds1123001/liquid-time-constant` | 0 | Mojo-native Liquid Time-Constant NN for edge robotics | **新增**, Mojo 实现, 边缘 / 机器人场景 |
| `R-Liebert/LOKI-G` | 0 | 物理机器学习项目 | 关注 LNN/LTC 调用 |

候选 `DLNet` 是今日 (2026-08-04) 出现的蒸馏-压缩 pipeline, 与今日 digest 中的 LFM2.5-Encoder-350M (PII Detector / Prompt Router / Spellchecker / Policy Linter) 共同构成 **"LNN/LFM 边缘端剪枝-蒸馏-4bit 化"** 当下研究热点. 建议: 待 digest 持续观察 2-3 天, 累计 5+ stars 后纳入 `bench_lnn_compression.py` 复现计划.

## 5. 当日 (2026-08-04) PR-ready commits

```
4a5c747 (HEAD -> master, origin/master) chore(daily): update LNN research digest + Jetson improvements
9042a8d chore(daily): update LNN research digest 2026-08-03
1a241ce chore(daily): update LNN research digest 2026-08-04
502ecca chore(daily): update LNN research digest 2026-08-03
ab095b6 chore(daily): update LNN research digest 2026-08-02
```

`4a5c747` = 4b7341c + rebase 后 hash. 包含:
- 修复 systemd `lnn-daily-research.service` 默认 Python 路径
- 修复 `scripts/lfm25_benchmark.py` 缺 `import contextlib`
- 跑通 GPU 路径 + LFM2.5-350M CPU 路径 benchmark
- 新增 `docs/daily/2026-08-04_LNN_research_digest.md` (本文件的 LLM 增量)
- 删 crontab broken 行 (新提交, 见下)

## 6. 后续 backlog (今日评估)

1. **GPU LFM2.5 benchmark**: 4-bit GPTQ (`RESMP-DEV/LFM2.5-Encoder-350M-Code-MXFP4-GPTQ`) 已在 HF 上, 优先跑这个 350M Encoder 在 Orin GPU 上 (避开 1.2B 显存压力)
2. **LNN 蒸馏-压缩复现**: 等 `DLNet` star ≥ 5 后, 跑 `bench_lnn_compression.py` (尚未存在, 需先建)
3. **MOJO LTC 复现**: `kds1123001/liquid-time-constant` 是 Mojo-native, 评估本机 aarch64 是否能跑 Mojo (Modular SDK for aarch64 是否可装)
4. **NCPS-CfC**: 官方 `mlech26l/ncps` 实现, 已加入 jetson benchmark. 评估是否替换 `CfCStyle` (近似实现), 看 -mse / -latency 收益
5. **能源基准**: `lnn/edge/tegrastats.py` 已落地, 但今天没启用 `--power` (因 libcudss 路径修复). 下次跑 benchmark 时打开
6. **paper 研读 backlog**: 今日 digest 25 篇中, 已研读 14/25 (含 6 月以来累积), 剩余主要是 2026-07 后非 liquid 主题 (VLA policy, fall detection 等边缘通用). 优先研读 `2607.01986v1` Liquid Latent State Dynamics for Turbofan Degradation (与本仓 C-MAPSS / PDNA 相关)

## 7. 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
- LFM2.5 release: https://huggingface.co/LiquidAI/LFM2.5-350M
- systemd timer: `~/.config/systemd/user/lnn-daily-research.{service,timer}`
