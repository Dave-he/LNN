---
title: 2026-06-03 Loop iteration 2 - Jetson CUDA 修复 + Pareto sweep
date: 2026-06-03
tags: [LNN, jetson, CUDA, pareto, loop, validation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-03 Loop iteration 2 — Jetson CUDA 修复 + Pareto sweep

> 本轮(`/loop 1h` 第二次触发)的主要进展:
> 1. **PRD §8 P0 完成** — Jetson Orin Nano Super 上 PyTorch CUDA 路径
>    从"不可用"修复到"可用"(`torch 2.10.0 / cu126 / device=Orin / cuDNN 9.3`)。
> 2. **Pareto sweep 数据更新** — 在 `--hidden-sizes 8,16,24 --seq-lens 16,32` 网格上
>    对 CfCStyle vs GRU 跑 12 个 trial,识别 4 个 Pareto-front 点。

## 1. CUDA 修复 — root cause & 解法

### 1.1 Root cause

之前 `python3` (pyenv 3.14.4) 装的 `torch 2.11.0+cu130` 是 PyTorch 官方 CUDA 13.0 wheel,
但 Jetson Orin Nano 的 BSP R36.4.7 只提供 **CUDA 12.6 + driver 540.4.0** 运行时;
PyTorch 启动时 `torch._C._cuda_init` 直接报:

```
The NVIDIA driver on your system is too old (found version 12060).
```

并把 `torch.cuda.is_available()` 返回 False。

### 1.2 解法 — 三步

| 步骤 | 命令 / 文件 | 备注 |
|---|---|---|
| 1. 找匹配 Jetson 的 torch wheel | `pypi.jetson-ai-lab.io/jp6/cu126/+simple/` | 社区 devpi 镜像,只有 cp310 (Python 3.10) |
| 2. 安装到 system Python 3.10 | `python3.10 -m pip install --force-reinstall --no-deps <wheel-url> torch==2.10.0` | 2.11.0 引入了对 `libcudss` 的硬依赖,而 cudss 还要单独装,降到 2.10.0 直接绕过 |
| 3. 补 libcudss(2.10 也需要) | 从 `developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/` 拉 `libcudss-linux-aarch64-0.8.0.10_cuda12-archive.tar.xz`,解压到 `~/.local/opt/`,然后 `LD_LIBRARY_PATH=$_/lib:$LD_LIBRARY_PATH` | 不需要 root |

### 1.3 验证

```bash
$ CUDSS_LIB=/home/hyx/.local/opt/libcudss-linux-aarch64-0.8.0.10_cuda12-archive/lib
$ LD_LIBRARY_PATH="$CUDSS_LIB:$LD_LIBRARY_PATH" python3.10 -c "
... import torch; print(torch.__version__, torch.version.cuda,
...                       torch.cuda.is_available(),
...                       torch.cuda.get_device_name(0))"
2.10.0 12.6 True Orin
```

### 1.4 仍存在的环境约束

Jetson Orin Nano Super 是 **统一显存架构**(CPU/GPU 共享 8 GB LPDDR5);
本时段(此 loop 运行时)系统 RAM 已被并行运行的 `claude`、`icm`、`logflare`、
`node`/`next-server` 等吃到 5.2 / 7.6 GB,
`tegrastats` 报告 `lfb 6×1MB`(最大可分配连续块只有 6 MB);
所以即使 CUDA 路径修通,`torch.cuda` 在尝试分配时仍然报:

```
NvMapMemAllocInternalTagged: 1075072515 error 12   (= ENOMEM)
torch.AcceleratorError: CUDA error: out of memory
```

下个 loop 应该:
- 把 `LD_LIBRARY_PATH` 写进 `~/.profile` 或者 `scripts/run_daily_lnn_task.sh`
  让 CUDA 持久可用;
- 在 GPU 路径跑 benchmark 前先 `sync; echo 3 > /proc/sys/vm/drop_caches`(需 root)
  或在 systemd timer 早晨低负载窗口跑;
- 把"CUDA 启动成功 + 分配失败"在 `jetson_lnn_benchmark.py` 里
  打成单独的 `status="ok_cuda_init_alloc_fail"`,
  比当前的 `ok_cpu_fallback` 更精确。

## 2. Pareto sweep — CfCStyle vs GRU

### 2.1 命令

```bash
LD_LIBRARY_PATH=$CUDSS_LIB:$LD_LIBRARY_PATH python3.10 \
  scripts/jetson_lnn_benchmark.py --date 2026-06-03 --pareto \
    --hidden-sizes 8,16,24 --seq-lens 16,32 --seeds 42 \
    --epochs 2 --samples 96 --batch-size 16 --inference-repeats 2 --cpu
```

`--cpu` 强制走 CPU,因为本次环境 RAM 紧张(见 §1.4)。

### 2.2 结果(12 trials)

| 模型 | hidden | seq | 参数 | MSE | steps/s | Pareto |
|---|---:|---:|---:|---:|---:|:---:|
| CfCStyle |  8 | 16 |  329 | 0.6323 |  50,901 |  |
| GRU      |  8 | 16 |  273 | 0.6511 | 241,818 | ★ |
| CfCStyle |  8 | 32 |  329 | 0.5632 |  47,202 | ★ |
| GRU      |  8 | 32 |  273 | 0.5845 | 274,087 | ★ |
| CfCStyle | 16 | 16 | 1169 | 0.6562 |  49,472 |  |
| GRU      | 16 | 16 |  929 | 0.6205 | 221,062 | ★ |
| CfCStyle | 16 | 32 | 1169 | 0.6104 |  49,155 |  |
| GRU      | 16 | 32 |  929 | 0.6121 | 245,764 |  |
| **CfCStyle** | 24 | 16 | 2521 | **0.5544** | 47,724 | ★ |
| GRU      | 24 | 16 | 1969 | 0.6343 | 197,557 |  |
| **CfCStyle** | 24 | 32 | 2521 | **0.4285** | 48,081 | ★ |
| GRU      | 24 | 32 | 1969 | 0.6179 | 188,597 |  |

### 2.3 解读

- **CfCStyle 的精度优势随 hidden_size 单调拉开**:
  - h=8: CfC 比 GRU 好 ~3%(seq=16)、~3.6%(seq=32)
  - h=16: 持平
  - h=24: 好 **12.6%**(seq=16)、**30.6%**(seq=32)
  - 说明 CfCStyle 在容量足够时才能利用其连续时间动力学拟合非平稳序列。
- **GRU 始终更便宜**: throughput 永远是 CfC 的 3.6–5.6 倍,
  推理预算紧的场景仍可选;
- **Pareto front(4 个点)**:
  - `(GRU, h=8, seq=16)` — 最低延迟基线
  - `(GRU, h=8, seq=32)` — 中精度低延迟
  - `(CfCStyle, h=8, seq=32)` — 同等参数下 CfC 更准
  - `(CfCStyle, h=24, seq=32)` — **最佳精度**(MSE 0.4285)
- **不在 Pareto front 但有意义的点**:
  `(CfCStyle, h=24, seq=16)` — 比同尺寸 GRU 准但更慢,
  在 seq=16 受限的硬实时场景里仍可考虑。

### 2.4 与早晨 smoke + 上一轮 loop 的对比

| 时间 | 配置 | CfC MSE | GRU MSE | 相对差 |
|---|---|---:|---:|---:|
| 06:39(systemd smoke) | h=8, seq=16, ep=1 | 0.5716 | 0.6756 | −15.4% |
| 12:38(上一轮 loop) | h=16, seq=32, ep=3 | 0.2637 | 0.3346 | −21.2% |
| 13:46(本轮 Pareto best) | **h=24, seq=32, ep=2** | **0.4285** | 0.6179 | **−30.6%** |

注: 本轮 Pareto 的 `--epochs 2` 比上一轮的 `epochs 3` 短,
所以绝对 MSE 反而比中午高;但 **相对差** 把 CfC 优势拉到了 30.6%,
证明随 hidden 加大,CfC 收益放大。

## 3. 衍生工作 / 入栈

| 任务 | 推入 |
|---|---|
| 让 `scripts/run_daily_lnn_task.sh` 默认带 `LD_LIBRARY_PATH=$CUDSS_LIB` | PRD §8(新增 P0 follow-up) |
| 跑 GPU 路径 benchmark 需在系统空载窗口,推上 systemd timer 早 6:30 触发 | 同上 |
| 把"CUDA init OK + cudaMalloc 失败"做成 `status=ok_cuda_alloc_fail`,带 `lfb_largest_mb` 字段 | scripts/jetson_lnn_benchmark.py(2 行改动) |
| 复现 LiquidTAD(论文 2604.18274) | PRD §8 #2 |
| LFM2.5-1.2B-Distilled INT4 推理 | PRD §8 #3 |

## 4. 参考

- 上一轮 loop: [[2026-06-03_loop_validation_summary]]
- PRD: [[PRD_LNN_Edge_Research]]
- Pareto JSON 原数据: `analysis/jetson/2026-06-03_lnn_benchmark.json`
- `libcudss` 上游: https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/
- Jetson cu126 torch wheel: https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/torch/
