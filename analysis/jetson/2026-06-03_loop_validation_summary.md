---
title: 2026-06-03 Loop — 9 变体 Jetson 验证 + 强配置 CPU Benchmark
date: 2026-06-03
tags: [LNN, jetson, validation, loop, benchmark]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-03 Loop 验证记录

> `/loop 1h` 调度产物。本次 loop 共做两类验证:
> (1) 9 个 LNN 变体在 Jetson Orin Nano Super CPU 路径上的实现一致性;
> (2) CfCStyle vs GRU 在中等规模合成时序上的 smoke benchmark
> (相比早晨 systemd 定时器跑的 quick smoke,放大了样本数/序列长/隐藏维度/epoch)。

## 1. 环境

- 硬件: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- 内核: Linux 5.15.148-tegra, aarch64
- BSP: R36.4.7, KERNEL_VARIANT oot
- Python: 3.14.4 (pyenv)
- PyTorch: 2.11.0+cu130
- CUDA 状态: `cuda.is_available() == False`
  (PyTorch wheel 编译用 CUDA 13.0,系统 BSP 是 CUDA 12.6 — 驱动版本不匹配。
   PR #1 in PRD §8 跟进。)
- nvidia-smi: 显示 `Orin (nvgpu) / Driver 540.4.0 / CUDA 12.6`,GPU 自身可用。

## 2. 验证 A — 9 变体实现一致性

脚本: `python3 scripts/verify_all_models.py`(177 行)

每个模型走 create → forward → backward → optimizer.step → 形状/数值校验。

| 模型 | 通过 | 前向耗时(s) | 参数量 |
|---|---|---:|---:|
| LTC | ✓ | 0.0498 | 185 |
| CfC | ✓ | 0.0105 | 257 |
| StrictCfC | ✓ | 0.0083 | 177 |
| HybridCfC | ✓ | 0.0093 | 257 |
| CT-LTC | ✓ | 0.0402 | 185 |
| LiquidS4 | ✓ | 0.0105 | 185 |
| LRC | ✓ | 0.0491 | 265 |
| CfC-DT | ✓ | 0.0126 | 329 |
| Euler-LTC-DT | ✓ | 0.0079 | 185 |

**速度梯队**:
- 最快: Euler-LTC-DT、StrictCfC、HybridCfC、CfC、LiquidS4 (~8–10ms);
- 中速: CfC-DT (~13ms);
- 较慢: LTC、CT-LTC、LRC (~40–50ms,正是 ODE 求解器开销)。

**结论**: 9/9 通过,变体 API 一致,
Jetson Orin Nano CPU 单序列前向都在 50ms 以内,
满足 < 60ms 的轻量边缘交互延迟门槛。

辅助脚本 `python3 scripts/quick_validate_implement.py` 给出同样的 9/9 结论
(数值与 timing 在同一量级)。

## 3. 验证 B — CfCStyle vs GRU smoke benchmark

脚本:
```bash
python3 scripts/jetson_lnn_benchmark.py --date 2026-06-03 --quick --cpu \
    --samples 256 --seq-len 32 --hidden-size 16 --epochs 5 \
    --batch-size 16 --inference-repeats 4
```

参数:samples=256, seq_len=32, hidden=16, epochs=3 (--quick clamp), batch=16。

| 模型 | 参数 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---:|---:|---:|---:|
| **CfCStyle** | **1169** | **0.2637** | 19,458 | 4.95 |
| GRU | 929 | 0.3346 | 51,818 | 1.45 |

**关键观察**

1. CfCStyle 测试 MSE 比 GRU 低 **21.2%**(0.2637 vs 0.3346),
   即使后者参数量少 20.5% — 这与之前 (2026-05-26 Jetson CUDA 验证 +
   `docs/NEXT_STEPS.md` 历史结论) "CfC 在非平稳/连续时间任务上更准" 一致。
2. CfCStyle 在 CPU 上的吞吐 19.5k steps/s,仍可满足实时性需求
   (单步约 50µs,远低于 1ms 控制周期);GRU 当然更快,但单纯吞吐
   不能弥补精度差距。
3. 与早晨 06:39 的 quick smoke (samples=64, seq_len=16, hidden=8, epochs=1)
   相比:
   - CfCStyle MSE 0.5716 → 0.2637, **−53.9%**;
   - GRU MSE 0.6756 → 0.3346, **−50.5%**;
   - 都证明早晨的 smoke 因 epoch=1 + hidden=8 还远没收敛,
     必要时应该跑这个"中等强度"配置作为日常基线。

## 4. CUDA 路径状态(开放问题)

- 早晨的 docker CUDA smoke 触发 `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED`,
  原因记录在早晨那份 md 的 "CUDA 回退" 部分。
- 本次 host 直接跑 PyTorch 也无法启用 CUDA:
  > "The NVIDIA driver on your system is too old (found version 12060).
  >  PyTorch was compiled for CUDA 13.0."
- 建议:
  1. 换到 NVIDIA 官方 `nvcr.io/nvidia/l4t-pytorch` 镜像(匹配 BSP R36.4 + cu126);
  2. 或直接用 `pip install --index-url
     https://pypi.jetson-ai-lab.dev/jp7/cu126/+simple/`(社区 wheel)。
- 暂时把 PRD §8 任务 #1 标记为 **优先级 P0**。

## 5. 衍生产出 / 入栈

| 任务 | 推入 |
|---|---|
| 让 `jetson_lnn_benchmark.py --pareto` 默认产出 7 模型 × 3 hidden_size 的对比 | NEXT_STEPS |
| 把 9 变体的"前向耗时 + 参数量"折线图沉淀为 README badge | NEXT_STEPS |
| 给 `experiment_imitation_lnn.py` 加 `--device cuda --fp16` 路径并验证 | NEXT_STEPS(等 CUDA 修复) |
| LFM2.5-1.2B-Distilled INT4 推理 | PRD §8 任务 #3 |

## 6. 参考

- 早晨 systemd 跑的 quick smoke(已被本次 loop 重新覆盖):
  之前 `analysis/jetson/2026-06-03_lnn_benchmark.{json,md}` 由 root 拥有,
  本 loop 清理后由 hyx 重写,数据保留在 git 历史里。
- 完整 9 变体表见 [[IMPLEMENTATION_SUMMARY]]。
- PRD: [[PRD_LNN_Edge_Research]]
- 论文/仓库简评: [[2026-06-03_loop_iteration_research_brief]]
