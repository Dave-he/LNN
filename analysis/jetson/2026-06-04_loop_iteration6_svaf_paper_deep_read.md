---
title: Jetson validation summary — iter#17 SVAF paper deep read + PRD §10 #9
date: 2026-06-04
platform: Linux hyx-desktop 5.15.148-tegra (Jetson Orin Nano Super, aarch64)
python: 3.14.4 (pyenv) — primary, CPU path
cuda: torch 2.11.0+cu130 → cuda.is_available()=False (BSP driver 12.060 < cu130)
tag: loop-iteration, jetson-cpu-only, svaf-deep-read, prd-10
---

# Jetson validation summary — iter#17

> 与 iter#17 论文研读 + PRD §10 #9 落地同批跑的环境验证。所有 CPU 路径必须 100%
> 通过,CUDA 路径按 iter#2 已知约束继续 disabled。

## 1. 环境快照

```
Platform : Jetson Orin Nano Super (Linux 5.15.148-tegra, aarch64)
Python   : 3.14.4 (pyenv primary)
Torch    : 2.11.0+cu130
CUDA     : disabled (BSP driver 12060 < cu130 最低要求)
Libcudss : n/a (CPU 路径不依赖)
```

iter#2 已修通 CUDA 路径(python3.10 + torch 2.10.0+cu126 + libcudss 0.8.0.10),
但 PyTorch 2.11.0+cu130 在 Jetson BSP 下 driver 不够新 → cudaMalloc 不可用。
本轮验证全部走 CPU 路径(等同于 iter#3 之后的回归基线)。

## 2. verify_all_models.py (9 变体 smoke)

```
LTC                 : ✓ 通过
CfC                 : ✓ 通过
StrictCfC           : ✓ 通过
HybridCfC           : ✓ 通过
CTLTC               : ✓ 通过
LiquidS4            : ✓ 通过
LRC                 : ✓ 通过
CfC-DT              : ✓ 通过
Euler-LTC-DT        : ✓ 通过
────────────────────────────────
✓ 所有模型测试通过!
```

每变体均验证: 模型创建 / 前向传播 / 输出形状 / 输出值合理性 / 反向传播 / 参数数量。
单模型前向时间 7–55 ms(CPU 路径),与 iter#9/10/11/12/13/14/15/16 报告区间一致
(无回归)。

## 3. quick_validate_implement.py (实现层 9 检查)

```
LTC                  ✓ 通过    前向 49.3 ms
CfC                  ✓ 通过    前向 11.3 ms
Strict CfC           ✓ 通过    前向  8.7 ms
Hybrid CfC           ✓ 通过    前向 10.4 ms
CT-LTC               ✓ 通过    前向 43.0 ms
LRC                  ✓ 通过    前向 51.8 ms
CfC-DT               ✓ 通过    前向 11.9 ms
Euler-LTC-DT         ✓ 通过    前向  7.5 ms
Liquid-S4            ✓ 通过    前向 10.4 ms
────────────────────────────────────
✓ 所有模型实现正确!
```

## 4. pytest 套件 smoke(46 tests)

```
tests/test_core.py + tests/test_liquid_tad_hierarchical.py
46 passed, 1 warning in 11.40s
```

- 46/46 PASS
- 1 warning 仅 CUDA 驱动版本警告(CPU 路径预期,非测试失败)
- 与 iter#14/15/16 一致(本仓新加的 6 个 LiquidTAD hierarchical 测试全过)

## 5. 与本周回退基线的对比

| 指标 | iter#14 | iter#15 | iter#16 | iter#17 (本次) |
|---|---:|---:|---:|---:|
| verify_all_models 9 变体 | 9/9 ✅ | 9/9 ✅ | 9/9 ✅ | **9/9 ✅** |
| quick_validate 9 检查 | 9/9 ✅ | 9/9 ✅ | 9/9 ✅ | **9/9 ✅** |
| pytest 套件 | 46/46 | 46/46 | 46/46 | **46/46** |

**0 回归**。本轮论文研读 + PRD 落地不修改任何 `lnn/` 代码,所以完全无回归属预期。

## 6. 任务意义

iter#17 的核心动作是**研读 SVAF + 把 τ 调制耦合算子挂进 PRD §10 #9** ——
本轮验证仅作环境健康检查(隔日必跑,确保 PRD §6 协议在每轮 loop 都被执行)。

真正的验证目标(τ 调制混合算子的最小复现)留到 iter#18:
- 2-agent toy mesh
- τ_i ∈ {1, 10, 60} 三组神经元
- N 步耦合后看 spectral diff(预期: fast τ 高度耦合 + slow τ 主权)
- 50 行核心代码 + 单元测试

## 7. 已知阻塞(无变化)

| 阻塞 | 来源 | 影响 |
|---|---|---|
| CUDA 不可用 | Jetson BSP driver 12060 < torch 2.11 cu130 | iter#2 修通的 py3.10+torch2.10 路径需要空载窗口,且 libcudss 0.8.0.10 需手动装 |
| RAM 1.7 GB available | 多 agents 并行 + 8GB 统一显存 | LFM2.5-1.2B INT8 推理 / 较大 hidden LNN sweep 受限 |
| THUMOS-14 数据未下载 | LiquidTAD stage C 真实数据 | 暂用 toy 长视频 |

本轮无新增阻塞。
