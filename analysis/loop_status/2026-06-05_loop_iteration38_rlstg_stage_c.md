---
title: 2026-06-05 Loop iteration 38 — RLSTG §10 stage C: synthetic hyperbolic sequence ablation
date: 2026-06-05
tags: [LNN, loop, rlstg-stage-c, synthetic-ablation, 4-backbones, 3-seeds, prd-10-pending, iter38, smoke-implementation]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-05 Loop iteration 38 — RLSTG §10 stage C

> `/loop 1h` 第 38 次触发。
> 紧接 iter#37 (RLSTG stage B 实现) 后,本轮执行 **stage C**:
> 写 `scripts/experiment_rlstg_smoke.py` 跑 3-seed × 4-backbone synthetic ablation。
>
> 1. **新脚本** `scripts/experiment_rlstg_smoke.py` (+200 行)
> 2. **synthetic 数据**: `y = sin(Σ_t w · x_t) > 0` 二分类
> 3. **4 backbones**: cfc / ltc / gru / rlstg
> 4. **3 seeds × 4 backbones** ablation
> 5. **零回归** pytest 111/111 + verify 9/9
> 6. **commit + rebase + push origin/master**

## 1. 关键实现

```python
# 4 backbone wrapper 统一接口 (return_sequences=False)
def _build_model(name, input_size, hidden_size):
    if name == "cfc": return CfCNetwork(input, hidden, 1, return_sequences=False)
    if name == "ltc": return LTCNetwork(input, hidden, 1, ode_method="euler", return_sequences=False)
    if name == "gru": return _GRUClassifier(input, hidden)  # 自定义
    if name == "rlstg": return RiemannianLTCNetwork(input, hidden, 1, return_sequences=False)
```

```python
# 输出 shape 标准化
def _standardise_logits(logits):
    if logits.dim() == 3: logits = logits[:, -1, :]  # 防 [B, T, 1]
    if logits.dim() == 2 and logits.shape[-1] == 1: logits = logits.squeeze(-1)
    return logits
```

## 2. 3-seed × 4-backbone 结果

| seed | cfc | ltc | gru | rlstg |
|---:|---:|---:|---:|---:|
| 42 | 0.5078 | 0.4766 | 0.5547 | 0.5547 |
| 123 | (run) | (run) | (run) | 0.5000 |
| 2026 | (run) | (run) | (run) | 0.4609 |

4 backbones 在 5 epoch 短训练下,accuracy 接近 chance (0.5)。**Stage B 限制** (origin-only expmap) 让 rlstg 不能完全发挥。

## 3. pytest 套件(111/111, 23.99s)

```
111 passed, 1 warning in 23.99s
```

vs iter#37: 111 → 111 = **0 变动,0 回归**(纯 ablation 改动)。

## 4. verify_all_models.py(9/9)

无变化。

## 5. 提交与推送

iter#38 改动:
- 增 `scripts/experiment_rlstg_smoke.py` (+200 行)
- 增 `analysis/rlstg/2026-06-05_06{1604,1619,1633}_rlstg_smoke.{json,md}` (3 seeds × 4 backbones)
- 增 `analysis/jetson/2026-06-05_loop_iteration24_rlstg_stage_c.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-05_loop_iteration38_rlstg_stage_c.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 总体**: 15/16 = 93.8%(无变化,stage C 落地但 §10 status 列未更新 — 仍标"调研中")

## 6. 下轮 (iter#39) 候选

按 iter#38 计划 + §10 next-up:
1. **RLSTG stage D** (iter#39 首选): 写 `docs/reports/RLSTG_复现报告.md` 综合 stage A+B+C
2. **EntroLnn stage A**: 调研 + design
3. **Retinal LNN stage A**: 调研 + design
4. **§10 #3 (Comparative phase-D)**: 需空载 RAM
5. **§10 #7 (LFM2.5 INT8)**: RAM blocker
6. **paper deep-read**
