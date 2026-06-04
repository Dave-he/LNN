---
title: 2026-06-04 Loop iteration 19 — PDNA stage A: PDNAPulseHead + 12 unit tests
date: 2026-06-04
tags: [LNN, loop, PRD-10-10, PDNA, CfC, pulse-module, code, stage-A, unit-tests]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 19 — PDNA stage A: PDNAPulseHead + 12 unit tests

> `/loop 1h` 第 19 次触发。
> 紧接 iter#18 (PDNA 研读) 后,本轮是**首次落地代码改动**(前 4 轮 iter#14-18
> 都是 paper 研读 + PRD 维护)。本轮执行 PRD §10 #10 **stage A**。
>
> 1. **`lnn/core/cfc.py` 加 `PDNAPulseHead` 类**(~80 行含 docstring):
>    完整实现 arXiv 2603.00153v1 §3.2-3.3 的两个 gated additive residual
> 2. **`tests/test_pdna_pulse.py` 加 12 个 unit test**:
>    shape / α/β init / ω log-uniform diversity / 残差 magnitude / 梯度流 / 端到端 CfC 集成
> 3. **pytest 58/58 (46 旧 + 12 新) + verify_all_models 9/9** — 0 回归
> 4. **commit + rebase + push origin/master**

## 1. 实现细节

### 1.1 PDNAPulseHead 接口

```python
from lnn.core.cfc import CfCNetwork, PDNAPulseHead

backbone = CfCNetwork(input_size, hidden_size, output_size, num_layers=1)
head = PDNAPulseHead(hidden_size, use_self_attend=True)  # α=β=0.01, ω ∈ [0.1, 10.0]
h_seq = backbone.cells[0](...)  # 拿到 [B, T, d] hidden state sequence
h_aug = head(h_seq)            # residual pulse + self-attend
y = backbone.output_proj(h_aug)
```

### 1.2 关键参数(对齐论文)

| 参数 | 默认值 | 论文引用 |
|---|---|---|
| `use_self_attend` | `True` | Variant E (full PDNA) |
| `omega_low/high` | 0.1 / 10.0 | §3.2 "log-uniform spacing" |
| `alpha_init` | 0.01 | §3.2 "initialized to 0.01" |
| `beta_init` | 0.01 | §3.3 "initialized to 0.01" |

### 1.3 实现要点(论文 §3.2-3.3 公式直译)

```python
# Pulse:  pulse(t, h) = A · sin(ω · t + φ(h))       Eq. 3
phase = self.phase_proj(h)                              # φ(h) = W_φ h + b_φ
angular = self.omega.view(1, 1, d) * t_b.unsqueeze(-1)  # ω · t
pulse = self.amplitude.view(1, 1, d) * torch.sin(angular + phase)
h = h + self.alpha * pulse                              # α-scaled residual Eq. 4

# Self-attend:  h + β · Wself · σ(h)                   Eq. 5-6
attended = self.self_attend_proj(torch.sigmoid(h))
h = h + self.beta * attended
```

论文 §8 (iii) 承认这是 **post-hoc augmentation** of full hidden-state tensor,
不是真 continuous-time dynamic evolving between input steps — 实现保留了
这个限制,文档明确说明。

## 2. 12 unit test 覆盖矩阵

| 维度 | 论文论据 | 测试 |
|---|---|---|
| Shape 不变性 | 论文 Eq. 4, 6 (additive residual) | #1, #2 |
| Gate init | §3.2-3.3 显式 "initialized to 0.01" | #3, #4, #5 |
| ω 多样性 | §3.2 "log-uniform spacing from 0.1 to 10.0" | #6, #7 |
| A 形状 | §3.2 "one per hidden dimension" | #9 |
| 残差 magnitude | 论文 §7 +5% wall-time 暗示 init 极小 | #8 |
| 梯度流 | 论文 §3.4 "trained end-to-end with standard backprop" | #10, #11 |
| 端到端集成 | 论文 §3.4 "CfC backbone → pulse → self-attend" | #12 |

## 3. pytest 套件(58/58,15.24s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed (新)
─────────────────────────────────────────────
58 passed, 1 warning in 15.24s
```

vs iter#18: 46 → 58 = **+12 新增,0 回归**。

## 4. verify_all_models.py(9/9)

无变化。`PDNAPulseHead` 改动**不触碰**任何 verify 路径。

## 5. 与本周回退基线对比

| 指标 | iter#15 | iter#16 | iter#17 | iter#18 | iter#19 (本次) |
|---|---:|---:|---:|---:|---:|
| verify_all_models | 9/9 | 9/9 | 9/9 | 9/9 | **9/9** |
| pytest 套件 | 46/46 | 46/46 | 46/46 | 46/46 | **58/58** (+12) |

## 6. 提交与推送

iter#19 改动:
- 改 `lnn/core/cfc.py` (+80 行 `PDNAPulseHead` class)
- 增 `tests/test_pdna_pulse.py` (196 行, 12 tests)
- 增 `analysis/jetson/2026-06-04_loop_iteration8_pdna_stage_a.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-04_loop_iteration19_pdna_stage_a.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归,见 §3-4)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 #10 状态**: **stage A ✅**,stage B/C pending
**PRD §10 总体**: 1/10 = 10%(首次有 §10 完成项)

## 7. 下轮 (iter#20) 候选

按 PRD §10 P 排序:

1. **§10 #10 PDNA stage B** (sMNIST Gapped 5 seed × 4 backbone): **本轮 stage A 已就绪**
   — 写 `scripts/experiment_pdna_smoke.py`,跑 ablation,把结果入 backbone matrix
2. **§10 #5 (loop_status --prd-status)**: 无阻塞,可与 #10 并行
3. **§10 #8 (loop_status README 标签云)**: 无阻塞,衍生 tooling
4. **§10 #9 (SVAF τ 调制算子)**: P2 mini-task,可与 #10 stage B 并行
5. §10 #6 (backbone matrix --export-readme-snippet): 无阻塞
6. §10 #1/#2 DynPMNN 复现:与 #10 PDNA 互不冲突
