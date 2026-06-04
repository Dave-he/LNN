---
title: 2026-06-04 Loop iteration 23 — DynPMNN §10 #1 stage A: FHNCell + DynPMNNNetwork
date: 2026-06-04
tags: [LNN, loop, PRD-10-1, DynPMNN, FitzHugh-Nagumo, ODE, Euler, paper-replication, stage-A]
parent: [[PRD_LNN_Edge_Research]]
---

# 2026-06-04 Loop iteration 23 — DynPMNN §10 #1 stage A

> `/loop 1h` 第 23 次触发。
> 紧接 iter#22 (SVAF τ-blend) 后,执行 iter#22 报告 next-up 中**最实质的
> 新增 backbone 任务**: §10 #1 (DynPMNN stage A) —— 实现 FHN ODE + Euler 积分。
>
> 1. **`lnn/core/dynpmnn.py`** (175 行): `FHNCell` (单层 FHN) + `DynPMNNNetwork` (堆叠)
> 2. **`tests/test_dynpmnn.py`** (165 行, 9 tests): shape / 零 init / 梯度流 / 稳定 / FHN 兴奋性 / multi-layer chain / 端到端 train
> 3. **FHN ODE** dV/dt = V - V³/3 - W + I, dW/dt = ε(V + a - bW) — 论文 §2.2 直译
> 4. **Euler 积分** n_euler_steps=5 步嵌入 autograd — 论文 §2.3 直译
> 5. **PRD §10 #1 stage A** ✅
> 6. **commit + rebase + push origin/master**

## 1. 实现细节

### 1.1 FHNCell(单层 FHN ODE)

```python
def forward(self, x, state=None):
    V, W = self.initial_state(B) if state is None else state
    I = self.input_proj(x)  # [B, hidden]
    V_seq = [V]
    for _ in range(self.n_euler_steps):
        dV = V - V**3/3.0 - W + I
        dW = self.epsilon * (V + self.a - self.b * W)
        V = V + self.dt * dV
        W = W + self.dt * dW
        V_seq.append(V)
    return V, W, torch.stack(V_seq, dim=1)
```

### 1.2 DynPMNNNetwork(堆叠多层)

- 每层有**自己独立**的 (a, b, epsilon, W_in) — 与 CfC 各 cell 独立参数同模式
- 序列处理: for t in range(T): V, W = cell(layer_input[:, t, :], (V, W))
- output_proj 在最后,V → output_size

### 1.3 关键设计点

- **dt = 1.0 / n_euler_steps** — 总积分时间 ~1.0,与论文 T 概念一致
- **FHN init values** 选 excitable regime (a=0.7, b=0.8, ε=0.08)
- **stateful 设计** — V/W 显式传,允许灵活 sequence chunking
- **多层堆叠** — num_layers > 1 时每层是独立 ODE,**深但窄**

## 2. pytest 套件(84/84, 11.18s)

```
tests/test_core.py                  : 46 passed
tests/test_liquid_tad_hierarchical.py: 6 passed
tests/test_pdna_pulse.py            : 12 passed
tests/test_loop_status_prd.py       :  8 passed
tests/test_svaf_tau_blend.py        :  9 passed
tests/test_dynpmnn.py               :  9 passed (iter#23 新增)
─────────────────────────────────────────────
84 passed, 1 warning in 11.18s
```

vs iter#22: 75 → 84 = **+9 新增,0 回归**。

## 3. verify_all_models.py(9/9)

无变化。

## 4. 提交与推送

iter#23 改动:
- 增 `lnn/core/dynpmnn.py` (175 行: FHNCell + DynPMNNNetwork)
- 增 `tests/test_dynpmnn.py` (165 行, 9 tests)
- 改 `docs/PRD_LNN_Edge_Research.md` (1 行: #10-1 stage A ✅)
- 增 `analysis/jetson/2026-06-04_loop_iteration12_dynpmnn_stage_a.md` (验证摘要)
- 增 `analysis/loop_status/2026-06-04_loop_iteration23_dynpmnn_stage_a.md` (本报告)

**PRD §6 verify 协议执行**: ✅ (验证 0 回归)
**PRD §9 状态**: 5/8 = 62.5%(无变化)
**PRD §10 #1 状态**: **stage A ✅**,stage B (matrix ingest) pending
**PRD §10 总体**: **5/10 = 50%**(#10-1 stage A + #10-5 + #10-9 stage A + #10-10 stage A+B)

## 5. 下轮 (iter#24) 候选

按 §10 next-up + iter#23 报告:

1. **§10 #1 stage B (DynPMNN 加到 backbone matrix)**: 加 `--backbone fhn_dynpmnn`
   到 ablation runner,跑 multi-seed 对比,**很可能给出 LNN 在 mackey_glass / gradual_multi_regime
   上的第 4 个 win** (iter#16 报告预期)
2. **§10 #6 (backbone matrix --export-readme-snippet)**: 无阻塞,~30 行
3. **§10 #8 (loop_status README 标签云)**: 无阻塞,衍生 tooling
4. **§10 #9 stage B (SVAF τ-blend 接 CfC backbone)**: toy 升级到 real sequence
5. **§10 #3 (Comparative phase-D)**: 需空载 RAM 窗口
