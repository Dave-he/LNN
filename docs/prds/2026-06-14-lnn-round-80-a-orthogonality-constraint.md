---
title: "PRD #10-37 — Orthogonality Constraint on Expert Representations"
id: prd-10-37
date: 2026-06-14
status: proposed
priority: P0
estimated: 3-4h
loop: round 80 (2026-06-14 下午, loop session #5)
related_papers:
  - arXiv:2606.03631v2 (AnchorMoE, 2026-06-02)
  - arXiv:2605.15403v1 (φ-Balancing, 2026-05-14)
  - arXiv:2605.18498v1 (DBES, 2026-05-18)
  - arXiv:2606.10703v1 (Causal Audit, 2026-06-09) — 反向证据
related_prior: round 76 #10-29 (n_tau), round 77 #10-24 (MR-MoE), round 78 #10-36 (FAME), round 79 #10-38 (sweep)
---

# PRD #10-37 — Orthogonality Constraint on Expert Representations

## 1. 背景与动机

### 1.1 round 79 sweep 暴露的硬阻塞

| Cell | K | top_k | mean loss | std | 备注 |
|---:|---:|---:|---:|---:|---|
| K=3, n_tau=1 | 3 | **1** | **0.7595** | **0.7906** | 部分 seed 发散到 1.86 |
| K=3, n_tau=1 | 3 | 2 | 0.0646 | 0.0130 | 稳定 |
| K=3, n_tau=1 | 3 | 3 | 0.0579 | 0.0105 | 稳定 |
| K=5, n_tau=1 | 5 | 1 | 0.2395 | 0.0993 | 同样发散风险 |

**结论**: top_k=1 (router argmax, 单 expert) 极不稳定, std 是 top_k=2/3 的 **60-100×**。这是 sweep 暴露的**硬阻塞**,必须解决才能让 FAME 稀疏 top-K 路径在 toy 上也稳定。

### 1.2 Causal Audit 反向证据 (arXiv:2606.10703)

- 观测指标 (utilization rate, activation norm, routing weight distribution) **不能预测 expert causal importance** (Cohen's d < 0.17)
- 即使 round 79 sweep 的 `activated_per_step` 看起来均匀, 也**不保证 expert 真的学到了异质表示**

### 1.3 AnchorMoE 模板 (arXiv:2606.03631)

- **"Geometric orthogonality constraint that penalizes representational redundancy, compelling distinct experts to specialize in heterogeneous predictive patterns"**
- 公式: `L_orth = Σ_{i<j} ||cos_sim(h_i, h_j)||²`
- 论文报告可解释性 + 分类精度双提升

### 1.4 叙事收益

- 立即复现 arXiv:2606.03631 的核心约束
- 解 round 79 sweep 暴露的 top_k=1 不稳定
- 给 K×top_k 全空间加**专家多样性保险**

---

## 2. 目标

新增 `lnn/core/orthogonality.py`:

- `orthogonality_loss(expert_outputs: list[Tensor], lambda_coeff: float = 0.01) -> Tensor`
- 公式: `L_orth = λ * Σ_{i<j} cos_sim(h_i, h_j)²`
- 默认 `lambda_coeff=0.01` (轻量, 跟 AnchorMoE 论文 spirit 一致, 不主导主 loss)
- `lambda_coeff=0.0` 时**完全无影响** (back-compat)

集成到 `FAMECfCCell`:
- `forward` 返回 `(hidden, aux_loss)` tuple, trainer 可选加 aux_loss 到主 loss
- `FAMECfCNetwork.forward` 累加 aux_loss, 同样以 tuple 返回

---

## 3. 设计

### 3.1 公式

设 K=3 experts, hidden dim H=16, 每次 forward:
- 收集 K expert 输出 `outs[k]`, shape `[B, H]`
- 计算 K×K cosine similarity 矩阵
- 取上三角 (i<j) 的平方和 → `L_orth = Σ_{i<j} cos_sim(outs[i], outs[j])²`
- 返回 `λ * L_orth` 作为 aux loss

### 3.2 API

```python
def orthogonality_loss(
    expert_outputs: list[torch.Tensor],   # K × [B, H]
    lambda_coeff: float = 0.01,
) -> torch.Tensor:
    """Geometric orthogonality constraint (AnchorMoE 2606.03631)."""
```

集成到 `FAMECfCCell`:
```python
def forward(self, x_t, h, dt=1.0):
    # ... existing ...
    h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]
    # New: also return per-expert outputs for orth loss
    return h_new, outs  # outs: K × [B, H]
```

`FAMECfCNetwork.forward`:
```python
def forward(self, x, ...):
    # ... existing ...
    out_proj = self.output_proj(layer_input)
    return out_proj, total_aux_loss
```

### 3.3 训练集成

```python
# 训练时
y_pred, aux_loss = model(x)
loss = task_loss(y_pred, y) + aux_loss  # aux_loss 已经乘 λ
```

---

## 4. 验收标准

### 4.1 单元测试 (`tests/test_orthogonality.py`)

| 测试 | 期望 |
|---|---|
| `test_orth_zero_when_lambda_0` | λ=0 时返回 0 (不依赖输入) |
| `test_orth_zero_for_orthogonal_outputs` | 完全正交 expert → loss ≈ 0 |
| `test_orth_high_for_duplicate_outputs` | 重复 expert → loss 高 |
| `test_orth_symmetric` | 交换 expert 顺序结果不变 |
| `test_orth_gradient_flow` | 梯度流到 expert outputs |
| `test_orth_lambda_scaling` | loss 与 λ 严格线性 |
| `test_fame_cell_returns_aux_loss` | FAMECfCCell.forward 返回 (h, outs) |
| `test_fame_network_accumulates_aux` | FAMECfCNetwork.forward 返回 (y, aux) |
| `test_fame_with_orthogonality_sin_smoke` | K=3 top_k=1 + orthogonality, 3 seed std < 0.05 |

### 4.2 回归

- `pytest tests/` 既有 132+ 测试零回归
- `test_fame_cfc.py` 等 back-compat (K=3 top_k=3 dense 等)

### 4.3 sweep 重跑验证 (关键)

- `scripts/bench_orthogonality.py` — K=3 top_k=1 with λ ∈ {0, 0.01, 0.1}
- 目标: λ > 0 时 K=3 top_k=1 std < 0.05 (vs current 0.79)
- 报告里写明 "round 79 sweep 暴露的 top_k=1 不稳定被 orthogonality 缓解"

### 4.4 文档

- `docs/research/2026-06-14_orthogonality_report.md` — bench 报告
- README.md — 加 orthogonality 简述

---

## 5. 实现步骤 (3-4h)

1. **写 `lnn/core/orthogonality.py`** (1h): `orthogonality_loss` + 单元测试
2. **改 `lnn/core/fame_cfc.py`** (45min): forward 返回 (h, outs), network 累加 aux
3. **写 `tests/test_orthogonality.py`** (1h): 9 个测试
4. **跑全量 pytest** (15min): 零回归
5. **写 `scripts/bench_orthogonality.py`** (30min): K=3 top_k=1 × λ sweep
6. **写报告 + 文档** (30min)
7. **commit + push** (15min)

---

## 6. 不在本次范围

- **#10-40 φ-Balancing EMA load balancing** — 互补方案, 留给下个 session
- **#10-41 DBES 5 指标诊断** — 文档级, 不上代码
- **修改现有 forward signature** — 本场只**加** aux loss 输出, 不破坏原 API

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `orthogonality_loss` 在 expert 输出极小时 (e.g. all-zero) 计算 cos_sim → NaN | 加 `eps=1e-8` 在 normalize 时 |
| λ=0.01 主导主 loss | 报告里**显式**记录 loss 分解 (task + aux) |
| 修改 FAMECfCCell.forward 签名破坏既有测试 | 既有测试调用 `cell(x_t, h, dt)`, 新签名 `cell(x_t, h, dt)[0]` 取 h — 但 back-compat 要求**直接**返回 h,不是 tuple |
| back-compat 解法: 保持 `cell.forward` 返回 h (旧), 加新方法 `cell.forward_with_aux(h)` 返回 (h, outs), network 默认调用旧方法 |  |

**最终 back-compat 方案**:
- `FAMECfCCell.forward` **保持返回 h** (单值, 不破坏既有测试)
- 加新方法 `FAMECfCCell.forward_with_aux` 返回 (h, expert_outputs)
- `FAMECfCNetwork.forward` 保持返回 y_pred (单值)
- 加新方法 `FAMECfCNetwork.forward_with_aux` 返回 (y_pred, aux_loss)
- 训练时用户显式调用 `forward_with_aux` 拿 aux_loss

---

## 8. 一句话总结

> **本 PRD 目标: 加 `orthogonality_loss` (AnchorMoE 2606.03631 模板) + 在 `FAMECfCCell` / `FAMECfCNetwork` 上提供 `forward_with_aux` 方法返回 expert 输出去算 aux loss, 3-4h 单 PR, 期望 sweep 重跑 K=3 top_k=1 时 std < 0.05 (vs round 79 当前 0.79), 解 round 79 暴露的 top_k=1 训练发散硬阻塞, 给 K×top_k 全空间加专家多样性保险。**
