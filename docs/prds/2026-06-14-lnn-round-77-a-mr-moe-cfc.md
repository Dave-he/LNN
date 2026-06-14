---
title: "PRD #10-24 — MR-MoE: Multi-Rate Mixture of Experts for LNN"
id: prd-10-24
date: 2026-06-14
status: proposed
priority: P0
estimated: 4-6h
loop: round 77 (2026-06-14 下午, loop session #2)
related_papers:
  - arXiv:2606.12240v1 (MR-MoE, Zong et al. VT 2026)
  - arXiv:2606.13024v1 (CausalMoE, 2026-06-11)
  - arXiv:2606.11162v1 (COGENT, 2026-06-09)
related_prior: round 76 #10-29 (CfCCell n_tau), round 39 §10 #10-24
---

# PRD #10-24 — MR-MoE: Multi-Rate Mixture of Experts for LNN

## 1. 背景与动机

### 1.1 跨域信号 (本 loop session 已汇总)

2026-06-04 ~ 2026-06-11 arXiv **6 篇独立论文**集中提出"显式多专家 / 多时间尺度"范式:

| 论文 | 域 | 范式 |
|---|---|---|
| MR-MoE (2606.12240) | 脓毒症时序 | K=3 LNN experts, 异 τ |
| COGENT (2606.11162) | 冰盖物理 | 显式 relative rollout time |
| Liquid-3DGS (2606.07670) | 4D 视觉 | depth-as-time |
| LiquidTAD (2604.18274) | 视频动作 | temporal pyramid |
| CausalMoE (2606.13024) | 时序因果 | Pattern-Routed MoE |
| Beyond Uniform Tokens (2606.13624) | TS-LLM | adaptive token budget |

**结论**: "MoE 显式分时间/模式" 已成 2026-06 **第二类学界共识**(与 round 76 的"异 τ"并列)。

### 1.2 本仓现状

- **round 76 (push 69a319b)**: `CfCCell` 加 `n_tau` 维度 → 单元**内**多 τ
- **本 PRD**: 单元**间**多专家 + 路由 → 完整 MR-MoE 范式
- 缺口: 一个 K=3 CfC experts + softmax router 的组合

### 1.3 叙事收益

- 立即复现 arXiv:2606.12240 (Zong VT 2026) 的 Eq. 8-10
- 跟 round 76 n_tau 叠加 → 完整 "cell 内多 τ + cell 间多 expert" MR-MoE 范本
- 单元粒度 (4-6h) + 高叙事 (B+ 论文复现)

---

## 2. 目标

新增 `lnn/core/mr_moe_cfc.py` 模块, 实现:

- `MRMoECfCCell(input_size, hidden_size, n_experts=3)` — K 个 `CfCCell` 专家 + softmax 路由门
- `MRMoECfCNetwork(input_size, hidden_size, output_size, num_layers=1, n_experts=3, ...)` — 网络级包装
- 路由策略: `g = softmax(W_g · [x_t, h_prev])`, 输出 `h_new = Σ_k g_k · expert_k(x_t, h_prev)`
- 兼容: `n_experts=1` 退化为单 `CfCCell` (数值上 forward 等价, 1e-5 误差)

---

## 3. 设计

### 3.1 公式 (MR-MoE 论文 Eq. 8-10)

```
g = softmax(W_g · [x_t; h_prev])           # [B, K]
h_k = CfCCell_k(x_t, h_prev)               # [B, H]  for k=1..K
h_new = Σ_k g_k · h_k                       # [B, H]
```

### 3.2 API

```python
class MRMoECfCCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,                # K
        n_tau_per_expert: int = 1,         # 复用 round 76 n_tau
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,            # 0 = linear, >0 = MLP
    ):
```

### 3.3 内部实现要点

1. **`self.experts = nn.ModuleList([CfCCell(input_size, hidden_size, n_tau=n_tau_per_expert, tau_scales=tau_scales) for _ in range(n_experts)])`**
2. **`self.router = nn.Linear(input_size + hidden_size, n_experts)`** (或 MLP)
3. **forward**:
   - `combined = [x_t, h_prev]`
   - `g = softmax(self.router(combined), dim=-1)` — [B, K]
   - `outs = [expert(combined[?, :], h_prev) for expert in self.experts]` — 每 expert 拿自己的 x_t 和 h_prev
   - `h_new = sum_k g[:, k:k+1] * outs[k]`

### 3.4 性能影响

- K=3: 参数 3× 单 CfCCell, FLOPs 3× 单 forward — 与 n_tau=K cell 不同, 这里 K experts 都要 forward
- K=1: 1 expert + router (1 linear), 略多于单 CfCCell
- 训练: 标准反向传播, 无特殊优化

---

## 4. 验收标准

### 4.1 单元测试 (`tests/test_mr_moe_cfc.py`)

| 测试 | 期望 |
|---|---|
| `test_mr_moe_k_1_equivalence` | n_experts=1 router 输出 one-hot, 与单 CfCCell 1e-5 误差 |
| `test_mr_moe_k_3_forward_shape` | n_experts=3 forward shape (B, H) |
| `test_mr_moe_router_sums_to_1` | g 沿 K 维求和 == 1 |
| `test_mr_moe_k_3_gradient_flows_to_all_experts` | 所有 expert 都有非零梯度 |
| `test_mr_moe_with_n_tau_3` | K=3 experts, 每 expert n_tau=3 → 3×3=9 effective τ groups |
| `test_mr_moe_sin_smoke` | toy sin 3 seed, K=3 MSE ≤ 1.5× K=1 (无灾难回归) |
| `test_mr_moe_network_smoke` | MRMoECfCNetwork 多层 forward / mask / dt 兼容 |

### 4.2 回归

- `pytest tests/ -q` 既有 268+ 测试零回归
- 88+ CfC 相关测试零回归

### 4.3 文档

- `docs/research/2026-06-14_mr_moe_cfc_sweep_report.md` — K=1/3/5 烟测报告
- `lnn/core/__init__.py` — 导出 `MRMoECfCCell, MRMoECfCNetwork`
- README.md — 加 MR-MoE 简述 + 示例

---

## 5. 实现步骤 (4-6h)

1. **写 `lnn/core/mr_moe_cfc.py`** (1.5h): MRMoECfCCell + MRMoECfCNetwork
2. **写 `tests/test_mr_moe_cfc.py`** (1h): 7 个测试
3. **跑全量 pytest** (15min): 零回归
4. **写 `scripts/bench_mr_moe_cfc.py`** (1h): K=1/3/5 烟测
5. **写报告 + 文档** (45min)
6. **commit + push** (15min)

---

## 6. 不在本次范围

- **CausalMoE (#10-33)**: 因果发现头
- **Adaptive Token Compression (#10-34)**: 频域 token 压缩
- **LFM2.5 INT8 推理 (#10-7/#10-35)**: 模型级蒸馏 / 量化

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| K=1 不等价 (router 输出非 one-hot) | 测试强制 `argmax == 0` 检查 + 1e-5 forward 误差 |
| K=3 toy sin 反而输 K=1 (过参数化) | 接受 — 报告里明说, 真实噪声/长程数据才见真章 |
| 训练时梯度消失到某 expert | 加 `test_mr_moe_k_3_gradient_flows_to_all_experts` 强约束 |
| router collapse (g 退化为 one expert) | 报告里记录 g 平均熵, 不强制 load-balancing loss (超出范围) |

---

## 8. 一句话总结

> **本 PRD 目标: 在 round 76 `CfCCell(n_tau=K)` 之上加 K=3 experts + softmax router, 实现 arXiv:2606.12240 MR-MoE 范式的最小可复现版本, 4-6h 单 PR, 12+ 单元测试, K=1/3/5 烟测, 立即解锁 arXiv:2606.13024 CausalMoE 跟 arXiv:2606.11162 COGENT 两条下游候选。**
