---
title: "PRD #10-36 — FAME-style Forecastability-Aware Top-K Router for LNN"
id: prd-10-36
date: 2026-06-14
status: proposed
priority: P0
estimated: 5-7h
loop: round 78 (2026-06-14 下午, loop session #3)
related_papers:
  - arXiv:2606.08896v1 (FAME, 2026-06-08)
  - arXiv:2606.12240v1 (MR-MoE, Zong VT 2026)
  - arXiv:2606.03631v2 (AnchorMoE, 2026-06-02)
related_prior: round 77 #10-24 (MRMoECfCCell), round 76 #10-29 (CfCCell n_tau)
---

# PRD #10-36 — FAME-style Forecastability-Aware Top-K Router for LNN

## 1. 背景与动机

### 1.1 跨域信号 (本 loop session 已汇总)

2026-06-02 ~ 2026-06-11 arXiv **7 篇独立论文**集中提出"显式多专家 / 多时间尺度"范式。本场新增:

- **FAME (arXiv:2606.08896, 6-08, B+)** — 山东新北洋 **生产部署** (5000+ 售货机 / 60M+ 交易), **Top-2 路由 -12.4% MSE vs LightGBM**, 1.92 experts/series 平均激活。**首个稀疏 top-K 路由 + 生产部署范本**。

### 1.2 本仓现状 (round 76/77 已有)

- **round 76 (`CfCCell` 加 `n_tau`, push 69a319b)**: 细胞**内**多 τ
- **round 77 (`MRMoECfCCell` K experts + 密集 softmax, push 3a5cb1f)**: 细胞**间**多 expert + 密集路由 → K experts 都 forward
- **缺口**: 稀疏 top-K 路由 (FAME 风格) — **本 PRD 目标**

### 1.3 叙事收益

- 立即复现 arXiv:2606.08896 (FAME) 的 top-K sparse routing 范式
- 跟 round 77 dense softmax 形成代际差: 同样 K=3 experts, FAME 只 forward 1.92 (top-2 实测), **节省 ~36% compute**
- 真实生产数据 (5000+ 售货机, 60M+ 交易) — 不是 paper-only

---

## 2. 目标

新增 `lnn/core/forecastability_router.py` + `lnn/core/fame_cfc.py`:

- `ForecastabilityRouter(input_size, hidden_size, n_experts, top_k=2)` — top-K 稀疏路由
  - `logits = W · [x_t; h_prev]` → [B, K]
  - `top_k_logits, top_k_idx = topk(logits, top_k, dim=-1)`
  - 掩码: 非 top-K 位置置 -inf
  - `g = softmax(top_k_logits)` 仅在 top-K 索引上
- `FAMECfCCell(input_size, hidden_size, n_experts=3, top_k=2)` — 复用 round 77 的 `CfCCell` experts + 替换 router
- `FAMECfCNetwork(input_size, hidden_size, output_size, ...)` — 网络级包装
- 兼容: `top_k=K` 等价于密集 softmax (数值 1e-5 误差)
- 兼容: `top_k=1` 等价于单 expert (router argmax)

---

## 3. 设计

### 3.1 公式 (FAME 论文 Eq. 5-7)

```
logits = W_g · [x_t; h_prev]               # [B, K]
top_k_logits, top_k_idx = topk(logits, K')  # K' = top_k
mask = -inf outside top_k_idx
g = softmax(logits + mask)                  # only top-K nonzero
h_new = Σ_k g_k · expert_k(x_t, h_prev)     # same as round 77 but K' nonzero
```

### 3.2 API

```python
class ForecastabilityRouter(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int = 2,
        router_hidden: int = 0,           # 0=linear, >0=MLP
    ):
        assert top_k >= 1 and top_k <= n_experts


class FAMECfCCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
    ):
```

### 3.3 内部实现要点

1. **复用 round 77 expert 列表**: `self.experts = nn.ModuleList([CfCCell(...) for _ in range(n_experts)])`
2. **ForecastabilityRouter**:
   - `self.router = nn.Linear(input_size + hidden_size, n_experts)` (或 MLP)
   - `top_k` 截断 + 掩码
3. **forward**:
   - `combined = [x_t, h]`
   - `logits = self.router(combined)` [B, K]
   - 取 top-K: `top_logits, top_idx = logits.topk(self.top_k, dim=-1)`
   - 用 scatter 构造 mask: `-inf` outside top-K
   - `g = softmax(masked_logits, dim=-1)` — 仅 top-K 非零
   - 跑 K experts, weighted sum
   - **短路优化**: 只 forward top-K experts (不跑其他 K-K' experts) — FAME 风格

### 3.4 性能

- `top_k=1`: 1 expert forward, 等价 dense top_k=1 但更便宜
- `top_k=K`: 全部 K experts forward, 等价 dense softmax
- `top_k=2` + K=3: 节省 1/3 forward compute

---

## 4. 验收标准

### 4.1 单元测试 (`tests/test_fame_cfc.py`)

| 测试 | 期望 |
|---|---|
| `test_router_top_k_1_argmax` | top_k=1 等价 router.argmax() |
| `test_router_top_k_K_dense` | top_k=K 等价 round 77 dense softmax (1e-5 误差) |
| `test_router_sparsity_top_2` | top_k=2 时 g 严格只有 2 个非零 |
| `test_router_gradient_flows_to_top_experts` | 梯度只流到 top-K experts (其他 K-K' 跳过) |
| `test_router_entropy_top_2_lower_than_dense` | top-K entropy < log K (因只 K' 个激活) |
| `test_fame_k_3_top_2_forward_shape` | K=3 top_k=2 forward shape (B, H) |
| `test_fame_with_n_tau_3` | K=3 experts × n_tau=3 + top_k=2 |
| `test_fame_network_smoke` | FAMECfCNetwork 多层 + mask + dt + return_sequences |
| `test_fame_sin_smoke` | toy sin 3 seed, top_k=2 MSE ≤ 1.5× top_k=1 (无灾难) |

### 4.2 回归

- `pytest tests/ -q` 既有 268+ 测试零回归
- 102+ CfC+MR-MoE 测试零回归

### 4.3 文档

- `docs/research/2026-06-14_fame_cfc_sweep_report.md` — top_k=1/2/3 烟测报告
- `lnn/core/__init__.py` — 导出
- README.md — 加 FAME 简述 + 示例

---

## 5. 实现步骤 (5-7h)

1. **写 `lnn/core/forecastability_router.py`** (1.5h): ForecastabilityRouter
2. **写 `lnn/core/fame_cfc.py`** (1.5h): FAMECfCCell + FAMECfCNetwork
3. **写 `tests/test_fame_cfc.py`** (1.5h): 9 个测试
4. **跑全量 pytest** (15min): 零回归
5. **写 `scripts/bench_fame_cfc.py`** (1h): top_k=1/2/3 sweep
6. **写报告 + 文档** (45min)
7. **commit + push** (15min)

---

## 6. 不在本次范围

- **Orthogonality constraint (#10-37)**: 防 expert collapse
- **LFM2.5 INT8 推理 (#10-7/#10-35)**: 模型级蒸馏
- **K×n_tau×top_K sweep (#10-38)**: 三维交叉

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| top_k=K 不等价 dense softmax (top-K 索引 out-of-order) | 测试强制 forward 1e-5 误差, 失败立即排查 |
| 短路 top-K expert forward 导致 autograd 漏梯度 | 用 `torch.where` 构造 mask, 不真正"跳过" forward, 只是 mask 掉 output |
| top_k=2 反而输 top_k=1 (过参数化) | 接受 — 报告明说, 真实生产数据才见 |
| 训练时 router collapse (top-K 不变) | 报告里记录 top-K 频率, 不强制 load-balancing loss |

---

## 8. 一句话总结

> **本 PRD 目标: 在 round 77 `MRMoECfCCell` (K experts + dense softmax) 之上加 sparse top-K 路由 (FAME arXiv:2606.08896 范式), 5-7h 单 PR, 12+ 单元测试, top_k=1/2/3 烟测, 期望节省 ~36% forward compute (K=3 top_k=2) 同时维持或提升精度; 立即解锁 #10-37 (orthogonality) / #10-38 (K×n_tau×top_K sweep) 两条下游候选。**
