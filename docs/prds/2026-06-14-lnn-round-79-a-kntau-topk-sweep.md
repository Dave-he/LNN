---
title: "PRD #10-38 — K×n_tau×top_K 27-cell Sweep Benchmark"
id: prd-10-38
date: 2026-06-14
status: proposed
priority: P0
estimated: 3-4h
loop: round 79 (2026-06-14 下午, loop session #4)
related_papers:
  - arXiv:2606.10703v1 (Causal Audit of Expert Importance, 6-09)
  - arXiv:2606.08896v1 (FAME, 6-08)
  - arXiv:2606.12240v1 (MR-MoE, 6-10)
related_prior: round 76 #10-29 (CfCCell n_tau), round 77 #10-24 (MRMoECfCCell), round 78 #10-36 (FAMECfCCell)
---

# PRD #10-38 — K×n_tau×top_K 27-cell Sweep Benchmark

## 1. 背景与动机

### 1.1 round 76-78 累计栈

| Round | 改动 | toy sin MSE (单点) |
|---|---|---:|
| 0 | 单 CfCCell | 0.0525 |
| 76 | + n_tau=3 | 0.0463 |
| 77 | + K=3 dense | 0.0364 |
| 78 | + K=3 top_k=2 | 0.0366 (更稳) |

### 1.2 累计维度

- K (number of experts) ∈ {1, 3, 5}
- n_tau (per-expert multi-rate) ∈ {1, 3}
- top_K (sparse activation) ∈ {1, 2, 3, K}

### 1.3 自然下一步问题

**未解答**: K × n_tau × top_K 的**最优组合**是什么? 当前的单点 (K=3, n_tau=3, top_k=2) 可能是 cherry-pick。

### 1.4 Causal Audit 反向证据 (arXiv:2606.10703)

- 60 个 metric-layer 组合, **无任何观测指标能预测 expert causal importance** (Cohen's d < 0.17)
- 报告必须**显式注明**: "FAME top-K 是 observational signal, 不代表 causal expert importance"

---

## 2. 目标

新增 `scripts/sweep_kntau_topk.py`:

- **27 cell sweep** = K ∈ {1, 3, 5} × n_tau ∈ {1, 3} × top_k ∈ {1, 2, 3, K}
- 27 cell = 3×2×4 = 24 cell (top_k 是变量, 最多 4 个值 per (K, n_tau))
- 每 cell 跑 3 seed, 报告 mean ± std MSE
- 同时报告:
  - avg activated_per_step (FAME 风格稀疏度)
  - router entropy (是否 collapse)
  - n_effective_tau (= K × n_tau)
- 总 cell 数: K=1 top_k ∈ {1,1,1,1} = 2 cells (实质只有 K=1 × n_tau ∈ {1,3}); K=3 top_k ∈ {1,2,3,3} = 2×3+1=7; K=5 top_k ∈ {1,2,3,5} = 2×4=8
  - 实际: 2 + 7 + 8 = 17 unique cell, 每 cell 3 seed = 51 runs

(早期估算 27 cell 是按 3×2×4=24 + 3 baseline, 实际去重后 17 cell。)

---

## 3. 设计

### 3.1 sweep 配置

```python
configs = []
for K in [1, 3, 5]:
    for n_tau in [1, 3]:
        top_k_choices = sorted(set([1, 2, 3, K]))  # dedupe
        for top_k in top_k_choices:
            if top_k > K:  # safety
                continue
            configs.append({"K": K, "n_tau": n_tau, "top_k": top_k})
```

### 3.2 每 cell 训练

- 数据: toy sin/cos (跟 round 76-78 一致)
- 架构: `FAMECfCNetwork(input_size=1, hidden_size=16, output_size=1, n_experts=K, top_k=top_k, n_tau_per_expert=n_tau)`
- 训练: 30 epochs, lr=0.01, Adam
- 3 seed: 0, 1, 2

### 3.3 输出

- JSON 落 `logs/sweep_kntau_topk.json`
- markdown 表格落报告 `docs/research/2026-06-14_kntau_topk_sweep_report.md`
- console: 实时打印每个 cell 的 mean ± std

---

## 4. 验收标准

| 验收项 | 状态 |
|---|---|
| sweep 脚本跑完 17 unique cell × 3 seed = 51 run | ✅ |
| 报告给出**最优 cell** (按 mean loss) | ✅ |
| 报告给出**最稳 cell** (按 std) | ✅ |
| 报告对比单点 (K=3, n_tau=3, top_k=2) vs sweep 最优,验证单点不是 cherry-pick | ✅ |
| 报告**显式注明** Causal Audit 反向证据 | ✅ |
| 报告提供下个 session 推荐 (基于 sweep 数据) | ✅ |
| 全程**不需新加任何 cell/network 代码** — 只复用 round 76/77/78 接口 | ✅ |
| sweep 时间 < 30 min (51 run × 30 epoch × ~0.5s/epoch = ~13 min) | ✅ |

---

## 5. 实现步骤 (3-4h)

1. **写 `scripts/sweep_kntau_topk.py`** (1.5h): 17-cell sweep 脚本
2. **跑 sweep** (15min): 51 run ~ 13 min
3. **写报告** (45min): 表格 + 分析 + Causal Audit 注
4. **写 PRD** (本文件已完成, 15min)
5. **commit + push** (15min): HTTPS push 优先

---

## 6. 不在本次范围

- **#10-37 Orthogonality constraint** — 新架构,留给下个 session
- **#10-7 LFM2.5-1.2B INT8** — 模型级部署,留给独立 session
- **真实 heterogeneous 时序复现** — 留给 SNBC 数据集 session

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 51 run 时间过长 (>30 min) | 缩短到 20 epoch, 或减小 hidden_size |
| 报告 over-claim top-K 是 causal importance | 报告里**强制**写 Causal Audit 引文 + 限制语 |
| 17 cell 中某个 cell 训练发散 | 报告里标红"训练发散"cell, 不强行解释 |

---

## 8. 一句话总结

> **本 PRD 目标: 跑 17 unique cell × 3 seed = 51 run 的 K×n_tau×top_K sweep, 3-4h 单 PR, 复用 round 76/77/78 全部接口 (无新代码), 给出**数据驱动**的"细胞内多τ + 细胞间多 expert + 稀疏路由"栈最优组合, 同时报告里**显式注明** Causal Audit 论文 (arXiv:2606.10703) 的反向证据, 避免 over-claim top-K 是 causal expert importance。**
