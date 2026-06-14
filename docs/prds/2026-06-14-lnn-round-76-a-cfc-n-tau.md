---
title: "PRD #10-29 — CfCCell 多时间尺度 (n_tau) 维度支持"
id: prd-10-29
date: 2026-06-14
status: proposed
priority: P0
estimated: 3-5h
loop: round 76 (2026-06-14 下午)
related_papers:
  - arXiv:2606.12240v1 (MR-MoE, Zong VT 2026)
  - arXiv:2606.11162v1 (COGENT, 2026-06-09)
  - arXiv:2606.07670v1 (Liquid-3DGS, 2026-06-04)
  - arXiv:2604.18274v2 (LiquidTAD, 2026-04-20)
related_prior: round 39 §10 #10-24 (MR-MoE 候选)
---

# PRD #10-29 — `CfCCell` 多时间尺度 (`n_tau`) 维度支持

## 1. 背景与动机

### 1.1 跨域信号 (本 loop session 已汇总)

2026-06-04 ~ 2026-06-10 arXiv **4 篇独立论文**集中提出"多时间尺度 ODE"范式:

| 论文 | 域 | 异 τ 体现 |
|---|---|---|
| MR-MoE (2606.12240) | 脓毒症时序 | K=3 LNN experts, τ1 ≪ τ2 ≪ τ3 |
| COGENT (2606.11162) | 冰盖物理 | 显式 relative rollout time + forcings 插值 |
| Liquid-3DGS (2606.07670) | 4D 视觉 | depth-as-time 多层 CfC stack |
| LiquidTAD (2604.18274) | 视频动作 | liquid-inspired temporal relaxation |

**结论**: 异 τ 多时间尺度已从"个别创新"升级为**领域共识**。

### 1.2 本仓现状

`lnn/core/cfc.py::CfCCell` 当前 `tau` 标量假设全部 hidden 维度同时间尺度:

```python
# 当前实现 (简化)
self.tau = nn.Parameter(torch.tensor(tau))  # 标量
# forward: dh/dt = -h/tau + f(x, u)
```

**阻塞**: 任何"显式分快/慢维度"的论文 (MR-MoE 全部 / COGENT 隐式) 都无法**直接**用本仓 `CfCCell` 复现 — 必须改源码。

### 1.3 叙事收益

- 立即解锁 4 篇 arXiv 跨域论文的"多 τ"主张
- 是本仓**最低成本**"叙事 + 实证"双升级
- 为 #10-30 (COGENTCell) / #10-24 (MR-MoE) / #10-28 (Timeflies) 铺路 — 它们都吃 `n_tau` 接口

---

## 2. 目标

`CfCCell` 在**不破坏现有行为**的前提下, 加 `n_tau: int = 1` 维度:

- `n_tau=1` (默认): 行为**完全等价**于现 `CfCCell` (1e-5 误差), 现有 7 篇研读 + 268+ 测试零回归
- `n_tau≥2`: hidden 维度按 `n_tau` 切分, 每支配不同 `tau` 标量, forward 时**并行**算 + 拼接

---

## 3. 设计

### 3.1 公式

设 `n_tau=K`, hidden dim `H`:
- 每支 hidden dim `H_i = H // K` (允许 H % K ≠ 0, 用 floor + 补零)
- 第 `i` 支时间常数 `τ_i = tau_scales[i]`, `i=0..K-1`
- 第 `i` 支 ODE: `dh_i/dt = -h_i/τ_i + f_i(x, u; θ_i)`
- 输出 `h = concat([h_0, h_1, ..., h_{K-1}])`

### 3.2 API 变更 (向后兼容)

```python
class CfCCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        # ... 现有参数
        n_tau: int = 1,                              # 新增, 默认 1
        tau_scales: tuple = (0.1, 1.0, 10.0),        # 新增, 仅 n_tau>1 用
    ):
```

### 3.3 内部实现要点

1. **n_tau=1 分支**: 走**原** `forward` 路径, 零回归
2. **n_tau=K 分支**:
   - `self.tau_per_branch = nn.ParameterList([nn.Parameter(torch.tensor(tau_scales[i])) for i in range(K)])`
   - `self.w_h_per_branch = nn.ModuleList([nn.Linear(...) for _ in range(K)])`
   - forward 时按 hidden 维度 split → 各自 ODE → concat
3. **gating**: 原 CfC 的 `g ⊙ (1-σ_τ) + h_cand ⊙ σ_τ` 保留, 仅 per-branch 独立
4. **不引入新依赖**, 纯 torch

### 3.4 性能影响

- `n_tau=1`: 零开销 (走原路径)
- `n_tau=K`: hidden 维度按 K 切分, **总参数量略增** (Linear 层 K 倍 vs 1 倍) — 但 hidden // K 每支, 总 FLOPS **相当**

---

## 4. 验收标准

### 4.1 单元测试 (`tests/test_cfc_n_tau.py`)

| 测试 | 期望 |
|---|---|
| `test_cfc_n_tau_1_equivalence` | `n_tau=1` 与现 `CfCCell` forward 输出 1e-5 误差 |
| `test_cfc_n_tau_3_dim` | `n_tau=3` 隐藏维度匹配, 3 支独立 τ |
| `test_cfc_n_tau_5_gradient` | `n_tau=5` 5 支 τ 都接受梯度, 反向传播无 NaN |
| `test_cfc_n_tau_3_sin_smoke` | n_tau=3 在简单 sin 上 3 seed 平均 MSE ≤ n_tau=1 (因 toy 无优势, 持平即可) |

### 4.2 回归测试

- `pytest tests/ -q` 全绿 (268+ tests 沿用)
- 烟测 `bench_suite.py::case_a` (toy sin) 用 n_tau=1 仍报告相同 MSE

### 4.3 文档

- `docs/research/2026-06-14_cfc_n_tau_sweep_report.md` — 烟测报告 (n_tau=1/3/5)
- README.md: 简述 `n_tau` 用法 + 示例代码
- CHANGELOG: 新增 "CfC 多时间尺度 (n_tau) 支持" 条目

---

## 5. 实现步骤 (3-5h)

1. **改 `lnn/core/cfc.py`** (1.5h):
   - `CfCCell.__init__` 加 `n_tau`, `tau_scales` 参数
   - 加 `_forward_n_tau_1` (原 forward 重命名) 和 `_forward_n_tau_K` (新多支)
   - 加 `forward` 路由
2. **写 `tests/test_cfc_n_tau.py`** (1h): 4 个测试
3. **跑全量 pytest** (15min): 验证零回归
4. **写烟测脚本** `scripts/bench_cfc_n_tau.py` (1h): case A sin, n_tau=1/3/5, 3 seed
5. **写报告 + 文档** (45min): 报告 + README + CHANGELOG
6. **commit + push** (15min): 含 IP 覆盖的 `GIT_SSH_COMMAND`

---

## 6. 不在本次范围

- **MR-MoE K=3 expert + gating**: 这是 #10-24 PRD, 本 PR 只做 cell 内部多 τ
- **COGENT 图结构**: 这是 #10-30 PRD
- **Timeflies observation head**: 这是 #10-28 PRD
- **LFM2 hybrid conv+GQA**: 这是 #10-31 PRD

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `n_tau=1` 不等价 (浮点误差 > 1e-5) | 单元测试强制 1e-5, 失败立即排查 |
| 多支切分 hidden 不均 (H % K ≠ 0) | `H_i = H // K`, 末尾用 zero-pad |
| 训练时 n_tau=3 反而输 n_tau=1 (toy 无优势) | 接受 — 报告里明说"toy 无优势, 真实场景才能见" |
| `tau_scales` 默认 (0.1, 1.0, 10.0) 不合理 | 暴露为可调参数, 烟测里只跑 default |

---

## 8. 一句话总结

> **本次 PRD 目标是给 `CfCCell` 加 `n_tau` 维度 (默认 1 零回归, ≥2 启用多时间尺度), 3-5h 单 PR, 立即解锁 4 篇 arXiv 跨域论文的"异 τ"主张, 为 #10-30/24/28 三条下游候选铺路。**
