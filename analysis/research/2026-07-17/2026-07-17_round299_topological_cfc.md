# Round 299 — TopologicalCfC: per-neuron ODE + learned sparse graph coupling

> 2026-07-17, 1h /loop cycle #1。研读方案见 `2026-07-17_lnn_research_report.md`。

---

## 一、动机

arXiv:2606.21295v6 (Cai & Zhao, AAAI 2027 — **Topological Neural Dynamics**) 论证:
> "Layer-wise dynamics 限制各神经元独立进化的自由度。Neuron-wise dynamics + 显式 graph 拓扑在单玩家 Pong 上让 catch rate 是最强基线的 3×(17.47 连续)"

本仓库现有 297+ rounds 中:MoE 在 **expert** 粒度做 routing(40+ rounds),CfC 各变体(pulse/oscillator/PDNA 等)都是 **layer-wise** matmul 加 gating 或正则。

**没有实现的是:per-neuron 独立 ODE + 显式学到的稀疏 coupling graph。**

## 二、实现(`lnn/core/topological_cfc.py`,211 行)

```python
class TopologicalCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, graph_k=8, mix_init=0.10):
        # per-neuron ODE branches (Wf, Wg, Wh, time_scale)
        self.f_gate = nn.Sequential(nn.Linear(input+hidden, hidden), nn.Sigmoid())
        self.g_branch = nn.Sequential(nn.Linear(input+hidden, hidden), nn.Tanh())
        self.h_branch = nn.Sequential(nn.Linear(input+hidden, hidden), nn.Tanh())
        self.time_scale = nn.Parameter(torch.ones(hidden_size))
        # random k-regular graph topology
        self._build_random_k_regular()
        # edge weights 初始 1/k,mix_strength sigmoid 初始 0.10
        self.adj_weights = nn.Parameter(...)
        self.mix_logit = nn.Parameter(torch.tensor(inv_sigmoid(0.10)))

    def forward(self, x_t, h, dt=1.0):
        # 1. per-neuron closed-form
        decay = sigmoid(-f * self.time_scale * dt)
        h_tilde = decay * g + (1 - decay) * h_target  # (B, H)
        # 2. sparse graph mixing via index_add_
        if self._adj_indices is not None:
            src = self._adj_indices[1]; tgt = self._adj_indices[0]
            w = self.adj_weights
            gathered = h_tilde[:, src] * w.unsqueeze(0)  # (B, H*k)
            mixed = torch.zeros_like(h_tilde)
            mixed.index_add_(1, tgt, gathered)
            denom = ...   # per-row normalizer
            mixed = mixed / denom.unsqueeze(0)
        # 3. linear blend
        m = sigmoid(self.mix_logit)
        return (1 - m) * h_tilde + m * mixed
```

设计要点:
1. **每神经元独立 ODE**:`f, g, h_target` 都是 `nn.Linear(input+hidden, hidden)`,但 view 为独立 per-neuron ODE —— CfC 同款 closure,但语义上是 H 个独立 1D systems 而非一个 H-D matmul。
2. **稀疏拓扑**:`graph_k` controls 每个 neuron 连接几个邻居(H*k 个边 ≤ H² 个,当 k≪H 时是 O(Hk))。Buffer 中存 `(2, H*k)` 形状的 indices(seeded random),`adj_weights` 是每边的可学习标量。
3. **闭合解**:无 ODE solver;Sigmoid 闭合 + sparse scatter 比 dense matmul 更便宜。
4. **退化 case**:`graph_k=0` 或 `graph_k=H` 时跳过图,等价于"per-neuron 独立 closed-form,no mixing"。
5. **mix_init=0.10** 起步:让 backbone 先 fit,graph term 慢慢长起来(同 PDNA α=0.01 模式)。

## 三、单元测试(`tests/test_topological_cfc.py`,20 tests,全部通过)

| 类别 | 测试数 | 验证 |
|---|---|---|
| 1. shape | 2 | (B, input) → (B, hidden) |
| 2. mix strength | 3 | 初始化 sigmoid(0.10) ✓,learnable param, mix=0 时 out=per-neuron h̃(atol=1e-4) |
| 3. graph topology | 5 | 无 self-loop, indices shape 正确, k clamp 到 H, learnable weights init=1/k, indices 在 state_dict |
| 4. edge cases | 2 | graph_k=0 / graph_k=H 时跳过邻接 |
| 5. gradient flow | 2 | backprop 通过 f/g/h/τ/adj_weights/mix_logit; mix=1 时梯度仍能传 |
| 6. misc | 6 | 0/0 输入稳定, network wrapper 工作, dt (B,1) broadcast, seed 确定性, conservation |

```
$ .venv312/bin/python -m pytest tests/test_topological_cfc.py -v
============================== 20 passed in 1.86s ==============================
```

## 四、设计诚实预期

参考本仓 round 91-101 audit(orth/MoE/ecology 91-99 + smoothness/diversity 95-101)模式:**神经元级结构替换(layer-wise → neuron-wise)在 1D toy 任务上不太可能战胜 baseline**。原因:
- CfC layer-wise matmul 在 toy_sin / structured / random 上已经是充分高效
- 把 H-D matmul 替换为 H 个独立 ODE + 稀疏 graph, 在低维小数据集上表达力通常 < 强耦合 baseline
- 91-101 系列已证明:大多数"正交"机制在 1D 玩具 honest-negative

**预期结果:**
- H1 ✓ shape / 接口与 CfC 兼容
- H2 ✓ 拓扑非平凡,不同 neuron 学习不同邻居
- H3 ✗ toy_sin 基线无显著改善(诚实预期)
- H4 (留作 honest candidate):与 FAME MoE 正交(后者 expert-粒度)

**应用域**(为何仍值得实现):
- 真实生产数据(EMMA rover, Henry Hub gas)有更高的 per-neuron 异质性
- 视觉域(D-3DGS, 3DGS deformation field)有 spatial topology 需求
- Quantum / Physics model(DynPMNN)有显式 neuron graph 建模先验

**下次 bench** (留作下一轮 push 后):
```python
# scripts/bench_topological_cfc.py
configs = [
    ('baseline_cfc',           CfcCell),
    ('topo_k4',                TopologicalCfCCell(graph_k=4)),
    ('topo_k8',                TopologicalCfCCell(graph_k=8)),
    ('topo_k_half_hidden',     TopologicalCfCCell(graph_k=hidden_size // 2)),
]
datasets = ['toy_sin', 'toy_structured', 'toy_random']
seeds = [0, 1, 2]
epochs = 100
# 主要看 test_mse 与 baseline 的差距
```

## 五、修改文件清单

| 文件 | 行数 | 状态 |
|---|---|---|
| `lnn/core/topological_cfc.py` | 211 | NEW |
| `tests/test_topological_cfc.py` | 207 | NEW |
| `pyrightconfig.json` | 12 | NEW (Pyright venv 修正) |
| `analysis/research/2026-07-17/2026-07-17_lnn_research_report.md` | 145 | NEW (研读方案) |
| `analysis/research/2026-07-17/2026-07-17_round299_topological_cfc.md` | 本文件 | NEW |
| `analysis/research/2026-07-17/2026-07-17_install_status.md` | 73 | NEW (opendataloader-pdf ✓ / UnlimitedOCR ✗) |

## 六、下一步(下一轮 round 300)

1. 跑 `scripts/bench_topological_cfc.py` 给出 H1-H4 实证;
2. 尝试 `git push origin master` (若网络恢复);
3. MEMORY 索引 round 299 摘要(这次手工 git 阻塞,记得标 ↓等下一轮手动 sync)。

— 2026-07-17, 1h /loop cycle #1
