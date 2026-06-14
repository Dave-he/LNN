> **兄弟项目**:[Dave-he/RoboticsResearch](https://github.com/Dave-he/RoboticsResearch) —— 通用机器人技术的同模式活知识库(本项目的范式扩展)

# LNN

PyTorch implementations, benchmarks, and research logs for Liquid Neural Networks.

`Dave-he/LNN` is a code-first research project. It contains a reusable Python
package under `lnn/` plus an auditable research archive under `docs/` and
`analysis/`. The package includes CfC, LTC, liquid neuron layers, continuous-time
variants, graph models, physics-informed models, multimodal models, and
noise-adaptive CfC backbones. The research archive records paper tracking,
ablation history, Jetson checks, and EMMA rover benchmark results.

Current package status: `0.1.0`. Core sequence models are covered by tests and
are intended for reuse. Timestamped research reports and analysis outputs are
evidence trails, not stable APIs.

## Install

```bash
git clone https://github.com/Dave-he/LNN.git
cd LNN

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
python -m pytest tests -q -m "not large_budget"
```

Run the large-budget EMMA rover tests explicitly when you want the slower
regime checks:

```bash
python -m pytest tests -q -m large_budget
```

## Minimal API Example

```python
import torch
from lnn import CfCNetwork

batch, steps, features = 8, 32, 3
x = torch.randn(batch, steps, features)
dt = torch.full((batch, steps, 1), 0.5)
mask = torch.ones(batch, steps, features)
mask[:, 10:12, :] = 0.0

model = CfCNetwork(
    input_size=features,
    hidden_size=32,
    output_size=1,
    return_sequences=False,
)

y = model(x, dt=dt, mask=mask)
print(y.shape)  # torch.Size([8, 1])
```

## Multi-Time-Scale CfC (n_tau ≥ 2)

`CfCCell` and `CfCNetwork` accept an optional `n_tau: int = 1` argument.  When
`n_tau == 1` (default) the cell is numerically equivalent to the legacy
single-τ path.  Setting `n_tau > 1` splits the hidden state into K independent
time-scale groups, each with its own τ, f_gate, g_branch, h_branch.  This is
the minimum-variance extension that aligns with the multi-τ pattern observed
in arXiv:2606.12240 (MR-MoE), arXiv:2606.11162 (COGENT),
arXiv:2606.07670 (Liquid-3DGS), and arXiv:2604.18274 (LiquidTAD).

```python
from lnn import CfCNetwork

# Three time-scale groups: τ ∈ {0.1, 1.0, 10.0}.
model = CfCNetwork(
    input_size=3,
    hidden_size=24,
    output_size=1,
    n_tau=3,
    tau_scales=(0.1, 1.0, 10.0),  # per-branch initial τ
)
y = model(x)  # hidden dim is split evenly across the 3 branches
```

Smoke-bench on toy sin/cos: `n_tau=3` reaches final MSE 0.0463 vs
`n_tau=1` 0.0535 (-13.4%, std 49% tighter) — see
`docs/research/2026-06-14_cfc_n_tau_sweep_report.md` and the unit
tests in `tests/test_cfc_n_tau.py`.

## Multi-Expert MR-MoE (n_experts ≥ 2)

`MRMoECfCCell` and `MRMoECfCNetwork` (PRD #10-24, round 77) wrap
K independent `CfCCell` experts behind a softmax router that produces
per-step mixture weights `g ∈ Δ^K`.  The cell output is
`Σ_k g_k · expert_k(x_t, h_prev)`.  This is the minimum-viable
implementation of the Multi-Rate Mixture-of-Experts pattern from
arXiv:2606.12240 (Zong et al., 2026) and the pattern-routed
heterogeneous-experts idea from arXiv:2606.13024 (CausalMoE, 2026-06-11).

```python
from lnn import MRMoECfCNetwork

# 3 CfC experts, each with n_tau=1 (combine with round 76 n_tau for
# multi-rate inside each expert: set n_tau_per_expert=3 for 3*3=9 effective τ groups).
model = MRMoECfCNetwork(
    input_size=3,
    hidden_size=24,
    output_size=1,
    n_experts=3,
    n_tau_per_expert=1,             # reuse round 76 n_tau per expert
    router_hidden=0,                # 0=linear router, >0=2-layer MLP
)
y = model(x)                       # [B, T, 1]
# Inspect router weights after a forward pass:
g = model.cells[0].last_g          # [B, K] softmax weights
```

Smoke-bench on toy sin/cos: `K=3` reaches final MSE 0.0364 vs
`K=1` 0.0525 (-30.7%, 2.3× the n_tau gain from round 76).  Router
entropy stays at ≈ log K after 30 epochs (no expert collapse).
See `docs/research/2026-06-14_mr_moe_cfc_sweep_report.md` and the
unit tests in `tests/test_mr_moe_cfc.py`.

## FAME-style Top-K Sparse MoE Routing

`ForecastabilityRouter` + `FAMECfCCell` + `FAMECfCNetwork` (PRD
#10-36, round 78) wrap the round 77 K-expert pool behind a **sparse
top-K router** that activates only `K'` of `K` experts per step.
This is the minimum-viable implementation of the
"cost-aware sparse router" from arXiv:2606.08896 (FAME, 2026-06-08)
— a production-deployed forecastability-aware MoE for retail
vending-machine sales (5000+ machines, 60M+ transactions, Top-2
routing −12.4% MSE vs LightGBM).

```python
from lnn import FAMECfCNetwork

# 3 CfC experts, top-2 sparse routing (FAME paper default).
# top_k=1 → argmax single expert; top_k=K → dense softmax (round 77).
model = FAMECfCNetwork(
    input_size=3,
    hidden_size=24,
    output_size=1,
    n_experts=3,
    top_k=2,                          # K' ∈ [1, K]
    n_tau_per_expert=1,               # round 76 n_tau per expert
)
y = model(x)                         # [B, T, 1]
# Inspect activated experts per step:
g = model.cells[0].last_g            # [B, K], exactly top_k nonzeros per row
idx = model.cells[0].last_top_idx    # [B, top_k] indices of activated experts
```

Smoke-bench on toy sin/cos: `top_k=2` reaches final MSE 0.0366
(≈ equal to `top_k=3` dense at 0.0364) but with **3.7× tighter
std** (0.0012 vs 0.0034) — the FAME paper's core "sparse is
more stable" claim, reproduced on a 1-seed-runner homely toy
setup.  Activated-experts-per-step is exactly `top_k` (sparsity
contract honoured).  See
`docs/research/2026-06-14_fame_cfc_sweep_report.md` and the unit
tests in `tests/test_fame_cfc.py`.

## Orthogonality Constraint (defence against expert collapse)

`orthogonality_loss` (PRD #10-37, round 80) implements the
"geometric orthogonality constraint that penalises representational
redundancy" idea from arXiv:2606.03631 (AnchorMoE, 2026-06-02).
Use it as an auxiliary loss to keep expert hidden states
decorrelated — the round 79 sweep showed that the K=3 top_k=1
(router-argmax single expert) cell is **unstable** without
orthogonality (0.76 ± 0.79 with 1/3 seeds diverging) but
**stable** with `lambda_coeff=0.001` (0.11 ± 0.05, 0/3 seeds
diverging).  This addresses the Causal Audit (arXiv:2606.10703)
warning that observational routing metrics do not predict expert
causal importance — the orthogonality constraint directly
intervenes in the expert representation space.

```python
from lnn import FAMECfCNetwork, orthogonality_loss

net = FAMECfCNetwork(input_size=3, hidden_size=24, output_size=1,
                    n_experts=3, top_k=1)  # unstable without orth
opt = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
for x, y in dataloader:
    opt.zero_grad()
    y_pred, expert_outs = net.forward_with_aux(x)
    task_loss = loss_fn(y_pred, y)
    # Use only the last step's expert outputs from the first layer.
    last_outs = expert_outs[0][-1]            # K × [B, H]
    aux = orthogonality_loss(last_outs, lambda_coeff=0.001)
    (task_loss + aux).backward()
    opt.step()
```

`net.forward(...)` (the back-compat path) still returns only the
prediction tensor; the orthogonality-aware `net.forward_with_aux(...)`
returns the nested `[num_layers][T][K]` expert outputs needed to
compute the auxiliary loss.  See
`docs/research/2026-06-14_orthogonality_report.md` and the unit
tests in `tests/test_orthogonality.py`.

## What Is Stable?

| Area | Path | Status |
|---|---|---|
| Core sequence models | `lnn/core/cfc.py`, `lnn/core/ltc.py`, `lnn/core/liquid_neuron.py` | Reusable, tested package code |
| Sequence utilities | `lnn/core/sequence_utils.py`, `lnn/core/trainer.py`, `lnn/data/timeseries.py` | Reusable helpers for experiments |
| Research backbones | `lnn/core/noise_adaptive_cfc.py`, `lnn/core/multimodal_physreg.py`, `lnn/core/dynpmnn.py`, `lnn/core/variants.py` | Tested, but APIs may change with new papers |
| Benchmarks and recipes | `scripts/`, `configs/` | Reproducibility entry points; CLI flags may evolve |
| Research archive | `docs/research/`, `docs/reports/`, `analysis/` | Timestamped evidence and iteration history |
| Knowledge workflow | `AGENTS.md`, `skills/` | Automation and paper-analysis workflow docs |

Use `lnn/` for library code. Use `docs/research/` and `analysis/` when you want
to inspect how a result was produced.

## Quick Paths

| Goal | Start here |
|---|---|
| Understand the current benchmark result | [LNN_TLDR.md](LNN_TLDR.md) |
| Learn LNN principles from zero | [docs/guides/LNN_PRINCIPLES_FOR_BEGINNERS.md](docs/guides/LNN_PRINCIPLES_FOR_BEGINNERS.md) |
| Reproduce the EMMA rover recipe | [LNN_QUICKSTART.md](LNN_QUICKSTART.md) |
| Compare supported model families | [LNN_MODEL_GUIDE.md](LNN_MODEL_GUIDE.md) |
| Audit LFM/LNN active-3B vs 30B+ LLM claims | [analysis/llm_battlecard/2026-06-04_llm_battlecard.md](analysis/llm_battlecard/2026-06-04_llm_battlecard.md) |
| Run the local LFM2.5 micro-eval | [analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md](analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_micro_eval.md) |
| Run the LFM2.5 HTTP endpoint micro-eval | [analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.md](analysis/llm_micro_eval/2026-06-04_lfm25_1_2b_instruct_q4_http_micro_eval.md) |
| Inspect the LFM2.5 DPO Q4 regression result | [analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.md](analysis/llm_micro_eval/2026-06-04_lfm25_dpo_s1_q4_micro_eval.md) |
| Inspect the LLM micro-eval leaderboard | [analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.md](analysis/llm_micro_eval/2026-06-04_llm_micro_leaderboard.md) |
| Read the product/research roadmap | [docs/PRD_LNN_Edge_Research.md](docs/PRD_LNN_Edge_Research.md) |
| Inspect multimodal design decisions | [docs/guides/LNN_MULTIMODAL_DESIGN.md](docs/guides/LNN_MULTIMODAL_DESIGN.md) |
| Understand the automation agents | [AGENTS.md](AGENTS.md) |

## Benchmark Snapshot

The current headline EMMA rover result is an adaptive freeze recipe using a
Bi-CfC-NAD style backbone:

```bash
python lnn/data/emma_rover_features.py
python scripts/benchmark_adaptive_freeze.py \
    --epochs 80 \
    --warmup-epochs 40 \
    --freeze-targets audio_only \
    --num-samples 200 \
    --hidden-size 64
```

The latest recorded run reports roughly `MSE ~= 0.31`, compared with a
`video_only` baseline around `0.87`. Treat this as an actively maintained
research benchmark; inspect `analysis/emma_rover/` and the linked research docs
for the full ablation trail.

For Jetson smoke checks:

```bash
RUN_BENCHMARK=1 COMMIT_AND_PUSH=0 ./scripts/run_daily_lnn_task.sh
python scripts/jetson_lnn_benchmark.py --quick --pareto
```

## Repository Layout

```text
LNN/
├── lnn/                  # Python package: models, datasets, utilities
├── tests/                # Unit and regime tests
├── scripts/              # Benchmarks, ablations, automation entry points
├── configs/              # Experiment configs
├── analysis/             # Generated benchmark outputs and plots
├── docs/                 # Research reports, PRD, living-review notes
├── papers/               # Paper tracking and archives
├── projects/             # External repo clones and reproduction work
└── skills/               # Vercel Skills-compatible research agents
```

This repository can also be opened as an Obsidian vault. The GitHub README is
intentionally shorter and code-first; the vault-style notes remain in `docs/`.

## Automation

Generate the daily LNN research digest without committing:

```bash
COMMIT_AND_PUSH=0 ./scripts/run_daily_lnn_task.sh
```

Install the local user-level systemd timer:

```bash
./scripts/install_daily_lnn_timer.sh
```

GitHub Actions also runs `.github/workflows/daily-lnn-research.yml` to generate
daily research summaries.

## Related Implementations

- [raminmh/liquid_time_constant_networks](https://github.com/raminmh/liquid_time_constant_networks)
- [raminmh/CfC](https://github.com/raminmh/CfC)
- [mlech26l/ncps](https://github.com/mlech26l/ncps)
- [emilierp/exact_lnn](https://github.com/emilierp/exact_lnn)
- [makramchahine/drone_causality](https://github.com/makramchahine/drone_causality)

## License

MIT. See [LICENSE](LICENSE).
