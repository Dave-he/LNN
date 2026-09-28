#!/usr/bin/env python3
"""Batch-append a 本仓具体实现路径 section to every report in docs/reports/ that doesn't have it. Idempotent: skips files that already have the section."""

from __future__ import annotations
import pathlib, datetime as dt

ROOT = pathlib.Path("/mnt/ssd/codespace/research/LNN")
REPORTS = ROOT / "docs" / "reports"
MARKER = "## 本仓具体实现路径"

# Universal footer that all templates include
FOOTER_TEMPLATE = """
**合规边界** (沿用 2026-06-09 用户偏好 critical 级 + AGENTS §约束):
- 仅合成数据 (`lnn/data/<synth>.py` 新建), 不接真机 / ROS / CAN / Modbus / mavlink / BMS / 真实电网
- 仅 in-house 模型 (基于本仓 `lnn/core/` 现有 ODE / CfC / LTC / 守恒 / 蒸馏栈)
- 任何负结果 (rollout fold / F1 < baseline / 长尾塌缩) → 进 `analysis/negative_results/` 而非默认报告
- 严禁触碰 8 条不可重复 claim ([[AGENTS]] §约束), 严禁宣称"AGI / 意识 / SOTA 横扫"

**维护说明**:
- 本实现路径段为 **standardized 模板**, grounding 到本报告核心方法论
- 实施时需按本报告 grounding 数字调整 λ, hidden, solver, seed 等超参
- 一旦实验落地, 把落地结果附在 `analysis/<新域>/<日期>_results.md` 并在本段维护交叉引用
"""


def classify(path: pathlib.Path) -> str:
    name = path.name.lower()
    # Order matters: more specific patterns first
    if "mt-lnn" in name or "awareliquid" in name:
        return "mtlnn"
    if "multi-rate" in name or "moe" in name or "mr_moe" in name or "mr-mo" in name:
        return "multi_rate"
    if "melo" in name:
        return "melotune"
    if "liquidtad" in name or "action_detection" in name or "2604.18274" in name:
        return "liquidtad"
    if "3dgs" in name or "deformation" in name:
        return "3dgs"
    if "sde" in name or "stochastic" in name or "2608.28702" in name:
        return "sde"
    if "mdn" in name:
        return "mdn"
    if "cfc" in name and "transferability" in name:
        return "cfc_transfer"
    if "cfc" in name or "closed-form" in name or "lechner" in name:
        return "cfc"
    if "ltc" in name or "time-constant" in name:
        return "ltc"
    if "lfm" in name or "liquidai" in name:
        return "lfm"
    if "ncp" in name or "neural_circuit" in name or "2609.10715" in name:
        return "ncp"
    if "sncp" in name or "crowdnav" in name:
        return "sncp"
    if "eeg" in name or "emotion" in name:
        return "eeg"
    if "fall" in name or "2607.12909" in name:
        return "fall"
    if "turbofan" in name or "2607.01986" in name:
        return "turbofan"
    if "transformer" in name or "temperature" in name or "2608.13260" in name:
        return "thermal"
    if "natural_gas" in name or "2604.24788" in name:
        return "natural_gas"
    if "lrfm" in name or "2606.15571" in name or "random_feature" in name or "pde" in name:
        return "lrfm"
    if "topological" in name or "2606.21295" in name:
        return "topological"
    if "loihi" in name:
        return "loihi"
    if "mdh" in name or "imitation" in name:
        return "mdh"
    if "antenna" in name or "beamforming" in name or "2604.07219" in name:
        return "antenna"
    if "conservation" in name or "2609.19674" in name:
        return "conservation"
    if "nsfd" in name or "structure_preserving" in name or "2607.10858" in name:
        return "nsfd"
    if "lss-ltcnet" in name or "foot_ulcer" in name or "2603.00459" in name:
        return "lss_ltcnet"
    if "bc_error" in name or "2604.14484" in name:
        return "bc_error"
    if "emma" in name or "2605.24047" in name:
        return "emma"
    if "lfnet" in name or "2606.26849" in name:
        return "lfnet"
    if "flowfake" in name or "2606.19579" in name:
        return "flowfake"
    if "gazelnn" in name or "2606.20491" in name:
        return "gazelnn"
    if "plan" in name or "2608.03041" in name:
        return "plan"
    if "tfp" in name or "2607.08283" in name:
        return "tfp"
    if "midpoint" in name:
        return "midpoint"
    if "ste" in name:
        return "ste"
    if "pdna" in name or "pulse-driven" in name:
        return "pdna"
    if "alpha_capacity" in name or "negative_result" in name:
        return "negative"
    if "dlnet" in name or "2601.06227" in name:
        return "dlnet"
    if "l4_liquid_s4" in name or "liquid_s4" in name:
        return "liquid_s4"
    if "comparative" in name or "lstm" in name or "rnn" in name or "2605.27467" in name:
        return "comparative"
    if "entro" in name:
        return "entrolnn"
    if "aegis" in name or "tvd-hl-ssm" in name or "2604.02149" in name:
        return "aegis"
    if "gcn" in name or "gcn-cfc" in name:
        return "gcn_cfc"
    if "training_paradigm" in name or "retention" in name or "mathematical_foundations" in name or "family_taxonomy" in name or "executive_summary" in name or "pipeline_quick_start" in name:
        return "survey"
    if "训练方向" in name:
        return "training_direction"
    if "round" in name or "loop_iteration" in name or "research_brief" in name or "research_digest" in name or "lnn_research_digest" in name:
        return "round_or_digest"
    return "generic"


TEMPLATES = {
    "cfc": """### 适配度
- **高**: 本仓 CfC 实现路径 (`lnn/core/closed_form_continuous_cell.py` 等同类) 是本论文落地基础
- **可与 r301-r305 (PLAN/STE/Midpoint 平行化) 合并**

### 实施步骤
1. **数据生成器** (`lnn/data/cfc_xxx_synth.py`):
   - Irregular Δt 序列 + 输入 mask (本仓 `lnn/data/timeseries.py` 已有基础)
   - 边界: 仅合成数据
2. **模型** (在 `lnn/core/` 新建 `cfc_xxx.py`):
   - 复用 CfC closed-form + gating 结构, 在 forward 加论文特定归纳偏置
3. **实验队列** (`analysis/cfc_xxx/`):
   - 5 seed × 3 regime = 15 run
   - 对照: vanilla CfC / LTC / Transformer
4. **诚实负结果预防**: 若论文主张的归纳偏置反而劣化 CfC baseline → 进 `analysis/negative_results/`""",
    "ltc": """### 适配度
- **高**: 本仓 LTC 基础在 `lnn/core/liquid_time_constant.py` 系列
- 与 CfC 对比维度: 求解器开销 / 闭式可微 / 训练稳定性

### 实施步骤
1. **数据生成器** (`lnn/data/ltc_xxx_synth.py`): 沿用 `lnn/data/timeseries.py`
2. **模型**: 在 `lnn/core/ltc_xxx.py` 套用本仓 LTC backbone
3. **实验** (`analysis/ltc_xxx/`): 5 seed × 多 regime, 与 CfC head-to-head
4. **诚实负结果预防**: 求解器慢 / gradient 爆炸 → 进 negative_results""",
    "lfm": """### 适配度
- **高**: 本仓 `projects/lfm25_orin_nano_smoke/` 与 `lnn/lfm2/` 是落地基础
- 已在 r304 (LFM2.5 + Parallel CfC Integration) 完成整合

### 实施步骤
1. **数据**: 走 LFM2.5 自带 tokenizer + WikiText-103 / SlimPajama 抽样 (本仓 `lnn/data/` 已有 generator)
2. **模型**: 沿用 `lnn/lfm2/parallel_integration.py` + `parallel_cfC` gate, 加本论文特定 adapter
4. **Jetson 实测**: `scripts/jetson_lnn_benchmark.py` 已有 LFM2.5 smoke baseline
5. **合规边界**: Jetson Orin Nano 已部署 (非真机 binder / 工业控制)
6. **诚实负结果预防**: PPL 比 baseline > +5% → 进 negative_results""",
    "sde": """### 适配度
- **中**: SDE 化需在 CfC/LTC 上加随机项, 已有 r306 探索 (进 negative-results)

### 实施步骤
1. **数据**: 沿用本仓 irregular Δt 数据集 (`lnn/data/timeseries.py`)
2. **模型** (`lnn/core/sde_cfc.py`): 在 CfC state 更新上加 Brownian noise 项 + Langevin 校正
4. **实验** (`analysis/sde_cfc/`): λ noise ∈ {0, 0.01, 0.1}, 与 r306 一致
5. **诚实负结果预防**: r306 已验证 toy benchmark 噪声略有害 → 本论文若声称"鲁棒性显著提升", 必须有 r306 全套实验对照""",
    "mdn": """### 适配度
- **低-中**: MDN 输出分布与本仓 `lnn/core/distribution_augmented_training` 协同

### 实施步骤
1. **数据**: 沿用 `lnn/data/multimodal.py` (mixture density 风格)
2. **模型** (`lnn/core/mdn_cfc.py`): CfC backbone + MDN head (Gaussian mixture)
3. **实验** (`analysis/mdn_cfc/`): r307 已记录 toy benchmark 复现失败 → 本论文必须先复现 r307 baseline
4. **诚实负结果预防**: toy_sin / random_irr 必须重做 baseline, 不能跳过""",
    "3dgs": """### 适配度
- **中**: 3DGS 是视觉应用, 本仓视觉方向落地在 `analysis/multimodal/`

### 实施步骤
1. **数据**: 合成 3D Gaussian 序列 (`lnn/data/multimodal_physreg.py` 已有 frame stacking)
2. **模型** (`lnn/core/cfc_deformation_field.py`): CfC backbone per Gaussian attribute
3. **实验** (`analysis/3dgs_cfc/`): 与 vanilla MLP deformation field 对照
4. **诚实负结果预防**: PSNR 与 MLP baseline 比 < +0.5 dB 在方差内 → 进 negative_results (r306 经验)""",
    "eeg": """### 适配度
- **高**: 本仓 EEG 长尾应用已落地 (`analysis/multimodal/` + `lnn/data/multimodal.py`)

### 实施步骤
1. **数据**: 合成 EEG 双时间尺度信号 (theta + gamma bands), 边界: 不接真机 EEG 设备
2. **模型** (`lnn/core/dual_timescale_cfc.py`): 双分支 CfC, 慢分支 θ, 快分支 γ, late fusion
3. **实验** (`analysis/dual_timescale_eeg/`): 5 seed × 3 noise regime, 与单尺度 CfC 对照
4. **合规边界**: 合成 EEG 数据, 严禁触真实 EEG 设备""",
    "fall": """### 适配度
- **高**: 本仓 `lnn/data/robotics.py` 已有物理引导数据生成器

### 实施步骤
1. **数据**: 合成 fall-detection 时序 (本仓 `lnn/data/timeseries.py` + 物理引导)
2. **模型** (`lnn/core/fall_dual_ltc.py`): 双 LTC, 边缘可部署 (hidden=8-16)
3. **实验** (`analysis/fall_dual_ltc/`): 5 seed × 3 noise level, on-device latency benchmark
4. **合规边界**: 合成数据, 严禁触真实监护设备""",
    "turbofan": """### 适配度
- **高**: C-MAPSS 是公开 benchmark, 本仓 `lnn/data/` 已有时间序列基础设施

### 实施步骤
1. **数据**: 沿用公开 NASA C-MAPSS 数据集 (`lnn/data/timeseries.py` + public loader)
2. **模型** (`lnn/core/liquid_latent_state.py`): CfC backbone + 退化潜在动态
3. **实验** (`analysis/turbofan_liquid/`): 5 seed, 与 LSTM / Transformer baseline 对照
4. **诚实负结果预防**: RMSE 比 LSTM 差 > 5% → 进 negative_results""",
    "thermal": """### 适配度
- **高**: 与本仓 DynPMNN 共享物理骨架

### 实施步骤
1. **数据** (`lnn/data/transformer_thermal_synth.py`): 合成变压器热模型 (油温 / 热点 / 功率曲线), 边界: 不接真实电网
2. **模型** (`lnn/core/physics_node_transformer.py`): 物理 backbone (对流 + 焦耳) + Neural residual
3. **实验** (`analysis/transformer_thermal_node/`): 3 物理参数 × 5 seed, 与 IEC 60076 / MLP 对照
4. **诚实负结果预防**: 24h rollout RMSE > IEC 60076 → 进 negative_results""",
    "natural_gas": """### 适配度
- **高**: 本仓 `lnn/data/natural_gas_generator.py` 已有合成器

### 实施步骤
1. **数据**: 沿用 `lnn/data/natural_gas_generator.py` + 季节性 / 节假日扰动
2. **模型** (`lnn/core/liquid_gas.py`): LTC backbone, 7-30 天预测窗口
3. **实验** (`analysis/natural_gas_liquid/`): 5 seed, 与 Prophet / LSTM 对照""",
    "lrfm": """### 适配度
- **高**: 本仓 LRFM 实现路径在 `lnn/core/lrfm.py` 系列 + `analysis/lrfm/`

### 实施步骤
1. **数据**: 合成 PDE 解 (heat / Burgers / Allen-Cahn), 沿用 `lnn/data/physics.py`
2. **模型** (`lnn/core/lrfm_xxx.py`): Random feature + ODE 残差
4. **实验** (`analysis/lrfm_xxx/`): 3 PDE × 3 noise × 3 seed = 27 run
5. **诚实负结果预防**: 比 MLP 残差差 → 进 negative_results""",
    "topological": """### 适配度
- **中**: 与 NCP 拓扑视角同源, 本仓 `analysis/sncp_ppo_lite/` 是承接域

### 实施步骤
1. **数据**: 合成 sequence 数据 (本仓 `lnn/data/long_sequence.py`)
2. **模型** (`lnn/core/topological_dynamics.py`): 单神经元拓扑动力学 + 守恒约束
3. **实验** (`analysis/topological_dynamics/`): 5 seed × 多尺度""",
    "loihi": """### 适配度
- **低-中**: Loihi 是神经形态硬件, 本仓不接真实 neuromorphic chip

### 实施步骤
1. **数据**: 沿用现有时序数据
2. **模型**: 在 `lnn/core/` 加 Loihi-style 脉冲编码
3. **仿真**: 仅在 CPU/GPU 上跑仿真, 不接 Intel Loihi 真机""",
    "mdh": """### 适配度
- **中**: 模仿学习, 本仓 `analysis/sncp_ppo_lite/` 是承接域

### 实施步骤
1. **数据**: 合成模仿学习轨迹 (`lnn/data/robotics.py`)
2. **模型** (`lnn/core/mdh_liquid.py`): MDH 蒸馏到 LTC backbone
3. **实验** (`analysis/mdh_liquid/`): 与 BC / 决策 Transformer 对照""",
    "antenna": """### 适配度
- **中**: 通信应用, 本仓 `analysis/multimodal/` 是承接域

### 实施步骤
1. **数据**: 合成 antenna pattern + beamforming input
2. **模型** (`lnn/core/liquid_antenna.py`): CfC backbone
3. **实验** (`analysis/antenna_liquid/`): 与 MLP 对照""",
    "conservation": """### 适配度
- **中**: 守恒约束 ODE, 与本仓 `lnn/core/conservation_*` 协同

### 实施步骤
1. **数据**: 合成 ODE 系统 (Lorenz, Van der Pol) with conservation penalty
2. **模型** (`lnn/core/conservation_lnn.py`): 加 symplectic / 守恒正则
3. **实验** (`analysis/conservation_lnn/`): 5 seed × 3 ODE system""",
    "nsfd": """### 适配度
- **高**: 结构保持 ODE, 本仓 LRFM / Physics-Modeled 路径可承接

### 实施步骤
1. **数据**: 合成 ODE 系统 (含 stiffness), 沿用 `lnn/data/physics.py`
2. **模型** (`lnn/core/nsfd_lnn.py`): NSFD 离散化 + 神经修正
3. **实验** (`analysis/nsfd_lnn/`): 比 vanilla RK4 / explicit Euler""",
    "lss_ltcnet": """### 适配度
- **中**: 医学图像分割 + LTC, 本仓 multimodal 域可承接

### 实施步骤
1. **数据**: 合成 mask refinement 数据 (随机 noise + ground truth mask)
2. **模型** (`lnn/core/lss_ltcnet.py`): U-Net backbone + LTC 序列精修
3. **实验** (`analysis/lss_ltcnet/`): 与 vanilla U-Net 对照""",
    "bc_error": """### 适配度
- **低**: 行为克隆误差分析是综述类, 主要用于设计选择

### 实施步骤
1. **数据**: 沿用 `lnn/data/robotics.py` 模仿学习轨迹
2. **模型**: 不直接落地, 仅作为 IL 算法设计指南
3. **实验** (`analysis/bc_error/`): 与 baseline BC 的对比误差""",
    "emma": """### 适配度
- **高**: 多模态物理参数提取, 本仓 `analysis/multimodal_physreg/` 是承接域

### 实施步骤
1. **数据**: 合成多模态物理轨迹 (`lnn/data/multimodal_physreg.py`)
2. **模型** (`lnn/core/emma_liquid.py`): 多模态融合 + 物理 ODE backbone
3. **实验** (`analysis/emma_liquid/`): 与 MLP baseline 对照""",
    "lfnet": """### 适配度
- **中**: 显著性检测, 视觉方向

### 实施步骤
1. **数据**: 合成 / 公开 SOD 数据集
2. **模型** (`lnn/core/lfnet_liquid.py`): 多模态融合 + liquid attention
3. **实验** (`analysis/lfnet_liquid/`): 与 SOTA SOD 对照""",
    "flowfake": """### 适配度
- **中**: 音频 deepfake 检测, 跨域应用

### 实施步骤
1. **数据**: 合成音频 deepfake 数据 (语音特征 + 频谱)
2. **模型** (`lnn/core/flowfake_ltc.py`): LTC + 频谱 attention
3. **实验** (`analysis/flowfake/`): 5 seed, 跨 dataset generalization 测试""",
    "gazelnn": """### 适配度
- **高**: 时序预测, 本仓 `lnn/data/timeseries.py` 已有基础

### 实施步骤
1. **数据**: 合成 gaze scanpath 时序 (本仓已有基础)
2. **模型** (`lnn/core/gazelnn.py`): CfC + 边缘可部署 (hidden=8)
3. **实验** (`analysis/gazelnn/`): 5 seed × 3 noise, 边缘 latency benchmark""",
    "plan": """### 适配度
- **高**: 已在 r301-r302 落地 (本仓核心路径之一)

### 实施步骤
1. **数据**: FJSP 合成 (本仓 `lnn/data/` 已有 FJSP generator)
2. **模型**: 沿用 `lnn/core/parallel_cfc.py` + PLAN attention
3. **实验**: 与 vanilla attention 对照, 已在 r302 完成""",
    "tfp": """### 适配度
- **高**: 与本仓 `lnn/core/memory_fusion_cfc.py` 同源

### 实施步骤
1. **数据**: 沿用本仓时序数据
2. **模型** (`lnn/core/tfp_cfc.py`): TFP + CfC 融合
3. **实验** (`analysis/tfp_cfc/`): 与 vanilla VLA 对照""",
    "midpoint": """### 适配度
- **高**: 已在 r305 落地

### 实施步骤
1. **数据**: 沿用 `lnn/data/timeseries.py`
2. **模型**: 沿用 `lnn/core/midpoint_cfc.py` + 论文特定归纳偏置
3. **实验**: 已完成, 见 r305 report""",
    "ste": """### 适配度
- **高**: 已在 r303 落地

### 实施步骤
1. **数据**: 沿用 `lnn/data/timeseries.py`
2. **模型**: 沿用 `lnn/core/ste_cfc.py`
3. **实验**: 已完成, 见 r303 report""",
    "pdna": """### 适配度
- **高**: 已有 `analysis/pdna_lra/` 路径

### 实施步骤
1. **数据**: 合成脉冲序列 (本仓 `lnn/data/`)
2. **模型** (`lnn/core/pdna.py`): PDNA backbone + LRA
3. **实验** (`analysis/pdna_lra/`): 与 SNCP / NCP 对照""",
    "negative": """### 适配度
- **N/A**: 本报告本身是诚实负结果, **不得**作为实施路径入口

### 维护说明
- 本报告是基线对照, 任何后续 paper 实施前必须先复现本报告结论""",
    "dlnet": """### 适配度
- **中**: 蒸馏路径, 与本仓 `analysis/cfc_nad/` 协同

### 实施步骤
1. **数据**: 沿用 `lnn/data/timeseries.py`
2. **模型** (`lnn/core/dlnet.py`): Teacher-student 蒸馏到 CfC
3. **实验** (`analysis/dlnet/`): 5 seed × Pareto sweep""",
    "liquid_s4": """### 适配度
- **中**: Liquid-S4 是 LNN + SSM 混合, 与 r304 LFM2.5 整合路径同源

### 实施步骤
1. **数据**: 沿用本仓时序数据
2. **模型** (`lnn/core/liquid_s4.py`): Liquid + S4 mixer
3. **实验** (`analysis/liquid_s4/`): 与 vanilla S4 / LNN 对照""",
    "comparative": """### 适配度
- **中**: 对比研究, 主要作为方法论指引

### 实施步骤
1. **数据**: 沿用 `lnn/data/timeseries.py`
2. **模型**: 跑对比矩阵 (CfC / LTC / LSTM / Transformer / ODE-RNN)
3. **实验** (`analysis/lstm_vs_lnn/`): head-to-head 矩阵""",
    "entrolnn": """### 适配度
- **中**: 熵引导可变形 LNN

### 实施步骤
1. **数据**: 沿用本仓时序数据
2. **模型** (`lnn/core/entro_lnn.py`): LNN + 可变形 attention head
3. **实验** (`analysis/entro_lnn/`)""",
    "aegis": """### 适配度
- **低-中**: TVD-HL-SSM 是新一代 SSM

### 实施步骤
1. **数据**: 沿用 `lnn/data/long_sequence.py`
2. **模型** (`lnn/core/aegis_ssm.py`): TVD-HL SSM + liquid 适配
3. **实验** (`analysis/aegis_liquid/`): 与 Mamba / S4 对照""",
    "gcn_cfc": """### 适配度
- **高**: 本仓 `analysis/cfc_nad/` 是承接域

### 实施步骤
1. **数据**: 合成图数据
2. **模型** (`lnn/core/gcn_cfc.py`): GCN + CfC backbone
3. **实验** (`analysis/gcn_cfc/`)""",
    "cfc_transfer": """### 适配度
- **高**: 本仓 `lnn/core/cfc_*` 是 CfC 主体

### 实施步骤
1. **数据**: 沿用 `lnn/data/timeseries.py`
2. **模型**: 复用 CfC + 跨 regime 蒸馏
3. **实验** (`analysis/cfc_transfer/`)""",
    "multi_rate": """### 适配度
- **高**: 已在 r304 + LFM2.5 整合完成

### 实施步骤
1. **数据**: 沿用 `lnn/data/`
2. **模型**: Multi-Rate + MoE 蒸馏到 CfC backbone
3. **实验** (`analysis/mr_moe/`)""",
    "melotune": """### 适配度
- **中**: 音乐推荐 / 端侧 CfC

### 实施步骤
1. **数据**: 合成音乐特征时序
2. **模型** (`lnn/core/melotune.py`): CfC + 端侧推理
3. **实验** (`analysis/melotune/`)""",
    "liquidtad": """### 适配度
- **中**: 视频时序动作检测

### 实施步骤
1. **数据**: 公开 / 合成视频动作检测
2. **模型** (`lnn/core/liquidtad.py`): Parallel Liquid + 视频 backbone
3. **实验** (`analysis/liquidtad/`)""",
    "mtlnn": """### 适配度
- **高**: 本仓 `analysis/cfc_nad/` + MT-LNN 整合

### 实施步骤
1. **数据**: 沿用 WikiText-103 / Pile (本仓 `lnn/data/`)
2. **模型**: MT-LNN M-series + O-series attention-free
3. **诚实负结果预防** (AGENTS §约束 #4):
   - **必须** 先在 README/论文里查 retraction 段
   - **不得** 重述 consciousness / AGI / new path to general intelligence 论调
   - 引用 5 条 validated + 4 条 retracted 时严格区分""",
    "ncp": """### 适配度
- **高**: 本仓 `analysis/sncp_ppo_lite/` 是承接域

### 实施步骤
1. **数据**: 合成 NCP 控制轨迹 (`lnn/data/robotics.py`)
2. **模型** (`lnn/core/ncp.py`): NCP backbone + 边缘可部署
3. **实验** (`analysis/ncp/`)""",
    "sncp": """### 适配度
- **高**: 本仓核心路径 (`analysis/sncp_ppo_lite/`)

### 实施步骤
1. **数据**: 沿用 `lnn/data/robotics.py`
2. **模型** (`lnn/core/sncp_ppo_lite.py`): SNCP-PPO actor
3. **实验** (`analysis/sncp_ppo_lite/`): 已落地, 见 r270+ report""",
    "survey": """### 适配度
- **N/A (综述类)**: 不直接实施, 仅作为方法论指引

### 维护说明
- 本报告是综合 survey / training paradigm / retention survey
- 实施时定位相应领域的 r* 报告作 entry point""",
    "training_direction": """### 适配度
- **N/A (训练方向指引类)**: 不直接实施, 给出多方向实施优先级

### 维护说明
- 本报告是训练方向可行报告 (机器人 / 边缘 / 长序列 / 物理 / 医疗金融 / 图时空)
- 实施时按本报告优先级排序进入各 `analysis/<方向>/`""",
    "round_or_digest": """### 适配度
- **N/A (round/digest 类)**: 本报告是 round 进展 / digest 综合

### 维护说明
- 本报告用于追踪迭代进展, 不直接作为实施入口
- 实施时定位具体 round / digest 引用的 paper 报告""",
    "generic": """### 适配度
- **待 grounding**: 本报告归类不明, 需先 grounding 到具体方法论

### 实施步骤
1. **数据**: 沿用 `lnn/data/` 现有合成器
2. **模型**: 在 `lnn/core/` 新建对应模块
3. **实验**: 进 `analysis/<新域>/`
4. **诚实负结果预防**: 任何 baseline 退化 → 进 negative_results""",
}


def build_section(report: pathlib.Path, category: str) -> str:
    template = TEMPLATES.get(category, TEMPLATES["generic"])
    today = dt.date.today().isoformat()
    return f"\n\n## 本仓具体实现路径 (in-house, 合成数据, 2026-09-28 批量化补丁)\n\n{template}{FOOTER_TEMPLATE}"


def main():
    patched = 0
    skipped = 0
    for report in sorted(REPORTS.glob("*.md")):
        text = report.read_text(encoding="utf-8")
        if MARKER in text:
            skipped += 1
            continue
        category = classify(report)
        section = build_section(report, category)
        report.write_text(text + section, encoding="utf-8")
        patched += 1
    print(f"Patched: {patched}, Skipped: {skipped}")


if __name__ == "__main__":
    main()