---
title: "Round 285 — Predictability-Gated Pulse Amplitude: 研读报告"
date: 2026-07-12
round: 285
prd: "docs/prds/2026-07-12-lnn-round-285-predictability-gated-pulse-amplitude-a.md"
paper: "arXiv:2603.00153 (Sharma 2026-03, Pulse-Driven Neural Architecture) + r280 blend gate (内部)"
status: "in_progress"
parent: "r284 pulse-augmented gated liquid τ"
---

# Round 285 — Predictability-Gated Pulse Amplitude

## 1. 研究动机（继承 r284）

**r284 实证结论**（2026-07-11）：
- pulse_sin 在 **structured** 上 Δ%=-19.5%（clean MSE）+ 6× 更鲁棒（gap_ratio 61 vs 368），是 *paper claim 结构 > 容量* 的正例。
- 但在 **random** 上 Δ%=+44.6%，learned amplitude 从 0.10 涨到 0.40（4×），pulse 把随机误差注入成了周期信号 —— **破坏了 r278-r280 参数无关门线 "不能追噪声" 的核心承诺**。

**r284 报告自己的建议**（原文，2026-07-11）：
> "The pulse chases noise because its amplitude is a free parameter that
> sees erratic input. The r278–r280 predictability gate already produces
> a per-step scalar `g_t ∈ (0,1]` that collapses on noise. **Gate the pulse
> amplitude by the same predictability gate** — `pulse_i = g_t · A_i ·
> sin(...)` — so the endogenous drive is suppressed exactly when the input
> is erratic (restoring noise safety) but active on predictable/gappy data
> (keeping the robustness)."

**r285 的核心问题**：gating pulse amplitude by `g_t ∈ (0,1]`（来自 r280 blend gate 的 per-step predictability score）能否同时
- (a) 保留 structured 的 gap-robustness（H3 2/3 datasets）
- (b) 恢复 noise safety（random Δ% ≤ +5%，H2）
- (c) 不引入新超参数（zero new knobs）

## 2. 理论预期

为什么 gating 应该 work？

**信息论角度**：paper 把 pulse 看作 "endogenous rhythm carries state through gaps"。但如果 pulse 是 *unconditional*（始终 drive），它就在 noise 上也 drive —— 等同于一个无监督的周期信号 source 在拟合误差。`g_t` 已经是 input predictability 的无参数 estimator（`g_t = max(exp(-β·vol_velocity), exp(-β·vol_acceleration))`），它 *已经* 学到了 input 当下是否 reliable：
- structured + gap → input 大部分可靠，g_t → 1（pulse 全开），通过 gap 的少数步由 endogenous rhythm 补偿
- random → input 不可靠，g_t → 0（pulse 关），等价于 r280 base cell

**实现零超参**：直接用 r280 已经算好的 `gate` 张量（per-step scalar, shape `(B,1)`）乘到 pulse term 上，不需要新参数、不需要新 loss、不需要新 schedule。

## 3. 论文延伸阅读（arXiv:2603.00153 关键摘录）

> "A learnable oscillatory pulse `A·sin(ω·t + φ(h))` is added to the
> hidden state so the network keeps a temporal rhythm even when input
> is erratic or absent. The headline control: a non-oscillatory
> perturbation of equal magnitude gives no benefit — the temporal
> STRUCTURE of the pulse is what matters, not added capacity."

r284 已经 replicate 了 structure > capacity claim（H5 on structured：sin 3.5× more gap-robust than noise control）。

论文没有讨论 *input-adaptive pulse gating*（一直是 unconditional pulse），所以 r285 是一个 *combining 内部 innovation (r280 gate) with 外部 paper claim (r284 pulse)* 的 paper-extension 工作。

## 4. 设计方案（PRD #10-126）

**类 `PredictabilityGatedPulseCfCCell`**，subclass `PulseGatedLiquidTauCfCCell`，仅修改 `_pulse_term` 签名：

```python
def _pulse_term(self, t, T, h, noise_drive, gate=None):
    """Return the (B, d_h) pulse contribution for timestep ``t``.

    If `gate` (B,1) is provided, the pulse amplitude is multiplied by
    g_t so the endogenous drive is suppressed exactly when the input
    is erratic (the r280 predictability score).
    """
    if self.pulse_strength == 0.0:
        return torch.zeros_like(h)
    amp = self.pulse_amp.unsqueeze(0)  # (1, d_h)
    if self.pulse_mode == "noise":
        out = self.pulse_strength * amp * noise_drive[t].unsqueeze(0) / math.sqrt(2.0)
    else:
        t_norm = t / max(T, 1)
        phase = self.pulse_phase0.unsqueeze(0)
        if self.state_phase:
            phase = phase + self.pulse_phase_proj(h)
        angle = 2.0 * math.pi * self.pulse_omega.unsqueeze(0) * t_norm + phase
        out = self.pulse_strength * amp * torch.sin(angle)
    if gate is not None:
        out = gate * out   # ← THE FIX: zero new parameters
    return out
```

**forward 修改**：在算完 `gate = max(g_vel, g_acc)` 之后，把 `gate` 传给 `_pulse_term(t, T, h, noise_drive, gate=gate)`。backward 自动通过 `gate` 流入 pulse 梯度。

**严格 superset**：
- `gate=1.0`（即关掉 gating，pulse 全开）= r284 行为
- `gate=None`（不传）= r284 行为
- pulse_strength=0 = r280 行为

**新增构造选项** `gate_pulse: bool = True`，默认 True 是 r285 提案；`gate_pulse=False` 即为 r284 对照。

## 5. 假设（H1-H5）

- **H1（headline）**：在 structured 上 gated_pulse 的 gap_ratio 保持 ≤ r284 pulse_sin（gate 没有破坏 robustness）
- **H2（safety, THE FIX）**：在 random 上 gated_pulse Δ% ≤ +5% vs r280 blend（gate 抑制了 noise 上的 amplitude 增长）
- **H3（结构保真）**：在 random 上 gated_pulse 的 learned amplitude 不再单调增长（与 r284 0.10→0.40 对照），说明 *pulse 不再被 noise 训练*
- **H4（superset）**：`gate_pulse=False` 行为 = r284 bit-for-bit（unit test）
- **H5（gating not just clamping）**：在 structured + gap 条件下，gate g_t 不崩塌到 0（仍然接近 1），所以 pulse 实际生效；如果 gating 只是 clamp 到 0，gap_ratio 会等于 r280（无提升）

## 6. 风险与诚实预期

- **风险 A**：如果 `g_t` 在 structured + gap 条件下也跌到接近 0（H5 失败），gating 就只是 *pulse clamp 到 0*，等价于 r280，此时 r285 是 HONEST NEGATIVE（gating 太激进）。
- **风险 B**：gate 乘到 pulse 会同时 gate 掉 `pulse_phase_proj(h)` 的 state→phase 投影（因为 phase 已经被 gate 乘 out）—— 这是 *feature* 不是 bug：noise 时 phase 也应该关掉，避免 state-dependent phase 把噪声放大。
- **风险 C**：gating 可能在 toy_sin 上 vs r284 微小退化（r284 toy_sin 是 init-noise 主导，无法判断），但 gap_ratio 在 toy_sin 上仍然应该 ≥ r280 blend 的 8.5× 优势（H1 的真正信号）。

## 7. 与既有研究的连接

- **r278 velocity gate** (strict-positive)：gating τ by predictability
- **r279 acceleration gate** (strict-positive)：同上 but on accel
- **r280 blend gate** (strict-positive, all-round)：max(vel, accel)
- **r281 mixed-regime gate**：gate 在 mixed 条件下的 transfer
- **r282 henry-hub gate**：真实数据集验证
- **r283 multiseries gate**：多序列 batch 验证
- **r284 pulse-augmented** (target-dep, +1 TD)：endogenous oscillatory drive
- **r285 gated-pulse**（本轮）：gating r284 pulse by r280 gate

r285 是 **22-layer LNN+MoE stack** 之外的 **liquid-τ line** 第 8 个 round，是 round 76-284 line 的直接延续。如果 H2 成立，r284 从 +1 TD 升级为 +1 SP（strictly-positive），机制 map 从 71/31/62 变成 72/31/62。

## 8. 实验设计（bench 草案）

24 cells = 4 modes × 3 datasets × 2 seeds：
- modes: `static_tau`, `blend_gated`（r280）, `pulse_sin`（r284）, `gated_pulse_sin`（r285）
- datasets: toy_sin, structured, random
- seeds: 2

每个 cell：
- 训练：hidden=128, T=48, 50 epochs, dense
- 评估：clean MSE + gap p=0.3 MSE + gap_ratio

report: `docs/research/2026-07-12_round285_predictability_gated_pulse_report.md`

## 9. 验收标准

- H1 ✓：structured gap_ratio (gated_pulse) ≤ 61（r284 的水平），且 ≤ blend (368)
- H2 ✓：random Δ% ≤ +5%（即随机数据集 noise safety 恢复）
- H3 ✓：random 上 learned `pulse_amp.abs().mean()` 终值 ≤ 0.20（vs r284 0.40）
- H4 ✓：unit test `gate_pulse=False` 输出 ≡ PulseGatedLiquidTauCfCCell 输出（bit-equal within float tolerance）
- H5 ✓：structured + gap 训练后，gate 的 `.mean() ≥ 0.5`（pulse 仍然生效）

如果 H1+H2+H3 都过：r285 是 **+1 SP** (strictly-positive, no caveat)，替换 r284 的 +1 TD。
如果 H1 ✓ 但 H2 ✗：HONEST NEGATIVE（gating 不够强，需更激进的 gate，如 EMA-smoothed g_t）。
如果 H1 ✗：HONEST NEGATIVE（gating 太激进，关掉了 structured 的 robustness）。

## 10. 实现难度

**M**（2-3h）。~50 LOC cell delta on top of r284 + ~12 unit tests + ~150 LOC bench delta（仅在 r284 bench 上加 1 个 mode 列）。Toy grid 24 cells，与 r284 一致。

## 11. 引用

- Sharma, P. (2026-03). *Pulse-Driven Neural Architecture: Learnable Oscillatory Dynamics for Robust Continuous-Time Sequence Processing*. arXiv:2603.00153.
- Hasani, R., et al. (2021). *Closed-form continuous-depth models*. Nature Machine Intelligence. (CfC 原始)
- Lechner, M., et al. (2020). *Designing worm-inspired neural networks for interpretable robotic control*. (LTC 原始)
- r280 blend gate internal report: `docs/research/2026-07-03_round280_blend_gated_liquid_tau_report.md`
- r284 pulse gate internal report: `docs/research/2026-07-11_round284_pulse_gated_liquid_tau_report.md`