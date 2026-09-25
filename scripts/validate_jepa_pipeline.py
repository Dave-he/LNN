#!/usr/bin/env python3
"""End-to-end validation of the JEPA decision pipeline.

This is the "smoke test" script — runs the entire pipeline in one go:
  1. Latency floor (Phase A)
  2. PD expert reach_rate SLO (Phase B)
  3. JEPA world model training MSE (Phase C.1)
  4. PD distillation (Phase C.2)
  5. QAT-lite distillation (Phase C.4)
  6. FP32 vs INT8 reach_rate (Phase C.3/4)
  7. ONNX export + onnxruntime cross-check (Phase C.6)

Writes a single summary JSON + MD to
analysis/decisions/<ts>_jepa_pipeline_validation.{json,md}.

SLO gates (all must pass for the pipeline to be considered validated):
  * PD reach_rate n_ped=0        >= 0.95
  * World model final MSE        <  0.005
  * Distilled reach_rate n_ped=0 >= 0.85
  * QAT reach_rate n_ped=0       >= 0.85
  * INT8 reach_rate n_ped=0      >= 0.50
  * ONNX bit-exact max_abs_diff  <  1e-4
  * PyTorch eager p99 latency    <  5 ms
  * ONNX runtime latency        <  0.5 ms
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from lnn.core.jepa_world_model import JEPAPolicy
from scripts.bench_decision_e2e_latency import (
    BACKBONES,
    _resolve_device,
    _percentile,
    run_one_config,
)
from scripts.bench_pid_expert_demo import PDExpert, collect_demos
from scripts.experiment_sncp_ppo_lite import PointMassNavLite


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


@dataclass
class Gate:
    name: str
    passed: bool
    value: float
    threshold: float
    unit: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "notes": self.notes,
        }


def gate_ge(name: str, value: float, threshold: float, unit: str = "", notes: str = "") -> Gate:
    return Gate(name, value >= threshold, value, threshold, unit, notes)


def gate_le(name: str, value: float, threshold: float, unit: str = "", notes: str = "") -> Gate:
    return Gate(name, value <= threshold, value, threshold, unit, notes)


def measure_p99(backbone_name: str, hidden_size: int, seq_len: int, n_iters: int = 100) -> float:
    device = _resolve_device("jetson-cpu")
    r = run_one_config(backbone=backbone_name, hidden_size=hidden_size, seq_len=seq_len,
                       precision="fp32", seed=42, device=device, warmup=10, measure=n_iters)
    return r.p99_ms


def onnx_bit_exact_diff(checkpoint: Path, obs_dim: int, n_trials: int = 5) -> float:
    """Export to ONNX, run via onnxruntime, return max diff vs PyTorch."""
    policy = JEPAPolicy(obs_dim=obs_dim, action_dim=2, latent_dim=8,
                        hidden_size=16, head_type="mse", n_tau=4)
    policy.load_state_dict(torch.load(checkpoint))
    policy.eval()
    import numpy as np
    import onnx
    import onnxruntime as ort
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = Path(f.name)
    try:
        torch.onnx.export(
            policy, torch.randn(1, obs_dim), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=14, do_constant_folding=True,
        )
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        diffs = []
        for _ in range(n_trials):
            x = torch.randn(4, obs_dim)
            with torch.no_grad():
                y_pt = policy.forward_inference(x, deterministic=True).numpy()
            y_ort = sess.run(None, {"obs": x.numpy()})[0]
            diffs.append(float(abs(y_pt - y_ort).max()))
        return max(diffs)
    finally:
        onnx_path.unlink(missing_ok=True)


def onnx_runtime_latency(checkpoint: Path, obs_dim: int, n_iters: int = 200) -> float:
    policy = JEPAPolicy(obs_dim=obs_dim, action_dim=2, latent_dim=8,
                        hidden_size=16, head_type="mse", n_tau=4)
    policy.load_state_dict(torch.load(checkpoint))
    policy.eval()
    import numpy as np
    import onnxruntime as ort
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = Path(f.name)
    try:
        torch.onnx.export(
            policy, torch.randn(1, obs_dim), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=14, do_constant_folding=True,
        )
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = np.random.randn(1, obs_dim).astype(np.float32)
        for _ in range(20):
            sess.run(None, {"obs": x})
        t0 = time.perf_counter()
        for _ in range(n_iters):
            sess.run(None, {"obs": x})
        return (time.perf_counter() - t0) / n_iters * 1000.0
    finally:
        onnx_path.unlink(missing_ok=True)


def evaluate_reach_rate(checkpoint: Path, n_episodes: int = 30, n_pedestrians: int = 0) -> float:
    from scripts.bench_pid_expert_demo import PDExpert
    policy = JEPAPolicy(obs_dim=17, action_dim=2, latent_dim=8,
                        hidden_size=16, head_type="mse", n_tau=4)
    policy.load_state_dict(torch.load(checkpoint))
    policy.eval()
    n_reached = 0
    for ep in range(n_episodes):
        env = PointMassNavLite(seed=7 + ep, n_pedestrians=n_pedestrians)
        obs = PDExpert.augmented_obs(env)
        for _ in range(50):
            with torch.no_grad():
                a = policy.forward_inference(obs.unsqueeze(0), deterministic=True).squeeze(0)
            r = env.step(a)
            obs = PDExpert.augmented_obs(env)
            if r.done:
                if r.info.get("reached"):
                    n_reached += 1
                break
    return n_reached / max(n_episodes, 1)


def apply_int8_ptq(checkpoint_in: Path, checkpoint_out: Path, obs_dim: int = 17) -> None:
    """Quantize a JEPAPolicy state dict in-place and save."""
    from lnn.core.quantization import quantize_model_inplace
    policy = JEPAPolicy(obs_dim=obs_dim, action_dim=2, latent_dim=8,
                        hidden_size=16, head_type="mse", n_tau=4)
    policy.load_state_dict(torch.load(checkpoint_in))
    quantize_model_inplace(policy, per_channel=True)
    torch.save(policy.state_dict(), checkpoint_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="path to a pre-trained QAT ckpt (skip training)")
    parser.add_argument("--quick", action="store_true",
                        help="smaller training (faster pipeline)")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args(argv)

    import tempfile
    work_dir = REPO_ROOT / "checkpoints" / "validate"
    work_dir.mkdir(parents=True, exist_ok=True)

    n_demos = 200 if args.quick else 500
    n_epochs = 5 if args.quick else 30
    n_ped = 0
    print(f"[validate] pipeline (quick={args.quick})", flush=True)

    # 1. PD reach_rate SLO
    print("[validate] 1/7 PD reach_rate SLO...", flush=True)
    pd_demos = collect_demos(n_episodes=n_demos, n_pedestrians=n_ped,
                             max_steps_per_episode=80, seed=42, controller=PDExpert())
    pd_reach_rate = pd_demos["summary"]["reach_rate"]

    # 2. World model training MSE
    print("[validate] 2/7 World model training...", flush=True)
    import json as _json
    transitions = pd_demos["transitions"]  # list of (obs, action, next_obs)
    obs_t = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
    actions_t = torch.tensor([t[1] for t in transitions], dtype=torch.float32)
    next_obs_t = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    obs_dim = obs_t.shape[1]
    # Quick inline training
    torch.manual_seed(42)
    wm = JEPAPolicy(obs_dim=obs_dim, action_dim=2, latent_dim=8, hidden_size=16,
                    head_type="mse", n_tau=4)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    N = obs_t.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(N)
        for s in range(0, N, 128):
            idx = perm[s:s + 128]
            x = obs_t[idx].unsqueeze(1)
            y = next_obs_t[idx].unsqueeze(1)
            a = actions_t[idx].unsqueeze(1)
            out = wm.forward_train(x, y, a)
            opt.zero_grad()
            out["mse"].backward()
            opt.step()
    world_model_mse = float(wm.forward_train(
        obs_t[:128].unsqueeze(1), next_obs_t[:128].unsqueeze(1), actions_t[:128].unsqueeze(1)
    )["mse"].item())

    # 3. PD distillation
    print("[validate] 3/7 PD distillation...", flush=True)
    n_distill_epochs = 50 if args.quick else 80
    torch.manual_seed(42)
    distill = JEPAPolicy(obs_dim=obs_dim, action_dim=2, latent_dim=8, hidden_size=16,
                         head_type="mse", n_tau=4)
    opt = torch.optim.Adam(distill.parameters(), lr=1e-3)
    actions_tensor = torch.tensor([t[1] for t in pd_demos["transitions"]], dtype=torch.float32)
    obs_tensor = torch.tensor([t[0] for t in pd_demos["transitions"]], dtype=torch.float32)
    for _ in range(n_distill_epochs):
        perm = torch.randperm(N)
        for s in range(0, N, 128):
            idx = perm[s:s + 128]
            x = obs_tensor[idx]
            y = actions_tensor[idx]
            z = distill.encoder.encode_step(x)
            pred = distill.policy_head(z, x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    distill_ckpt = work_dir / "distill.pt"
    torch.save(distill.state_dict(), distill_ckpt)
    distill_reach = evaluate_reach_rate(distill_ckpt, n_episodes=20 if args.quick else 30)

    # 4. QAT-lite distillation
    print("[validate] 4/7 QAT distillation...", flush=True)
    from scripts.train_jepa_quant_aware import fake_quantize_inplace
    n_qat_epochs = 60 if args.quick else 100
    torch.manual_seed(42)
    qat = JEPAPolicy(obs_dim=obs_dim, action_dim=2, latent_dim=8, hidden_size=16,
                     head_type="mse", n_tau=4)
    opt = torch.optim.Adam(qat.parameters(), lr=1e-3)
    for epoch in range(n_qat_epochs):
        perm = torch.randperm(N)
        for s in range(0, N, 128):
            idx = perm[s:s + 128]
            x = obs_tensor[idx]
            y = actions_tensor[idx]
            z = qat.encoder.encode_step(x)
            pred = qat.policy_head(z, x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 3 == 0:
            fake_quantize_inplace(qat, num_bits=8)
    qat_ckpt = work_dir / "qat.pt"
    torch.save(qat.state_dict(), qat_ckpt)
    qat_reach = evaluate_reach_rate(qat_ckpt, n_episodes=20 if args.quick else 30)

    # 5. INT8 PTQ reach_rate on QAT ckpt
    print("[validate] 5/7 INT8 PTQ on QAT ckpt...", flush=True)
    int8_ckpt = work_dir / "qat_int8.pt"
    apply_int8_ptq(qat_ckpt, int8_ckpt, obs_dim=obs_dim)
    int8_reach = evaluate_reach_rate(int8_ckpt, n_episodes=20 if args.quick else 30)

    # 6. PyTorch eager p99 latency
    print("[validate] 6/7 PyTorch eager latency...", flush=True)
    pytorch_p99 = measure_p99("cfcs", hidden_size=16, seq_len=1, n_iters=200)

    # 7. ONNX (skip in quick or if --skip-onnx)
    onnx_diff = float("nan")
    onnx_latency = float("nan")
    if not args.skip_onnx:
        print("[validate] 7/7 ONNX export + onnxruntime...", flush=True)
        onnx_diff = onnx_bit_exact_diff(qat_ckpt, obs_dim, n_trials=5)
        onnx_latency = onnx_runtime_latency(qat_ckpt, obs_dim, n_iters=200)
    else:
        print("[validate] 7/7 ONNX skipped", flush=True)

    # Build gates.
    gates = [
        gate_ge("PD reach_rate n_ped=0", pd_reach_rate, 0.95, "", "PD expert demos."),
        gate_le("World model final MSE", world_model_mse, 0.005, "", "n_ped=0 demos."),
        gate_ge("Distilled PD reach_rate", distill_reach, 0.85, "", "Vanilla BC ckpt."),
        gate_ge("QAT reach_rate", qat_reach, 0.85, "", "QAT-lite ckpt."),
        gate_ge("INT8 reach_rate (QAT PTQ)", int8_reach, 0.50, "", "INT8 weight-only PTQ on QAT ckpt."),
        gate_le("PyTorch eager p99 latency", pytorch_p99, 5.0, "ms", "CfC h=16."),
        gate_le("ONNX max_abs_diff", onnx_diff, 1e-4, "", "PyTorch vs onnxruntime."),
        gate_le("ONNX runtime latency", onnx_latency, 0.5, "ms", "batch=1 inference."),
    ]

    all_passed = all(g.passed for g in gates)

    # Outputs.
    stamp = _ts()
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}_jepa_pipeline_validation.json"
    md_path = out_dir / f"{stamp}_jepa_pipeline_validation.md"
    payload = {
        "gates": [g.to_dict() for g in gates],
        "all_passed": all_passed,
        "ckpts": {
            "distill": str(distill_ckpt),
            "qat": str(qat_ckpt),
            "qat_int8": str(int8_ckpt),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))

    md = [
        f"# JEPA pipeline validation ({stamp})",
        "",
        f"Overall: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "| gate | value | threshold | unit | passed |",
        "|---|---|---|---|---|",
    ]
    for g in gates:
        mark = "OK" if g.passed else "X"
        unit = g.unit or ""
        thr = g.threshold if not unit else f"{g.threshold:g}"
        val = f"{g.value:.4f}" if not unit else f"{g.value:.4f}{unit}"
        md.append(f"| {g.name} | {val} | {thr}{unit} | {unit} | {mark} |")

    md_path.write_text("\n".join(md) + "\n")

    # Print summary.
    for g in gates:
        mark = "OK" if g.passed else "FAIL"
        print(f"[validate] {mark:4s} {g.name}: {g.value:.4f} (threshold {g.threshold})", flush=True)
    print(f"[validate] overall: {'PASS' if all_passed else 'FAIL'}", flush=True)
    print(f"[validate] wrote {json_path}", flush=True)
    print(f"[validate] wrote {md_path}", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())