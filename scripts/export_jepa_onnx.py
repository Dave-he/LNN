#!/usr/bin/env python3
"""ONNX export for the JEPA decision policy.

Phase C.6 of PRD-MSD-LNN. Closes the docstring-debt noted in the
agent inventory: ``scripts/jetson_lnn_benchmark.py --export-trt` was
advertised but never implemented. This script provides a real
``torch.onnx.export`` path for the JEPA policy and validates the
exported model against the PyTorch reference.

The exported graph fuses:
* JEPAEncoder.forward_step (CfC closed-form, no solver)
* JEPAPolicyHead.forward (Linear over concat[z, obs])

Caveats for ONNX:
* JEPAPolicy.forward_inference uses the **skip-connection** head.
  ONNX supports concat natively, but the input to the head must be
  supplied as a single tensor — we route it through a torch.cat
  inside the policy. We verify by exporting and re-running.
* CfC cells are static (no ODE solver), so the graph is 100% static.
* For TensorRT conversion, run ``trtexec`` on the exported ONNX
  manually (not in this script).

Typical usage:
    python scripts/export_jepa_onnx.py --checkpoint ckpt.pt \\
        --out je_policy.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401 — required for tensor.numpy() under torch 2.2+
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.jepa_world_model import JEPAPolicy  # noqa: E402


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def export_to_onnx(policy: JEPAPolicy, obs_dim: int, out_path: Path) -> None:
    """Export JEPAPolicy.forward_inference to ONNX.

    The exported graph signature is:
        forward(obs_1xN) -> action_1xA

    We trace with a single example input (batch=1) but declare
    dynamic batch dim so the model can serve any batch size at
    inference time.
    """
    policy.eval()
    example = torch.randn(1, obs_dim)
    torch.onnx.export(
        policy,
        example,
        str(out_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )


def validate_onnx(policy: JEPAPolicy, onnx_path: Path, obs_dim: int) -> dict:
    """Run the ONNX model via onnxruntime and compare to PyTorch reference.

    Returns a dict of {max_abs_diff, mean_abs_diff} on a few random inputs.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        return {"error": f"onnxruntime not installed: {e}"}

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    diffs = []
    for _ in range(10):
        x = torch.randn(4, obs_dim)
        with torch.no_grad():
            y_pt = policy.forward_inference(x, deterministic=True).cpu().numpy()
        y_ort = sess.run(None, {"obs": x.cpu().numpy()})[0]
        diffs.append(float(abs(y_pt - y_ort).max()))
    return {
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "n_inputs_checked": 10,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("jepa_policy.onnx"))
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--skip-validation", action="store_true",
                        help="skip onnxruntime cross-check (faster)")
    args = parser.parse_args(argv)

    policy = JEPAPolicy(
        obs_dim=args.obs_dim, action_dim=args.action_dim,
        latent_dim=args.latent_dim, hidden_size=args.hidden_size,
        head_type=args.head_type, n_tau=args.n_tau,
    )
    policy.load_state_dict(torch.load(args.checkpoint))
    policy.eval()

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(policy, args.obs_dim, out_path)
    print(f"[export-onnx] wrote {out_path}")

    valid = {}
    if not args.skip_validation:
        valid = validate_onnx(policy, out_path, args.obs_dim)
        if "error" in valid:
            print(f"[export-onnx] validation skipped: {valid['error']}")
        else:
            print(
                f"[export-onnx] onnxruntime cross-check: max_abs_diff={valid['max_abs_diff']:.6f} "
                f"mean_abs_diff={valid['mean_abs_diff']:.6f}",
            )

    # Persist artefacts.
    stamp = _ts()
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}_jepa_onnx_export.json"
    md_path = out_dir / f"{stamp}_jepa_onnx_export.md"
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "onnx_path": str(out_path),
        "onnx_size_bytes": out_path.stat().st_size,
        "validation": valid,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md = [
        f"# JEPA ONNX export — {stamp}",
        "",
        f"* checkpoint: `{args.checkpoint.name}`",
        f"* exported to: `{out_path}`",
        f"* ONNX size: {out_path.stat().st_size} bytes",
        f"* validation: {valid}",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(f"[export-onnx] wrote {json_path}")
    print(f"[export-onnx] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())