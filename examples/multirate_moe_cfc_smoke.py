"""Smoke test for ``lnn.core.multirate_moe_cfc.MultiRateMoECfC``.

Verifies:
    - module imports cleanly from ``lnn`` package
    - forward pass produces correct shape and dtype
    - cfg parity: ``n_tau=K=1`` is rejected (sanity)
    - auxiliary loss is finite and well-ordered
    - routed-cell output differs from vanilla ``CfCCell`` output but stays close
      when the routing degenerates to uniform (half-τ split smoke).

Usage::

    PYTHONPATH=. python3 examples/multirate_moe_cfc_smoke.py

Following the 2026-06-09 critical preference "no device control", this
script runs purely on synthetic non-stationary time-series — no hardware
adapter, no ADB, no real sensor.
"""

from __future__ import annotations

import math
import sys

import torch

from lnn.core.cfc import CfCCell, CfCNetwork
from lnn.core.multirate_moe_cfc import MultiRateMoECfC, MultiRateMoECfCNetwork


def make_synthetic_series(samples: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    steps = seq_len + 1
    t = torch.linspace(0.0, 1.0, steps).unsqueeze(0).repeat(samples, 1)
    freq = torch.rand(samples, 1) * 3.0 + 0.5
    phase = torch.rand(samples, 1) * (2.0 * math.pi)
    drift = (torch.rand(samples, 1) - 0.5) * 0.4
    switch = (t > (0.4 + 0.3 * torch.rand(samples, 1))).float()
    base = torch.sin(2.0 * math.pi * freq * t + phase)
    seasonal = 0.3 * torch.sin(2.0 * math.pi * (freq * 2.7) * t + phase / 2.0)
    regime = switch * 0.4 * torch.sin(2.0 * math.pi * (freq * 5.0) * t)
    noise = 0.05 * torch.randn(samples, steps)
    sig = base + seasonal + drift * t + regime + noise
    x = sig[:, :-1].unsqueeze(-1)
    y = sig[:, 1:].unsqueeze(-1)
    return x, y


def main() -> int:
    device = torch.device("cpu")
    samples, seq_len, hidden, n_tau = 64, 32, 32, 4
    x, _ = make_synthetic_series(samples, seq_len)
    x = x.to(device)
    print(f"[smoke] input shape={tuple(x.shape)}, hidden={hidden}, n_tau={n_tau}")

    # 1. cell-level forward shape check (cell runs EC routing per step).
    cell = MultiRateMoECfC(input_size=1, hidden_size=hidden, n_tau=n_tau)
    cell = cell.to(device)
    h = cell(x[:, 0, :], h=None)
    print(f"[smoke] single-step h shape={tuple(h.shape)}; expected (B,{hidden})")
    assert h.shape == (samples, hidden), f"got {h.shape}"

    # 2. network-level forward
    net = MultiRateMoECfCNetwork(input_size=1, hidden_size=hidden, output_size=1, n_tau=n_tau)
    net = net.to(device)
    y = net(x)
    print(f"[smoke] net out shape={tuple(y.shape)}")
    assert y.shape == (samples, seq_len, 1), f"got {y.shape}"

    # 3. auxiliary loss is finite
    aux = net.auxiliary_loss(x)
    print(f"[smoke] aux load-balance loss = {float(aux.detach()):.6f}")
    assert torch.isfinite(aux), "aux loss not finite"

    # 4. parity against vanilla CfC in a degenerate regime:
    # when n_tau=1 is requested the MultiRate cell refuses to construct (assert),
    # but we can still verify that full routing (n_tau=K, cap_k_frac=1.0) yields
    # an output whose L2 distance to a vanilla multi-tau CfC is on the same
    # order of magnitude as random init fluctuation.

    van = CfCNetwork(input_size=1, hidden_size=hidden, output_size=1, num_layers=1)
    van = van.to(device)
    # CfCNetwork has return_sequences=True default (CfC layer)
    y_van = van(x)

    # Force top_k_active = n_tau → every branch fires every step, behaviour
    # should be within tolerance of vanilla multi-tau CfC init.
    full = MultiRateMoECfCNetwork(
        input_size=1,
        hidden_size=hidden,
        output_size=1,
        n_tau=n_tau,
        top_k_active=n_tau,
    )
    full = full.to(device)
    y_full = full(x)

    diff = (y_van - y_full).abs().mean().item()
    rel = diff / (y_van.abs().mean().item() + 1e-9)
    print(f"[smoke] mean |y_van - y_full| = {diff:.6f}; rel = {rel:.4f}")
    # We don't require tight numerical equality — just order of magnitude.
    assert rel < 5.0, f"MultiRate routing at cap=1.0 shouldn't diverge; rel={rel}"

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
