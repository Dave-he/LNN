# JEPA distillation (pd) — 2026-09-25_142435

Target source: **pd** (16000 states)
Epochs: 30, batch: 128, lr: 0.003

Initial MSE: **1.24048**
Final MSE:   **1.21783**
Improvement: **1.02x**

Next step: ``python scripts/bench_jepa_e2e.py --checkpoint <ckpt>`` to confirm reach_rate > 0.
