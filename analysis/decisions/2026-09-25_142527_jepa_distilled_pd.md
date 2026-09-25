# JEPA distillation (pd) — 2026-09-25_142527

Target source: **pd** (16000 states)
Epochs: 50, batch: 128, lr: 0.001

Initial MSE: **1.22810**
Final MSE:   **1.07292**
Improvement: **1.14x**

Next step: ``python scripts/bench_jepa_e2e.py --checkpoint <ckpt>`` to confirm reach_rate > 0.
