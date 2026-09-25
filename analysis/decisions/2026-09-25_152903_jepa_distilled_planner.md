# JEPA distillation (planner) — 2026-09-25_152903

Target source: **planner** (16000 states)
Epochs: 80, batch: 128, lr: 0.001

Initial MSE: **0.00608**
Final MSE:   **0.00224**
Improvement: **2.71x**

Next step: ``python scripts/bench_jepa_e2e.py --checkpoint <ckpt>`` to confirm reach_rate > 0.
