# JEPA distillation (pd) — 2026-09-25_142715

Target source: **pd** (16000 states)
Epochs: 100, batch: 128, lr: 0.001

Initial MSE: **1.08123**
Final MSE:   **0.00001**
Improvement: **93261.32x**

Next step: ``python scripts/bench_jepa_e2e.py --checkpoint <ckpt>`` to confirm reach_rate > 0.
