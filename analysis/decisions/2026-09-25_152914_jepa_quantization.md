# JEPA quantization benchmark (2026-09-25_152914)

Checkpoint: `jepa_distilled_planner_2026-09-25_152903.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 0.000 (0/30) | 0.572 | 0.783 | 4596 | baseline |
| float32 | 0.000 (0/30) | 0.571 | 0.852 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
