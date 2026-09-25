# JEPA quantization benchmark (2026-09-25_143634)

Checkpoint: `jepa_distilled_pd_2026-09-25_143626.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 1.000 (30/30) | 0.544 | 0.671 | 4596 | baseline |
| float32 | 0.000 (0/30) | 0.550 | 0.662 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
