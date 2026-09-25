# JEPA quantization benchmark (2026-09-25_153318)

Checkpoint: `jepa_distilled_pd_2026-09-25_153307.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 1.000 (30/30) | 0.539 | 0.637 | 4596 | baseline |
| float32 | 1.000 (30/30) | 0.539 | 0.644 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
