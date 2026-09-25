# JEPA quantization benchmark (2026-09-25_143243)

Checkpoint: `jepa_distilled_pd_2026-09-25_142715.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 1.000 (30/30) | 0.566 | 0.759 | 4596 | baseline |
| float32 | 0.000 (0/30) | 0.557 | 0.713 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
