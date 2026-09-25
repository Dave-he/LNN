# JEPA quantization benchmark (2026-09-25_143425)

Checkpoint: `jepa_qat_8bit_2026-09-25_143343.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 0.000 (0/30) | 0.559 | 1.012 | 4596 | baseline |
| float32 | 0.000 (0/30) | 0.562 | 0.736 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
