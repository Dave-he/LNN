# JEPA quantization benchmark (2026-09-25_143400)

Checkpoint: `jepa_qat_8bit_2026-09-25_143343.pt`
Episodes per mode: 50

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 1.000 (50/50) | 0.539 | 0.658 | 4596 | baseline |
| float32 | 1.000 (50/50) | 0.545 | 0.681 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
