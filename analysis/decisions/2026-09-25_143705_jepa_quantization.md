# JEPA quantization benchmark (2026-09-25_143705)

Checkpoint: `jepa_qat_8bit_2026-09-25_143659.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 1.000 (30/30) | 0.565 | 0.713 | 4596 | baseline |
| float32 | 1.000 (30/30) | 0.562 | 0.685 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
