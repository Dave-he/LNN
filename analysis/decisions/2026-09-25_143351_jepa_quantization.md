# JEPA quantization benchmark (2026-09-25_143351)

Checkpoint: `jepa_qat_8bit_2026-09-25_143343.pt`
Episodes per mode: 30

| precision | reach_rate | p50_ms | p99_ms | n_params | notes |
|---|---:|---:|---:|---:|---|
| float32 | 1.000 (30/30) | 0.552 | 0.737 | 4596 | baseline |
| float32 | 1.000 (30/30) | 0.548 | 0.724 | 4596 | weight-only int8 PTQ; expect brittle for overfit BC ckpts |
