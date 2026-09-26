# JEPA pipeline validation (2026-09-25_153623)

Overall: **FAIL**

| gate | value | threshold | unit | passed |
|---|---|---|---|---|
| PD reach_rate n_ped=0 | 1.0000 | 0.95 |  | OK |
| World model final MSE | 0.0000 | 0.005 |  | OK |
| Distilled PD reach_rate | 0.0000 | 0.85 |  | X |
| QAT reach_rate | 0.0000 | 0.85 |  | X |
| INT8 reach_rate (QAT PTQ) | 0.0000 | 0.5 |  | X |
| PyTorch eager p99 latency | 0.1311ms | 5ms | ms | OK |
| ONNX max_abs_diff | nan | 0.0001 |  | X |
| ONNX runtime latency | nanms | 0.5ms | ms | X |
