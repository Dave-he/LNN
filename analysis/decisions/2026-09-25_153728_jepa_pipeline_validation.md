# JEPA pipeline validation (2026-09-25_153728)

Overall: **PASS**

| gate | value | threshold | unit | passed |
|---|---|---|---|---|
| PD reach_rate n_ped=0 | 1.0000 | 0.95 |  | OK |
| World model final MSE | 0.0000 | 0.005 |  | OK |
| Distilled PD reach_rate | 1.0000 | 0.85 |  | OK |
| QAT reach_rate | 1.0000 | 0.85 |  | OK |
| INT8 reach_rate (QAT PTQ) | 1.0000 | 0.5 |  | OK |
| PyTorch eager p99 latency | 0.1685ms | 5ms | ms | OK |
| ONNX max_abs_diff | 0.0000 | 0.0001 |  | OK |
| ONNX runtime latency | 0.0636ms | 0.5ms | ms | OK |
