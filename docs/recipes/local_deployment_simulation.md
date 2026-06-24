---
title: 本地部署模拟
date: 2026-06-24
tags: [LNN, deployment, simulation, device-control, safety]
---

# 本地部署模拟

`scripts/local_deployment_sim.py` 是设备操控 4-case harness 的本地部署彩排层。它复用
`scripts/experiment_device_control_cases.py` 的合成数据与模型 smoke，生成 `sim://`
部署 manifest、mock 部署审计日志和目标预算判定。

## 安全边界

该脚本只做本机模拟，不触发真机：

- 不调用 `adb`、`xcrun devicectl`、ROS、MAVLink、CAN、串口、蓝牙、短信、付款或登录能力。
- 不连接 Jetson、iOS 设备、无人机、机器人、BMS 或 MCU。
- 不生成真实提交单；manifest 中固定 `real_device_access=false`、`submit_allowed=false`。
- 审计日志的接口只允许 `python`、`filesystem`、`sim://`。

## 快速运行

```bash
# 单 case: 工业控制本地 smoke 部署模拟
python scripts/local_deployment_sim.py --case industrial --quick --steps 8

# 4 case 全部彩排，使用 Jetson-like CPU 预算做本地判定
python scripts/local_deployment_sim.py --case all --target jetson_orin_cpu --quick --steps 8

# MCU-like 严格预算；PyTorch smoke artifact 失败是正常结果，用于暴露导出/量化缺口
python scripts/local_deployment_sim.py --case industrial --target mcu_tiny --quick --steps 8
```

## 产物

默认输出目录：

```text
analysis/local_deployment_sim/
  YYYY-MM-DD_HHMMSS_local_deployment_sim_<target>_<case>.json
  latest_local_deployment_simulation.json
```

单个 JSON 包含：

- `source_report`: 合成 harness 的 per-seed 结果与均值。
- `manifest`: `sim://` artifact、mock target、参数量、估算 FP32 内存和安全声明。
- `audit`: package / transfer / load / warmup / inference_loop / rollback_check 六段审计。
- `budget_check`: case 支持性、延迟预算、内存预算判定。
- `status`: `pass`、`fail` 或 `blocked`。

## 目标配置

| target | 用途 | 说明 |
|---|---|---|
| `local_cpu_smoke` | 默认本地彩排 | 预算宽松，用来证明 packaging 与本机推理路径能跑通。 |
| `jetson_orin_cpu` | Jetson-like CPU 预算 | 不碰 Jetson/CUDA，只用本机 smoke 指标做预算门禁。 |
| `mcu_tiny` | Cortex-M4-like 严格预算 | 只支持 `industrial` / `battery`，用于暴露 PyTorch artifact 距离 MCU 部署的差距。 |

## 与现有 harness 的关系

`experiment_device_control_cases.py` 证明 4 个设备操控场景的 LNN wiring 可以在合成数据上
forward/backward。`local_deployment_sim.py` 在它之上补上部署前应有的 manifest、mock
runtime lifecycle、预算判定和安全审计。它不是 Jetson/MCU/iOS 真部署，也不替代
`scripts/jetson_lnn_benchmark.py` 的真机或边缘性能数据。
