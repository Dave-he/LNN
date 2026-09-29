---
title: LNN 每日自动化健康度排查 - 2026-09-29
date: 2026-09-29
tags: [LNN, automation, systemd, jetson, benchmark, honest-reporting]
---

# LNN 每日自动化健康度排查 - 2026-09-29

> 排查范围：定时任务是否每日执行、最近 14 天执行完整性、今日研究产物完整性。
> 结论：**调度层健康（14/14 天无缺口），但 Jetson benchmark 存在 7 天静默 CPU 回退的性能证据失效问题。**

---

## 一、调度层：健康

| 检查项 | 状态 | 证据 |
|---|---|---|
| timer 已启用 | ✅ `enabled` | `systemctl --user is-enabled lnn-daily-research.timer` |
| 开机自启 | ✅ `Linger=yes` | `loginctl show-user hyx -p Linger`（无此值 user timer 不会在未登录时触发） |
| 今日已执行 | ✅ 04:33:53 → 04:34:48（55s） | journalctl |
| 下次触发 | ✅ 2026-09-30 04:35:59 CST | `list-timers` |
| crontab 冲突 | ✅ 已禁用 | crontab 内 LNN 行仍为注释（9/01 起） |
| GitHub Actions 兜底 | ✅ 连续 3 次 ok | `Daily LNN Research` runs |
| Weekly Verify | ✅ 连续 4 次 ok | `LNN Weekly Verify` runs |

### 双写者设计（预期行为，非 race）

| 时间 | 写者 | commit |
|---|---|---|
| 04:34 | systemd timer（heyongxian） | `7fa1945` digest + jetson + watchlist |
| 06:33 | GitHub Actions（LNN Cron Bot） | `ee06610` digest + 研读报告 |
| 06:37 | GitHub Actions | `388903e` watchlist |
| 06:41 | GitHub Actions | `9c0ab41` reproduce imitation LNN |

本地 04:30 先跑、GH Action 06:30 后跑，**2 小时错峰**，今日 `HEAD == origin/master`（`9c0ab41`），无 push 冲突、无 rebase 冲突。方案 E（`sync_with_origin` + 5 次 push 重试）持续有效。

### 路径一致性

systemd unit 的 `WorkingDirectory=/home/hyx/codespace/research/LNN` 与本 session cwd `/mnt/ssd/codespace/research/LNN`
inode 相同（`66304:73146687`），是同一目录的两个挂载点，**不存在双仓库分叉**。

---

## 二、产物完整性：近 14 天

| 日期 | digest | watchlist | jetson benchmark | reproduce |
|---|---|---|---|---|
| 09-16 ~ 09-22 | Y (7/7) | Y (7/7) | **缺** | 09-22 |
| 09-23 ~ 09-29 | Y (7/7) | Y (7/7) | Y (7/7) | 09-29 |

- 09-16 ~ 09-22 无 benchmark：已知 `libcudss.so.0` 导入崩溃 + docker 镜像未拉，`\|\| true` 兜住不阻塞 commit（记忆 `lnn-2026-09-22-daily-status`）。**09-23 起已恢复。**
- `reproduce`（`analysis/control/*_imitation_lnn.json`）由 GH Action 跑，近期落在 09-22 / 09-29。

### 今日研读覆盖率：25/25

`papers/daily/2026-09-29_lnn_research.json` 共 25 篇候选，全部已有独立研读报告。

> ⚠️ 朴素的文件名匹配脚本会误报 3 篇「未覆盖」（`2605.08176` / `2603.00459` / `2603.00153`），
> 因为这三篇的报告用的是描述性文件名而非 arXiv ID。实际对应：
> - `2605.08176` → `Physics-Modeled_Neural_Networks_DynPMNN_研读报告.md`
> - `2603.00459` → `LSS-LTCNet_Foot_Ulcer_Segmentation_研读报告.md`
> - `2603.00153` → `Pulse-Driven_Neural_Architecture_PDNA_研读报告.md`
>
> **覆盖率检查脚本应按 arXiv ID 全文匹配报告正文，而非匹配文件名**，否则会持续误报。

---

## 三、异常发现：Jetson benchmark 连续 7 天静默 CPU 回退

### 现象

`analysis/jetson/2026-09-23 ~ 09-29` 共 7 份报告，**7/7 全部**在「CUDA 回退」段落声明已回退 CPU，
`device: cpu`。对照历史：`2026-08-03-gpu-pareto` 与 `2026-08-04` 确实跑在 `device: cuda`。
即 **09-23 起 GPU 路径再未成功过**。

### 根因：系统 RAM 耗尽，不是显存不足

错误串看起来像 CUDA OOM，实际不是：

```text
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
torch.AcceleratorError: CUDA error: out of memory
```

Orin Nano 是**统一内存架构**，GPU carve-out 由 nvmap 从系统 RAM 里划。
`error 12` 出现在 **CUDA context 创建阶段**（连一个 8 元素张量都分配不出来），
而 benchmark 的模型只有 2.5K~3.2K 参数 —— 显存根本不是瓶颈。

实测（排查时）：

```
Mem: total 7619 MB | free 327 MB | available 1307 MB
```

8 GB 的机器上，04:30 定时任务运行时 `available` 仅 ~700 MB，nvmap 拿不到连续物理页 → context 创建失败。
常驻占用大户：`baidu-pcs-rust-build` 1.1 GB、`rsface-server` 550 MB、`rustfs` 486 MB、`ollama serve` 453 MB。

### 关键证据：这是间歇性抖动，不是确定性配置错误

排查过程中用**完全相同的脚本、完全相同的机器**重跑：

| 时间 | 内存 available | 结果 |
|---|---|---|
| 04:33（定时任务） | ~700 MB | `device: cpu`（回退） |
| 排查时手动重跑 | 1482 MB | **`device: cuda`（成功）** |

同一份代码、同一台机器，7 小时后 CUDA 路径就通了。**说明成败取决于 context 创建瞬间的系统内存余量**，
不是配置、不是镜像、不是依赖。

### 为什么没人发现

`scripts/run_daily_lnn_task.sh` 的 benchmark 调用带 `|| true`，失败与回退都不影响 exit code，
日志只有一行 `[warn]`。日更索引也不会提到。

### 影响范围（诚实边界）

| 数据 | 是否有效 | 说明 |
|---|---|---|
| 功耗（VDD_IN / VDD_CPU_GPU_CV / VDD_SOC mJ） | ✅ 有效 | tegrastats 直接读 Jetson 传感器，与 device 无关 |
| 温度（soc0/1/2, tj） | ✅ 有效 | 同上 |
| 推理步/秒、训练秒、测试 MSE | ❌ **不是边缘性能证据** | CPU smoke，只是链路连通性验证 |

即：**报告里那张 `139440 步/秒` 的表不能用来支撑任何 "LNN 在 Jetson 上快" 的结论。**
按 [[AGENTS.md]] §禁止重复的 claim 之 #6，LNN 本就未在标准主基准上超过 transformer；
这里连基准本身都没跑在 GPU 上，更不能外推。

---

## 四、本次修复

`scripts/jetson_lnn_benchmark.py`：

1. 新增 `_read_meminfo()`，把 `MemTotal / MemFree / MemAvailable` 收进 `detect_environment()`。
2. 报告「环境」段新增一行 `- 系统内存：total / free / available MB`。
   **今后一眼能看出 nvmap 是否有页可用**，无需再手工 `free -m`。
3. 「CUDA 回退」段新增显式告警：速度/精度列标注为非边缘性能证据，功耗/温度仍有效。

验证：语法检查通过；临时日期端到端跑通（CUDA 路径成功、无回退段），scratch 产物已清理。

## 五、待决（未擅自执行，需用户拍板）

要让定时任务稳定跑在 GPU，需释放 04:30 时段的系统内存。候选：

| 方案 | 收益 | 代价/风险 |
|---|---|---|
| A. 定时任务前临时停 `ollama serve`（453 MB，04:30 基本空闲） | 大概率够 context 创建 | 动用户服务，需授权；可做成 `ExecStartPre` 停 / `ExecStopPost` 起 |
| B. 拉 `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` 镜像走 docker 路径 | 隔离内存，最稳 | 镜像 ~20 GB（磁盘 142 GB 够），拉取耗流量 |
| C. 什么都不做，只靠新增的内存行暴露抖动 | 零风险 | benchmark 继续随机在 cpu/cuda 间跳 |

**当前按 C 执行**：报告已自带足够信息判断某天数据是否可引用，无需改动用户服务。

---

## 相关

- [[LNN_持续研究协议]]
- [[docs/LNN_深度研读报告]]
- 记忆：`lnn-2026-09-22-daily-status`（benchmark 退化）、`lnn-daily-race-fix-2026-08-22`（方案 E）
