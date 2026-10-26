# 2026-10-26 自动化排查：HF 抓取失败的真因是 IPv6 通路，候选池从未全局排序

> 本次排查起因：2026-10-26 04:38 的定时任务日志里 5 条 Hugging Face 查询
> 全部 `Connection reset by peer`，候选模型数 **0**；而 2026-10-24 的 digest
> 也带着同样的 `models=0` 进了 origin 且从未修正。
> 逐层挖下来发现 **8 个独立缺陷**，其中 4 个会把"抓取失败/选错解释器/装错 wheel"伪装成
> "领域事实 / 本机没有 GPU"。

---

## 0. 一句话结论

| # | 现象 | 真因 | 修复 |
|---|---|---|---|
| B1 | HF 查询全部 `Connection reset by peer` | **本机 IPv6 通路坏**（TCP 握手能过、TLS 被 RST），`socket.create_connection` 以为连上了所以不回退 IPv4 | `prefer_ipv4()` 强制走 IPv4 |
| B2 | HF 抓取失败后 digest 仍写"本次未发现可记录的 Hugging Face 模型" | 渲染层不区分"领域事实"与"覆盖盲区" | 摘要行/空结果行显式打 ⚠️ 标记 |
| B3 | HF 查询**完全没有重试**（arXiv 侧有 `request_text_retry`） | 单次瞬时故障即可清空整轮候选池 | `request_json_retry` + 退避 (5/15/40s) |
| B4 | 摘要的 arXiv 表首行是 10-08，**全池第二新的论文（10-22）排在第 48 位** | 候选池是**各查询组的拼接**，只有组内有序，拼接后无全局排序 | `sort_papers_by_date()`，且放在 `main()` 回退合并之后 |
| B5 | PDF 下载器只拿到 **34** 个 arxiv id，而候选池有 93 篇 | `download_arxiv_pdfs.collect_arxiv_ids()` 只读 digest 的**渲染表**（前 12 行） | 改读 `papers/daily/*_lnn_research.json` **全量**候选池 → 122 个 id |
| B6 | Jetson benchmark 永远是 `ok_cpu`；换镜像后 `torch.cuda.is_available()==True` 但**任何 kernel 都跑不起来** | ① 镜像名写死一个**本机从未 pull 过**的镜像 → 静默降级；② cu126 索引的 dev 夜轮 torch **只编 sm_80/sm_90 且显式排除 8.7**，Orin Nano 是 sm_87 | 镜像候选列表 + 本地存在性检查；新增 `Dockerfile.lnn-sm87` |
| B7 | 产物 16 天全是 `ok_cpu`，环境行写 `CUDA：False` —— 但**本机其实有一个能跑 sm_87 kernel 的 torch**（`/usr/bin/python3.10` 的 2.10.0，CUDA 12.6） | `resolve_bench_python()` 的判据只测 `import torch`，而候选排序把 cu130 的 `python3` 排在前面 → 永远选不到那个 cu126 解释器 | 判据改成**真跑一个 kernel**（`bench_cuda_kernel_ok`），优先级改为"能跑 kernel > 能 import > 兜底"；当日首次拿到真实 GPU 数据 |
| B8 | 本仓 `Dockerfile.lnn-sm87` 写"PyPI 官方 aarch64 wheel 带 sm_87 SASS"，据此建出的 `lnn-sm87:local` 镜像里却是 **torch 2.7.0+cpu** | **PyPI / TUNA 的 aarch64 torch wheel 本身就是 CPU 构建**（99.0 MB vs 同 tag x86_64 的 865.0 MB；METADATA 里零个 nvidia-* 依赖）—— 文档里的 claim 与产物互相矛盾 | 该 Dockerfile 的 claim 更正；**构建期加 CUDA 自检**（装完没有 CUDA 直接构建失败）；脚本的镜像预检**失败即跳过该镜像**（不再拿坏镜像继续跑），默认镜像列表里去掉 `lnn-sm87:local` |

> B5 是 B4 的同型复发：**下游脚本只读渲染产物，必然继承渲染层的截断**。
> 这是本项目第三次踩同一个坑（`select_papers_for_report` → 改读 JSON；
> 索引按行截断 → 改按天；现在是 PDF 下载器）。**判据：任何读 digest markdown
> 的自动化都要问一句"它需要的是渲染结果还是全量池"。**


---

## 1. B1 — HF "限流" 其实是本地 IPv6 通路坏（根因）

### 1.1 症状

```
04:39:04 [warn] Hugging Face query failed for 'LiquidAI': <urlopen error [Errno 104] Connection reset by peer>
04:39:04 [warn] Hugging Face query failed for 'LFM2': ...
... (5 条全灭)
```

第一反应是"HF 限流"，但两个现象否掉了这个解释：

1. 04:38 那一轮 5 条全灭，**06:31 重跑又全好了**（`errors: []`，20 个模型）—— 限流不会这样；
2. 失败发生在**每条查询的第 4 秒**，而且 5 条查询一个接一个立刻失败——限流应该是 429，不是连接重置。

### 1.2 定位

```bash
$ python3 -c "import socket; [print(r[0].name, r[4]) for r in socket.getaddrinfo('huggingface.co',443,type=socket.SOCK_STREAM)]"
AF_INET6 ('2600:9000:2078:e400:17:b174:6d00:93a1', 443, 0, 0)   # × 8 个 AAAA
AF_INET  ('13.35.202.34', 443)                                  # × 4 个 A

$ curl -4 -sS https://huggingface.co/api/models?search=LFM2 → http=200   ✅
$ curl -6 -sS https://huggingface.co/api/models?search=LFM2 → curl (35)  # ❌
```

**关键机制**：本机到 HF 的 IPv6 路径 **TCP 握手能成功、TLS 握手被 RST**。
`socket.create_connection` 只在 TCP connect 失败时才回退下一个地址 —— 这里 TCP
"连上了"，所以 urllib 认为连接已建立，直接死在 SSL 层，**永远不会尝试 IPv4**。

对照组（解释为什么只有 HF 出事）：

| 主机 | AAAA | A | 表现 |
|---|---|---|---|
| `huggingface.co` | 8 | 4 | ❌ IPv6-first → RST |
| `arxiv.org` | 0 | 4 | ✅ 纯 IPv4，一直正常 |
| `export.arxiv.org` | 0 | 4 | ✅ 纯 IPv4 |
| `api.github.com` | 0 | 1 | ✅ 纯 IPv4 |

arXiv / GitHub **根本没有 AAAA 记录**，所以从来不受这条通路影响 —— 这也解释了
为什么"HF 限流"看起来像一个独立故障：只有 HF 一个源会被本机 IPv6 打中。

### 1.3 修复

`scripts/daily_lnn_research.py` 新增 `prefer_ipv4()` 上下文管理器，在
`request_json()`（HF + GitHub 共用）内临时把 `socket.getaddrinfo` 的
AF_INET6 结果过滤掉。arXiv 走的 `request_text()` 不需要动（无 AAAA）。

修复后实测：**8 个模型、0 错误、零重试瞬时返回**。

### 1.4 教训

- "限流 / 403 / 连接重置"这类词很容易被当成**远端状态**写进文档，实际上
  先要分清是"对端在限我"还是"我的链路坏了"。区分方法很便宜：`curl -4` vs `curl -6`。
- 退避重试**救不了**这条故障：每次重试走的还是同一条坏掉的 IPv6 通路
  （B3 的重试在 10-24 上跑满 4 次仍全灭，见 §2）。重试只能救"对端窗口抖动"。

---

## 2. B2 + B3 — 抓取失败必须与领域事实分开

### 2.1 缺陷

B3（无重试）：`fetch_huggingface_models` 直接调 `request_json`，而 arXiv 侧
早在 2026-10-14 就加过 `request_text_retry`（退避 8/30/60s）。同一个脚本里
两个源的容错策略不一致，HF 侧成了短板。

B2（渲染层）：`render_daily_digest` 的摘要行与空结果分支都是无条件文案：

```
- Hugging Face 候选模型：0 个
## Hugging Face 候选模型
- 本次未发现可记录的 Hugging Face 模型。
```

**"0 个 / 本次未发现" 读起来就是"领域没有新模型"（领域事实）**，而实际是
"我们没抓到"（覆盖盲区）—— 正是 AGENTS.md 约束 #7 要求区分的那两件事。
10-24 那份 digest 的错误虽然记在「数据源状态」小节里，但摘要行已经把
读者带偏了。

### 2.2 修复

- `request_json_retry(url, ..., backoff=HF_RETRY_BACKOFF)`，退避 (5, 15, 40)s；
  `fetch_huggingface_models` 的列表查询与 detail 查询都改走它。
- 新增 `_count_source_failures(errors)`，把错误按 arXiv / GitHub / Hugging Face
  归类；`_source_note()` 产出摘要行尾注：
  - 失败且计数为 0 → `⚠️（**本次抓取失败**，空结果**不代表领域没有新条目**）`
  - 失败且计数非 0 → `⚠️（N 条查询失败，本次覆盖不全，计数偏少）`
  - 空结果小节同样加 ⚠️ 文案，不再写"本次未发现"。
- 「数据源状态」小节开头补一句总纲：这些是**抓取失败**记录，空结果**不得读作
  「领域没有新条目」**。

### 2.3 补抓工具

新增 `scripts/backfill_huggingface.py`：**只补 HF，不重抓 arXiv/GitHub**。

为什么不能直接 `--date 2026-10-24` 重跑：arXiv 的 `/api/query` 按当前时间窗
返回，今天重跑会把 10-25/10-26 的新论文灌进 10-24 的历史池，**archive 漂移**。
补抓工具保留 JSON 里其余字段原样，只替换 `huggingface_models` 并清掉 HF 错误，
再用脚本里同一套渲染函数重生成 digest / watchlist，索引按日期去重（幂等）。

实测：`2026-10-24` 补回 **21 个模型**（原 0），`errors` 清空，digest / watchlist /
两个索引文件同步更新。

---

## 3. B4 — 候选池从未全局排序，"最近进展"被系统性藏起来

### 3.1 缺陷

`fetch_arxiv()` 逐组查询并把各组结果**依次 append**，只有**组内**按
`sortBy=submittedDate` 有序，**拼接后没有全局排序**。而 digest 只渲染
`papers[:12]`。

后果（2026-10-26 实测，全池 93 篇）：

| 名次 | 论文 | 日期 | 组 |
|---:|---|---|---|
| 0 | ContiLNN | 2026-10-08 | 主线-LNN |
| **48** | **Query-Conditioned Liquid Temporal Grounding** | **2026-10-22** | 主线-CfC |
| 57 | Digital Twins for Patient Care | 2023-07-08 | 主线-CfC |

全池**第二新**的论文（10-22）在第 48 位；而渲染表首行是 10-08 的 ContiLNN。
**渲染表实际只覆盖了第一个查询组**（主线-LNN）的 12 条，另外 7 个组的成果
——包括 10-22 的 CfC 音频定位、08-13 的 Lagrangian NN、07-16 的 LFM2 tokenizer
扩展、07-14 的跌倒检测、06-27 的 LNN-Fly —— 全部落在折叠线以下。

读者（和任何只读渲染表的自动化）会得出"领域自 10-08 起没动静"的结论，
**而这是查询组织的产物，不是领域事实**。与 AGENTS.md 约束 #9 修掉的
GitHub 侧"排序不按研究价值"是同一类问题。

### 3.2 修复

新增 `sort_papers_by_date()`，**放在 `main()` 的"保留前次结果"回退合并之后**
——因为走回退路径时候选池来自旧文件，若只在 `fetch_arxiv` 内部排序，
降级当天照样是乱的。排序键 `(published, updated, id)` 降序，保证确定性。

修复后摘要表首行即为 2026-10-22 的 Query-Conditioned Liquid Temporal Grounding。

---

## 4. 覆盖度自检（协议 §3.3）

自检方法：**不用本仓的 `/api/query`**，改用 arXiv **搜索 UI HTML**
（`https://arxiv.org/search/?searchtype=all&query=...`），即协议要求的
"不要只信 API 返回"。

| 查询词 | UI 结果数 | 解析到 | **最新一篇** | 判定 |
|---|---:|---:|---|---|
| `"liquid neural"` | 39 | 39 | 2610.12337（2026-10-08）ContiLNN | 与候选池一致 ✅ |
| `"closed-form continuous-time"` | 17 | 17 | **2610.31252（2026-10-22）** | 在候选池，但曾被排在第 48 位 ⚠️→已修 |
| `"liquid time-constant"` | 28 | 28 | 2607.12909（2026-07-14） | 在候选池 ✅ |
| `LFM2` | 6 | 6 | 2607.15232（2026-07-16） | 在候选池 ✅ |

**结论**：
- "LNN 主线自 2026-10-08 起无新投稿"经**独立通道**（arXiv 搜索 UI）复核成立，
  是**领域事实**而非覆盖盲区；
- 但 "closed-form continuous-time" 组在 10-22 有新论文（Query-Conditioned
  Liquid Temporal Grounding for Long-Form Audio），它一直在候选池里，
  只是被渲染表的截断 + 无全局排序藏住了 —— 这是**覆盖盲区 (b)**，
  由 B4 修复，不是领域没动静。

两条结论同时成立，这正是 §3.3 要求做的区分。

---

## 5. B5 — PDF 下载器也是截断盲区（第三次同型复发）

`scripts/download_arxiv_pdfs.py` 的 `collect_arxiv_ids()` 从三个地方收 id：
研读报告、`papers/daily/*.md`、以及 `docs/daily/*.md`（digest）。
**最后一个是渲染表** —— 只渲染前 12 行。

实测：修复前只收得到 **34** 个 id，候选池里第 13 名之后的论文**永远拿不到 PDF**。
修复后从 `papers/daily/*_lnn_research.json`（**全量**候选池，字段 `papers[].id`
形如 `2610.31252v1`）取 id → **122** 个（+88）。

---

## 6. B6 — Jetson benchmark 的两道静默降级

### 6.1 降级一：镜像名指向一个本机没有的镜像

`run_daily_lnn_task.sh` 原来写死 `JETSON_DOCKER_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`，
而**本机从来没 pull 过它** → `docker image inspect` 失败 → 每天静默走 host Python
分支 → 宿主 torch 是 cu130（驱动 540.4.0 最高只支持 12.6）→ 起不来 CUDA →
产物永远是 `ok_cpu`。**两级降级都只写一行"CUDA 不可用"，不留痕。**

修复：改成 `JETSON_DOCKER_IMAGES` 候选列表，取**第一个本地存在**的镜像；
一个都没有时**显式打印**"本机没有 Jetson CUDA 镜像 + 宿主是 cu130 所以本次只能是
CPU 数据 + 怎么现场构建"。

### 6.2 降级二：cu126 夜轮 torch 在 sm_87 上无 kernel

按上面的提示构建了 `analysis/jetson/Dockerfile.lnn-cuda126`
（`torch==2.13.0.dev20260610+cu126`，来自 download.pytorch.org 的 cu126 索引
——该索引对 aarch64 **只有 dev 夜轮版**）。镜像构建成功，容器内：

```
torch 2.13.0.dev20260610+cu126
cuda_avail True
cuda_ver 12.6
dev Orin
```

**看起来完全正常**，但真跑 kernel 就炸：

```
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
No published PyTorch CUDA builds for release 2.13.0.dev20260610+cu126 support this GPU.

torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**该夜轮版只编了 sm_80 与 sm_90，并显式排除 8.7，也没有编 PTX 供 JIT 回退。**
Orin Nano 是 sm_87 → 一条 kernel 都没有。

> **教训**：`torch.cuda.is_available() == True` **不等于** CUDA 可用。
> 它只说明驱动和 runtime 起来了，不代表这个 torch build 有你 GPU 的 SASS。
> 验收必须真跑一个 kernel（`torch.randn(n,n,device='cuda'); x@x`），
> 只看 `is_available()` 会把一个完全跑不起来的镜像判成"可用"。
> 这条已修正本项目 memory 里"lnn-cuda126:local 可用"的旧结论。

### 6.3 正确路线：PyPI 官方 aarch64 wheel

PyPI 上 torch 有 aarch64 官方 wheel（2.5.1~2.10.0），它们是 JetPack 上唯一
长期带 **sm_87 SASS** 的发行版：

| torch | 默认 CUDA | 与驱动 540.4.0 (≤12.6) |
|---|---|---|
| 2.7.0 / 2.7.1 | cu126 | ✅ 版本正好对齐 |
| 2.6.0 / 2.5.1 | cu124 | ✅ 靠 CUDA minor-version 兼容 |
| 2.8.0+ | cu128 | ❌ 超出驱动上限 |

新增 `analysis/jetson/Dockerfile.lnn-sm87`（`ARG TORCH_SPEC=2.7.0`，走 PyPI 默认索引）。

> **2026-10-26 晚更正**：镜像路线**不是必需的** —— 宿主 `/usr/bin/python3.10` 的
> torch 2.10.0+cu126 本身就带 sm_87 SASS 且能跑 kernel，详见 §7。于是"PyPI 官方
> aarch64 wheel 是唯一长期带 sm_87 SASS 的发行版"这句话并不成立（宿主 2.10.0+cu126
> 就是反例）。本轮放弃构建镜像（下载 99MB wheel 在本机 ≈20 分钟），
> 改去修解释器选择。Dockerfile 保留作隔离环境备选。

### 6.4 另一个独立问题：系统 RAM 耗尽导致 nvmap 分配失败

第一次跑容器时连 **64×64（16KB）** 的分配都报
`NvMapMemAllocInternalTagged: error 12` + `CUDA error: out of memory`。
`tegrastats` 显示 `RAM 6177/7620MB, SWAP 10214/20194MB, lfb 1x256kB` ——
**最大连续空闲块只有 256kB**，nvmap 要连续物理内存，分配不出 carve-out。
等系统内存回落到 `RAM 4569/7620MB, lfb 2x2MB` 后，同一条分配立刻成功，
随后才暴露出真正的 sm_87 kernel 缺口。

> 与 2026-09-29 记录的现象同源（RAM 耗尽 → 静默 CPU 回退）。
> 本次是**两件事叠在一起**：先是 RAM 压力伪装成"CUDA 不可用"，
> 内存回落后才看到真正的 kernel 缺口。**排查要分层，别把第一层症状当结论。**

---

## 7. B7 — 解释器选择只看 `import torch`，于是**永远选不到能跑 CUDA 的那个**

### 7.1 症状与根因

10-12 ~ 10-26 每一天的产物都是 `status=ok_cpu`，环境行写着
`PyTorch：2.11.0+cu130` / `CUDA：False (13.0)`。本仓的长期记忆里因此记着
"**本机无任何可用 CUDA**"。

真因在 `scripts/run_daily_lnn_task.sh:resolve_bench_python()`：候选解释器按
`python3 → /usr/bin/python3.10 → …` 排序，**判据只测 `import torch` 能不能成功**。
本机两个解释器都能 import：

| 解释器 | torch | wheel 的 CUDA | 驱动 540.4.0 (上限 12.6) | 实际 |
|---|---|---|---|---|
| `python3` (pyenv 3.14) | 2.11.0+**cu130** | 13.0 | 超出上限 | `is_available()` = **False** → CPU |
| `/usr/bin/python3.10` | 2.10.0+**cu126** | 12.6 | 正好 | 能跑 sm_87 kernel → **GPU** |

判据太松 → 排序里靠前的 cu130 永远胜出 → 每天静默产出一份 CPU 数据。
**这不是"本机没 CUDA"，是"本机有一个我们从没选过的解释器有 CUDA"。**

### 7.2 修正：判据从"能 import"改成"能跑一个 kernel"

新增 `bench_cuda_kernel_ok()`：真跑一次 64×64 matmul 并 `cuda.synchronize()`。
`resolve_bench_python()` 的优先级随之改为
**(1) 能跑 kernel → (2) 能 import torch → (3) 兜底**。

> 判据必须跟着"要证明什么"走：要证明"这个解释器能用 CUDA 跑训练/推理"，
> 就得真跑一个 kernel。`import torch` 只证明 wheel 装上了；
> `torch.cuda.is_available()` 只证明驱动 + runtime 起来了；
> **两者都不证明这个 wheel 编了本机 sm_87 的 SASS。**

### 7.3 当日产物：首次拿到真实 GPU 数据

2026-10-26 用 `/usr/bin/python3.10` 跑出 `status=ok` / `device=cuda`：
64 samples / seq 16 / hidden 8 / 1 epoch，**含 tegrastats 功耗与温度**
（VDD_IN mean 7046 mW / peak 7615 mW，GPU 峰值 53.4 °C，GPU 利用率 mean 8% / peak 27%）。
推理步/秒：PDNAPulse 12276 / CfCStyle 11065 / LTC 2378 / GRU 204976。
这是本仓第一次拿到**真实边缘 GPU 性能数据**（此前 16 天全是 CPU 数据）。

### 7.4 仍然存在的一层：内存压力下的 NvMap 分配失败

同一天下午先失败过一次：64×64（16KB）都报
`NvMapMemAllocInternalTagged: 1075072515 error 12` + `CUDA error: out of memory`；
`mem_get_info` 却报 free 2165 MB。**几分钟后同一条命令就成功了。**
这与 §6.4 是同一件事：nvmap 要**连续物理内存**，系统被 codex/Hoper 批任务压满时
就分配不出来。所以：

- **"CUDA 可见" / "CUDA 可用" / "CUDA 现在分配得到内存" 是三件事**；
- 失败与否**随系统内存压力波动**，不是稳定的能力边界；
- 修法不是"绕开 CUDA"，而是**让失败可解释**：宿主路径不再传
  `--no-cpu-fallback`，让脚本自己回退并把失败原因写进产物。

### 7.5 附带修掉的"结论方向相反"缺陷

`write_report()` 原来只在 `status ∈ {ok, ok_cpu, ok_cpu_fallback}` 时渲染正文，
而 `cuda_fallback_reason` **只写进 JSON**。只读 md 的人看到
"CUDA was unavailable" 会得出"这台机器没有 GPU"，而真实情况是
"CUDA 在，分配失败" —— **两个相反的结论**。已让失败原因同时出现在 md 横幅里，
并给成功的 GPU 运行补上 status_note（原来 GPU 成功时横幅写
"（本次运行未记录 status_note）"）。

### 7.6 对镜像路线的影响

`lnn-sm87:local`（`Dockerfile.lnn-sm87`）**不再是必需品** —— 宿主解释器已经能跑
sm_87 kernel。镜像只剩"隔离环境"这一个价值，而它要下载 ~345MB wheel（本机网络
约 80 KB/s，≈1 小时）。因此本轮**放弃重建**，改为修解释器选择。

> ⚠️ 这里曾写过"宿主 2.10.0+cu126 就是'PyPI 官方 aarch64 wheel'的反例" ——
> **这句也不准确**。宿主那份 torch 的 wheel 标签是 `cp310-cp310-linux_aarch64`
> （**不是** manylinux），`direct_url.json` 指向
> `file:///tmp/torch-2.10.0-cp310-cp310-linux_aarch64.whl`，即**本地 wheel 文件**；
> 它的 `torch.__version__` 是 `2.10.0`（无 `+cu126` 本地版本号），所以也不是
> download.pytorch.org 那个 `2.10.0+cu126`。**版本号一样，出身不一样**，
> 别把两者混为一谈（详见 §7.7）。

### 7.7 B8 — 本仓自己的 Dockerfile 写着与产物相反的 claim

`Dockerfile.lnn-sm87` 上一版写着："**PyPI 上的官方 aarch64 wheel 默认带 sm_87
SASS，2.7.0 → 默认 cu126**"。据此构建出的镜像实测是：

```
$ docker run --rm --runtime nvidia --gpus all lnn-sm87:local \
    python3 -c "import torch; x=torch.randn(64,64,device='cuda')"
AssertionError: Torch not compiled with CUDA enabled
# torch.__version__ == "2.7.0+cpu"，get_arch_list() == []
```

**根因**：**PyPI（及 TUNA 镜像）上的 aarch64 torch wheel 本身就是 CPU 构建**。

| torch 2.7.0 / cp312 | 体积 | 是否含 CUDA |
|---|---:|---|
| `torch-2.7.0-cp312-cp312-manylinux_2_28_aarch64.whl` | **99.0 MB** | ❌ CPU（METADATA 里零个 `nvidia-*` 依赖） |
| `torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl` | **865.0 MB** | ✅ CUDA（nvidia 库打包在内） |

> **教训 1：wheel 体积是判断 aarch64 wheel 是否含 CUDA 的最快信号**
> （~99 MB = CPU-only，800+ MB = CUDA）。更硬的办法是拉 wheel 的 `.metadata`
> 段：CUDA 版会列一串 `nvidia-*-cu12` 依赖，CPU 版一个都没有。
> 这条正好推翻了本仓长期记忆里"正确路径 = PyPI 官方 aarch64 wheel"的说法。

同时更正 2026-10-19 记的"cu126 索引对 aarch64 只有 dev 夜轮"——**也是错的**：
实测该索引的 aarch64 稳定版有 2.6.0 / 2.9.0 / 2.9.1 / 2.10.0 / 2.11.0 /
2.12.0 / 2.12.1 / 2.13.0 / 2.14.1 共 9 个（另加 1 个 dev 夜轮）。
**教训 2：`pip index versions` 会把没有 aarch64 wheel 的版本也列出来**，
判断有无 wheel 必须去索引页解析 `*_aarch64.whl` 文件名。

**已做的三处修改**：

1. `Dockerfile.lnn-sm87` 的注释更正为事实，并且**构建期加 CUDA 自检**
   （装完的 torch 没有 CUDA 就直接构建失败）——宁可构建失败，也不要留一个
   "看着装好 torch、实则是 CPU 版"的镜像；
2. 脚本的镜像预检从"失败只报错、继续用该镜像"改成
   **失败即跳过该镜像**（并继续试下一个候选，最后回退宿主）；
3. 默认镜像候选列表里去掉 `lnn-sm87:local`（当前本机那份是 CPU-only 的坏镜像）。

**仍然没验证的**：cu126 索引的 `torch-2.10.0+cu126-cp312-...-aarch64.whl`
**确实是 CUDA 构建**（已核 `.metadata`：cublas 12.6.4.1 / cudnn 9.10.2.21 /
nccl 2.27.5 / triton 3.6.0 …），但**它有没有编 sm_87 的 SASS 没验证** ——
官方 aarch64 构建常只编 sm_80/sm_90，那就会重演 dev 夜轮那个
`no kernel image is available`。所以 Dockerfile 的验收标准写成
**必须跑出 `KERNEL_OK`**，而不是"装上了就算好"。

---

## 8. 附带发现：`.git/rebase-merge` 残留了两个月，2026-08-27 的中断 rebase

排查过程中 `git commit` 突然报
`You are currently editing a commit while rebasing branch 'master' on '2d9a23a'`，
而 reflog 里**没有任何 rebase 记录**。查 `.git/rebase-merge/` 的 mtime：
**2026-08-27 04:35** —— 两个月前的一次 rebase 中途夭折（机器重启或 `reset --hard`
打断了），状态目录留在 `.git/` 里没人清理。

它引用的 `orig-head = eefd3e6`（"chore(daily): update LNN research digest 2026-08-27"）
**已经不在 master 历史里**。先确认没丢东西：该 commit 改的 5 个文件
（`docs/daily/2026-08-27_*.md`、`papers/daily/2026-08-27_*.json`、watchlist、
两个索引）**全部还在 master 树里** —— 那次 rebase 被中断后，**同样的内容被
用另一个 commit 重新提交过**。所以是纯垃圾状态，不是内容丢失。

危险在于：留着它，任何人执行 `git rebase --continue` 都会去重放一个**已经不
在历史上**的提交；而 `git commit` / `git status` 又会一直打出 "rebasing" 的
误导性状态。已用 `git rebase --quit` 清除（quit 只删状态文件，**不动 HEAD 与
工作区**）。

> **判据**：看到 git 报 "rebasing" 但 reflog 里没有对应 rebase 记录时，
> 先看 `.git/rebase-merge/` 的 mtime —— 那是**陈旧状态**而不是进行中的操作。
> 处理前必须先确认内容没丢（`git merge-base --is-ancestor <orig-head> master`
> + 核对受影响文件是否仍在树里），再 `git rebase --quit`，
> **绝不要 `--continue`**。

---

## 9. 验证

```bash
python3 scripts/check_negative_results.py        # [OK] 9 条条目, grounding 全部存在
python3 scripts/check_repo_integrity.py          # [OK] 无冲突标记, JSON 全部可解析
python3 scripts/select_papers_for_report.py --date 2026-10-26
#   pool_source=json, n_total=93, n_skipped_reported=44, pending=3
python3 scripts/audit_arxiv_ids.py               # 8 处 mismatch 全在 2026-06 旧文件, 非今日新增
```

修复后的当日 digest：`papers/repos/models = 93/42/21`，`errors: []`。
