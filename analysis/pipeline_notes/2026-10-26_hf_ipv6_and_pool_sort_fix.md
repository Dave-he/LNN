# 2026-10-26 自动化排查：HF 抓取失败的真因是 IPv6 通路，候选池从未全局排序

> 本次排查起因：2026-10-26 04:38 的定时任务日志里 5 条 Hugging Face 查询
> 全部 `Connection reset by peer`，候选模型数 **0**；而 2026-10-24 的 digest
> 也带着同样的 `models=0` 进了 origin 且从未修正。
> 逐层挖下来发现 **4 个独立缺陷**，其中 2 个会把"抓取失败"伪装成"领域事实"。

---

## 0. 一句话结论

| # | 现象 | 真因 | 修复 |
|---|---|---|---|
| B1 | HF 查询全部 `Connection reset by peer` | **本机 IPv6 通路坏**（TCP 握手能过、TLS 被 RST），`socket.create_connection` 以为连上了所以不回退 IPv4 | `prefer_ipv4()` 强制走 IPv4 |
| B2 | HF 抓取失败后 digest 仍写"本次未发现可记录的 Hugging Face 模型" | 渲染层不区分"领域事实"与"覆盖盲区" | 摘要行/空结果行显式打 ⚠️ 标记 |
| B3 | HF 查询**完全没有重试**（arXiv 侧有 `request_text_retry`） | 单次瞬时故障即可清空整轮候选池 | `request_json_retry` + 退避 (5/15/40s) |
| B4 | 摘要的 arXiv 表首行是 10-08，**全池第二新的论文（10-22）排在第 48 位** | 候选池是**各查询组的拼接**，只有组内有序，拼接后无全局排序 | `sort_papers_by_date()`，且放在 `main()` 回退合并之后 |

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

## 5. 验证

```bash
python3 scripts/check_negative_results.py        # [OK] 9 条条目, grounding 全部存在
python3 scripts/check_repo_integrity.py          # [OK] 无冲突标记, JSON 全部可解析
python3 scripts/select_papers_for_report.py --date 2026-10-26
#   pool_source=json, n_total=93, n_skipped_reported=44, pending=3
python3 scripts/audit_arxiv_ids.py               # 8 处 mismatch 全在 2026-06 旧文件, 非今日新增
```

修复后的当日 digest：`papers/repos/models = 93/42/21`，`errors: []`。
