#!/usr/bin/env python3
"""Daily Liquid Neural Networks research tracker.

The script intentionally uses only Python's standard library so it can run on
GitHub Actions, a Jetson device, or a clean research workstation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import pathlib
import re
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
USER_AGENT = "LNN-research-tracker/1.0 mailto:hyx@nous.local (https://github.com/Dave-he/LNN)"

# arXiv `/api/query` 按查询串长度限流: 单条 21 词 OR (~810 字符) 会稳定触发 406/429。
# 所以按 protocol §3 的分组拆成多个小查询, 每组独立重试, 一组挂掉不牵连其他组。
#
# ⚠️ 语法要点 (2026-10-20 实测, 见协议 §3.2):
#   arXiv API 的 `all:"..."` 是**连续短语**匹配, 而搜索 UI 的 `"X" AND "Y"` 是
#   **布尔**匹配 (X 和 Y 各自出现即可, 位置不限)。两者不等价:
#     API  all:"LNN edge"   -> 0 篇 (要求 "LNN edge" 连续出现)
#     UI   "LNN" AND "edge" -> 2 篇 (LNN 与 edge 各自出现即可)
#   所以应用场景组必须写成布尔式 `all:LNN AND all:edge`, 不能写成 `all:"LNN edge"` ——
#   否则该组会**静默返回 0 篇**, 看起来像"该场景没论文", 实际是查询写错了。
#
# 各组词的选择依据 (arXiv 搜索 UI 全库实测, 2026-10-20):
#   基础模型: LFM2=6 篇 / "liquid foundation model"=3 篇
#             —— 含从未进池的 2511.23404 LFM2 Technical Report、2603.03517 MMAI Gym
#   相关方法: "continuous-time RNN"=6 篇 —— 含 1811.00321
#   应用场景: LNN time series=8 / robotics=4 / forecasting=3 / edge=2 / medical=2
#
# 被剔除的协议词 (附实测理由, 避免以后被重新加回来):
#   "Liquid AI"          -> 1 篇, 但唯一命中是 "Illiquid Markets" 金融论文 (纯误伤)
#   "neural ODE"         -> 715 篇, 噪声淹没主线, 且 selector 的 STRONG_KEYWORDS
#                           本就含 neural[- ]ode, 放开会把 715 篇灌进研读队列
#   "state-space liquid" -> 0 篇
#   "LNN finance"        -> 0 篇
#   "NCP liquid"         -> 0 篇
ARXIV_TERM_GROUPS: list[tuple[str, list[str]]] = [
    ("主线-LNN", ['all:"liquid neural network"', 'all:"liquid neural networks"']),
    ("主线-LTC", ['all:"liquid time-constant"', 'all:"liquid time constant"']),
    ("主线-CfC", ['all:"closed-form continuous-time"', 'all:"closed form continuous time"']),
    ("主线-NCP", ['all:"neural circuit policy"', 'all:"neural circuit policies"']),
    ("主线-state-space", ['all:"liquid structural state-space"']),
    ("基础模型", ['all:"liquid foundation model"', "all:LFM2"]),
    ("相关方法", ['all:"continuous-time RNN"']),
    # 布尔式: LNN 缩写与场景词各自出现即命中 (实测 UI 计数 LNN time series=8,
    # LNN robotics=4, LNN forecasting=3, LNN edge=2, LNN medical=2)
    ("应用场景", [
        "all:LNN AND all:edge",
        "all:LNN AND all:\"time series\"",
        "all:LNN AND all:forecasting",
        "all:LNN AND all:medical",
        "all:LNN AND all:robotics",
    ]),
]

def _strip_all_prefix(fragment: str) -> str:
    """`all:"foo bar"` -> `"foo bar"`;  `all:LFM2` -> `LFM2`; 其余原样。"""
    if not fragment.startswith("all:"):
        return fragment
    body = fragment[len("all:"):]
    if len(body) >= 2 and body.startswith('"') and body.endswith('"'):
        body = body[1:-1]
    return body


# 扁平**短语**列表 (供文档 / 探针脚本引用). **从 ARXIV_TERM_GROUPS 派生**, 避免两份
# 清单各自漂移 —— 2026-10-20 的审计根因就是"协议 §3 有一组词, 但脚本里没有"。
# 只收纯短语; 布尔 fragment (all:LNN AND all:edge) 不是短语, 由 ARXIV_TERM_GROUPS 承载。
ARXIV_TERMS: list[str] = [
    _strip_all_prefix(fragment)
    for _group, _terms in ARXIV_TERM_GROUPS
    for fragment in _terms
    if " AND " not in fragment
]

# RSS fallback categories — used when /api/query is throttled by Varnish (HTTP 406).
# These mirror the categories most likely to contain LNN / Neural ODE papers.
ARCHIVE_RSS_CATEGORIES = [
    "cs.LG",
    "cs.AI",
    "cs.NE",
    "cs.RO",
    "cs.SY",
]

# arXiv 重试退避 (秒). Varnish 的 406/read-timeout 窗口通常在 30~60s 内清掉.
ARXIV_RETRY_BACKOFF = (8, 30, 60)

# arXiv 查询组之间的间隔 (秒). 2026-10-20 实测 /api/query 处于 429 硬限流时,
# 连续请求会把所有组一起推进限流窗口; 留 3s 让 Varnish 窗口有机会滑动.
ARXIV_GROUP_PAUSE_S = 3

# 每日索引保留的**天数**上限 (子条目跟随其日期条目一起保留, 不单独计数).
INDEX_MAX_DAYS = 30
# digest markdown 表格最多渲染多少行; 摘要里的总数是全量, 表格是前 N, 必须同时标注
DIGEST_TABLE_LIMIT = 12

GITHUB_QUERIES = [
    '"liquid neural network"',
    '"liquid neural networks"',
    '"liquid time constant"',
    '"liquid time-constant"',
    '"closed-form continuous-time"',
    '"neural circuit policy"',
    "LTC CfC neural network",
    "LiquidAI LFM2",
]

HF_QUERIES = [
    "LiquidAI",
    "LFM2",
    "LFM2.5",
    "liquid neural",
    "closed-form continuous-time",
]

KEYWORD_RE = re.compile(
    r"liquid neural|liquid time[- ]constant|closed[- ]form continuous[- ]time|"
    r"neural circuit polic|liquid structural state[- ]space|\bCfC\b|\bLTC\b|\bLFM2|"
    # 2026-10-20: 补上基础模型 / 相关方法 / 应用场景三组的判定词, 否则新加的查询组
    # 抓回来的论文会被 keyword_score 判 0 分丢弃 (score<=0 直接 continue)。
    # 只包含实测有真实命中的词 —— "liquid ai" 唯一命中是金融 "Illiquid Markets" 论文,
    # 属误伤, 不收; "neural ode" 715 篇噪声, 交给 selector 的 STRONG_KEYWORDS 处理。
    r"liquid foundation model|continuous[- ]time rnn|\bLNN\b",
    re.IGNORECASE,
)


def request_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _rate_limit_wait(exc: Exception) -> int:
    """403 里还剩多少秒可等 —— 只认 GitHub 自己的限额头, 认不出就返回 0.

    `x-ratelimit-remaining: 0` 是一级限额 (search API 未认证 10 次/分);
    `retry-after` 多见于二级限额 (突发流量封禁). 两者都不是的话不当成限额,
    因为普通 403 (token 无效 / 权限不足) 等再久也没用。
    """
    if not isinstance(exc, urllib.error.HTTPError) or exc.code != 403:
        return 0
    head = {k.lower(): v for k, v in (exc.headers or {}).items()}
    if head.get("retry-after", "").isdigit():
        return min(int(head["retry-after"]), 120)
    if head.get("x-ratelimit-remaining", "").strip() == "0":
        try:
            reset = int(head["x-ratelimit-reset"])
        except (KeyError, ValueError):
            return 0
        # 限额窗口按 UTC 分钟走, 多等 2s 避免踩在边界上被再次拒.
        return max(0, min(reset - int(time.time()) + 2, 120))
    return 0


def request_json_rate_limited(
    url: str, headers: dict[str, str] | None = None, attempts: int = 2
) -> Any:
    """撞 GitHub 速率限额时等窗口重开, 而不是直接放弃这条查询.

    未认证的 search API 只有 10 次/分且**按出口 IP 计**, 一台机器上别的工具
    随手调一次 GitHub 就能把整轮 digest 的查询预算吃光。2026-10-19 就有 3/8 条
    查询这么被吞掉, 且失败只进 stderr, 产物里看不出来 (见 fetch_github_repos)。
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return request_json(url, headers)
        except Exception as exc:
            last = exc
            wait = _rate_limit_wait(exc)
            if not wait or attempt == attempts - 1:
                raise
            print(
                f"[warn] GitHub 速率限额, {wait}s 后重试 {attempt + 1}/{attempts - 1}: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise last  # pragma: no cover - 循环内必然 return 或 raise


def request_text(url: str, headers: dict[str, str] | None = None) -> str:
    # arXiv 返回 406 when Accept: */* 与复杂 OR 查询混用, 显式声明 atom+xml 即可.
    merged = {"User-Agent": USER_AGENT, "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.5"}
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, headers=merged)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def request_text_retry(url: str, backoff: tuple[int, ...] = ARXIV_RETRY_BACKOFF) -> str:
    """带退避重试的 arXiv 文本抓取.

    arXiv 前置的 Varnish 缓存会偶发 406 / read timeout —— 实测同一 URL 反复请求
    0.3~2.3s 即成功, 说明是窗口期抖动而非持续故障。旧实现只尝试一次, 一次抖动就
    让整轮退回前一日候选池 (09-22、09-23、10-12/13/14 连续命中), 退避按 Varnish
    清理窗口取到 60s。最后一次仍失败则抛出原异常, 由调用方走 RSS / 前日 fallback。
    """
    last: Exception | None = None
    for attempt, delay in enumerate((0, *backoff)):
        if delay:
            print(
                f"[warn] arXiv 请求失败 ({last}); {delay}s 后重试 {attempt + 1}/{len(backoff)}",
                file=sys.stderr,
            )
            time.sleep(delay)
        try:
            return request_text(url)
        except Exception as exc:
            last = exc
    assert last is not None
    raise last


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def slugify(value: str, limit: int = 96) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:limit] or "untitled"


def arxiv_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def keyword_score(title: str, summary: str) -> int:
    text = f"{title} {summary}"
    score = len(KEYWORD_RE.findall(text))
    lower = text.lower()
    if "cfc" in lower and "continuous" in lower and "time" in lower:
        score += 2
    if "ltc" in lower and ("neural" in lower or "network" in lower):
        score += 1
    return score


def fetch_arxiv(max_results: int, errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Query arXiv with SEVERAL small term groups instead of one giant OR.

    Why not one big OR: arXiv's `/api/query` throttles (406/429) aggressively as the
    query string grows. A 21-term OR is ~810 URL chars and reliably trips the
    throttle, which is exactly the failure the 2026-10-20 audit ran into.
    Splitting into themed groups keeps each request small, and a group that
    throttles no longer takes the other groups down with it.

    Groups mirror `docs/LNN_持续研究协议.md` §3 keyword table.

    **部分失败必须留痕**: 某组 429 限流时, 其余组照常产出, 但失败必须记进 `errors`
    —— 否则 digest 会把"只查到 5/8 组"显示成"查询全成功", 读者无从判断漏了几路。
    (与 GitHub/HF 侧同一纪律, 见 2026-10-19 的 a556476 修复。)
    """
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()

    for group_name, terms in ARXIV_TERM_GROUPS:
        group_papers, group_error = _fetch_arxiv_group(terms, max_results)
        if group_error:
            message = f"arXiv group {group_name!r} failed: {group_error}"
            print(f"[warn] {message}", file=sys.stderr)
            if errors is not None:
                errors.append(message)
            continue
        added = 0
        for paper in group_papers:
            if paper["id"] in seen:
                continue
            seen.add(paper["id"])
            papers.append(paper)
            added += 1
        print(
            f"[info] arXiv group {group_name!r}: {len(group_papers)} hits, {added} new",
            file=sys.stderr,
        )
        # arXiv 对连续请求敏感: 组间留间隔, 避免把后面的组也推进限流窗口
        time.sleep(ARXIV_GROUP_PAUSE_S)

    if not papers and errors is not None:
        raise RuntimeError("all arXiv term groups failed")
    return papers


def build_arxiv_query(terms: list[str]) -> str:
    """Build one arXiv `search_query` string from a list of fragments.

    A fragment may be:
      - a bare phrase          -> wrapped as `all:"phrase"`
      - an already-quoted phrase (`"foo bar"`) -> wrapped as `all:"foo bar"` (不重复加引号)
      - a complete fragment    -> used as-is (already contains `all:` / `AND` / `OR`)

    Fragments containing `AND` get explicit parentheses: when several fragments are
    joined with `OR`, we should not depend on arXiv's operator precedence to keep
    `A AND B` from splitting across the OR boundary.
    """
    fragments = []
    for term in terms:
        if "all:" in term or " AND " in term or " OR " in term:
            fragments.append(f"({term})" if " AND " in term and " OR " not in term else term)
        elif len(term) >= 2 and term.startswith('"') and term.endswith('"'):
            # 已经带引号的短语, 直接补 all: 前缀, 不要套第二层引号
            fragments.append(f"all:{term}")
        else:
            fragments.append(f'all:"{term}"')
    return " OR ".join(fragments)


def _fetch_arxiv_group(terms: list[str], max_results: int) -> tuple[list[dict[str, Any]], str]:
    """Fetch one arXiv term group. Returns (papers, error_message)."""
    query = build_arxiv_query(terms)
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        feed = request_text_retry(url)
        root = ET.fromstring(feed)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entry in root.findall("atom:entry", ns):
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        score = keyword_score(title, summary)
        if score <= 0:
            continue

        paper_id = clean_text(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = arxiv_id_from_url(paper_id)
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        authors = [
            clean_text(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        links = {link.attrib.get("title") or link.attrib.get("rel", ""): link.attrib.get("href", "") for link in entry.findall("atom:link", ns)}
        pdf_url = links.get("pdf", "")
        abs_url = links.get("alternate", paper_id)
        categories = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns)]
        papers.append(
            {
                "id": arxiv_id,
                "title": title,
                "authors": [author for author in authors if author],
                "published": clean_text(entry.findtext("atom:published", default="", namespaces=ns))[:10],
                "updated": clean_text(entry.findtext("atom:updated", default="", namespaces=ns))[:10],
                "summary": summary,
                "categories": [cat for cat in categories if cat],
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "keyword_score": score,
                "arxiv_group": query,
            }
        )
    return papers, ""


def fetch_arxiv_rss(max_results: int) -> list[dict[str, Any]]:
    """Fallback to arXiv category RSS feeds when /api/query is throttled (HTTP 406).

    Walks each category in ARCHIVE_RSS_CATEGORIES, extracts <item> entries, applies
    the same keyword filter, dedups by arxiv id. RSS feeds return atom-style items
    so parsing mirrors fetch_arxiv() but iterates ``atom:item`` inside ``rss/channel``.
    """
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    for category in ARCHIVE_RSS_CATEGORIES:
        url = f"https://export.arxiv.org/rss/{category}"
        try:
            feed = request_text_retry(url)
        except Exception as exc:
            print(f"[warn] RSS fetch failed for {category}: {exc}", file=sys.stderr)
            continue
        try:
            root = ET.fromstring(feed)
        except ET.ParseError as exc:
            print(f"[warn] RSS parse failed for {category}: {exc}", file=sys.stderr)
            continue
        channel = root.find("channel")
        if channel is None:
            continue
        for item in channel.findall("item"):
            title = clean_text(item.findtext("title", default=""))
            summary = clean_text(item.findtext("description", default=""))
            score = keyword_score(title, summary)
            if score <= 0:
                continue
            link = clean_text(item.findtext("link", default=""))
            arxiv_id = arxiv_id_from_url(link)
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            authors_raw = clean_text(item.findtext("author", default=""))
            authors = [authors_raw] if authors_raw else []
            pub = clean_text(item.findtext("pubDate", default=""))
            papers.append(
                {
                    "id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "published": pub[:10],
                    "updated": "",
                    "summary": summary,
                    "categories": [category],
                    "abs_url": link,
                    "pdf_url": link.replace("/abs/", "/pdf/") if "/abs/" in link else "",
                    "keyword_score": score,
                    "source": f"rss:{category}",
                }
            )
            if len(papers) >= max_results:
                break
        if len(papers) >= max_results:
            break
    return papers


def fetch_github_repos(per_query: int, errors: list[str] | None = None) -> list[dict[str, Any]]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        # search API 未认证只有 10 次/分 (且按出口 IP 计), 8 条查询挤在同一分钟里
        # 很容易被自己或同机其他工具的调用吃光预算。值得在 stderr 明说一次, 否则
        # 只看产物的人无法区分"限额"和"查询真的没结果"。
        print(
            "[warn] GITHUB_TOKEN 未设置: GitHub 查询按未认证限额 (10 次/分) 执行, "
            "失败概率显著升高。给 systemd service 配 EnvironmentFile 可根治。",
            file=sys.stderr,
        )

    repos: dict[str, dict[str, Any]] = {}
    for query in GITHUB_QUERIES:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_query,
            }
        )
        url = f"https://api.github.com/search/repositories?{params}"
        try:
            payload = request_json_rate_limited(url, headers=headers)
        except Exception as exc:
            print(f"[warn] GitHub query failed for {query!r}: {exc}", file=sys.stderr)
            # 未认证的 search API 限额很容易触发 (HTTP 403); 不记进 errors 的话,
            # 部分覆盖会以"查询全成功"的假象进 digest, 读者无从判断缺了几路查询.
            if errors is not None:
                errors.append(f"GitHub query failed for {query!r}: {exc}")
            continue

        for item in payload.get("items", []):
            full_name = item.get("full_name")
            if not full_name:
                continue
            current = repos.get(full_name)
            candidate = {
                "full_name": full_name,
                "description": clean_text(item.get("description") or ""),
                "html_url": item.get("html_url"),
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "is_fork": bool(item.get("fork")),
                "language": item.get("language"),
                "updated_at": (item.get("updated_at") or "")[:10],
                "pushed_at": (item.get("pushed_at") or "")[:10],
                "topics": item.get("topics", []),
                "query": query,
            }
            if current is None or candidate["stars"] > current.get("stars", 0):
                repos[full_name] = candidate
        time.sleep(1)

    return rank_repos(list(repos.values()))


def rank_repos(repos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse fork spam and order by research value, not by API order.

    2026-10-20 审计: GitHub 侧 42 个候选里有 8 个是同一个 `bebelm` 仓库的 fork
    (不同 owner、同名、0 star、同描述), 而 digest 表只渲染前 12 行 —— 结果
    **10 行是 0-star fork**, 真正有价值的 `Ipsedo/LiquidNetworks` (29★) 被挤到
    第 13 行, `Noah-hjl/ContiLNN` (官方 ContiLNN 实现) 被挤到第 21 行, 人类完全看不见。

    两步修:
    1. **折叠 fork 簇**: 同名 (repo 名 stem 相同) 的多个 owner 视为同一上游的 fork 群,
       保留 star 最多的那个为主条目, 并记录 `fork_variants` 计数 (不丢信息)。
       真正的 fork (GitHub `fork: true`) 一律降级到主条目后面。
    2. **按研究价值排序**: star 为主, 最近更新为辅 —— 而不是 API 的 "best match" 顺序。
    """
    # --- 1) 按 repo 名 stem 分组, 找出 fork 簇 ---------------------------------
    groups: dict[str, list[dict[str, Any]]] = {}
    for repo in repos:
        stem = (repo.get("full_name") or "").split("/")[-1].lower()
        groups.setdefault(stem, []).append(repo)

    kept: list[dict[str, Any]] = []
    for stem, members in groups.items():
        if len(members) == 1:
            repo = dict(members[0])
            repo["fork_variants"] = 0
            kept.append(repo)
            continue

        # 簇内: 非 fork 优先, 其次 star 多者为主条目
        def primary_key(repo: dict[str, Any]) -> tuple:
            return (
                not repo.get("is_fork"),
                repo.get("stars", 0) or 0,
                repo.get("updated_at") or "",
            )

        ordered = sorted(members, key=primary_key, reverse=True)
        head = dict(ordered[0])
        head["fork_variants"] = len(members) - 1
        head["fork_family"] = stem
        kept.append(head)
        # 簇内其余成员保留在池里 (JSON 里仍可查), 但明确标记为变体
        for variant in ordered[1:]:
            v = dict(variant)
            v["fork_variants"] = 0
            v["fork_of"] = head.get("full_name")
            v["is_fork_variant"] = True
            kept.append(v)

    # --- 2) 按研究价值排序 ----------------------------------------------------
    # star 降序; 同 star 用最近更新优先。is_fork_variant 永远排到普通条目之后。
    def value_key(repo: dict[str, Any]) -> tuple:
        return (
            not repo.get("is_fork_variant"),
            repo.get("stars", 0) or 0,
            repo.get("updated_at") or "",
        )

    return sorted(kept, key=value_key, reverse=True)


def fetch_huggingface_models(per_query: int, errors: list[str] | None = None) -> list[dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for query in HF_QUERIES:
        params = urllib.parse.urlencode({"search": query, "limit": per_query, "sort": "lastModified", "direction": -1})
        url = f"https://huggingface.co/api/models?{params}"
        try:
            payload = request_json(url)
        except Exception as exc:
            print(f"[warn] Hugging Face query failed for {query!r}: {exc}", file=sys.stderr)
            if errors is not None:
                errors.append(f"Hugging Face query failed for {query!r}: {exc}")
            continue

        for item in payload:
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            current = models.get(model_id)
            candidate = {
                "model_id": model_id,
                "author": item.get("author"),
                "last_modified": (item.get("lastModified") or "")[:10],
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "pipeline_tag": item.get("pipeline_tag"),
                "tags": item.get("tags", [])[:16],
                "url": f"https://huggingface.co/{model_id}",
                "query": query,
            }
            if current is None or candidate["downloads"] > current.get("downloads", 0):
                models[model_id] = candidate
        time.sleep(0.5)

    return sorted(models.values(), key=lambda model: (model.get("last_modified") or "", model.get("downloads") or 0), reverse=True)


def download_pdfs(papers: list[dict[str, Any]], output_dir: pathlib.Path, max_downloads: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for paper in papers[:max_downloads]:
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            continue
        name = f"{paper.get('published', 'unknown')}_{slugify(paper['title'])}_{paper['id']}.pdf"
        target = output_dir / name
        if target.exists():
            downloaded.append(str(target.relative_to(ROOT)))
            continue
        try:
            request = urllib.request.Request(
                pdf_url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
            downloaded.append(str(target.relative_to(ROOT)))
            time.sleep(1)
        except Exception as exc:
            print(f"[warn] PDF download failed for {paper['id']}: {exc}", file=sys.stderr)
    return downloaded


def md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def truncate(value: str, limit: int = 280) -> str:
    value = clean_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def render_daily_digest(
    run_date: str,
    papers: list[dict[str, Any]],
    repos: list[dict[str, Any]],
    models: list[dict[str, Any]],
    downloaded: list[str],
    errors: list[str],
) -> str:
    lines = [
        "---",
        f"title: LNN 每日研究追踪 - {run_date}",
        f"date: {run_date}",
        "tags: [LNN, daily, automation, arxiv, github, huggingface]",
        "---",
        "",
        f"# LNN 每日研究追踪 - {run_date}",
        "",
        "> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。",
        "",
        "## 摘要",
        f"- arXiv 候选论文：{len(papers)} 篇",
        f"- GitHub 候选仓库：{len(repos)} 个",
        f"- Hugging Face 候选模型：{len(models)} 个",
        f"- 已下载 PDF：{len(downloaded)} 个",
        f"- 下表每类仅渲染前 {DIGEST_TABLE_LIMIT} 条；全量候选池见 `papers/daily/{run_date}_lnn_research.json`，研读选择器以该 JSON 为准。",
    ]
    if errors:
        lines.extend(["", "## 数据源状态"])
        lines.extend(f"- `{error}`" for error in errors)
        lines.append("- 若当天已有历史结果，脚本会保留上一轮成功获取的数据，避免 transient API 错误清空候选池。")

    lines.extend(["", "## arXiv 候选论文"])

    if papers:
        lines.extend(["| 日期 | 论文 | 作者 | 摘要 |", "|---|---|---|---|"])
        for paper in papers[:DIGEST_TABLE_LIMIT]:
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += " 等"
            lines.append(
                "| {date} | [{title}]({url}) | {authors} | {summary} |".format(
                    date=paper.get("published") or paper.get("updated") or "",
                    title=md_escape(paper.get("title", "")),
                    url=paper.get("abs_url", ""),
                    authors=md_escape(authors),
                    summary=md_escape(truncate(paper.get("summary", ""))),
                )
            )
    else:
        lines.append("- 本次未发现通过关键词过滤的新候选论文。")

    lines.extend(["", "## GitHub 候选仓库"])
    if repos:
        lines.extend(["| 更新 | 仓库 | Star | Fork | 语言 | 说明 |", "|---|---|---:|---:|---|---|"])
        for repo in repos[:DIGEST_TABLE_LIMIT]:
            fork_note = ""
            if repo.get("fork_variants"):
                fork_note = f" (另有 {repo['fork_variants']} 个同名 fork)"
            lines.append(
                "| {updated} | [{name}]({url}){fnote} | {stars} | {forks} | {language} | {desc} |".format(
                    updated=repo.get("updated_at") or repo.get("pushed_at") or "",
                    name=md_escape(repo.get("full_name", "")),
                    url=repo.get("html_url", ""),
                    fnote=fork_note,
                    stars=repo.get("stars", 0),
                    forks=repo.get("forks", 0),
                    language=repo.get("language") or "",
                    desc=md_escape(truncate(repo.get("description", ""), 160)),
                )
            )
        variants = sum(1 for repo in repos if repo.get("is_fork_variant"))
        if variants:
            lines.append(
                f"- 同名 fork 已折叠 {variants} 条（见 `papers/daily/{run_date}_lnn_research.json` 的 "
                "`fork_variants` / `fork_of` 字段）"
            )
    else:
        lines.append("- 本次未发现可记录的 GitHub 仓库。")

    lines.extend(["", "## Hugging Face 候选模型"])
    if models:
        lines.extend(["| 更新 | 模型 | 下载 | Likes | 任务 |", "|---|---|---:|---:|---|"])
        for model in models[:DIGEST_TABLE_LIMIT]:
            lines.append(
                "| {updated} | [{model}]({url}) | {downloads} | {likes} | {task} |".format(
                    updated=model.get("last_modified") or "",
                    model=md_escape(model.get("model_id", "")),
                    url=model.get("url", ""),
                    downloads=model.get("downloads", 0),
                    likes=model.get("likes", 0),
                    task=model.get("pipeline_tag") or "",
                )
            )
    else:
        lines.append("- 本次未发现可记录的 Hugging Face 模型。")

    if downloaded:
        lines.extend(["", "## PDF 归档"])
        lines.extend(f"- `{path}`" for path in downloaded)

    lines.extend(
        [
            "",
            "## 建议动作",
            "- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。",
            "- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。",
            "- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。",
            "",
            "## 数据源",
            "- arXiv API: https://export.arxiv.org/api/query",
            "- GitHub Search API: https://docs.github.com/rest/search/search",
            "- Hugging Face Models API: https://huggingface.co/docs/hub/api",
            "",
        ]
    )
    return "\n".join(lines)


def render_repo_watchlist(run_date: str, repos: list[dict[str, Any]], models: list[dict[str, Any]]) -> str:
    lines = [
        "---",
        f"title: LNN 开源生态观察 - {run_date}",
        f"date: {run_date}",
        "tags: [LNN, repo-watchlist, automation]",
        "---",
        "",
        f"# LNN 开源生态观察 - {run_date}",
        "",
        "## GitHub 仓库",
    ]
    for repo in repos[:20]:
        topics = ", ".join(repo.get("topics", [])[:8])
        fork_note = ""
        if repo.get("is_fork_variant"):
            fork_note = f"- 归属：同源 fork 簇，主条目 `{repo.get('fork_of')}`"
        elif repo.get("fork_variants"):
            fork_note = f"- 同源 fork：另有 {repo['fork_variants']} 个同名 fork 已折叠"
        lines.extend(
            [
                f"### [{repo.get('full_name')}]({repo.get('html_url')})",
                f"- 更新：{repo.get('updated_at') or repo.get('pushed_at') or 'unknown'}",
                f"- Star / Fork：{repo.get('stars', 0)} / {repo.get('forks', 0)}",
                f"- 语言：{repo.get('language') or 'unknown'}",
                f"- Topics：{topics or '未标注'}",
                f"- 说明：{repo.get('description') or '无'}",
            ]
        )
        if fork_note:
            lines.append(fork_note)
        lines.append("")

    lines.append("## Hugging Face 模型")
    for model in models[:20]:
        tags = ", ".join(model.get("tags", [])[:8])
        lines.extend(
            [
                f"### [{model.get('model_id')}]({model.get('url')})",
                f"- 更新：{model.get('last_modified') or 'unknown'}",
                f"- 下载 / Likes：{model.get('downloads', 0)} / {model.get('likes', 0)}",
                f"- 任务：{model.get('pipeline_tag') or 'unknown'}",
                f"- Tags：{tags or '未标注'}",
                "",
            ]
        )
    return "\n".join(lines)


def update_marked_section(path: pathlib.Path, marker: str, content: str) -> None:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = f"{start}\n{content.rstrip()}\n{end}"
    if path.exists():
        original = path.read_text(encoding="utf-8")
    else:
        original = ""

    if start in original and end in original:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
        updated = pattern.sub(block, original)
    else:
        suffix = "\n\n" if original and not original.endswith("\n\n") else ""
        updated = f"{original}{suffix}{block}\n"
    path.write_text(updated, encoding="utf-8")


DAY_ENTRY_RE = re.compile(r"- \*\*\d{4}-\d\d-\d\d")


def split_index_entries(block: str) -> list[list[str]]:
    """把索引区块切成条目: 每条 = 一个顶格日期行 + 其后所有缩进/续行.

    旧实现按行过滤 ``- **`` 前缀再截断前 29 行, 有两个数据丢失通道:
      1. 缩进子条目 (每日复盘明细) 整段被丢弃;
      2. 老条目里 ``- **workaround**`` 这类**续行**也被当成条目计数, 29 行的容量
         先被续行吃光, 于是 10-06~10-01 这些真实日期条目被挤出索引 (2026-10-14
         重跑时实测: 42 个顶格 ``- **`` 里 12 个是续行)。

    **结构行必须丢掉, 不能当条目**: 区块开头的 ``## 4. 自动化每日追踪索引``
    不是日期条目, 落到孤儿分支后会变成一条独立条目, 而写回时 merge_existing
    又自己补一个标题 —— 于是每跑一天索引里就多一个重复标题 (10-16 4 个,
    10-19 涨到 7 个)。标题和 HTML 注释在这里没有任何信息量, 直接剔除。
    """
    entries: list[list[str]] = []
    for raw in block.splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#") or raw.lstrip().startswith("<!--"):
            continue  # 标题 / 注释: 由 merge_existing 统一重建
        if raw.startswith("- **") and DAY_ENTRY_RE.match(raw):
            entries.append([raw])
        elif entries:
            entries[-1].append(raw)
        else:
            entries.append([raw])
    return entries


def cap_index_entries(rows: list[str], max_days: int = INDEX_MAX_DAYS) -> list[str]:
    """按**天数**截断索引行 (子条目跟随所属日期条目, 不单独占容量).

    被挤掉的日期条目必须连同它的缩进明细一起丢弃, 否则会在上一个保留条目下面
    留下孤儿子项, 比原来的静默截断更难读。
    """
    days = 0
    skipped = False
    kept: list[str] = []
    for row in rows:
        if DAY_ENTRY_RE.match(row):
            skipped = days >= max_days
            if skipped:
                continue
            days += 1
        elif skipped:
            continue
        kept.append(row)
    return kept


def update_indexes(run_date: str, digest_path: pathlib.Path, papers: list[dict[str, Any]], repos: list[dict[str, Any]], models: list[dict[str, Any]]) -> None:
    rel_digest = digest_path.relative_to(ROOT).as_posix()
    summary_path = ROOT / "docs" / "Liquid_Neural_Networks_Latest_Papers_Summary.md"
    deep_path = ROOT / "docs" / "LNN_深度研读报告.md"
    line = f"- **{run_date}**：[[{rel_digest}|每日追踪]]，候选论文 {len(papers)} 篇，仓库 {len(repos)} 个，模型 {len(models)} 个。"

    def merge_existing(path: pathlib.Path, heading: str) -> str:
        existing: list[str] = []
        if path.exists():
            text = path.read_text(encoding="utf-8")
            match = re.search(r"<!-- daily-lnn-index:start -->(.*?)<!-- daily-lnn-index:end -->", text, re.DOTALL)
            if match:
                for chunk in split_index_entries(match.group(1)):
                    if any(f"**{run_date}**" in row for row in chunk):
                        continue  # 今日条目由本次运行重写
                    existing.extend(chunk)
                existing = cap_index_entries(existing)
        items = [line, *existing]
        return f"## {heading}\n\n" + "\n".join(items)

    update_marked_section(summary_path, "daily-lnn-index", merge_existing(summary_path, "4. 自动化每日追踪索引"))
    update_marked_section(deep_path, "daily-lnn-index", merge_existing(deep_path, "4. 自动化追踪与待研读队列"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--max-results", type=int, default=25, help="Maximum arXiv results to request.")
    parser.add_argument("--per-query", type=int, default=8, help="Maximum GitHub/Hugging Face results per query.")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDFs for top arXiv hits.")
    parser.add_argument("--max-pdf-downloads", type=int, default=5, help="Maximum PDFs to download when --download-pdfs is set.")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub Search API calls.")
    parser.add_argument("--skip-huggingface", action="store_true", help="Skip Hugging Face model API calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_date = args.date
    (ROOT / "docs" / "daily").mkdir(parents=True, exist_ok=True)
    (ROOT / "papers" / "daily").mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis" / "repo_watchlist").mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "papers" / "daily" / f"{run_date}_lnn_research.json"
    previous_payload: dict[str, Any] = {}
    if json_path.exists():
        try:
            previous_payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_payload = {}

    errors: list[str] = []
    papers: list[dict[str, Any]] = []
    try:
        # errors 传进去: 单组失败要留痕, 否则"5/8 组成功"会被显示成"查询全成功"
        papers = fetch_arxiv(args.max_results, errors)
        if not papers:
            raise RuntimeError("arXiv /api/query returned 0 papers (likely HTTP 406 throttle)")
    except Exception as exc:
        primary_error = f"arXiv /api/query failed: {exc}"
        print(f"[warn] {primary_error}", file=sys.stderr)
        errors.append(primary_error)
        # Fallback to RSS feeds (less likely to be Varnish-throttled).
        try:
            rss_papers = fetch_arxiv_rss(args.max_results)
            if rss_papers:
                papers = rss_papers
                errors.append(
                    f"arXiv RSS fallback recovered {len(rss_papers)} papers across "
                    f"{', '.join(ARCHIVE_RSS_CATEGORIES)}"
                )
            else:
                errors.append("arXiv RSS fallback returned 0 papers (likely weekend or empty category)")
        except Exception as rss_exc:
            errors.append(f"arXiv RSS fallback failed: {rss_exc}")

    repos = []
    if not args.skip_github:
        repos = fetch_github_repos(args.per_query, errors)

    models = []
    if not args.skip_huggingface:
        models = fetch_huggingface_models(args.per_query, errors)

    previous_papers = previous_payload.get("papers") or []
    previous_repos = previous_payload.get("github_repos") or []
    previous_models = previous_payload.get("huggingface_models") or []
    if previous_papers and len(papers) < len(previous_papers):
        print("[warn] Keeping previous arXiv result set because the current run returned fewer items.", file=sys.stderr)
        papers = previous_papers
    if previous_repos and len(repos) < len(previous_repos):
        print("[warn] Keeping previous GitHub result set because the current run returned fewer items.", file=sys.stderr)
        repos = previous_repos
    if previous_models and len(models) < len(previous_models):
        print("[warn] Keeping previous Hugging Face result set because the current run returned fewer items.", file=sys.stderr)
        models = previous_models

    daily_dir = ROOT / "papers" / "daily"
    downloaded: list[str] = []
    if args.download_pdfs:
        downloaded = download_pdfs(papers, daily_dir / run_date, args.max_pdf_downloads)

    payload = {
        "date": run_date,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "errors": errors,
        "papers": papers,
        "github_repos": repos,
        "huggingface_models": models,
        "downloaded_pdfs": downloaded,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    digest_path = ROOT / "docs" / "daily" / f"{run_date}_LNN_research_digest.md"
    digest_path.write_text(render_daily_digest(run_date, papers, repos, models, downloaded, errors), encoding="utf-8")

    watchlist_path = ROOT / "analysis" / "repo_watchlist" / f"{run_date}_lnn_open_source_watchlist.md"
    watchlist_path.write_text(render_repo_watchlist(run_date, repos, models), encoding="utf-8")

    update_indexes(run_date, digest_path, papers, repos, models)

    print(
        textwrap.dedent(
            f"""
            Daily LNN research update complete.
            - digest: {digest_path.relative_to(ROOT)}
            - data: {json_path.relative_to(ROOT)}
            - repo watchlist: {watchlist_path.relative_to(ROOT)}
            - papers/repos/models: {len(papers)}/{len(repos)}/{len(models)}
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
