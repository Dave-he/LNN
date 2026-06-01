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
USER_AGENT = "LNN-research-tracker/1.0 (https://github.com/Dave-he/LNN)"

ARXIV_TERMS = [
    "liquid neural network",
    "liquid neural networks",
    "liquid time-constant",
    "liquid time constant",
    "closed-form continuous-time",
    "closed form continuous time",
    "neural circuit policy",
    "neural circuit policies",
    "liquid structural state-space",
]

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
    r"neural circuit polic|liquid structural state[- ]space|\bCfC\b|\bLTC\b|\bLFM2",
    re.IGNORECASE,
)


def request_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def request_text(url: str, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


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


def fetch_arxiv(max_results: int) -> list[dict[str, Any]]:
    query = " OR ".join(f'all:"{term}"' for term in ARXIV_TERMS)
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
    feed = request_text(url)
    root = ET.fromstring(feed)
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
            }
        )
    return papers


def fetch_github_repos(per_query: int) -> list[dict[str, Any]]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

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
            payload = request_json(url, headers=headers)
        except Exception as exc:
            print(f"[warn] GitHub query failed for {query!r}: {exc}", file=sys.stderr)
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
                "language": item.get("language"),
                "updated_at": (item.get("updated_at") or "")[:10],
                "pushed_at": (item.get("pushed_at") or "")[:10],
                "topics": item.get("topics", []),
                "query": query,
            }
            if current is None or candidate["stars"] > current.get("stars", 0):
                repos[full_name] = candidate
        time.sleep(1)

    return sorted(repos.values(), key=lambda repo: (repo.get("updated_at") or "", repo.get("stars") or 0), reverse=True)


def fetch_huggingface_models(per_query: int) -> list[dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for query in HF_QUERIES:
        params = urllib.parse.urlencode({"search": query, "limit": per_query, "sort": "lastModified", "direction": -1})
        url = f"https://huggingface.co/api/models?{params}"
        try:
            payload = request_json(url)
        except Exception as exc:
            print(f"[warn] Hugging Face query failed for {query!r}: {exc}", file=sys.stderr)
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
            request = urllib.request.Request(pdf_url, headers={"User-Agent": USER_AGENT})
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
    ]
    if errors:
        lines.extend(["", "## 数据源状态"])
        lines.extend(f"- `{error}`" for error in errors)
        lines.append("- 若当天已有历史结果，脚本会保留上一轮成功获取的数据，避免 transient API 错误清空候选池。")

    lines.extend(["", "## arXiv 候选论文"])

    if papers:
        lines.extend(["| 日期 | 论文 | 作者 | 摘要 |", "|---|---|---|---|"])
        for paper in papers[:12]:
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
        lines.extend(["| 更新 | 仓库 | Star | 语言 | 说明 |", "|---|---|---:|---|---|"])
        for repo in repos[:12]:
            lines.append(
                "| {updated} | [{name}]({url}) | {stars} | {language} | {desc} |".format(
                    updated=repo.get("updated_at") or repo.get("pushed_at") or "",
                    name=md_escape(repo.get("full_name", "")),
                    url=repo.get("html_url", ""),
                    stars=repo.get("stars", 0),
                    language=repo.get("language") or "",
                    desc=md_escape(truncate(repo.get("description", ""), 160)),
                )
            )
    else:
        lines.append("- 本次未发现可记录的 GitHub 仓库。")

    lines.extend(["", "## Hugging Face 候选模型"])
    if models:
        lines.extend(["| 更新 | 模型 | 下载 | Likes | 任务 |", "|---|---|---:|---:|---|"])
        for model in models[:12]:
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
        lines.extend(
            [
                f"### [{repo.get('full_name')}]({repo.get('html_url')})",
                f"- 更新：{repo.get('updated_at') or repo.get('pushed_at') or 'unknown'}",
                f"- Star / Fork：{repo.get('stars', 0)} / {repo.get('forks', 0)}",
                f"- 语言：{repo.get('language') or 'unknown'}",
                f"- Topics：{topics or '未标注'}",
                f"- 说明：{repo.get('description') or '无'}",
                "",
            ]
        )

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
                existing = [item.strip() for item in match.group(1).splitlines() if item.strip().startswith("- **")]
        existing = [item for item in existing if f"**{run_date}**" not in item]
        items = [line, *existing[:29]]
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
    try:
        papers = fetch_arxiv(args.max_results)
    except Exception as exc:
        papers = []
        errors.append(f"arXiv fetch failed: {exc}")
        print(f"[warn] {errors[-1]}", file=sys.stderr)

    repos = []
    if not args.skip_github:
        repos = fetch_github_repos(args.per_query)

    models = []
    if not args.skip_huggingface:
        models = fetch_huggingface_models(args.per_query)

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
