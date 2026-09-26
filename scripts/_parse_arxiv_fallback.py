#!/usr/bin/env python3
"""
手工 fallback: 把 curl 抓到的 /tmp/arxiv_<date>.xml 解析为 digest 内的 arxiv 表格,
同时输出 JSON 给后续研读报告阶段使用.

不做网络请求, 只读本地 XML.
"""
import argparse, json, re, sys, pathlib, datetime, html
import xml.etree.ElementTree as ET

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
KEYWORD_RE = re.compile(
    r"liquid neural|liquid time[- ]constant|closed[- ]form continuous[- ]time|"
    r"neural circuit polic|liquid structural state[- ]space|\bCfC\b|\bLTC\b|\bLFM2|"
    r"neural[- ]ode|continuous[- ]depth|continuous[- ]time",
    re.IGNORECASE,
)

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()

def keyword_score(title: str, summary: str) -> int:
    text = f"{title} {summary}"
    score = len(KEYWORD_RE.findall(text))
    lower = text.lower()
    if "cfc" in lower and ("continuous" in lower) and ("time" in lower):
        score += 2
    if "ltc" in lower and ("neural" in lower or "network" in lower):
        score += 1
    return score

def parse(xml_path: pathlib.Path):
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    rows = []
    for entry in root.findall("atom:entry", NS):
        title = clean(entry.findtext("atom:title", default="", namespaces=NS))
        summary = clean(entry.findtext("atom:summary", default="", namespaces=NS))
        score = keyword_score(title, summary)
        if score <= 0:
            continue
        paper_id = clean(entry.findtext("atom:id", default="", namespaces=NS))
        arxiv_id = paper_id.rstrip("/").split("/")[-1]
        published = clean(entry.findtext("atom:published", default="", namespaces=NS))[:10]
        authors = [clean(a.findtext("atom:name", default="", namespaces=NS))
                   for a in entry.findall("atom:author", NS)]
        pdf_url = abs_url = ""
        for link in entry.findall("atom:link", NS):
            t = link.attrib.get("title") or link.attrib.get("rel", "")
            h = link.attrib.get("href", "")
            if t == "pdf":
                pdf_url = h
            elif t == "alternate":
                abs_url = h
        rows.append({
            "id": arxiv_id,
            "date": published,
            "title": title,
            "authors": ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""),
            "summary": summary,
            "abs_url": abs_url or f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
            "score": score,
        })
    rows.sort(key=lambda r: (r["score"], r["date"]), reverse=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    rows = parse(pathlib.Path(args.xml))
    pathlib.Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] kept {len(rows)} arxiv candidates with score>0")
    for r in rows[:5]:
        print(f"  score={r['score']} {r['date']} {r['id']} :: {r['title'][:80]}")

if __name__ == "__main__":
    main()
