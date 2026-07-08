#!/usr/bin/env python3
"""Inspect the 1 un-reported digest paper to see whether it deserves an LNN
   研读报告 on a relaxed rule."""
import urllib.request
import re

ARXIV_ID = "2606.21295"
try:
    with urllib.request.urlopen(
        f"https://export.arxiv.org/api/query?id_list={ARXIV_ID}", timeout=15
    ) as r:
        text = r.read().decode("utf-8", "ignore")
    title_m = re.search(r"<title>(.*?)</title>", text, re.S)
    abs_m = re.search(r"<summary>(.*?)</summary>", text, re.S)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else "?"
    summary = re.sub(r"\s+", " ", abs_m.group(1)).strip() if abs_m else "?"
    print("TITLE:", title)
    print("SUMMARY:", summary[:1500])
    blob = (title + " " + summary).lower()
    kws = [
        "liquid", "cfc", "ltc", "ncp", "closed-form", "continuous-time",
        "neural ode", "continuous-depth", "neural circuit polic",
    ]
    for kw in kws:
        if kw in blob:
            print(f"  HIT: {kw!r}")
except Exception as e:
    print("err", e)
