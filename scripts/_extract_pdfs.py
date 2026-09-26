#!/usr/bin/env python3
"""Extract full text from each arXiv PDF via PyMuPDF, save to /tmp."""
import fitz, pathlib, sys

pdfs = [
    ("/tmp/2608.28702.pdf", "/tmp/2608.28702.txt"),
    ("/tmp/2609.28716.pdf", "/tmp/2609.28716.txt"),
]

for src, dst in pdfs:
    if not pathlib.Path(src).exists():
        print(f"missing {src}"); sys.exit(1)
    doc = fitz.open(src)
    pages_out = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        pages_out.append(f"\n\n=== PAGE {i+1}/{len(doc)} ===\n\n" + txt)
    body = "".join(pages_out)
    pathlib.Path(dst).write_text(body, encoding="utf-8")
    print(f"wrote {dst}  pages={len(doc)}  bytes={len(body)}")
    doc.close()
