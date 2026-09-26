#!/usr/bin/env python3
"""probe what PDF libs are available"""
import sys
for m in ["fitz", "pypdf", "pdfplumber", "PyPDF2"]:
    try:
        __import__(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"no  {m}: {type(e).__name__}")
