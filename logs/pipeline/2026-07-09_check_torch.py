#!/usr/bin/env python3
"""Quick check: which python on this host has torch installed?"""
import subprocess
import sys

candidates = [
    "/Users/hyx/miniconda3/bin/python3",
    "/usr/local/bin/python3.13",
    "/usr/local/bin/python3.12",
    "/Users/hyx/workspace/LNN/.venv/bin/python3",
    "/usr/local/bin/python3",
]
for c in candidates:
    try:
        r = subprocess.run(
            [c, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=20,
        )
        out = (r.stdout or "") + (r.stderr or "")
        ok = r.returncode == 0
        print(f"  {c}  {'OK' if ok else 'FAIL'}: {out.strip()[:160]}")
    except Exception as e:
        print(f"  {c}  ERR {e}")
