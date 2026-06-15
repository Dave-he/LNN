"""pytest bootstrap for examples/lnn_local_service.

The service is intentionally *not* a top-level package — it lives under
``examples/`` which is not declared as a Python package in this repo
(otherwise every ``import examples.foo`` from random scripts would start
to matter).  This conftest makes ``server`` importable from the test
module by adding this directory to ``sys.path`` at collection time.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
