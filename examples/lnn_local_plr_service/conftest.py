"""pytest bootstrap for examples/lnn_local_plr_service.

Mirrors round 133's lnn_local_service pattern: this directory is **not**
a top-level package, so the test module can't ``import server`` directly.
We add this directory to ``sys.path`` at collection time.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
