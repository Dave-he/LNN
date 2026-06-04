"""Tests for scripts/loop_status.py --prd-status parser (iter#21, PRD §10 #5).

Covers:
1. Section header detection (## 8., ## §9., ## §10.)
2. ID format tolerance (pure int + N-M)
3. Done-marker detection in title and last-cell (status column)
4. Pending-marker detection
5. Blocker keyword extraction
6. End-to-end CLI smoke
"""

import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Add scripts/ to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import loop_status  # noqa: E402


def _with_fake_prd(tmp_path: Path, body: str):
    """Create a fake ROOT/docs/PRD_LNN_Edge_Research.md and return rows.

    The parser hardcodes `ROOT / "docs" / "PRD_LNN_Edge_Research.md"`, so we
    must build that exact layout inside the temp dir.
    """
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    prd = docs / "PRD_LNN_Edge_Research.md"
    prd.write_text(body)
    orig_ROOT = loop_status.ROOT
    loop_status.ROOT = tmp_path
    try:
        return loop_status._parse_all_prd_sections()
    finally:
        loop_status.ROOT = orig_ROOT


# -------------------------------------------------------- 1. section header
def test_section_header_in_real_prd():
    prd = Path(__file__).resolve().parents[1] / "docs" / "PRD_LNN_Edge_Research.md"
    assert prd.exists()
    text = prd.read_text(encoding="utf-8")
    n8 = len(re.findall(r"^##\s*8\.\s", text, re.MULTILINE))
    n9 = len(re.findall(r"^##\s*§?9\.\s", text, re.MULTILINE))
    n10 = len(re.findall(r"^##\s*§?10\.\s", text, re.MULTILINE))
    assert n8 + n9 + n10 >= 3, "PRD must contain §8/§9/§10"


# ---------------------------------------------------------- 2. ID format
def test_pure_int_id():
    with tempfile.TemporaryDirectory() as d:
        rows = _with_fake_prd(Path(d), textwrap.dedent("""\
            # PRD
            ## §8. P0
            | # | 任务 | 出口物 |
            |---:|---|---|
            | 1 | task one | x |
            | 2 | task two ✅ | y |
            """))
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[1]["id"] == 2
        assert rows[1]["status"] == "completed"


def test_dash_id():
    with tempfile.TemporaryDirectory() as d:
        rows = _with_fake_prd(Path(d), textwrap.dedent("""\
            # PRD
            ## §10. Third wave
            | # | 任务 | 出口物 | 状态 |
            |---:|---|---|---|
            | 10-1 | task one | x | pending |
            | 10-2 | task two | y | ✅ done |
            """))
        assert len(rows) == 2
        assert rows[0]["id_str"] == "10-1"
        assert rows[0]["id"] == 1
        assert rows[0]["status"] == "pending"
        assert rows[1]["status"] == "completed"


# ---------------------------------------------------------- 3. done markers
def test_checkmark_in_title():
    with tempfile.TemporaryDirectory() as d:
        rows = _with_fake_prd(Path(d), textwrap.dedent("""\
            ## §8. P0
            | # | 任务 | 出口物 |
            |---:|---|---|
            | 1 | task ✅ | x |
            | 2 | task [done] | y |
            | 3 | task loop#3 ✅ | z |
            | 4 | task plain | w |
            """))
        assert [r["status"] for r in rows] == ["completed", "completed", "completed", "pending"]


def test_checkmark_in_status_column():
    with tempfile.TemporaryDirectory() as d:
        rows = _with_fake_prd(Path(d), textwrap.dedent("""\
            ## §10. Wave
            | # | 任务 | 出口物 | 状态 |
            |---:|---|---|---|
            | 10-1 | task | x | pending |
            | 10-2 | task | y | ✅ (iter#21) |
            """))
        assert rows[0]["status"] == "pending"
        assert rows[1]["status"] == "completed"


# ---------------------------------------------------------- 4. pending markers
def test_pending_in_title_lowercase():
    with tempfile.TemporaryDirectory() as d:
        rows = _with_fake_prd(Path(d), textwrap.dedent("""\
            ## §8. P0
            | # | 任务 | 出口物 |
            |---:|---|---|
            | 1 | task pending (RAM blocker) | x |
            """))
        assert rows[0]["status"] == "pending"
        assert "RAM" in rows[0]["blockers"]


# ---------------------------------------------------------- 5. blocker
def test_multiple_blocker_hints():
    with tempfile.TemporaryDirectory() as d:
        rows = _with_fake_prd(Path(d), textwrap.dedent("""\
            ## §8. P0
            | # | 任务 | 出口物 |
            |---:|---|---|
            | 1 | LFM2.5 in 空载 window with CUDA + THUMOS-14 data | x |
            """))
        assert "空载" in rows[0]["blockers"]
        assert "CUDA" in rows[0]["blockers"]
        assert "THUMOS-14" in rows[0]["blockers"]


# ---------------------------------------------------------- 6. CLI smoke
def test_prd_status_cli_runs():
    """End-to-end: invoking --prd-status on the real PRD must succeed."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "loop_status.py"),
         "--prd-status", "--no-write"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    out = result.stdout
    assert "§8:" in out
    assert "§9:" in out
    assert "§10:" in out
