"""Tests for lnn.core.arxiv_catalog (round 72).

The catalog is offline-first; all tests use ``offline=True`` so the test
suite does not depend on the public arXiv API being reachable.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from lnn.core.arxiv_catalog import (
    ArxivCatalog,
    CatalogEntry,
    REPRO_STATUS_FAILED,
    REPRO_STATUS_REPRODUCED,
    REPRO_STATUS_UNTESTED,
    REPRO_STATUS_VENDOR,
    REPRO_STATUS_VALUES,
    VerifyResult,
    build_default_catalog,
)


# --------------------------------------------------------------------------- #
# 1. seed catalog has the foundational entries
# --------------------------------------------------------------------------- #
def test_seed_catalog_has_foundational_lnn_papers():
    cat = build_default_catalog()
    assert cat.get("2106.13898") is not None  # CfC
    assert cat.get("2006.04439") is not None  # LTC
    assert cat.get("1706.01350") is not None  # NCP
    assert cat.get("2111.00396") is not None  # S4
    assert cat.get("2312.00752") is not None  # Mamba
    assert cat.get("2605.24047") is not None  # EMMA CVPR'26


# --------------------------------------------------------------------------- #
# 2. the three known mismatches are flagged
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arxiv_id",
    ["2003.06236", "2203.14343", "2002.08071"],
)
def test_known_mismatches_flagged_offline(arxiv_id):
    cat = build_default_catalog()
    res = cat.verify_id(arxiv_id, offline=True)
    assert res.catalog_match is False, f"{arxiv_id} should be flagged as MISMATCH"
    assert res.is_mismatch is True


# --------------------------------------------------------------------------- #
# 3. correct entries verify clean
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arxiv_id",
    ["2106.13898", "2006.04439", "1706.01350", "2111.00396", "2312.00752"],
)
def test_correct_entries_pass_offline(arxiv_id):
    cat = build_default_catalog()
    res = cat.verify_id(arxiv_id, offline=True)
    assert res.catalog_match is True
    assert res.is_mismatch is False
    # API not called in offline mode
    assert res.api_match is None
    assert res.api_actual_title is None


# --------------------------------------------------------------------------- #
# 4. unknown ID is accepted as not-in-catalog (not a mismatch)
# --------------------------------------------------------------------------- #
def test_unknown_id_not_mismatch():
    cat = build_default_catalog()
    res = cat.verify_id("9999.99999", offline=True)
    assert res.catalog_match is False  # not in catalog
    assert res.api_match is None  # offline
    # Without API title to compare, we treat unknown as not-yet-mismatch;
    # CLI scripts flag unknown IDs separately.
    assert res.is_mismatch is False


# --------------------------------------------------------------------------- #
# 5. add_reproduction_status updates the entry
# --------------------------------------------------------------------------- #
def test_add_reproduction_status_updates_existing():
    cat = build_default_catalog()
    cat.add_reproduction_status(
        "2106.13898", REPRO_STATUS_FAILED, notes="CfC failed to beat ODE-RNN in our re-run"
    )
    e = cat.get("2106.13898")
    assert e.reproduction_status == REPRO_STATUS_FAILED
    assert "CfC failed to beat" in e.notes


def test_add_reproduction_status_creates_stub_for_unknown():
    cat = build_default_catalog()
    e = cat.add_reproduction_status("2602.01234", REPRO_STATUS_VENDOR)
    assert e.arxiv_id == "2602.01234"
    assert e.reproduction_status == REPRO_STATUS_VENDOR
    # Title is the stub
    assert "unknown" in e.expected_title.lower()


# --------------------------------------------------------------------------- #
# 6. invalid reproduction_status raises
# --------------------------------------------------------------------------- #
def test_invalid_reproduction_status_raises():
    cat = build_default_catalog()
    with pytest.raises(ValueError):
        cat.add_reproduction_status("2106.13898", "garbage_status")


# --------------------------------------------------------------------------- #
# 7. invalid arXiv ID format raises on add()
# --------------------------------------------------------------------------- #
def test_add_invalid_arxiv_id_raises():
    cat = build_default_catalog()
    with pytest.raises(ValueError):
        cat.add(CatalogEntry(arxiv_id="not-an-id", expected_title="x"))


# --------------------------------------------------------------------------- #
# 8. status_counts aggregates correctly
# --------------------------------------------------------------------------- #
def test_status_counts_aggregate():
    cat = build_default_catalog()
    counts = cat.status_counts()
    assert set(counts.keys()) == REPRO_STATUS_VALUES
    # Sanity: at least one entry per status is present in the seed.
    assert counts[REPRO_STATUS_REPRODUCED] >= 1
    assert counts[REPRO_STATUS_FAILED] >= 1
    assert counts[REPRO_STATUS_VENDOR] >= 1
    assert counts[REPRO_STATUS_UNTESTED] >= 1
    assert sum(counts.values()) == len(cat.entries)


# --------------------------------------------------------------------------- #
# 9. known_mismatches returns exactly the three flagged IDs
# --------------------------------------------------------------------------- #
def test_known_mismatches_returns_three():
    cat = build_default_catalog()
    mm = cat.known_mismatches()
    ids = {e.arxiv_id for e in mm}
    assert ids == {"2003.06236", "2203.14343", "2002.08071"}


# --------------------------------------------------------------------------- #
# 10. scan_text extracts IDs from free-form text
# --------------------------------------------------------------------------- #
def test_scan_text_finds_arxiv_ids():
    cat = build_default_catalog()
    text = (
        "CfC is at 2106.13898, LTC at 2006.04439. "
        "A bad citation might be 2003.06236. "
        "And another: arxiv.org/abs/2312.00752 is Mamba. "
        "Same ID again 2106.13898 should not duplicate."
    )
    found = cat.scan_text(text)
    assert found == ["2106.13898", "2006.04439", "2003.06236", "2312.00752"]


# --------------------------------------------------------------------------- #
# 11. cache file round-trip persists API results
# --------------------------------------------------------------------------- #
def test_cache_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "cache.json"
        cat1 = ArxivCatalog(cache_path=cache_path)
        # Manually populate the in-memory cache and flush.
        cat1._api_cache["2106.13898"] = (1_700_000_000.0, "CfC title", True)
        cat1._flush_cache()
        assert cache_path.exists()

        cat2 = ArxivCatalog(cache_path=cache_path)
        # Fresh verify_id (offline) should use the warm cache.
        # We do not call the API — we just check the in-memory map.
        assert "2106.13898" in cat2._api_cache
        ts, title, ok = cat2._api_cache["2106.13898"]
        assert title == "CfC title"
        assert ok is True


# --------------------------------------------------------------------------- #
# 12. titles_loosely_match handles punctuation + case
# --------------------------------------------------------------------------- #
def test_titles_loosely_match_helper():
    m = ArxivCatalog._titles_loosely_match
    assert m("Mamba: Linear-Time Sequence Modeling", "Mamba Linear Time Sequence Modeling") is True
    assert m("Completely Different Title", "Another Title Entirely") is False
    # Substring is allowed (handles MISMATCH suffix).
    assert m("Diagonal State Spaces", "Diagonal State Spaces MISMATCH") is True


# --------------------------------------------------------------------------- #
# 13. harvest_from_daily_json adds stubs for unknown IDs
# --------------------------------------------------------------------------- #
def test_harvest_from_daily_json_adds_unknown_ids(tmp_path):
    daily = {
        "date": "2026-06-06",
        "papers": [
            {
                "id": "2604.10815v1",
                "title": "MeloTune: On-Device Arousal Learning",
                "authors": ["Alice", "Bob"],
                "published": "2026-04-10",
                "categories": ["cs.SD", "cs.AI"],
                "keyword_score": 4,
            },
            {
                "id": "2604.18274v2",
                "title": "LiquidTAD: Efficient Temporal Action Detection",
                "authors": ["Carol"],
                "published": "2026-04-20",
                "categories": ["cs.CV"],
                "keyword_score": 5,
            },
        ],
    }
    p = tmp_path / "daily.json"
    p.write_text(json.dumps(daily))

    cat = build_default_catalog()
    new = cat.harvest_from_daily_json(p)

    assert set(new.keys()) == {"2604.10815", "2604.18274"}
    e1 = cat.get("2604.10815")
    assert e1 is not None
    assert e1.expected_title.startswith("MeloTune")
    assert e1.reproduction_status == REPRO_STATUS_VENDOR
    assert "daily.json" in e1.notes  # source filename mentioned in provenance
    assert e1.year == "2026"


# --------------------------------------------------------------------------- #
# 14. harvest does NOT override existing entries
# --------------------------------------------------------------------------- #
def test_harvest_does_not_override_existing(tmp_path):
    daily = {
        "papers": [
            {
                "id": "2106.13898v1",
                "title": "WRONG TITLE — should not override",
                "authors": ["X"],
                "published": "2021-01-01",
                "categories": [],
                "keyword_score": 0,
            },
        ],
    }
    p = tmp_path / "daily.json"
    p.write_text(json.dumps(daily))

    cat = build_default_catalog()
    new = cat.harvest_from_daily_json(p)
    # 2106.13898 is in seed; should not be added again
    assert new == {}
    e = cat.get("2106.13898")
    assert "Closed-form Continuous-Depth" in e.expected_title  # original title


# --------------------------------------------------------------------------- #
# 15. harvest strips arXiv version suffix (v1, v2, ...)
# --------------------------------------------------------------------------- #
def test_harvest_strips_version_suffix(tmp_path):
    daily = {
        "papers": [
            {
                "id": "2604.18274v2",
                "title": "LiquidTAD v2",
                "authors": ["X"],
                "published": "2026-04-20",
                "categories": [],
            }
        ]
    }
    p = tmp_path / "daily.json"
    p.write_text(json.dumps(daily))
    cat = build_default_catalog()
    new = cat.harvest_from_daily_json(p)
    assert "2604.18274" in new
    # The stored id should NOT include the v2
    assert "2604.18274v2" not in cat.entries


# --------------------------------------------------------------------------- #
# 16. harvest skips malformed IDs
# --------------------------------------------------------------------------- #
def test_harvest_skips_malformed_ids(tmp_path):
    daily = {
        "papers": [
            {"id": "not-an-id", "title": "X"},
            {"id": "", "title": "Y"},
            {"id": "2605.12345", "title": "Z"},
        ]
    }
    p = tmp_path / "daily.json"
    p.write_text(json.dumps(daily))
    cat = build_default_catalog()
    new = cat.harvest_from_daily_json(p)
    assert "2605.12345" in new
    assert "not-an-id" not in new
    assert "" not in new


# --------------------------------------------------------------------------- #
# 17. harvest_arxiv_catalog convenience helper
# --------------------------------------------------------------------------- #
def test_harvest_arxiv_catalog_helper(tmp_path):
    from lnn.core.arxiv_catalog import harvest_arxiv_catalog

    (tmp_path / "a.json").write_text(
        json.dumps({"papers": [{"id": "2604.10815v1", "title": "A", "authors": []}]})
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"papers": [{"id": "2604.18274v2", "title": "B", "authors": []}]})
    )
    cat = harvest_arxiv_catalog([tmp_path / "a.json", tmp_path / "b.json"])
    assert cat.get("2604.10815") is not None
    assert cat.get("2604.18274") is not None


# --------------------------------------------------------------------------- #
# 18. harvested entries verify clean in lint (no longer UNKNOWN)
# --------------------------------------------------------------------------- #
def test_harvested_entries_classify_as_ok(tmp_path):
    daily = {"papers": [{"id": "2604.10815v1", "title": "MeloTune"}]}
    p = tmp_path / "daily.json"
    p.write_text(json.dumps(daily))
    cat = build_default_catalog()
    cat.harvest_from_daily_json(p)

    res = cat.verify_id("2604.10815", offline=True)
    assert res.catalog_match is True
    assert res.is_mismatch is False
    # expected_title is set
    assert "MeloTune" in res.expected_title
