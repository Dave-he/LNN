"""Tests for scripts/build_backbone_matrix.py _dedupe_keep_higher_n (iter#25).

The original implementation replaced the whole row by the higher-n_seeds
one, which silently dropped data when one row had more backbones but
fewer seeds. The fix is per-backbone max: for each (row_key, backbone)
pair, keep the entry with the highest n_seeds.

Cases verified:
1. Two rows, same row_key, disjoint backbones → both kept in one merged row
2. Two rows, same row_key, overlapping backbones → per-backbone max wins
3. n_seeds of merged row = max across backbones (matches original semantics)
4. Single row passes through unchanged
5. Non-backbone fields (domain, metric, source_path) come from the first row
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_backbone_matrix  # noqa: E402


# ---------------------------------------------------------- 1. disjoint
def test_disjoint_backbones_merged():
    rows = [
        {
            "row_key": "task1",
            "domain": "timeseries",
            "metric": "test_mse",
            "metric_direction": "lower_is_better",
            "n_seeds": 3,
            "backbones": {
                "cfc": {"n": 3, "median_mse": 0.10},
                "ltc": {"n": 3, "median_mse": 0.12},
            },
            "source_path": "a.json",
        },
        {
            "row_key": "task1",
            "domain": "timeseries",
            "metric": "test_mse",
            "metric_direction": "lower_is_better",
            "n_seeds": 6,
            "backbones": {
                "fhn_dynpmnn": {"n": 6, "median_mse": 0.20},
            },
            "source_path": "b.json",
        },
    ]
    out = build_backbone_matrix._dedupe_keep_higher_n(rows)
    assert len(out) == 1
    assert set(out[0]["backbones"].keys()) == {"cfc", "ltc", "fhn_dynpmnn"}
    assert out[0]["backbones"]["cfc"]["n"] == 3
    assert out[0]["backbones"]["ltc"]["n"] == 3
    assert out[0]["backbones"]["fhn_dynpmnn"]["n"] == 6
    assert out[0]["n_seeds"] == 6  # max across backbones


# -------------------------------------------------- 2. overlapping per-bb max
def test_overlapping_backbones_per_bb_max():
    """Same backbone in two rows, different n_seeds — keep the higher."""
    rows = [
        {
            "row_key": "task2",
            "domain": "timeseries",
            "metric": "test_mse",
            "metric_direction": "lower_is_better",
            "n_seeds": 3,
            "backbones": {
                "cfc": {"n": 3, "median_mse": 0.05},  # better
                "gru": {"n": 3, "median_mse": 0.07},
            },
            "source_path": "a.json",
        },
        {
            "row_key": "task2",
            "domain": "timeseries",
            "metric": "test_mse",
            "metric_direction": "lower_is_better",
            "n_seeds": 6,
            "backbones": {
                "cfc": {"n": 6, "median_mse": 0.08},  # worse but more seeds
                "ltc": {"n": 6, "median_mse": 0.06},
            },
            "source_path": "b.json",
        },
    ]
    out = build_backbone_matrix._dedupe_keep_higher_n(rows)
    assert len(out) == 1
    # cfc: keep the 3-seed entry (higher n wins per-backbone)
    assert out[0]["backbones"]["cfc"]["n"] == 6, \
        "per-backbone max should pick 6-seed cfc entry"
    assert out[0]["backbones"]["cfc"]["median_mse"] == 0.08
    # ltc: only in row 2, take it
    assert out[0]["backbones"]["ltc"]["n"] == 6
    # gru: only in row 1, take it
    assert out[0]["backbones"]["gru"]["n"] == 3
    assert out[0]["n_seeds"] == 6


# ---------------------------------------------------- 3. n_seeds = max
def test_n_seeds_is_max_across_backbones():
    rows = [
        {
            "row_key": "task3", "domain": "x", "metric": "y",
            "metric_direction": "lower_is_better", "n_seeds": 2,
            "backbones": {"a": {"n": 2, "v": 1}},
            "source_path": "a",
        },
        {
            "row_key": "task3", "domain": "x", "metric": "y",
            "metric_direction": "lower_is_better", "n_seeds": 5,
            "backbones": {"b": {"n": 5, "v": 2}},
            "source_path": "b",
        },
    ]
    out = build_backbone_matrix._dedupe_keep_higher_n(rows)
    assert out[0]["n_seeds"] == 5


# ---------------------------------------------------- 4. single row
def test_single_row_passes_through():
    rows = [{
        "row_key": "task4", "domain": "x", "metric": "y",
        "metric_direction": "lower_is_better", "n_seeds": 7,
        "backbones": {"x": {"n": 7, "v": 1}},
        "source_path": "a",
    }]
    out = build_backbone_matrix._dedupe_keep_higher_n(rows)
    assert len(out) == 1
    assert out[0] == rows[0]


# ---------------------------------------------------- 5. non-bb fields from first
def test_non_backbone_fields_from_first_row():
    """domain, metric, source_path should come from the first matching row."""
    rows = [
        {
            "row_key": "task5", "domain": "timeseries", "metric": "test_mse",
            "metric_direction": "lower_is_better", "n_seeds": 3,
            "backbones": {"cfc": {"n": 3, "median_mse": 0.1}},
            "source_path": "first.json",
        },
        {
            "row_key": "task5", "domain": "smnist_gap", "metric": "acc",  # ignored
            "metric_direction": "higher_is_better", "n_seeds": 5,
            "backbones": {"gru": {"n": 5, "acc": 0.9}},
            "source_path": "second.json",
        },
    ]
    out = build_backbone_matrix._dedupe_keep_higher_n(rows)
    assert out[0]["domain"] == "timeseries"  # from first
    assert out[0]["source_path"] == "first.json"  # from first
    # Both backbones present
    assert set(out[0]["backbones"].keys()) == {"cfc", "gru"}
