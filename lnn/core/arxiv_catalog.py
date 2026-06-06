"""arXiv catalog hygiene for the LNN research repo (round 72).

Background
----------
The 2026-06-06 deep-research pass (3-vote adversarial verification) uncovered
three arXiv-ID misattributions that have been floating in the broader LNN
literature commentary:

  2003.06236 → an unrelated hydrocarbon Monte Carlo study
                (NOT Hasani et al. CfC, which is correctly 2106.13898,
                 Nature Machine Intelligence 4:992-1003, 2022)
  2203.14343 → Gupta, Gu, Berant "Diagonal State Spaces" (2022)
                (NOT Lockhart et al. "Adaptive Solvers for Neural ODEs")
  2002.08071 → Massaroli et al. "Dissecting Neural ODEs" (NeurIPS 2020)
                (NOT Lienen & Günnemann "torchode")

This module gives the LNN pipeline a small, offline-first catalog that:

1. Knows the canonical arXiv ID ↔ title mapping for ~12 foundational LNN
   papers (CfC, LTC, NCP, DSS, S4, Mamba-adjacent, etc.).
2. Flags the three known misattributions above as ``MISMATCH`` so that any
   downstream researcher who copy-pastes one of them gets an immediate
   warning.
3. Tracks a per-paper ``reproduction_status`` field with four values:
   ``vendor_claim`` / ``third_party_reproduced`` / ``failed_to_reproduce``
   / ``untested``. This was the second gap the deep-research report
   called out — the catalog was front-loaded with vendor claims and
   underrepresented honest negative results.
4. Optionally queries the public arXiv API (no key needed) to verify an
   arbitrary ID, with a 3s throttle and a local JSON cache. Network is
   never required: tests run with ``offline=True`` against the embedded
   seed.

The class is deliberately small — this is a 1-2 hour round-72 deliverable,
not a full bibliographic service.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

# Status taxonomy: the four values are mutually exclusive. ``untested`` is
# the default and is not a negative result — it just means no one has run
# the reproduction in this repo (or anywhere we know of).
REPRO_STATUS_VENDOR = "vendor_claim"
REPRO_STATUS_REPRODUCED = "third_party_reproduced"
REPRO_STATUS_FAILED = "failed_to_reproduce"
REPRO_STATUS_UNTESTED = "untested"
REPRO_STATUS_VALUES = frozenset(
    {
        REPRO_STATUS_VENDOR,
        REPRO_STATUS_REPRODUCED,
        REPRO_STATUS_FAILED,
        REPRO_STATUS_UNTESTED,
    }
)


@dataclass
class CatalogEntry:
    """One arXiv paper in the local catalog.

    The ``expected_title`` field is what we *think* the paper actually is.
    The ``reproduction_status`` is updated by hand or by ``add_reproduction_status``.
    ``notes`` is free-form text shown by ``show_reproduction_status.py``.
    """

    arxiv_id: str
    expected_title: str
    authors: str = ""
    year: str = ""
    reproduction_status: str = REPRO_STATUS_UNTESTED
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifyResult:
    """Result of verifying an arXiv ID against the catalog + (optionally) arXiv API."""

    arxiv_id: str
    catalog_match: bool
    api_match: Optional[bool] = None  # None if API not called (offline)
    api_actual_title: Optional[str] = None
    expected_title: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_mismatch(self) -> bool:
        """A mismatch means: catalog has the ID but flags it as a known
        wrong attribution (``expected_title`` contains ``MISMATCH``), OR
        the arXiv API returned a different title than the catalog claims
        for an ID that *is* in the catalog.

        An ID that is simply not in the catalog is ``catalog_match=False``
        but **not** a mismatch — the CLI scripts flag unknowns separately
        (the lint can optionally surface them, but they are not the same
        as a known misattribution)."""
        if self.expected_title is not None and "MISMATCH" in self.expected_title:
            return True
        if self.api_match is False:
            return True
        return False


# Seed catalog. arXiv IDs without ``(MISMATCH)`` are correct.
# ``expected_title`` is the title the arXiv API returns for that ID.
SEED_CATALOG: dict[str, CatalogEntry] = {
    # --- CfC / LTC / NCP / LNN foundational papers (correct attributions) ---
    "2106.13898": CatalogEntry(
        arxiv_id="2106.13898",
        expected_title=(
            "Closed-form Continuous-Depth Models: a class of "
            "Liquid Time-Constant networks"
        ),
        authors="Hasani, Lechner, Amini, Rus, Grosu",
        year="2022",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes=(
            "Nature Machine Intelligence 4:992-1003 (2022), "
            "DOI 10.1038/s42256-022-00556-7. CfC is a tightly-bounded "
            "closed-form APPROXIMATION of LTC, not an exact closed form."
        ),
    ),
    "2006.04439": CatalogEntry(
        arxiv_id="2006.04439",
        expected_title=(
            "Liquid Time-Constant Networks"
        ),
        authors="Hasani, Lechner, Amini, Ghomi, Grosu, Rus",
        year="2020",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes=(
            "Original LTC paper. Repository lnn/core/ltc.py implements "
            "a from-scratch version."
        ),
    ),
    "2003.06236": CatalogEntry(
        arxiv_id="2003.06236",
        # Wrong attribution: this ID is a hydrocarbon Monte Carlo study,
        # NOT Hasani et al. CfC. Catalogued as a known mismatch so the
        # lint script flags it if any project file references it.
        expected_title=(
            "Monte Carlo simulation of hydrocarbon adsorption [MISMATCH — "
            "NOT Hasani CfC; correct CfC ID is 2106.13898]"
        ),
        authors="(misattributed)",
        year="2020",
        reproduction_status=REPRO_STATUS_FAILED,
        notes=(
            "Known misattribution: this ID has been mis-cited as the CfC "
            "paper in some downstream commentary. The actual CfC paper is "
            "2106.13898. Catalogued here as a trap so lint flags it."
        ),
    ),
    "2203.14343": CatalogEntry(
        arxiv_id="2203.14343",
        # Wrong attribution: this is DSS, not Lockhart adaptive solvers.
        expected_title=(
            "Diagonal State Spaces are as Effective as Structured "
            "[MISMATCH — NOT Lockhart adaptive solvers]"
        ),
        authors="Gupta, Gu, Berant",
        year="2022",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes=(
            "DSS paper. Purely diagonal A matches S4 on LRA (avg 81.88 "
            "vs 80.21) and Speech Commands (98.2 vs 98.1). 2022 historical "
            "comparison only — does NOT include Mamba/Mamba-2/RetNet."
        ),
    ),
    "2002.08071": CatalogEntry(
        arxiv_id="2002.08071",
        # Wrong attribution: this is Massaroli et al. Dissecting Neural
        # ODEs, NOT the torchode library.
        expected_title=(
            "Dissecting Neural ODEs [MISMATCH — NOT Lienen & Günnemann "
            "torchode]"
        ),
        authors="Massaroli, Poli, Park, Yamashita, Asama, Kosec, Murray",
        year="2020",
        reproduction_status=REPRO_STATUS_UNTESTED,
        notes=(
            "NeurIPS 2020. Foundational context for CfC/LTC training "
            "optimization, but NOT a CfC/LTC paper. Catalogued as a known "
            "misattribution."
        ),
    ),
    # --- SSM / S4 family ---
    "2111.00396": CatalogEntry(
        arxiv_id="2111.00396",
        expected_title=(
            "Efficiently Modeling Long Sequences with Structured State Spaces"
        ),
        authors="Gu, Goel, Ré",
        year="2021",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes="S4 paper.",
    ),
    "2312.00752": CatalogEntry(
        arxiv_id="2312.00752",
        expected_title=(
            "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
        ),
        authors="Gu, Dao",
        year="2023",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes="Mamba paper. CfC vs Mamba head-to-head is open work (round 73 candidate).",
    ),
    # --- NCP / Liquid structural ---
    "1706.01350": CatalogEntry(
        arxiv_id="1706.01350",
        expected_title=(
            "Neural Circuit Policies: Enabling Robust Learning of "
            "Sensorimotor Skills"
        ),
        authors="Lechner, Hasani, Grosu, Rus, Grosu",
        year="2017",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes="Original NCP paper. Sparse wiring is foundational for LNN interpretability.",
    ),
    # --- Multimodal LNN ---
    "2605.24047": CatalogEntry(
        arxiv_id="2605.24047",
        expected_title=(
            "EMMA: Extracting Multiple physical parameters from "
            "Multimodal Data"
        ),
        authors="Shaikh, Banerjee, Gupta",
        year="2026",
        reproduction_status=REPRO_STATUS_VENDOR,
        notes=(
            "CVPR 2026 submission. Uses LTC backbone for physics-informed "
            "multimodal parameter recovery. Reproduction in repo via "
            "lnn/core/multimodal_physreg.py + scripts/benchmark_emma_*."
        ),
    ),
    "2601.14115": CatalogEntry(
        arxiv_id="2601.14115",
        expected_title=(
            "Riemannian Liquid Spatio-Temporal Graph Network"
        ),
        authors="(see repo docs/research)",
        year="2026",
        reproduction_status=REPRO_STATUS_REPRODUCED,
        notes=(
            "WWW '26. Reproduced in round 65-70 as "
            "lnn/core/riemannian_ltc.py."
        ),
    ),
}


class ArxivCatalog:
    """In-memory + JSON-persisted arXiv catalog with hygiene checks."""

    def __init__(
        self,
        entries: Optional[dict[str, CatalogEntry]] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        # Start from a copy of the seed so callers can mutate freely.
        self.entries: dict[str, CatalogEntry] = dict(entries or SEED_CATALOG)
        self.cache_path = (
            Path(cache_path) if cache_path else Path.home() / ".lnn_arxiv_cache.json"
        )
        # local API verify cache: arxiv_id -> (ts, api_title, ok?)
        self._api_cache: dict[str, tuple[float, str, bool]] = {}
        if self.cache_path.exists():
            try:
                raw = json.loads(self.cache_path.read_text())
                for k, v in raw.items():
                    self._api_cache[k] = (v["ts"], v["title"], v["ok"])
            except (json.JSONDecodeError, KeyError, OSError):
                # Cache corruption is non-fatal; just start fresh.
                self._api_cache = {}

    def _flush_cache(self) -> None:
        """Write the in-memory API cache to disk (best-effort).

        Used by tests to verify round-trip behaviour without going through
        a real network call. Production code calls this from inside
        ``verify_id`` automatically.
        """
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        k: {"ts": v[0], "title": v[1], "ok": v[2]}
                        for k, v in self._api_cache.items()
                    }
                )
            )
        except OSError:
            pass

    # ---- catalog CRUD ----------------------------------------------------

    def add(self, entry: CatalogEntry) -> None:
        """Add or overwrite a catalog entry. Raises if the entry is malformed."""
        if not ARXIV_ID_RE.match(entry.arxiv_id):
            raise ValueError(f"Invalid arXiv ID format: {entry.arxiv_id!r}")
        if entry.reproduction_status not in REPRO_STATUS_VALUES:
            raise ValueError(
                f"reproduction_status must be one of {sorted(REPRO_STATUS_VALUES)}, "
                f"got {entry.reproduction_status!r}"
            )
        self.entries[entry.arxiv_id] = entry

    def get(self, arxiv_id: str) -> Optional[CatalogEntry]:
        return self.entries.get(self._norm(arxiv_id))

    def add_reproduction_status(
        self,
        arxiv_id: str,
        status: str,
        notes: str = "",
    ) -> CatalogEntry:
        """Update the reproduction_status of an existing entry (or add a stub).

        Returns the updated entry. Raises if the status is not in the
        taxonomy.
        """
        if status not in REPRO_STATUS_VALUES:
            raise ValueError(
                f"reproduction_status must be one of {sorted(REPRO_STATUS_VALUES)}, "
                f"got {status!r}"
            )
        arxiv_id = self._norm(arxiv_id)
        if arxiv_id not in self.entries:
            # Auto-create a stub so callers don't have to know the title.
            self.entries[arxiv_id] = CatalogEntry(
                arxiv_id=arxiv_id,
                expected_title="(unknown — needs API verification)",
                reproduction_status=status,
                notes=notes,
            )
        else:
            self.entries[arxiv_id].reproduction_status = status
            if notes:
                self.entries[arxiv_id].notes = (
                    (self.entries[arxiv_id].notes + "\n" + notes).strip()
                )
        return self.entries[arxiv_id]

    def known_mismatches(self) -> list[CatalogEntry]:
        """Return all entries flagged as MISMATCH in their expected_title."""
        return [
            e for e in self.entries.values() if "MISMATCH" in e.expected_title
        ]

    def harvest_from_daily_json(
        self, json_path: str | os.PathLike, *, default_status: str = REPRO_STATUS_VENDOR
    ) -> dict[str, CatalogEntry]:
        """Add catalog entries for every paper in a daily-research JSON.

        The daily JSON produced by ``scripts/daily_lnn_research.py`` has the
        shape ``{"papers": [{"id": "2604.10815v1", "title": "...", ...}]}``.
        For each paper we:

        1. Normalize the arXiv ID (strip ``v1``/``v2`` suffix).
        2. Skip if the ID is already a known MISMATCH (so we never
           silently override a known-bad attribution).
        3. Skip if the ID is already in the catalog and is correctly
           attributed.
        4. Otherwise add a stub ``CatalogEntry`` with the title from the
           daily JSON, marked with ``default_status`` (vendor_claim by
           default — the daily JSON is itself a vendor/aggregator
           source, not a third-party reproduction).

        Returns a dict ``{arxiv_id: CatalogEntry}`` containing only the
        newly added entries.
        """
        import json

        with open(json_path) as f:
            data = json.load(f)
        new_entries: dict[str, CatalogEntry] = {}
        for p in data.get("papers", []):
            raw_id = p.get("id", "")
            m = ARXIV_ID_RE.search(raw_id)
            if not m:
                continue
            aid = m.group(1)
            if aid in self.entries:
                # Don't override existing entries (correct OR mismatch).
                continue
            title = p.get("title", "(no title in daily JSON)").strip()
            if not title:
                title = "(no title in daily JSON)"
            entry = CatalogEntry(
                arxiv_id=aid,
                expected_title=title,
                authors=", ".join(p.get("authors", []))[:200],
                year=str(p.get("published", ""))[:4],
                reproduction_status=default_status,
                notes=(
                    f"Harvested from {Path(str(json_path)).name}; "
                    f"keyword_score={p.get('keyword_score', '?')}; "
                    f"categories={','.join(p.get('categories', []))}"
                ),
            )
            self.entries[aid] = entry
            new_entries[aid] = entry
        return new_entries

    def status_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in REPRO_STATUS_VALUES}
        for e in self.entries.values():
            counts[e.reproduction_status] = counts.get(e.reproduction_status, 0) + 1
        return counts

    # ---- verification ----------------------------------------------------

    def verify_id(
        self, arxiv_id: str, *, offline: bool = False, throttle_s: float = 3.0
    ) -> VerifyResult:
        """Verify an arXiv ID against the local catalog and (optionally) the
        public arXiv API.

        ``offline=True`` skips the network call entirely — useful in unit
        tests and CI. The local catalog alone is enough to flag the three
        known misattributions.
        """
        arxiv_id = self._norm(arxiv_id)
        entry = self.entries.get(arxiv_id)
        catalog_match = entry is not None and "MISMATCH" not in entry.expected_title
        result = VerifyResult(
            arxiv_id=arxiv_id,
            catalog_match=catalog_match,
            expected_title=entry.expected_title if entry else None,
        )

        if offline:
            return result

        # Cache check: skip the network if we hit the cache within 7 days.
        if arxiv_id in self._api_cache:
            ts, title, ok = self._api_cache[arxiv_id]
            if time.time() - ts < 7 * 24 * 3600:
                result.api_match = ok
                result.api_actual_title = title
                return result

        try:
            url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
                xml_bytes = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            result.error = f"arXiv API error: {exc}"
            return result

        # Tiny XML pull — we only want the first <title> after the entry.
        title_match = re.search(
            rb"<entry>.*?<title>(.*?)</title>", xml_bytes, re.DOTALL
        )
        if not title_match:
            result.error = "arXiv API returned no <entry> for this ID"
            return result
        api_title = re.sub(r"\s+", " ", title_match.group(1).decode("utf-8").strip())
        result.api_actual_title = api_title

        if entry is None:
            # Not in catalog — we accept whatever arXiv says.
            result.api_match = True
        else:
            # Compare loosely: ignore case + whitespace + punctuation.
            result.api_match = self._titles_loosely_match(api_title, entry.expected_title)

        # Update cache.
        self._api_cache[arxiv_id] = (time.time(), api_title, bool(result.api_match))
        try:
            self.cache_path.write_text(
                json.dumps(
                    {k: {"ts": v[0], "title": v[1], "ok": v[2]} for k, v in self._api_cache.items()}
                )
            )
        except OSError:
            # Cache write failures are non-fatal.
            pass

        # Be a good citizen of the arXiv API.
        time.sleep(throttle_s)
        return result

    # ---- scanning --------------------------------------------------------

    def scan_text(self, text: str) -> list[str]:
        """Find all arXiv IDs in ``text`` (used by the lint CLI).

        Returns the IDs in document order, with duplicates removed.
        """
        seen: list[str] = []
        seen_set: set[str] = set()
        for m in ARXIV_ID_RE.finditer(text):
            aid = m.group(1)
            if aid not in seen_set:
                seen.append(aid)
                seen_set.add(aid)
        return seen

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _norm(arxiv_id: str) -> str:
        return ARXIV_ID_RE.search(arxiv_id).group(1) if ARXIV_ID_RE.search(arxiv_id) else arxiv_id

    @staticmethod
    def _titles_loosely_match(a: str, b: str) -> bool:
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

        na, nb = norm(a), norm(b)
        if na == nb:
            return True
        # Allow one to be a substring of the other (handles "MISMATCH" suffix).
        if na in nb or nb in na:
            return True
        # Token overlap >= 60% of the smaller.
        ta, tb = set(na.split()), set(nb.split())
        if not ta or not tb:
            return False
        overlap = len(ta & tb) / min(len(ta), len(tb))
        return overlap >= 0.6


def build_default_catalog() -> ArxivCatalog:
    """Helper for CLI scripts — gives a fresh catalog from the seed."""
    return ArxivCatalog()


def harvest_arxiv_catalog(json_paths: list[str | os.PathLike]) -> ArxivCatalog:
    """Convenience: build a default catalog then harvest from one or more
    daily-research JSON files.

    Equivalent to::

        cat = build_default_catalog()
        for p in json_paths:
            cat.harvest_from_daily_json(p)
        return cat
    """
    cat = build_default_catalog()
    for p in json_paths:
        cat.harvest_from_daily_json(p)
    return cat
