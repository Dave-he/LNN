# Knowledge Frameworks for Living Field Research

Use this reference when designing or revising a repository-level research protocol.

## Living Systematic Review Pattern

Use for fast-moving fields where new papers and code can change conclusions.

- Define the review question, scope, search sources, search frequency, and update triggers before running the loop.
- Maintain continual evidence surveillance: scheduled searches, alerts, or API-based pulls.
- Update the field synthesis when new evidence changes a claim, benchmark, taxonomy, or implementation priority.
- Keep a visible "what changed" trail in the daily digest, field index, or changelog-like section.

Source basis: Cochrane describes living systematic reviews as continually updated reviews with new evidence incorporated as it becomes available, with explicit search methods and anticipated update frequency in the protocol.

Reference: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-22

## PRISMA-Lite Traceability

Use PRISMA as a transparency checklist, not as a badge.

Track:

- Sources searched.
- Exact search strings or API parameters.
- Date of each search.
- Candidate counts before and after de-duplication.
- Screening criteria.
- Included, excluded, and deferred items with reasons.
- Links to raw data, PDFs, reports, and synthesis pages.

Use full PRISMA 2020 only when the deliverable is a formal systematic review. Otherwise, keep the lightweight traceability fields above.

Source basis: PRISMA 2020 provides checklist items and an expanded checklist for transparent systematic-review reporting.

Reference: https://www.prisma-statement.org/prisma-2020-checklist

## Snowballing

Use for finding important papers missed by keyword search.

- Start from a high-quality seed set.
- Backward snowballing: inspect references cited by the seed paper.
- Forward snowballing: inspect later papers that cite the seed paper.
- Decide inclusion or exclusion before using a newly found paper as a snowballing seed.
- Run one iteration at a time so the path from seed to inclusion remains traceable.
- Stop when a full iteration yields no meaningful new inclusions, or when effort exceeds the value for the current research question.

Source basis: Wohlin's snowballing guideline emphasizes backward/forward snowballing, removing already examined candidates, deciding inclusion before further snowballing, and preserving traceability one iteration at a time.

Reference: https://www.wohlin.eu/ease14.pdf

## Zettelkasten and Evergreen Notes

Use for durable knowledge, not raw capture.

- Create atomic notes for reusable concepts, mechanisms, claims, or distinctions.
- Give each note a clear title that states the idea.
- Link concept notes to source reports and neighboring concepts.
- Build maps of content for domains, subfields, method families, datasets, and experiment themes.
- Avoid converting every paper section into an atomic note; extract only ideas likely to be reused.

Source basis: Zettelkasten emphasizes atomicity and connectivity: keep one topic per note and connect notes into a navigable knowledge web.

Reference: https://zettelkasten.de/overview/

## Progressive Summarization

Use to prevent daily digests from becoming a dead archive.

Recommended layers:

1. Raw source data: JSON, PDFs, repository metadata, model cards.
2. Daily digest: short human-readable scan.
3. Triage: status and reason.
4. Deep report: structured paper/repo/model analysis.
5. Concept note: reusable idea extracted from one or more reports.
6. Field synthesis: roadmap, comparison table, taxonomy, or experiment plan.

Source basis: Progressive summarization is a layered method for making notes discoverable and increasingly distilled as they are revisited.

Reference: https://fortelabs.com/blog/second-brain-case-study-progressive-summarization-in-the-intelligence-community/

## GitHub Knowledge-Base Mapping

Suggested mapping for research repositories:

- `README.md`: public entry, current status, and most important navigation links.
- `AGENTS.md`: agent roles, automation design, and skill installation/use.
- `docs/<DOMAIN>_研究协议.md`: scope, search protocol, taxonomy, and update rules.
- `docs/daily/`: daily or scheduled search digests.
- `papers/daily/`: raw search JSON and optionally downloaded PDFs by date.
- `docs/reports/`: deep single-paper reports.
- `docs/concepts/`: atomic concept notes when the repo has enough synthesis to justify them.
- `analysis/repo_watchlist/`: open-source repositories, models, datasets, and benchmark watchlists.
- `analysis/`: experiment results, benchmark plots, and structured outputs.
- `projects/`: cloned or vendored reproduction projects.
- `scripts/`: deterministic automation for repeated search, extraction, benchmark, and report generation.
