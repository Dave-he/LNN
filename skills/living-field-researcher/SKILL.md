---
name: "living-field-researcher"
description: "Builds and maintains a living research knowledge base in a GitHub repository for a chosen scientific or technical field. Use when the user wants recurring searches for papers, repositories, models, benchmarks, or frontier updates; wants to turn daily tracking into structured reports, indices, and experiment queues; or wants to organize domain knowledge with PRISMA-lite, living review, snowballing, Zettelkasten, or progressive-summarization workflows."
---

# Living Field Researcher

## Role
You turn a repository into a continuously updated research knowledge base for one domain. The output must be searchable, auditable, useful in GitHub, and friendly to Obsidian-style linked notes.

Default language: use the user's language. In this repository, default to Chinese for narrative docs and keep technical terms bilingual when helpful.

## Operating Rules

- Work from the current repository first. Reuse existing scripts, docs, folders, naming conventions, and skills before creating new structures.
- Keep the workflow domain-general, but specialize the profile for the current field. For this LNN repo, treat Liquid Neural Networks / LTC / CfC / NCP / LFM as the current domain.
- Preserve evidence. Store raw search outputs separately from human summaries; every digest or report should link to source URLs, local PDFs, data files, or repo paths.
- Record search queries, source names, run date, inclusion/exclusion reasons, and confidence. Do not claim formal systematic-review compliance unless the full checklist and protocol are actually maintained.
- Prefer small, iterative updates: daily capture, triage, focused deep reading, concept extraction, weekly synthesis, and keyword refinement.
- If a task requires deep paper reading, invoke or follow `skills/paper-analyzer`. If it requires translation, invoke or follow `skills/paper-translator`.

## Core Workflow

1. **Define the field protocol**
   - Identify domain name, slug, scope, research questions, seed papers/repos, keywords, sources, search cadence, and inclusion/exclusion rules.
   - If no protocol exists, create one under `docs/` using `references/domain-profile-template.md`.
   - For existing repositories, update the nearest equivalent document instead of duplicating it.

2. **Run or design recurring surveillance**
   - Search papers, preprints, code, models, datasets, standards, official blogs, and benchmark/news sources relevant to the field.
   - For ML/AI domains, common sources are arXiv, Semantic Scholar/OpenAlex, OpenReview, ACL Anthology, Papers with Code, GitHub, Hugging Face, organization blogs, and benchmark leaderboards.
   - Persist machine-readable data to `papers/daily/YYYY-MM-DD_<domain>_research.json` or the repository's existing equivalent.
   - Produce a human digest at `docs/daily/YYYY-MM-DD_<DOMAIN>_research_digest.md`.

3. **Triage candidates**
   - De-duplicate by DOI, arXiv ID, title, repository full name, or model ID.
   - Score relevance with explicit signals: keyword match, author/lab importance, citation/Star/download trend, novelty, reproducibility, benchmark impact, and fit to the repo's roadmap.
   - Mark each item as `read_now`, `watch`, `repo_analyze`, `experiment`, or `ignore`, and include a short reason.

4. **Create deep-reading artifacts**
   - For each selected paper, create `docs/reports/<paper_file_or_slug>_研读报告.md`.
   - Required sections: `元数据`, `核心问题`, `方法论与核心思路`, `核心公式提取`, `关键成果与贡献`, `局限性与未来展望`.
   - Add `复现线索` when code, datasets, metrics, or implementation details are available.
   - Link the report from the global field index, such as `docs/<DOMAIN>_深度研读报告.md`.

5. **Synthesize into durable knowledge**
   - Convert high-value reports into concept notes, maps of content, comparison tables, timelines, or experiment hypotheses.
   - Use progressive summarization: raw capture -> daily digest -> paper/repo report -> concept note -> field synthesis.
   - Use Zettelkasten-style notes only for reusable ideas, not every fact. Each concept note should have one central claim and links to supporting reports.

6. **Connect research to implementation**
   - Track reusable repositories under `projects/` or `analysis/repo_watchlist/`.
   - Convert promising claims into experiment tickets, scripts, benchmark configs, or analysis outputs under `scripts/`, `configs/`, and `analysis/`.
   - When a paper makes an empirical claim relevant to the repo, record the reproduction target: dataset, baseline, metric, expected result, and blocking dependencies.

7. **Review and improve the loop**
   - Weekly: inspect search misses, false positives, stale keywords, unprocessed high-priority items, and open experiment questions.
   - Monthly: update the protocol, research taxonomy, seed set, source list, and global index.
   - When a new subfield emerges, add it to the protocol and link it from the field index instead of scattering isolated notes.

8. **Improve this skill when the pattern generalizes**
   - If a repeated improvement applies to any research field, update this skill or its `references/` templates.
   - If an improvement is specific to one domain, update that domain's protocol or automation script instead.
   - Keep `SKILL.md` focused on the core workflow; move long examples, frameworks, and templates into `references/`.
   - After changing the skill, check that repository entry points such as `README.md` and `AGENTS.md` still describe it accurately.

## Knowledge Frameworks

Use these as lightweight working methods, not as decoration:

- **Living systematic review**: keep search frequency, sources, and update triggers explicit for fast-moving fields.
- **PRISMA-lite**: maintain transparent source/query/screening/inclusion records; use full PRISMA only when the user asks for a formal systematic review.
- **Backward/forward snowballing**: expand from high-quality seed papers through references and citing papers, one iteration at a time.
- **Zettelkasten / evergreen notes**: distill durable concepts into atomic linked notes with stable titles.
- **Progressive summarization**: layer summaries so each revisit creates a more compressed and reusable artifact.

Read `references/frameworks.md` when you need source rationale or to design a new protocol. Read `references/document-templates.md` when creating docs from scratch.

## LNN Default Profile

When this skill is used in the current LNN repository, start from this profile unless the user overrides it:

- Domain: Liquid Neural Networks, continuous-time neural networks, and liquid foundation models.
- Keywords: `Liquid Neural Networks`, `liquid neural network`, `Liquid Time-Constant`, `LTC`, `Closed-form Continuous-time`, `CfC`, `Neural Circuit Policy`, `NCP`, `Liquid AI`, `LFM`, `LFM2`, `state-space liquid`, `neural ODE`.
- Existing automation: `scripts/daily_lnn_research.py`, `scripts/run_daily_lnn_task.sh`, `.github/workflows/daily-lnn-research.yml`.
- Existing outputs: `docs/daily/`, `papers/daily/`, `analysis/repo_watchlist/`, `docs/reports/`, `docs/LNN_深度研读报告.md`.
- Preferred next step after a daily digest: select 1-3 high-signal items, create deep reports, then update the global index and experiment queue.

## Done Criteria

A field-research update is complete only when:

- The search/update scope and date are explicit.
- Raw evidence and human summaries are both saved or linked.
- Important candidates have triage status and reasons.
- Deep reports or synthesis updates are linked from the field index.
- Any recommended experiment has a concrete target, metric, and output path.
