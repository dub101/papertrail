# CLAUDE.md — PaperTrail project guide

> **Before anything else, read the canonical brief**: [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md). That document is the authoritative spec. This file is the operational guide on top of it.

## How to work with this user

This is a multi-session learning project. The user is preparing for the **Claude Certified Architect – Foundations** exam. The 7-step teaching pattern is **non-negotiable** on every step:

1. Explain the concept in plain language before any code
2. Ask user how they would approach it
3. Listen, correct gaps, confirm understanding
4. Walk through the logic before writing anything
5. Write code with comments explaining every decision
6. After each implementation, ask a cert-exam-style question on what we just built
7. Wait for user's answer before moving to the next step

After each component, **map it to a cert domain and task statement** (e.g., "This is D1, TS 1.1"). Mappings must be **literal** — cite a TS only if its actual content matches. For foundation code not yet LLM-facing, write *"Foundation for D2 TS 2.1 — applies when wrapped as a tool in arch_01"* rather than over-claiming. If the user says "just do it" or "skip the explanation," push back — that defeats the purpose. See memory `[[feedback-teaching-pattern]]`.

## Cert guide — authoritative reference

The official exam guide PDF lives at **`CCA_Foundations_Guide.pdf`** in the repo root. **Gitignored** — local-only, not for distribution. 40 pages.

**Session start protocol — non-negotiable.** Before doing anything substantive in a new session, read all of the following:

1. **The cert guide PDF in full.** `CCA_Foundations_Guide.pdf` at repo root, 40 pages. Read it via `Read` with `pages="1-20"` then `pages="21-40"` (20-page Read limit). The PDF is the authoritative source; do not skip it on the assumption that the synopsis is enough.
2. **The cert guide synopsis** — `[[reference-cert-guide-synopsis]]` in memory. Auto-loads via MEMORY.md. It is a *quick-reference index* on top of the PDF, not a replacement.
3. **The working-style feedback memories** — also auto-loaded via MEMORY.md. Confirm the rules are present in context before the first code or explanation lands: `[[feedback-teaching-pattern]]`, `[[feedback-code-snippet-summaries]]`, `[[feedback-concept-introduction]]`, `[[feedback-session-workflow]]`, `[[feedback-quiz-answer-parity]]`, `[[feedback-cost-awareness]]`, `[[feedback-dry-run-sentinels]]`.

Treat the PDF as foundational reading every session, not as a "lookup when uncertain" resource. The 2026-05-15 incident — where I worked through the agentic-loop implementation without having read the PDF — is exactly what this protocol prevents.

Domain titles and weights from the guide:

| | Domain | Weight |
|---|---|---|
| **D1** | Agentic Architecture & Orchestration | 27% |
| **D2** | Tool Design & MCP Integration | 18% |
| **D3** | Claude Code Configuration & Workflows | 20% |
| **D4** | Prompt Engineering & Structured Output | 20% |
| **D5** | Context Management & Reliability | 15% |

D5 is **context engineering and multi-agent error propagation**, NOT "operations" or testing. Testing/CI strategy is *not* in the cert at all.

PaperTrail is closest to **Scenario 3 (Multi-Agent Research System)** in the cert guide — that's the scenario PaperTrail's architectures exercise most directly, and Scenario 3's primary domains (D1, D2, D5) are the project's primary cert hooks.

## Stack

- Python 3.11+, `uv` for dependencies, Docker + docker-compose
- `anthropic` SDK (raw, **no wrappers**), `mcp` SDK (server built from scratch)
- pydantic v2, httpx async, asyncio, structlog → JSON, rich
- pytest + pytest-asyncio, mypy strict, ruff, `act` for local CI

## Models policy

- `claude-haiku-4-5` — all agents in all architectures by default
- `claude-sonnet-4-6` — Evaluator Agent only
- `claude-opus` — **never used, in any architecture, under any circumstance**

Rationale in ADR-0003.

## Layout

```
.
├── src/papertrail/
│   ├── __init__.py
│   └── architectures/        # arch_00 through arch_06 land here
├── tests/
├── apps/                     # legacy code; moves into arch_00 in step 4
├── qdrant_storage/           # Qdrant volume; preserve, gitignored
├── docs/
│   ├── PROJECT_BRIEF.md      # canonical spec
│   └── decisions/            # ADRs
├── docker-compose.yml        # Qdrant service
├── pyproject.toml            # deps + ruff/mypy/pytest/coverage config
└── .github/workflows/ci.yml  # ruff + mypy + pytest
```

## Branching

- `main` protected, `dev` protected (integration)
- `feature/*` per architecture (`feature/arch-01-sequential`, etc.)
- Conventional commits: `feat(arch-01): ...`, `test(tools): ...`, `fix(mcp): ...`, `chore(ci): ...`

## Engineering standards

- Full type hints; **mypy strict** must pass
- All public functions/classes have docstrings
- Every new module has a corresponding test file
- Coverage floor 80% in CI, target 90%+
- Ruff zero-warning before commit
- No hardcoded secrets — `.env` only
- Anthropic API calls **always mocked** in unit tests
- After Session 1 scaffolding, no work directly on `main`/`dev` — always cut a `feature/*` branch first. Merging back is fine via PR *or* local merge (`git merge --no-ff` + push); the rule is about *where* you edit, not *how* the merge happens. Delete the feature branch (local + remote) after merge.

## Architectures roadmap

| ID | Pattern | Status |
|---|---|---|
| arch_00 | Naive baseline (no LLM) | not yet adapted |
| arch_01 | Sequential Pipeline | not started |
| arch_02 | Map-Reduce | not started |
| arch_03 | Parallel Fan-Out | not started |
| arch_04 | Plan-and-Execute | not started |
| arch_05 | Critic-Revisor Loop | not started |
| arch_06 | Hierarchical + MCP | not started |

## Session continuity — five-layer persistence

See memory `[[reference-session-continuity]]`.

1. **This file** (auto-loaded, operational guide, **current cursor**)
2. **Memory** at `~/.claude/projects/.../memory/` (auto-loaded, collaboration rules + state)
3. **ADRs** in `docs/decisions/` (durable decisions with rationale)
4. **Per-arch READMEs** in `src/papertrail/architectures/arch_NN_*/README.md`
5. **Git** + conventional commits

## Current cursor

**Session 5 in progress — arch_01 stages 1-4 shipped; stage 5 (executive summary) is next.** Step ordering:

1. ✅ Project scaffolding — infrastructure (no cert mapping)
2. ✅ Benchmark interface — `BenchmarkResult` pydantic + `Architecture` ABC (foundation for D4 TS 4.3, applies when wrapped as tool output)
3. ✅ Shared tool layer — `arxiv_search` / `arxiv_fetch` / `format_citation` in `src/papertrail/tools/` (foundation for D2 TS 2.1; D2 TS 2.2 already hit by the 429-retry policy)
4. ✅ arch_00 baseline — `src/papertrail/architectures/arch_00_baseline/` with `BaselineArchitecture`. Zero Claude tokens; one arxiv call; ≤3-bucket date partition; hybrid per-field policy; `confidence=0.5` uniform; `citation_*=None`. Raises `BaselineTooFewResultsError` if arxiv returns <8 usable papers (arch_00's own strict floor, decoupled from the schema by ADR-0006). (No cert mapping — the floor.)
5. ✅ Evaluator + benchmark runner + first real run — `src/papertrail/evaluator.py`, `src/papertrail/prompts/evaluator_v1.md`, `src/papertrail/runner.py`, `scripts/run_benchmark.py`. First real arch_00 + Sonnet run on "self-attention" → overall=0.18. Cert: **D4 TS 4.6**, **D4 TS 4.3**. See ADR-0004.
6. 🟡 arch_01 sequential pipeline — stages 1-3 shipped on `dev`, stages 4-6 pending. Cert hooks across the architecture: D1 TS 1.1, D1 TS 1.6, D4 TS 4.3, D4 TS 4.4, D5 TS 5.1, D5 TS 5.3, D5 TS 5.6.
   - ✅ **Stage 1 — `SearchAgent`** (`src/papertrail/architectures/arch_01_sequential/search.py`). Agentic loop over `arxiv_search`; compact tool_result hand-back; partial-results recovery via `ErrorRecord(recovered=True)` when `stop_reason != "end_turn"` but ≥ `MIN_PAPERS_TO_PROCEED=4` papers collected. Cert: **D1 TS 1.1**, **D5 TS 5.1**, **D5 TS 5.3**. See ADR-0006 (schema floor relaxation 8 → 4).
   - ✅ **Stage 2 — `TriageAgent`** (`src/papertrail/architectures/arch_01_sequential/triage.py`). Single forced `tool_use` call producing one decision per candidate (included|rejected + reason). Four exception classes for fail-loud classification; `TriageInsufficientQualityError` distinct from `TriageInvalidOutputError`. Full audit trail via `CandidateRecord` flowing to `Telemetry.candidates`. Cert: **D4 TS 4.3**, **D4 TS 4.1**, **D5 TS 5.6**.
   - ✅ **Stage 3 — `SynthesisAgent`** (`src/papertrail/architectures/arch_01_sequential/synthesis.py`). Parallel batched synthesis via `asyncio.gather` + `_safe_batch` wrapper. Balanced batch sizes near `TARGET_BATCH_SIZE=5`. One retry round on missing papers; permanent misses get distinct per-dimension disclaimer text + `ErrorRecord(recovered=True)`. Hard-fail on zero-success. Cert: **D1 TS 1.6**, **D4 TS 4.3**, **D4 TS 4.4** (first introduction; un-defers the policy left open in ADR-0005), **D5 TS 5.1**, **D5 TS 5.3**.
   - ✅ **Stage 4 — `EraPartitionAgent`** (`src/papertrail/architectures/arch_01_sequential/era_partition.py`). Single forced `tool_use` call partitioning the synthesised papers into **2-4 content-driven eras** (`ERAS_MIN=2`, `ERAS_MAX=4` per user direction). `EraEntry` schema: lowercase-slug `era_id`, year-int range (overlap across eras allowed; end-year nullable for the frontier era), narrative bounded to 200-900 chars (encodes the soft 80-120 word target). Four cross-field invariants enforced post-parse: no fabricated paper_ids, **paper in exactly one era** (Option A — schema unchanged), unique era_ids, year ordering. Three exceptions: `EraPartitionError` base, `EraPartitionRefusedError`, `EraPartitionInvalidOutputError`. Cert: **D4 TS 4.3**, **D1 TS 1.6** (first cross-paper integration pass), **D5 TS 5.6** (provenance via paper_ids audit trail; narrative speaks at concept level, no paper-name-dropping).
   - ⬜ **Stage 5 — Executive summary.** **Resume here.** Single forced `tool_use` call producing `Deliverable.overall_summary` (~150-200 words). Input: topic + era partition + per-paper syntheses. Re-use of the forced-tool pattern (no new theory section needed per `[[feedback-concept-introduction]]`); short "what's new vs. prior" callout suffices.
   - ⬜ **Stage 6 — Assembly + `SequentialArchitecture` orchestrator.** Pure-code stage that wires all six stages into one `Architecture.run(topic) -> BenchmarkResult`. Includes provenance injection, telemetry aggregation, year-int → `date(...)` lifting for era ranges, and confidence default (open item below).
- Auxiliary: `src/papertrail/pricing.py` extracted in stage 2's session as the third caller appeared (Evaluator + SearchAgent + TriageAgent); `compute_cost_usd` + `MODEL_PRICING_PER_MTOK` now live there.
- Test count: **221 passing**, ruff + mypy strict clean across all of `src/`. arch_00 + arch_01 stages 1-3 covered.

**Open/deferred items:**

- **ADR-0005 D5 TS 5.1 framing nitpick**: "structured handoff between stages" wording should be tightened to "context preservation via compact tool_result." Not blocking; cleanup commit.
- **ADR documenting D4 TS 4.4 un-deferral**: stage 3's retry-with-feedback was introduced without a follow-up ADR. Either a short ADR-0007 or an addendum to ADR-0005.
- **`PaperEntry.confidence` handling in stage 6 assembly**: deliberately not emitted by stage 3 per user direction ("confidence is further down the line"). Stage 6 will set a constant or simple heuristic for arch_01; user expected to weigh in when stage 6 lands.
- **Citation lookup integration**: deferred for arch_01 (would double stage 1's tool surface). Revisit for arch_02 or as a later tool upgrade.
- **`MODEL_PRICING_PER_MTOK`** (`src/papertrail/pricing.py`): hand-maintained, verified 2026-05-15. Update when Anthropic publishes new rates.
- **Architecture-side cost tracking**: `BenchmarkResult.telemetry.total_cost_usd` populated by each architecture (arch_00 always 0). Evaluator's cost on `EvaluatorScore.usage.cost_usd_estimated` separately — by design.
- **Provenance helpers in `BaselineArchitecture`** (move to harness): open since arch_00; stage 6 / orchestrator work would be a natural moment to do it.
- **`apps/` legacy code deletion**: open; whenever convenient.

**Recently resolved (this session):**

- ADR-0006 — `Deliverable.papers` floor 8 → 4, decoupling arch_00's local guard from the schema.
- D4 TS 4.4 retry-with-feedback — un-deferred for stage 3 (single retry on missing papers).
- Pricing module extracted to `papertrail.pricing` (triage was the third caller).
- Full cert guide (`CCA_Foundations_Guide.pdf`) read for the first time; session-start protocol now mandates this. See memory `[[reference-cert-guide-synopsis]]` + `[[feedback-concept-introduction]]`.

## Update protocol

At the end of each session:
- Produce a four-list end-of-session summary in chat: **done overall** / **this session** / **next session** / **outstanding** (see memory `[[feedback-session-workflow]]`). This is the resumption anchor for the next session — without it, the next session pays ~30 minutes re-deriving state.
- Update the **Current cursor** section above to match the summary
- Save new memories or update existing ones
- Write/update ADRs for any decisions made
- Commit on the appropriate feature branch with a conventional message, merge into `dev`, delete the feature branch
