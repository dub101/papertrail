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

**arch_01 complete (all 6 stages + runner wired); first real run pending arxiv rate-limit cooldown.** Resume tomorrow with a single command; see "Next session" below. Step ordering:

1. ✅ Project scaffolding — infrastructure (no cert mapping)
2. ✅ Benchmark interface — `BenchmarkResult` pydantic + `Architecture` ABC (foundation for D4 TS 4.3, applies when wrapped as tool output)
3. ✅ Shared tool layer — `arxiv_search` / `arxiv_fetch` / `format_citation` in `src/papertrail/tools/`. **arxiv now respects a 4 s inter-request pace + 5/15/45 s exponential backoff on 429** (`papertrail.tools.arxiv._pace_request` + `_RATE_LIMIT_BACKOFF_SECONDS`). Cert: D2 TS 2.1, D2 TS 2.2 (transient-retryable error handling).
4. ✅ arch_00 baseline — `src/papertrail/architectures/arch_00_baseline/`; zero Claude tokens; provenance now imports from shared `papertrail.provenance`.
5. ✅ Evaluator + benchmark runner + first real run — first real arch_00 + Sonnet run on "self-attention" → overall=0.18. Cert: **D4 TS 4.6**, **D4 TS 4.3**. See ADR-0004.
6. ✅ **arch_01 sequential pipeline — 6/6 stages merged on `dev`.** Cert hooks across the architecture: D1 TS 1.1, D1 TS 1.4, D1 TS 1.6, D4 TS 4.1, D4 TS 4.3, D4 TS 4.4, D5 TS 5.1, D5 TS 5.3, D5 TS 5.6.
   - ✅ **Stage 1 — `SearchAgent`** (agentic loop over `arxiv_search`; compact tool_result; partial-results recovery; `MIN_PAPERS_TO_PROCEED=4`). Cert: D1 TS 1.1, D5 TS 5.1, D5 TS 5.3.
   - ✅ **Stage 2 — `TriageAgent`** (forced tool_use; quality gate; four exception classes; `CandidateRecord` audit). Cert: D4 TS 4.3, D4 TS 4.1, D5 TS 5.6.
   - ✅ **Stage 3 — `SynthesisAgent`** (parallel batched + one retry; per-paper `confidence: float`; notes channel; disclaimer-fill with confidence=0.0 on permanent failures). Cert: D1 TS 1.6, D4 TS 4.3, **D4 TS 4.4** (first use), D5 TS 5.1, D5 TS 5.3.
   - ✅ **Stage 4 — `EraPartitionAgent`** (forced tool_use; `ERAS_MIN=2`, `ERAS_MAX=4`; content-driven boundaries; year ints; four cross-field invariants). Cert: D4 TS 4.3, D1 TS 1.6, D5 TS 5.6.
   - ✅ **Stage 5 — `ExecutiveSummaryAgent`** (forced tool_use; 150–200-word soft target; self-rated confidence; eras-first user-message ordering for D5 TS 5.1 mitigation). Cert: D4 TS 4.3, D1 TS 1.6, D5 TS 5.1, D5 TS 5.6.
   - ✅ **Stage 6 — `SequentialArchitecture` orchestrator** (`orchestrator.py`). Pure code; wires the five LLM stages; lifts year-ints → `date(year, 1, 1)` / `date(year, 12, 31)` for `TimelineEra`; sums per-stage usage; `PaperEntry.confidence` propagates directly from `PaperSynthesis.confidence`. Cert: D1 TS 1.6 (primary), D1 TS 1.4, D5 TS 5.3, D5 TS 5.6.
- Auxiliary modules: `papertrail.pricing` (3rd-caller extraction; `compute_cost_usd` + `MODEL_PRICING_PER_MTOK`); `papertrail.provenance` (shared `build_provenance` for any architecture).
- Schema additions (additive, no breaking changes): `Telemetry.synthesis_notes: list[SynthesisNote]`; `SynthesisNote` defined in `benchmark.py`.
- Runner: registry switched from `type[Architecture]` to factory functions `(AsyncAnthropic | None) -> Architecture` so arch_00 (no client) and arch_01 (needs client) coexist. CLI builds one shared client when either the architecture or the evaluator needs it.
- Test count: **289 passing**, ruff + mypy strict clean across 21 source files.

**Next session — resume here:**

1. **Quick arxiv check first:**
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" "https://export.arxiv.org/api/query?search_query=all:test&max_results=1"
   ```
   Expect `200`. If `429`: arxiv is still throttling our IP from the 2026-05-17 attempts; wait longer.
2. **Once arxiv is green, run arch_01 end-to-end:**
   ```bash
   uv run python scripts/run_benchmark.py \
       --topic "Attention mechanism for text generation" \
       --architecture arch_01_sequential --verbose
   ```
   Expected cost ~$0.07–0.08 (Haiku stages ~$0.04, Sonnet evaluator ~$0.03). Worst-case time including arxiv backoffs: a few minutes.
3. **Examine the run:** deliverable structure, per-paper confidences, era partition, executive summary, evaluator score breakdown. Compare implicit/explicit signals against arch_00's earlier `self-attention` result (overall=0.18).
4. **Optional follow-ups** after the first successful run: arch_01 on `self-attention` for an A/B against arch_00; or arch_00 on `Attention mechanism for text generation` for the inverse A/B.

**Open / deferred items:**

- **arxiv was throttling our IP on 2026-05-17.** Three back-to-back attempts (one arch_01 + one arch_00 diagnostic + a re-run after the pacing fix shipped) all hit 429s, with arxiv slow-walking responses up to 15 s before sending the 429. No `Retry-After` header. The pacing/backoff fix is the right architectural answer (in production we will hit arxiv much less aggressively), but the cooldown after our test bursts is real. Tomorrow's wait should clear it.
- **No ADR** yet for: D4 TS 4.4 retry-with-feedback un-deferral (stage 3); the arxiv pacing+backoff design (D2 TS 2.2 territory). Either small standalone ADRs or one combined "session 5/6 follow-up" ADR.
- **ADR-0005 D5 TS 5.1 framing nitpick** — "structured handoff between stages" wording could be tightened to "context preservation via compact tool_result." Cosmetic; not blocking.
- **arch_00's previous run was on "self-attention".** Tomorrow's arch_01 run is on "Attention mechanism for text generation". They are not directly comparable; pick one shared topic if you want a clean A/B.
- **Citation lookup integration**: still deferred for arch_01 (would double stage 1's tool surface). Revisit for arch_02 or as a later tool upgrade.
- **`MODEL_PRICING_PER_MTOK`**: hand-maintained, verified 2026-05-15. Update when Anthropic publishes new rates.
- **`apps/` legacy code deletion**: open; whenever convenient.
- **Per-stage `TraceStep` entries in `Telemetry.trace`**: deferred during stage 6; arch_01 currently emits `trace=[]`. Future "observability pass" lands them.

**Recently resolved (this session, spanning 2026-05-15..17):**

- arch_01 stages 1–6 all merged on `dev`; arch_01 invocable end-to-end via the CLI runner.
- ADR-0006 (schema floor 8 → 4).
- `papertrail.pricing` and `papertrail.provenance` extracted from their original homes.
- `PaperSynthesis.confidence: float` added; stage 6 propagates it directly into `PaperEntry.confidence` (no heuristic — disclaimer-filled syntheses carry 0.0 as the code-emitted floor).
- arxiv 429 handling overhauled: 4 s inter-request pacing + 5/15/45 s exponential backoff. Tests get a `tests/tools/conftest.py` autouse fixture that bypasses the sleep for speed.
- Full cert guide PDF read for the first time; session-start protocol mandates this. See `[[reference-cert-guide-synopsis]]`.
- New / updated memories: `[[feedback-concept-introduction]]`, `[[feedback-commit-message-brevity]]`, tighter `[[feedback-code-snippet-summaries]]` and `[[feedback-quiz-answer-parity]]`.

## Update protocol

At the end of each session:
- Produce a four-list end-of-session summary in chat: **done overall** / **this session** / **next session** / **outstanding** (see memory `[[feedback-session-workflow]]`). This is the resumption anchor for the next session — without it, the next session pays ~30 minutes re-deriving state.
- Update the **Current cursor** section above to match the summary
- Save new memories or update existing ones
- Write/update ADRs for any decisions made
- Commit on the appropriate feature branch with a conventional message, merge into `dev`, delete the feature branch
