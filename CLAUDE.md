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

The official exam guide PDF lives at **`CCA_Foundations_Guide.pdf`** in the repo root. **Gitignored** — local-only, not for distribution. Read it (via the `Read` tool with `pages` parameter) whenever you need to verify a cert mapping rather than guessing.

Domain titles and weights from the guide:

| | Domain | Weight |
|---|---|---|
| **D1** | Agentic Architecture & Orchestration | 27% |
| **D2** | Tool Design & MCP Integration | 18% |
| **D3** | Claude Code Configuration & Workflows | 20% |
| **D4** | Prompt Engineering & Structured Output | 20% |
| **D5** | Context Management & Reliability | 15% |

D5 is **context engineering and multi-agent error propagation**, NOT "operations" or testing. Testing/CI strategy is *not* in the cert at all.

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

**Session 2 in progress.** Step ordering:

1. ✅ Project scaffolding — infrastructure (no cert mapping)
2. ✅ Benchmark interface — `BenchmarkResult` pydantic + `Architecture` ABC (foundation for D4 TS 4.3, applies when wrapped as tool output)
3. ✅ Shared tool layer — `arxiv_search` / `arxiv_fetch` / `format_citation` in `src/papertrail/tools/` (foundation for D2 TS 2.1; D2 TS 2.2 already hit by the 429-retry policy)
4. ⬜ arch_00 adapter — wrap legacy `apps/` code as the deterministic no-LLM baseline, zero Claude tokens (no cert mapping; the floor)
5. ⬜ First benchmark run + Evaluator stub (Evaluator is LLM-based → D4 TS 4.4)

**Deferred decision (live at step 4):** arch_00 timeline/impact stubs — deterministic heuristics (year buckets, category counts) or null fields (Evaluator floors at zero on those dimensions)?

## Update protocol

At the end of each session:
- Produce a four-list end-of-session summary in chat: **done overall** / **this session** / **next session** / **outstanding** (see memory `[[feedback-session-workflow]]`). This is the resumption anchor for the next session — without it, the next session pays ~30 minutes re-deriving state.
- Update the **Current cursor** section above to match the summary
- Save new memories or update existing ones
- Write/update ADRs for any decisions made
- Commit on the appropriate feature branch with a conventional message, merge into `dev`, delete the feature branch
