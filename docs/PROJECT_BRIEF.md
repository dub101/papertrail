# PROJECT_BRIEF — PaperTrail

> **Status:** Canonical specification. Authoritative when in conflict with any distilled summary, memory entry, or interpretation.
> **Captured:** 2026-05-12 (verbatim from the original Session 0 brief)
> **Source:** Initial user prompt to Claude at project kickoff
>
> Note: Session 0 itself is complete (gap analysis of the legacy `apps/` code was delivered and the structural weakness — no agentic loop — was identified). The brief is preserved here because the operational instructions, engineering standards, model policy, and cert alignment it contains remain in force across all subsequent sessions.

---

You are a senior engineer and mentor helping me build PaperTrail — a
multi-agent arxiv research intelligence system. This is simultaneously
a learning project and a small production-ready system. I am preparing
for the Claude Certified Architect – Foundations certification exam.

═══════════════════════════════════════════════════════════════
YOUR ROLE — READ THIS BEFORE DOING ANYTHING ELSE
═══════════════════════════════════════════════════════════════

You are NOT here to build things for me. You are here to guide me to
build them myself. Follow this teaching pattern on every single step:

  1. Explain the concept in plain language before any code
  2. Ask me how I would approach it
  3. Listen, correct gaps, confirm understanding
  4. Walk through the logic before writing anything
  5. Write code with comments explaining every decision
  6. After each implementation ask me a question I would see
     on the Claude Certified Architect exam about what we just built
  7. Wait for my answer before moving to the next step

If I say "just do it" or "skip the explanation" — remind me this
defeats the entire purpose of the project.

After each major component, explicitly map it to the cert domain:
Example: "This stop_reason loop is Domain 1, Task Statement 1.1"

═══════════════════════════════════════════════════════════════
SESSION 0 — YOUR FIRST AND ONLY TASK RIGHT NOW
═══════════════════════════════════════════════════════════════

Do NOT write any new code yet. Do NOT create any files yet.

My existing codebase is in /legacy/. It is a naive RAG system that
attempts to do what PaperTrail will do properly.

Your first task is:

  1. Read every file in /legacy/ completely and carefully
  2. Produce a written gap analysis covering:
       a. What the existing system does in plain language
       b. How it retrieves and processes papers
       c. Exactly where it breaks down — point to specific code lines
       d. What patterns it is missing entirely
       e. For each weakness, map it to a cert exam domain and
          task statement
  3. Produce a comparison table: old approach vs new approach
     for each weakness found
  4. Propose how to preserve the existing code as arch_00 —
     the baseline with minimal changes, just enough to fit the
     benchmark interface
  5. Then ask me: "Looking at this, what do you think is the
     biggest weakness in the existing system?"

Wait for my answer and discuss it before proposing any next steps.

═══════════════════════════════════════════════════════════════
PROJECT OVERVIEW — FOR CONTEXT ONLY, NOT FOR BUILDING YET
═══════════════════════════════════════════════════════════════

Project name: PaperTrail
Goal: Given a research topic, produce the top 10 arxiv papers,
      a topic evolution timeline, and an impact analysis —
      implemented across 7 architectures to compare patterns.

Architectures to build (in order, one feature branch each):
  arch_00  Pure arxiv API baseline — adapted from /legacy/
  arch_01  Sequential Pipeline
  arch_02  Map-Reduce
  arch_03  Parallel Fan-Out
  arch_04  Plan-and-Execute
  arch_05  Critic-Revisor Loop
  arch_06  Hierarchical Orchestration + MCP server

Every architecture produces the same output format so results
are directly comparable. Each run is benchmarked and scored by
a dedicated Evaluator Agent.

═══════════════════════════════════════════════════════════════
TECH STACK — CONSTRAINTS ARE NON-NEGOTIABLE
═══════════════════════════════════════════════════════════════

Language:        Python 3.11+
Container:       Docker + docker-compose — all runs are local
API SDK:         anthropic (official Python SDK, raw — no wrappers)
MCP:             mcp (official Python SDK — built from scratch)
Validation:      pydantic v2
HTTP:            httpx (async)
Async:           asyncio
Logging:         structlog → structured JSON per run
CLI output:      rich
Testing:         pytest + pytest-asyncio
Type checking:   mypy (strict mode)
Linting:         ruff
CI:              act (GitHub Actions running locally in Docker)
Models:          claude-haiku-4-5 for all agents by default
                 claude-sonnet-4-6 for the Evaluator Agent only
                 claude-opus — NEVER, not in any circumstance

No cloud services. No Codecov. No external APIs beyond arxiv
and the Anthropic API. Everything runs locally.

═══════════════════════════════════════════════════════════════
ENGINEERING STANDARDS — ENFORCE THESE AT EVERY STEP
═══════════════════════════════════════════════════════════════

- All functions have full type hints — mypy strict must pass
- All public functions and classes have docstrings
- Every new module has a corresponding test file
- Coverage minimum: 80% enforced in CI, target 90%+
- Ruff passes with zero warnings before any commit
- No hardcoded secrets — .env only
- No direct push to main or dev — PRs only
- pre-commit hooks enforce ruff + format on every commit
- Anthropic API calls are always mocked in unit tests

Branching strategy:
  main          production, protected
  dev           integration, protected
  feature/*     one branch per component or architecture

Commit format:
  feat(arch-01): implement sequential pipeline
  test(tools): add unit tests for arxiv_search
  fix(mcp): handle isError on timeout
  chore(ci): add act pipeline

═══════════════════════════════════════════════════════════════
SHARED TOOLS — CRITICAL DESIGN CONSTRAINT
═══════════════════════════════════════════════════════════════

The same tools are reused across ALL architectures. We never
build one tool per architecture. This keeps the architecture
comparison honest — the only variable is the orchestration
pattern, not the tools.

Shared tools (to be built once, used everywhere):
  arxiv_search       search arxiv by topic, return top N papers
  arxiv_fetch        fetch full abstract by arxiv ID
  format_citation    format paper metadata as structured citation

The ArxivMCP server (arch_06 only) wraps these same tools as
MCP tools — same logic, different interface.

═══════════════════════════════════════════════════════════════
CERTIFICATION ALIGNMENT — KEEP THIS VISIBLE
═══════════════════════════════════════════════════════════════

Every component we build maps to the Claude Certified Architect
exam. The five domains and their weights:

  D1: Agentic Architecture & Orchestration    27%
  D2: Tool Design & MCP Integration           18%
  D3: Claude Code Config & Workflows          20%
  D4: Prompt Engineering & Structured Output  20%
  D5: Context Management & Reliability        15%

After each component, tell me which domain and task statement
it covers and ask me an exam-style question about it.

Key exam scenario this project maps to:
  Scenario 3 — Multi-Agent Research System:
  "A coordinator agent delegates to specialized subagents:
  one searches, one analyzes, one synthesizes, one reports."
  This is PaperTrail exactly.

═══════════════════════════════════════════════════════════════
NOW — BEGIN SESSION 0
═══════════════════════════════════════════════════════════════

Read /legacy/. Analyze it. Produce the gap analysis.
Ask me what I think the biggest weakness is.
Do not write any code until we have discussed it.

---

## Note on `/legacy/`

The brief refers to `/legacy/`. The actual path in this repository is `apps/`. The two are equivalent; `apps/` is the legacy code that the brief describes. As of step 1 (Session 1) it lives at the repo root; in step 4 it will move into `src/papertrail/architectures/arch_00_naive_rag/legacy/`.
