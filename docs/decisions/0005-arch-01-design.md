# 0005 — arch_01 design: sequential pipeline with agentic-loop search

- **Status:** Accepted (design); implementation in progress
- **Date:** 2026-05-14
- **Deciders:** dub101
- **Updated by:** [ADR-0006](0006-deliverable-papers-floor.md) — `Deliverable.papers` floor loosened 8 → 4; `MIN_PAPERS_TO_PROCEED` in this ADR is correspondingly 4, not 8.

## Context

arch_01 is the first LLM-driven architecture in PaperTrail. arch_00 set the no-LLM floor; arch_01 establishes what an LLM baseline looks like before we move to harder patterns (map-reduce, plan-and-execute, hierarchical). The Evaluator scored arch_00 at 0.18 overall; arch_01 needs to actually fill the per-paper synthesis fields and era narratives that arch_00 disclaimed.

Two cert hooks were available for the search stage in particular:

- **D1 TS 1.6** (prompt chaining / sequential pipelines) — fits trivially if we chain stages
- **D1 TS 1.1** (agentic loop) — fits only if some stage uses `tool_use` with `stop_reason`-driven iteration

The user's framing — "let me put a paragraph of the topic instead of a keyword so the agent can make several calls with different search queries" — made stage 1 the natural place to put the agentic loop. The pipeline as a whole is sequential, but stage 1 internally is iterative.

## Decision

### Six stages

| # | Stage | Type | Tool? | Output to next stage |
|---|---|---|---|---|
| 1 | Search | LLM **agentic loop** (D1 TS 1.1) | `arxiv_search` | ~25-35 deduped candidate papers |
| 2 | Triage & select | LLM, single forced `tool_use` | none | 8-12 chosen, with one-line justifications |
| 3 | Per-paper synthesis | LLM, batched **5 papers/call** | none | All 5 summary fields per paper |
| 4 | Era partition + narrative | LLM | none | 2-4 eras with `paper_ids` + narrative |
| 5 | Executive summary | LLM | none | `overall_summary` (~150-200 words) |
| 6 | Assembly | pure code | none | Valid `BenchmarkResult` |

### Stage 1: agentic loop, not one-shot

`arch_01.run(topic)` accepts a `topic` that may be a paragraph. The model decomposes it into multiple `arxiv_search` queries and decides itself when coverage is adequate. Loop terminates on `stop_reason == "end_turn"`. Safety cap `MAX_SEARCH_ITERATIONS=8` is a circuit breaker only, never the primary stopping mechanism (TS 1.1 anti-pattern: "setting arbitrary iteration caps as the primary stopping mechanism").

### Partial-results recovery in stage 1 only

If an unexpected `stop_reason` (`max_tokens`, `pause_turn`, `refusal`, etc.) fires after the loop has accumulated `>= MIN_PAPERS_TO_PROCEED` (8 — matches the `Deliverable.papers` schema floor), the pipeline records an `ErrorRecord(recovered=True)` and proceeds. Below 8 papers, it raises `SearchInsufficientResultsError`. Same logic if the safety cap is hit.

This is the **D5 TS 5.3 pattern**: structured error context, neither silently suppressed nor blanket-terminated. Stages 2-5 do not get this treatment — they lack a partial-result mode (every schema field has `min_length=1` or non-empty list requirements) so failure there means failure of the run.

### Compact tool_result hand-back in stage 1

The model sees per paper: `arxiv_id`, `title`, first sentence of abstract, year, `primary_category` (~65 tokens). Authors, full abstract, URLs, DOIs, secondary categories are stripped. Plus a running `Total unique papers so far: N` counter and a duplicates-filtered count per call. ~10x reduction in per-iteration token cost (~$0.005-$0.015 per stage 1 vs ~$0.030 without compaction on a 4-iteration loop).

### Citations deferred

`PaperEntry.citation_count` / `citation_source` stay `None` in arch_01 as they do in arch_00. Adding a citation-lookup tool (Semantic Scholar / OpenAlex) doubles stage 1's tool surface and isn't core to the sequential-pipeline cert hook. Revisit for arch_02 or later.

### Structured output via forced tool_use everywhere else

Stages 2, 3, 4, 5 use the same pattern as the Evaluator: one tool per stage, `tool_choice={"type":"tool","name":...}`, pydantic-derived `input_schema`. **D4 TS 4.3**.

### Fail-loud on stages 2-5

Stage-specific exception types (`TriageError`, `SynthesisError`, `EraPartitionError`, `ExecutiveSummaryError`) so failures surface with a clear stage label. Retry-with-feedback (D4 TS 4.4) deferred, consistent with [ADR-0004](0004-evaluator-design.md).

### Model

`claude-haiku-4-5` per [ADR-0003](0003-haiku-default.md) for every LLM stage.

### Cost envelope (per run, on Haiku)

- Stage 1 (agentic loop): ~$0.005-$0.015
- Stage 2 (triage): ~$0.007
- Stage 3 (synthesis, 2 calls): ~$0.016
- Stage 4 (era partition): ~$0.007
- Stage 5 (executive summary): ~$0.003
- **arch_01 total**: ~$0.04-$0.05
- + Evaluator (Sonnet) ~$0.03
- **Total per benchmark run**: ~$0.07-$0.08

## Cert mappings (literal)

- **D1 TS 1.1** — stage 1 agentic loop (`stop_reason` inspection + iteration)
- **D1 TS 1.6** — overall pipeline = prompt chaining
- **D4 TS 4.3** — forced `tool_use` with JSON schemas on every LLM stage
- **D5 TS 5.1** — structured handoff between stages, compact tool_result hand-back inside stage 1
- **D5 TS 5.3** — partial-results recovery on stage 1 with `ErrorRecord(recovered=True)`; stage-typed exceptions across the pipeline

Five distinct cert hooks in one architecture is high but each is independently realized — they're not double-counting.

## Consequences

**Easier:**

- Clean comparison to arch_00 (same shape; arch_01 only has to clear the disclaimer-field floor).
- Five distinct cert hooks satisfied in one architecture.
- Each stage is independently testable.
- Per-stage cost is bounded; one bad call doesn't poison the rest.
- Paragraph-topic input gives the model room to decompose the search task itself, instead of pre-tokenizing the topic into a single keyword.

**Harder / traded away:**

- Stage 1 cost is variable (~$0.005-$0.015) depending on how many queries the model issues.
- Compact tool_result format strips authors and full abstracts; if a later use case needs them, we'd add a `arxiv_fetch` tool (already exists in `tools/arxiv.py`) and grant it to a downstream stage.
- No retry on stages 2-5 means a single bad output crashes the whole run; that's the no-retry policy from ADR-0004 generalized.
- arch_01 is the first architecture that can fail mid-run; the failure-mode surface area (refusal, max_tokens, validation, network) is genuinely new and may require iteration on prompts and error handling.

## Alternatives considered

- **Programmatic search (no LLM in stage 1)** — rejected: identical to arch_00, leaves TS 1.1 to a later architecture, and forfeits the paragraph-topic decomposition value.
- **One-shot `tool_use` search (no loop)** — rejected: would have hit D4 TS 4.3 only, not D1 TS 1.1. Same paragraph-decomposition reason as above.
- **Merge era partition + executive summary into one stage** — rejected: doubles context size and risks "lost in the middle" per D5 TS 5.1. The narrower stage 5 prompt also reduces fabrication risk in the executive summary.
- **Per-paper synthesis at one call per paper** — rejected as too expensive (10 calls vs 2); batching 5 papers/call balances cost against lost-in-middle risk.
- **Hand back full `ArxivPaper` in stage 1 tool_result** — rejected; ~10x token cost for marginal model utility. The triage stage (2) can request full abstracts if we later see triage misses traced to missing context.
- **Retry-with-feedback for stage failures (D4 TS 4.4)** — deferred. Add when real failures observed, same call as in ADR-0004.
- **Citation lookup integrated as part of stage 2 triage** — deferred. Doubles stage 2 surface and isn't core to the cert hook.
- **One large synthesis stage that produces papers + eras + executive summary in one call** — rejected; collapses three distinct cognitive tasks into one massive context and is exactly the anti-pattern the cert flags.

## Open at time of writing

- Implementation is **not started**. Next session resumes by writing `src/papertrail/architectures/arch_01_sequential/search.py` (the `SearchAgent` for stage 1).
- The compact format and tool definition shape are designed but not coded — see the session 4 transcript for the markdown example and `ARXIV_SEARCH_TOOL_DEF` sketch.
