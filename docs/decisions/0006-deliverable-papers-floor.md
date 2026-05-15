# 0006 — Loosen `Deliverable.papers` floor from 8 to 4

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** dub101
- **Supersedes:** part of ADR-0005 (specifically the line "8 — matches the `Deliverable.papers` schema floor")

## Context

`BenchmarkResult.deliverable.papers` was ratified in Step 2 with
`Field(min_length=8, max_length=12)`. The floor of 8 reflected an
implicit assumption that "any topic worth analysing has ≥8 arXiv papers."

While designing arch_01 stage 1 (the agentic search loop), the question
came up: what happens when the model legitimately decides — via clean
`stop_reason == "end_turn"` — that fewer than 8 unique papers cover the
topic? Two scenarios where this is the *right* answer, not a failure:

- A niche topic with only 4-7 papers on arXiv but each is high-signal
  (e.g. a freshly-published sub-field).
- The model exhausts query diversity and the duplicate rate dominates
  before 8 unique results accumulate.

With the old floor, arch_01 would either have to raise an error in
stage 1 (rejecting the run entirely) or pass through and crash at the
final assembly step with a less informative `pydantic.ValidationError`.

The user's framing: "a new topic with 4 super-important papers should
pass and allow them to be used." That is incompatible with `min_length=8`.

## Decision

Change `src/papertrail/benchmark.py` only:

```python
# before:
papers: list[PaperEntry] = Field(min_length=8, max_length=12)
# after:
papers: list[PaperEntry] = Field(min_length=4, max_length=12)
```

The upper bound `max_length=12` stays — it is a comparability constraint,
not a quality floor, and 12 papers is already a stretch for a single
synthesis pass.

The new lower bound `min_length=4` is the smallest count at which:

- An era partition produces ≥1 meaningful era (`TimelineEra.paper_ids`
  requires `min_length=1`).
- The stage 3 per-paper synthesis batch of 5 papers/call still produces
  one reasonably-sized batch.
- The Evaluator can score per-dimension across enough samples to be
  meaningful rather than noisy.

Below 4 papers, the deliverable does not carry enough signal to be a
useful benchmark artifact, and the run should fail.

### What does *not* change

- **arch_00 keeps its own strict 8-floor.** `BaselineTooFewResultsError`
  in `src/papertrail/architectures/arch_00_baseline/baseline.py` still
  raises if arxiv returns <8 papers. That floor is arch_00's *self-imposed*
  honesty floor for the no-LLM baseline ("we can't pretend to produce a
  rich timeline from 4 arxiv hits"). It is no longer the schema floor;
  it is an architecture-level discipline. The benchmark.py schema
  constraint and arch_00's local guard are now intentionally decoupled.

- **arch_01 aim is still 8-12 papers.** The "aim of 8" lives in the
  arch_01 search prompt (`arch_01_search_v1.md`): the model is instructed
  to target 8-12 unique papers and to accept fewer only when query
  diversity is genuinely exhausted. The schema floor of 4 is the
  *minimum* the architecture can flow through, not the *target*.

- **arch_01 stage 1 still raises** when it ends with <4 unique papers
  via `SearchInsufficientResultsError`. The exception is the new
  schema-aware guard: it fails fast at stage 1 instead of letting an
  empty/near-empty result poison stages 2-5 before crashing at assembly.

### Asymmetry between arch_00 and arch_01

For a topic with 4-7 arxiv results, arch_00 will raise
`BaselineTooFewResultsError` while arch_01 will succeed. This asymmetry
is intentional and informative: it surfaces the LLM-driven architecture's
ability to handle thin-data topics that the no-LLM floor cannot. Calling
this out explicitly so future readers don't "fix" the asymmetry by
loosening arch_00 in sympathy.

## Cert mappings

- No new cert hooks. This is a schema-contract change, not an architecture
  pattern. ADR-0005's existing mappings (D1 TS 1.1, D1 TS 1.6, D4 TS 4.3,
  D5 TS 5.1, D5 TS 5.3) all remain valid.

## Consequences

**Easier:**

- arch_01 can flow through thin-data topics end-to-end instead of failing
  at stage 1 or crashing at assembly.
- The schema's role becomes clearer: it enforces *structural* validity
  (≥4 papers, ≥1 era, fields populated), not *target quality*. Target
  quality lives in prompts and architecture-local guards.

**Harder / traded away:**

- arch_00 and arch_01 are no longer directly comparable on thin-data
  topics — arch_00 refuses to run, arch_01 produces output. The
  asymmetry must be communicated in the eventual cross-architecture
  benchmark report.
- Future architectures (arch_02 …) inherit the looser floor and must
  decide for themselves whether they need a local guard like arch_00's.

## Alternatives considered

- **Keep `min_length=8`, raise in stage 1.** Rejected — would reject
  the user-relevant "4 important papers" case at run time.
- **Loosen to `min_length=1`.** Rejected — 1, 2, or 3 papers cannot
  carry the structure the rest of the deliverable assumes (eras,
  synthesis across papers, executive summary). 4 is the smallest count
  where the schema and the architecture's design assumptions still hold
  together.
- **Loosen arch_00 in sympathy.** Rejected — arch_00 is the no-LLM
  floor; its 8-floor was a deliberate honesty constraint, not a side
  effect of the schema. Decoupling them lets each layer carry its own
  semantics.

## Open / follow-up

- Update test `tests/test_benchmark.py` to pin the new bound
  (`papers=[3]` → ValidationError, `papers=[4]` → accepts).
- arch_01 `SearchAgent` triggers `SearchInsufficientResultsError`
  on `len(unique_papers) < 4`, not `< 8`.
