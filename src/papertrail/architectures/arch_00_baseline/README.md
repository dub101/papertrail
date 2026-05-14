# arch_00 — Baseline (no-LLM floor)

The bottom row of the benchmark table. One arxiv search, no Claude calls.

## What it does

```
topic ──► arxiv_search (over-fetch 20)
       └► filter: post-2007 arxiv ids only
       └► trim to Modes.target_paper_count (default 10)
       └► partition by publication date into ≤3 eras
       └► dress each paper in the PaperEntry schema (heuristic|disclaimer)
       └► return a fully valid BenchmarkResult with zeroed token counters
```

## Per-field policy

The five `PaperEntry` summary fields each get one of two treatments —
**heuristic** (derivable from the arxiv response, no synthesis) or
**disclaimer** (a constant string that the Evaluator scores low):

| Field | Treatment | Value |
|---|---|---|
| `summary_about` | heuristic | `"{title}. {first two sentences of abstract}"` |
| `summary_relation_to_topic` | heuristic | `"Returned by arXiv relevance search for query '{topic}' (rank N of M)."` |
| `summary_problem` | disclaimer | `"Baseline architecture — no problem analysis."` |
| `summary_approach` | disclaimer | `"Baseline architecture — no approach analysis."` |
| `summary_impact` | disclaimer | `"Baseline architecture — no impact analysis."` |
| `era.narrative` | disclaimer | `"Baseline architecture — no era narrative."` |
| `overall_summary` | heuristic | factual aggregate: count, year range, top categories, top authors, baseline self-disclosure |

`confidence` is `0.5` for every paper — "no opinion" — so we don't leak
arxiv's relevance signal into our confidence field.

`citation_count` and `citation_source` are both `None` because we don't
look up citations.

## Failure mode

If arxiv returns fewer than 8 usable papers for the topic (after the
post-2007 filter), `BaselineArchitecture.run` raises
`BaselineTooFewResultsError`. We do **not** pad — padding would mean
fabricating papers and would violate the no-LLM contract.

## Cert mapping

None — arch_00 is the floor by design. arch_01 onward are the cert-relevant
architectures; arch_00 exists so we can quantify what the LLM-driven
pipelines actually contribute.

## Smoke test

```python
import asyncio
from papertrail.architectures.arch_00_baseline import BaselineArchitecture

arch = BaselineArchitecture()
result = asyncio.run(arch.run("transformer attention"))
print(result.deliverable.papers[0].arxiv_id)
print(result.telemetry.total_input_tokens)   # 0
print(result.telemetry.tool_call_count)      # 1
print(len(result.deliverable.timeline))      # ≤ 3
```

## Open follow-up for Step 5

Provenance (git SHA, package version) is currently discovered by the
architecture itself via `subprocess` + `importlib.metadata`. Once the
benchmark harness exists, it should *inject* provenance — the architecture
should not be in the filesystem-introspection business. The local helpers
become override defaults at that point.
