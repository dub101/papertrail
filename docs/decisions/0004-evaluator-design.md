# 0004 — Evaluator design: forced tool_use, fail-loud, exact tokens + estimated cost

- **Status:** Accepted
- **Date:** 2026-05-14
- **Deciders:** dub101

## Context

PaperTrail compares 7 research architectures (arch_00 .. arch_06). To make those comparisons meaningful we need a uniform, repeatable grader that scores every architecture's `Deliverable` on the same dimensions. ADR-0003 already pinned this grader to `claude-sonnet-4-6`; this ADR is about everything else — output enforcement, error handling, observability, and prompt management.

Inputs to the decision:

- Cert mappings — the grader is **D4 TS 4.6** (independent review instance, calibrated confidence) and **D4 TS 4.3** (structured output via tool use and JSON schemas) verbatim.
- Cost awareness — the user explicitly asked that LLM-touching subsystems surface their cost. Sonnet is ~$0.03 / call for an arch_00 deliverable.
- Comparability — scores must be on the same scale across architectures and across time.
- Failure modes — `tool_choice` + a JSON schema eliminates syntax errors but not refusals or semantic-validation errors.

## Decision

1. **Model:** `claude-sonnet-4-6` is the default (per ADR-0003). The constructor accepts a `model` override — only for ad-hoc prompt iteration. The runner always uses the default; canonical scores are Sonnet scores.
2. **Output enforcement:** Forced tool use. One tool named `submit_evaluation`, its `input_schema` is `EvaluatorVerdict.model_json_schema()`, and `tool_choice={"type": "tool", "name": "submit_evaluation"}` forces the model to call it. This is the canonical D4 TS 4.3 pattern.
3. **Failure handling:** Fail loud. Two distinct exception types so callers (and future retry policies) can branch:
   - `EvaluatorRefusedError` — response had no matching `tool_use` block. Carries `stop_reason` + content summary.
   - `EvaluatorInvalidOutputError` — `tool_use` block was present but its input failed `EvaluatorVerdict` validation. Carries the pydantic message.
   No automatic retry. Retry-with-feedback (D4 TS 4.4) is a deferred decision — we want to *see* the failure modes before adding recovery.
4. **Verdict schema:** Nine graded dimensions plus a holistic `overall`, a self-reported `confidence`, and a free-text `critique`:
   - `selection_relevance`, `timeline_quality`, `timeline_veracity` — top-level dimensions
   - `synthesis.{about, relation_to_topic, problem, approach, impact}` — one per per-paper summary field, aggregated across all papers
   - `executive_summary` — grade of `Deliverable.overall_summary`
   `confidence` is the calibrated-review hook (D4 TS 4.6). The `critique` field forces narrative justification.
5. **Cost observability:** Token counts come from the API response (`Message.usage.input_tokens` / `output_tokens`) — exact, authoritative. `cost_usd_estimated` is computed client-side from `MODEL_PRICING_PER_MTOK` and **labelled with `_estimated`** in the field name. The pricing dict has a verification-date comment and unknown models fall back to `0.0` (visibly wrong, prompts a fix).
6. **Prompts as data:** `src/papertrail/prompts/` is a versioned prompt store; the Evaluator loads `evaluator_v1.md` via `importlib.resources`. Bumping a prompt is a code change with a visible diff; the version slug lands in `Telemetry.prompt_versions` for reproducibility.
7. **Dry-run path:** `DryRunEvaluator` mirrors the `evaluate(...) -> EvaluatorScore` signature, emits **sentinel** scores (`0.123` everywhere, all rationales contain "DRY RUN"), and stamps provenance as `evaluator_model="dry-run-no-llm"`. Persisted dry-run JSON is self-identifying.

## Consequences

**Easier:**

- **Cost is predictable.** No retry means no unbounded cost from a single bad call. The CLI shows exact tokens + estimated $ at-a-glance.
- **Verdicts are schema-guaranteed.** `tool_use` + pydantic = the verdict either validates or raises a specific exception with a debuggable message. No `JSONDecodeError` paths.
- **Easy to test.** The Evaluator is fully exercised in unit tests with a duck-typed `AsyncAnthropic` fake (`tests/test_evaluator.py`). Zero network calls.
- **Self-labeling outputs.** A glance at persisted JSON reveals whether it's a dry-run (sentinel scores + `dry-run-no-llm` model id) or a real grade.
- **Comparability across runs.** Holding the model + prompt version constant means observed deltas are meaningful.

**Harder / traded away:**

- **No auto-recovery.** A refusal or invalid output crashes the run. The trade is intentional — we want to fix root causes (prompt, schema, deliverable) rather than mask them. Adding retry later is straightforward: catch `EvaluatorInvalidOutputError`, append the validation message to a follow-up turn, recall the API. Skipping for now.
- **Pricing can drift.** `MODEL_PRICING_PER_MTOK` is hand-maintained. Anthropic price changes between updates produce a small lie in `cost_usd_estimated`; the `_estimated` suffix on the field name keeps that lie legible.
- **Prompts vs code coupling.** Prompts live next to code. Editing one requires a code review; a non-engineer can't tune the grader without a PR. Acceptable for a cert-prep learning project; revisit if collaborators need looser ownership.
- **One arch model = Sonnet costs scale linearly with runs.** ~$0.03/run is cheap, but 1000 runs is $30 — we're not at that volume yet, but a future cost gate (max runs/month) may be needed.

## Alternatives considered

- **Free-text JSON parsing instead of forced tool_use** — rejected. The cert PDF (TS 4.3) is unambiguous: tool_use with a JSON schema is *the* canonical pattern for guaranteed structured output. Free-text JSON parsing reintroduces syntax errors as a failure mode.
- **`tool_choice={"type": "any"}`** — rejected. Multiple tools would be required to justify `any`; we have one tool, so naming it explicitly is more honest and prevents future drift if more tools are added.
- **Retry-with-error-feedback for `EvaluatorInvalidOutputError` (D4 TS 4.4)** — deferred, not rejected. Worth adding once we have data on real failure rates. The exception split (`Refused` vs `InvalidOutput`) leaves room.
- **Per-call cost from the Anthropic API** — not available. The standard messages response returns token counts but no per-call dollar amount; the Admin API has aggregate billing with reporting lag. Client-side estimate is the only per-call option.
- **Tokens only, no `cost_usd_estimated`** — rejected. The user is cost-aware and the at-a-glance dollar read is the primary observability lever. The `_estimated` label makes the trade-off explicit at the data layer.
- **Inline system prompt in `evaluator.py`** — rejected. `prompts/` as a versioned data folder (the user's framing) makes prompt bumps a visible diff and ties them to the version slug recorded in `Telemetry.prompt_versions`.
- **Plugin-discovery architecture registry** — rejected for `runner.py`. The 7-architecture bound is small and known; an explicit dict produces a clear error on typos rather than silently mis-selecting an architecture.
