# 0003 — Model policy: Haiku 4.5 default, Sonnet 4.6 for Evaluator, never Opus

- **Status:** Accepted
- **Date:** 2026-05-12
- **Deciders:** dub101

## Context

PaperTrail uses Claude across 7 architectures plus a dedicated Evaluator Agent. Model choice affects:

- **Cost** — Opus >> Sonnet >> Haiku
- **Speed** — Haiku is fastest, Opus slowest
- **Reasoning quality** — Opus is strongest on complex multi-step reasoning
- **Comparability across architectures** — the only variable should be the orchestration pattern, not the model strength

Available Claude models (knowledge cutoff January 2026):

- `claude-opus-4-7` — strongest reasoning
- `claude-sonnet-4-6` — balanced
- `claude-haiku-4-5-20251001` — fastest, cheapest

## Decision

- **`claude-haiku-4-5`** — all agents in all architectures by default
- **`claude-sonnet-4-6`** — Evaluator Agent ONLY
- **`claude-opus`** — never used, in any architecture, under any circumstance

## Consequences

**Easier:**
- **Comparable architectures.** Holding the model constant means observed differences come from orchestration, not model power. This is the central premise of the experiment.
- **Cost-bounded.** Many runs across many architectures stay affordable.
- **Faster iteration loop** — Haiku is fast.
- **Meaningful asymmetry** between agents (Haiku) and Evaluator (Sonnet): a stronger judge scoring weaker actors mirrors realistic deployment.

**Harder / traded away:**
- Some architectures might benefit from Sonnet- or Opus-level reasoning; we will not measure that. That's the point — we want to know which orchestration patterns are robust on Haiku.
- If Haiku is consistently insufficient for a pattern, we surface "this orchestration pattern fails on Haiku" rather than the easier "Opus solves it."

## Alternatives considered

- **All-Sonnet** — rejected; too expensive for the planned run volume
- **Per-architecture model choice** — rejected; kills comparability
- **Opus for hard reasoning tasks** — rejected; undermines the experimental design and inflates cost without informing orchestration questions
