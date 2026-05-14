"""Tests for the prompts package loader."""

from __future__ import annotations

import pytest

from papertrail.prompts import load_prompt


def test_load_prompt_returns_evaluator_v1() -> None:
    text = load_prompt("evaluator", "v1")
    # Smoke checks on the body — full-text compare is brittle as the prompt
    # iterates. These two substrings are load-bearing instructions in v1.
    assert "You are the Evaluator agent" in text
    assert "submit_evaluation" in text


def test_load_prompt_missing_file_raises() -> None:
    # The missing-prompt path is a bug, not a runtime condition — let the
    # underlying filesystem error propagate so the stack trace points at
    # the typo'd name/version.
    with pytest.raises(FileNotFoundError):
        load_prompt("evaluator", "v999")
