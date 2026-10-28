"""Parsing, token accounting helpers, and scoring."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from reasonzipper.schemas import parse_error_line

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def extract_reasoning_span(text: str) -> tuple[Optional[str], bool, bool]:
    """Return (reasoning_text, missing_close, truncated_open)."""
    if THINK_OPEN not in text:
        return None, False, False
    after = text.split(THINK_OPEN, 1)[1]
    if THINK_CLOSE not in after:
        return after, True, False
    return after.split(THINK_CLOSE, 1)[0], False, False


def count_reasoning_tokens(text: str, tokenizer: Any) -> dict[str, Any]:
    """Count tokens between <think>...</think>; codes included if present."""
    span, missing_close, _ = extract_reasoning_span(text)
    if span is None:
        return {
            "reasoning_tokens": 0,
            "has_think": False,
            "missing_close": False,
            "total_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        }
    r_ids = tokenizer.encode(span, add_special_tokens=False)
    return {
        "reasoning_tokens": len(r_ids),
        "has_think": True,
        "missing_close": missing_close,
        "total_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
    }


def score_prediction(
    pred_text: str,
    gold: int | Literal["NONE"],
) -> dict[str, Any]:
    parsed = parse_error_line(pred_text)
    missing_close = False
    if THINK_OPEN in pred_text and THINK_CLOSE not in pred_text:
        missing_close = True
    # Format failure = cannot parse ERROR_LINE. An unclosed </think> alone
    # (e.g. hit max_new_tokens) should not void a valid parsed answer.
    malformed = parsed is None
    correct = (not malformed) and parsed == gold
    return {
        "parsed": parsed,
        "gold": gold,
        "correct": bool(correct),
        "malformed": bool(malformed),
        "missing_close": missing_close,
    }


def build_prompt(problem: str, solution_lines: list[str], thinking: bool = True) -> str:
    lines = "\n".join(solution_lines)
    instruction = (
        "A student wrote a short solution. Identify the earliest incorrect line. "
        "If the solution is fully correct, answer ERROR_LINE=NONE. "
        "Otherwise answer ERROR_LINE=<N> with N the 1-based line number. "
        "You may reason first, but your FINAL line must be exactly "
        "ERROR_LINE=NONE or ERROR_LINE=<N> with no other text on that line."
    )
    user = f"{instruction}\n\nProblem:\n{problem}\n\nStudent solution:\n{lines}"
    if thinking:
        return user + (
            "\n\nRemember: end with a single final line of the form ERROR_LINE=..."
        )
    return user + "\n\nAnswer with only ERROR_LINE=..."

_CODE_RE = re.compile(r"<\|r\d{3}\|>")


def count_code_tokens_in_text(text: str) -> int:
    return len(_CODE_RE.findall(text))
