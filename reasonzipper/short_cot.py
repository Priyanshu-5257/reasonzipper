"""Deterministic short chain-of-thought from verified traces."""

from __future__ import annotations

from typing import Any, Literal


def build_short_reasoning(verification_trace: list[dict[str, Any]]) -> str:
    """Compact check list ending at the first error (or NONE)."""
    lines: list[str] = []
    for step in verification_trace:
        n = int(step["line"])
        if step.get("valid", False):
            lines.append(f"L{n}: OK ({step.get('op', '?')})")
        else:
            lines.append(
                f"L{n}: FAIL expected={step.get('expected')} stated={step.get('stated')}"
            )
            break
    if not lines:
        lines.append("empty")
    return "\n".join(lines)


def build_short_completion(
    verification_trace: list[dict[str, Any]],
    first_error_line: int | Literal["NONE"],
) -> str:
    reasoning = build_short_reasoning(verification_trace)
    if first_error_line == "NONE":
        answer = "ERROR_LINE=NONE"
    else:
        answer = f"ERROR_LINE={int(first_error_line)}"
    return f"<think>\n{reasoning}\n</think>\n{answer}"
