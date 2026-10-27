"""Independent verification of first-error labels from structured traces."""

from __future__ import annotations

from typing import Any, Literal


def first_error_from_trace(trace: list[dict[str, Any]]) -> int | Literal["NONE"]:
    """Return 1-indexed first failing step, or NONE if all valid."""
    for step in trace:
        if not step.get("valid", False):
            return int(step["line"])
    return "NONE"


def verify_example(
    solution_lines: list[str],
    verification_trace: list[dict[str, Any]],
    first_error_line: int | Literal["NONE"],
) -> tuple[bool, str]:
    """Cross-check stored label against the verification trace."""
    if len(verification_trace) != len(solution_lines):
        return False, "trace length != solution_lines length"

    for i, (line, step) in enumerate(zip(solution_lines, verification_trace), start=1):
        if int(step.get("line", -1)) != i:
            return False, f"trace line index mismatch at {i}"
        if step.get("stated_line") is not None and step["stated_line"] != line:
            return False, f"stated_line mismatch at {i}"

    derived = first_error_from_trace(verification_trace)
    if derived != first_error_line:
        return False, f"label {first_error_line!r} != derived {derived!r}"

    # After first error, later steps may be marked valid_wrt_wrong_prefix
    if first_error_line != "NONE":
        idx = int(first_error_line) - 1
        if verification_trace[idx].get("valid", True):
            return False, "marked first error step is valid"
        for later in verification_trace[idx + 1 :]:
            if later.get("valid", False) and not later.get("consistent_with_wrong_prefix", False):
                # Later "valid" without noting wrong-prefix consistency is suspicious
                # but allowed if they recompute from ground truth — we forbid that.
                return False, "post-error step marked unconditionally valid"

    return True, "ok"
