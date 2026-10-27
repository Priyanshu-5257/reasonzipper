"""Data schemas for synthetic first-error localization examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

ErrorType = Literal[
    "arithmetic",
    "invalid_transform",
    "wrong_sign",
    "order_of_operations",
    "fraction",
    "none",
]

SplitName = Literal["train", "val", "test_id", "test_ood"]


@dataclass
class Example:
    id: str
    template_id: str
    split: SplitName
    problem: str
    solution_lines: list[str]
    first_error_line: int | Literal["NONE"]
    error_type: ErrorType
    verification_trace: list[dict[str, Any]]
    gold_answer: str
    source: str = "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def format_gold_answer(first_error_line: int | Literal["NONE"]) -> str:
    if first_error_line == "NONE":
        return "ERROR_LINE=NONE"
    return f"ERROR_LINE={int(first_error_line)}"


def parse_error_line(text: str) -> Optional[int | Literal["NONE"]]:
    """Parse final-answer region; returns None if malformed."""
    import re

    # Prefer last ERROR_LINE= occurrence (after </think> if present)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = re.search(r"ERROR_LINE\s*=\s*(NONE|\d+)\b", text.strip(), re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).upper()
    if val == "NONE":
        return "NONE"
    n = int(val)
    if n < 1:
        return None
    return n
