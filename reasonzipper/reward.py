"""Correctness-first length reward for GRPO."""

from __future__ import annotations

from typing import Sequence


def correctness_first_rewards(
    corrects: Sequence[bool],
    lengths: Sequence[int],
    *,
    alpha: float = 0.2,
    budget_b: float = 512.0,
) -> list[float]:
    """R_i = correct_i - alpha * all_correct * min(L_i / B, 1)."""
    if len(corrects) != len(lengths):
        raise ValueError("corrects and lengths length mismatch")
    all_correct = all(corrects) and len(corrects) > 0
    out: list[float] = []
    for c, L in zip(corrects, lengths):
        r = float(c) - float(alpha) * float(all_correct) * min(float(L) / budget_b, 1.0)
        out.append(r)
    return out
