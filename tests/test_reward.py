"""Stage B reward helper tests (placeholder logic mirrored here until train_grpo)."""

from __future__ import annotations


def correctness_first_reward(
    correct: bool, length: int, all_group_correct: bool, alpha: float = 0.2, B: float = 512.0
) -> float:
    return float(correct) - alpha * float(all_group_correct) * min(length / B, 1.0)


def test_length_only_when_all_correct():
    assert correctness_first_reward(True, 100, False) == 1.0
    r_short = correctness_first_reward(True, 100, True, alpha=0.2, B=512)
    r_long = correctness_first_reward(True, 400, True, alpha=0.2, B=512)
    assert r_short > r_long


def test_incorrect_not_rewarded_for_shortness():
    assert correctness_first_reward(False, 10, False) == 0.0
    assert correctness_first_reward(False, 10, True) < 0.0
