from __future__ import annotations

from reasonzipper.metrics import score_prediction
from reasonzipper.schemas import parse_error_line


def test_parse_plain():
    assert parse_error_line("ERROR_LINE=3") == 3
    assert parse_error_line("ERROR_LINE=NONE") == "NONE"


def test_parse_after_think():
    text = "<think>checking line 2</think>\nERROR_LINE=2"
    assert parse_error_line(text) == 2


def test_parse_after_empty_and_extra_think_closes():
    # Qwen A0 often emits empty think + duplicate closes before the answer.
    text = "<think>\n</think>\n\n</think>\n\nreasoning...\nERROR_LINE=4"
    assert parse_error_line(text) == 4


def test_parse_prefers_last_error_line():
    text = "<think>x</think>\nERROR_LINE=1\nwait\nERROR_LINE=3"
    assert parse_error_line(text) == 3


def test_malformed():
    assert parse_error_line("the answer is line 2") is None
    scored = score_prediction("<think>oops", gold=1)
    assert scored["malformed"] is True
    assert scored["correct"] is False


def test_missing_close_still_scores_if_error_line_present():
    scored = score_prediction("<think>almost done\nERROR_LINE=2", gold=2)
    assert scored["missing_close"] is True
    assert scored["malformed"] is False
    assert scored["correct"] is True


def test_score_correct():
    scored = score_prediction("<think>x</think>\nERROR_LINE=NONE", gold="NONE")
    assert scored["correct"] is True
