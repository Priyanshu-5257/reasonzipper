from __future__ import annotations

from reasonzipper.metrics import score_prediction
from reasonzipper.schemas import parse_error_line


def test_parse_plain():
    assert parse_error_line("ERROR_LINE=3") == 3
    assert parse_error_line("ERROR_LINE=NONE") == "NONE"


def test_parse_after_think():
    text = "<think>checking line 2</think>\nERROR_LINE=2"
    assert parse_error_line(text) == 2


def test_malformed():
    assert parse_error_line("the answer is line 2") is None
    scored = score_prediction("<think>oops", gold=1)
    assert scored["malformed"] is True
    assert scored["correct"] is False


def test_score_correct():
    scored = score_prediction("<think>x</think>\nERROR_LINE=NONE", gold="NONE")
    assert scored["correct"] is True
