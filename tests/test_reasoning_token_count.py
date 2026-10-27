from __future__ import annotations

from reasonzipper.metrics import count_code_tokens_in_text, count_reasoning_tokens


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        # whitespace tokenize
        return text.split() if text.strip() else []


def test_count_reasoning_tokens():
    text = "<think>one two three</think>\nERROR_LINE=1"
    stats = count_reasoning_tokens(text, FakeTok())
    assert stats["has_think"] is True
    assert stats["reasoning_tokens"] == 3
    assert stats["missing_close"] is False


def test_code_token_count():
    assert count_code_tokens_in_text("<|r000|> <|r012|> hello") == 2
