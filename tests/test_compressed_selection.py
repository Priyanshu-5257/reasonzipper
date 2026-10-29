from __future__ import annotations

from reasonzipper.compressed_selection import (
    combine_scores,
    effective_codebook_size,
    _zscore,
)
from reasonzipper.existing_vocab import (
    answer_label_code_ids,
    assert_ids_roundtrip,
    build_think_completion_from_ids,
)


def test_combine_scores_func_primary():
    assert combine_scores(0.99, -1.0, alpha=0.0, beta=1.0) == -1.0
    assert combine_scores(0.1, -0.5, alpha=0.0, beta=1.0) > combine_scores(
        0.9, -2.0, alpha=0.0, beta=1.0
    )


def test_zscore_zero_var():
    assert _zscore([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_zscore_orders():
    z = _zscore([1.0, 2.0, 3.0])
    assert z[0] < z[1] < z[2]


def test_effective_codebook_size():
    u = effective_codebook_size([0, 0, 0, 0], 8)
    assert u["n_used"] == 1.0
    assert u["top1_share"] == 1.0
    d = effective_codebook_size([0, 1, 2, 3], 8)
    assert d["n_used"] == 4.0
    assert d["effective_n"] > u["effective_n"]


def test_answer_label_codes_deterministic():
    cands = list(range(10, 74))
    a = answer_label_code_ids(cands, 2, n_codes=2)
    b = answer_label_code_ids(cands, 2, n_codes=2)
    assert a == b
    none_ids = answer_label_code_ids(cands, "NONE", n_codes=1)
    assert none_ids[0] == cands[0]
    assert answer_label_code_ids(cands, 1, n_codes=1)[0] == cands[1]


class _RoundTripTok:
    """Tokens 'A','B' are single-id stable; chat template wraps completion."""

    def encode(self, text, add_special_tokens=False):
        if text == "A":
            return [1]
        if text == "B":
            return [2]
        if text == "AB":
            return [1, 2]
        if "AB" in text and "ERROR_LINE" in text:
            return [10, 1, 2, 11, 12]
        if text.startswith("<think>"):
            return [10, 1, 2, 11, 12]
        return [99]

    def decode(self, ids, skip_special_tokens=False):
        mapping = {1: "A", 2: "B"}
        return "".join(mapping[int(i)] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, enable_thinking=True):
        # Simulate Qwen wrapping: still contains the assistant completion body.
        user = messages[0]["content"]
        asst = messages[1]["content"]
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{asst}"


def test_assert_ids_roundtrip_ok():
    body = assert_ids_roundtrip(_RoundTripTok(), [1, 2])
    assert body == "AB"


def test_build_think_completion_span():
    completion, ids = build_think_completion_from_ids(
        _RoundTripTok(), [1, 2], "ERROR_LINE=1"
    )
    assert ids == [1, 2]
    assert "<think>" in completion
    assert "ERROR_LINE=1" in completion


def test_build_think_completion_chat_template():
    completion, ids = build_think_completion_from_ids(
        _RoundTripTok(),
        [1, 2],
        "ERROR_LINE=1",
        prompt="Find the error.",
        verify_chat_template=True,
    )
    assert ids == [1, 2]
    assert "ERROR_LINE=1" in completion
