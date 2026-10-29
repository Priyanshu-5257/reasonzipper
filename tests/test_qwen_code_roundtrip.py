"""Integration: selected existing IDs survive Qwen chat templating.

Skipped when the tokenizer cannot be loaded offline.
"""

from __future__ import annotations

import pytest

from reasonzipper.existing_vocab import (
    assert_ids_roundtrip,
    build_candidate_token_ids,
    build_think_completion_from_ids,
)


@pytest.fixture(scope="module")
def qwen_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B", trust_remote_code=True, local_files_only=True
        )
    except Exception:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B", trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Qwen tokenizer unavailable: {e}")
    return tok


def test_qwen_candidate_concat_and_chat_template(qwen_tok):
    cands = build_candidate_token_ids(qwen_tok, k=32, seed=0)
    # Find a short sequence that round-trips as a body
    ids = None
    for a in cands:
        for b in cands:
            if a == b:
                continue
            try:
                assert_ids_roundtrip(qwen_tok, [a, b])
                ids = [a, b]
                break
            except AssertionError:
                continue
        if ids:
            break
    if ids is None:
        # Fall back to single token
        for a in cands:
            try:
                assert_ids_roundtrip(qwen_tok, [a])
                ids = [a]
                break
            except AssertionError:
                continue
    assert ids is not None

    completion, out_ids = build_think_completion_from_ids(
        qwen_tok,
        ids,
        "ERROR_LINE=2",
        prompt="A student wrote a short solution. Identify the earliest incorrect line.",
        verify_roundtrip=True,
        verify_formatted=True,
        verify_chat_template=True,
        enable_thinking=True,
    )
    assert out_ids == ids
    assert "ERROR_LINE=2" in completion
