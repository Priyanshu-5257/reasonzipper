from reasonzipper.codes import (
    Codebook,
    build_code_completion,
    build_filler_completion,
    count_code_tokens,
    make_token_list,
)


def test_token_list_unique():
    toks = make_token_list(128)
    assert len(toks) == 128
    assert len(set(toks)) == 128
    assert toks[0] == "<|r000|>"
    assert toks[127] == "<|r127|>"


def test_codebook_stops_at_first_error():
    cb = Codebook(n_tokens=128, codes_per_block=2, max_codes_per_solution=16)
    trace = [
        {"line": 1, "valid": True, "op": "add", "expected": "3", "stated": "3"},
        {"line": 2, "valid": False, "op": "add", "expected": "5", "stated": "6"},
        {"line": 3, "valid": False, "op": "answer", "expected": "5", "stated": "6"},
    ]
    text, ids = build_code_completion(trace, 2, cb)
    assert text.startswith("<think>")
    assert "ERROR_LINE=2" in text
    assert "L1" not in text  # unreadable
    assert len(ids) == 4  # 2 blocks * 2 codes
    assert count_code_tokens(text) == 4


def test_filler_matched_budget():
    t = build_filler_completion(3, "NONE", filler_id=0)
    assert count_code_tokens(t) == 3
    assert t.endswith("ERROR_LINE=NONE")
