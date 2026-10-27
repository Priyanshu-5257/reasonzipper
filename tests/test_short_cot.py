from reasonzipper.short_cot import build_short_completion


def test_short_cot_stops_at_first_error():
    trace = [
        {"line": 1, "valid": True, "op": "add", "expected": "3", "stated": "3"},
        {"line": 2, "valid": False, "op": "add", "expected": "5", "stated": "6"},
        {"line": 3, "valid": False, "op": "answer", "expected": "5", "stated": "6"},
    ]
    text = build_short_completion(trace, 2)
    assert "<think>" in text and "</think>" in text
    assert "L1: OK" in text
    assert "L2: FAIL" in text
    assert "L3:" not in text
    assert text.strip().endswith("ERROR_LINE=2")
