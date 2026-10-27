from __future__ import annotations

from reasonzipper.generate import make_example
from reasonzipper.verify import first_error_from_trace, verify_example


def test_first_error_propagated_still_points_to_earliest():
    ex = make_example(
        ex_id="prop",
        template_id="word_two_step",
        split="train",
        seed=99,
        mode="single",
    )
    assert ex.first_error_line == first_error_from_trace(ex.verification_trace)
    ok, msg = verify_example(ex.solution_lines, ex.verification_trace, ex.first_error_line)
    assert ok, msg
    # At least one later line may be consistent_with_wrong_prefix
    if ex.first_error_line != "NONE":
        idx = int(ex.first_error_line) - 1
        assert ex.verification_trace[idx]["valid"] is False


def test_many_templates_verify():
    for tid in [
        "add_chain",
        "order_ops",
        "linear_eq",
        "fraction_add",
        "mixed_ops_linear",
        "nested_order_ops",
    ]:
        for mode in ("none", "single", "multi"):
            ex = make_example(
                ex_id=f"{tid}_{mode}",
                template_id=tid,
                split="val",
                seed=3,
                mode=mode,
            )
            ok, msg = verify_example(
                ex.solution_lines, ex.verification_trace, ex.first_error_line
            )
            assert ok, f"{tid}/{mode}: {msg}"
