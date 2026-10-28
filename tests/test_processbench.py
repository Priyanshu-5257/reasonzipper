from reasonzipper.processbench import (
    convert_processbench_row,
    processbench_label_to_gold,
    steps_to_solution_lines,
)


def test_label_mapping():
    assert processbench_label_to_gold(-1) == "NONE"
    assert processbench_label_to_gold(0) == 1
    assert processbench_label_to_gold(3) == 4


def test_steps_numbering():
    lines = steps_to_solution_lines(["alpha", "beta"])
    assert lines == ["1. alpha", "2. beta"]


def test_convert_row():
    row = {
        "id": "x",
        "generator": "t",
        "problem": "P",
        "steps": ["a", "b"],
        "final_answer_correct": False,
        "label": 0,
    }
    ex = convert_processbench_row(row, split="gsm8k")
    assert ex["first_error_line"] == 1
    assert ex["solution_lines"][0].startswith("1. ")
    assert ex["split"] == "processbench"
