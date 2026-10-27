from __future__ import annotations

from reasonzipper.generate import BUILDERS, ID_TEMPLATES, OOD_TEMPLATES, generate_dataset, make_example
from reasonzipper.verify import verify_example


def test_all_builders_produce_3_to_5_steps():
    import random

    rng = random.Random(0)
    for name, builder in BUILDERS.items():
        problem, steps = builder(rng)
        assert problem
        assert 3 <= len(steps) <= 5, name


def test_make_example_none_and_error():
    for mode in ("none", "single", "multi"):
        ex = make_example(
            ex_id=f"t_{mode}",
            template_id="add_chain",
            split="train",
            seed=1,
            mode=mode,
        )
        ok, msg = verify_example(ex.solution_lines, ex.verification_trace, ex.first_error_line)
        assert ok, msg
        if mode == "none":
            assert ex.first_error_line == "NONE"
            assert ex.gold_answer == "ERROR_LINE=NONE"
        else:
            assert ex.first_error_line != "NONE"
            assert ex.gold_answer == f"ERROR_LINE={ex.first_error_line}"


def test_ood_templates_disjoint():
    assert not set(ID_TEMPLATES) & set(OOD_TEMPLATES)


def test_small_dataset_splits():
    ds = generate_dataset(
        seed=7, n_train=20, n_val=10, n_test_id=10, n_test_ood=12
    )
    assert len(ds["train"]) == 20
    assert len(ds["test_ood"]) == 12
    id_t = {e.template_id for s in ("train", "val", "test_id") for e in ds[s]}
    ood_t = {e.template_id for e in ds["test_ood"]}
    assert not id_t & ood_t


def test_full_default_counts_no_hang():
    # Regression: wrong_sign on 0 used to infinite-loop during OOD generation.
    ds = generate_dataset(
        seed=42, n_train=8000, n_val=1000, n_test_id=1000, n_test_ood=2000
    )
    assert sum(len(v) for v in ds.values()) == 12000


def test_zero_value_wrong_sign_case():
    ex = make_example(
        ex_id="test_ood_fraction_mul_sub_00173",
        template_id="fraction_mul_sub",
        split="test_ood",
        seed=42,
        mode="single",
    )
    assert ex.first_error_line != "NONE" or ex.gold_answer == "ERROR_LINE=NONE"
    assert 3 <= len(ex.solution_lines) <= 5
