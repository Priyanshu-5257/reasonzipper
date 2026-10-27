from reasonzipper.reward import correctness_first_rewards


def test_length_only_when_all_correct():
    r = correctness_first_rewards([True, True], [100, 400], alpha=0.2, budget_b=512)
    assert r[0] > r[1]
    assert r[0] == 1.0 - 0.2 * min(100 / 512, 1.0)


def test_incorrect_not_rewarded_for_shortness_alone():
    r = correctness_first_rewards([False, False], [10, 400], alpha=0.2, budget_b=512)
    assert r[0] == 0.0
    assert r[1] == 0.0


def test_mixed_group_no_length_term():
    r = correctness_first_rewards([True, False], [10, 10], alpha=0.2, budget_b=512)
    assert r[0] == 1.0
    assert r[1] == 0.0
