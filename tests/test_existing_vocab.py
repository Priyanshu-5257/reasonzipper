from __future__ import annotations


class FakeTok:
    def __len__(self):
        return 10


class FakeModel:
    def __init__(self, n: int, cfg_n: int | None = None):
        self._n = n
        self.config = type("C", (), {"vocab_size": cfg_n if cfg_n is not None else n})()

    def get_input_embeddings(self):
        return type("E", (), {"weight": type("W", (), {"shape": (self._n, 8)})()})()


def test_assert_original_vocabulary_ok():
    from reasonzipper.existing_vocab import assert_original_vocabulary

    meta = assert_original_vocabulary(FakeModel(10), FakeTok())
    assert meta["tokenizer"] == 10


def test_assert_original_vocabulary_mismatch():
    from reasonzipper.existing_vocab import assert_original_vocabulary
    import pytest

    with pytest.raises(AssertionError):
        assert_original_vocabulary(FakeModel(11), FakeTok())
