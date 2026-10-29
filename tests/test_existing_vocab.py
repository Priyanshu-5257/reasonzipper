from __future__ import annotations

import pytest


class FakeTok:
    def __init__(self, n: int = 10):
        self._n = n

    def __len__(self):
        return self._n


class FakeModel:
    def __init__(self, n: int, cfg_n: int | None = None):
        self._n = n
        self.config = type("C", (), {"vocab_size": cfg_n if cfg_n is not None else n})()

    def get_input_embeddings(self):
        return type("E", (), {"weight": type("W", (), {"shape": (self._n, 8)})()})()


def test_assert_allows_stock_qwen_padding():
    from reasonzipper.existing_vocab import assert_original_vocabulary

    # tokenizer smaller than padded embeddings (stock Qwen layout)
    meta = assert_original_vocabulary(FakeModel(151936), FakeTok(151669))
    assert meta["tokenizer"] == 151669
    assert meta["embeddings"] == 151936


def test_assert_rejects_tokenizer_growth_beyond_embeddings():
    from reasonzipper.existing_vocab import assert_original_vocabulary

    with pytest.raises(AssertionError, match="new token"):
        assert_original_vocabulary(FakeModel(10), FakeTok(12))


def test_assert_baseline_unchanged():
    from reasonzipper.existing_vocab import assert_original_vocabulary

    base = assert_original_vocabulary(FakeModel(20), FakeTok(16))
    assert_original_vocabulary(FakeModel(20), FakeTok(16), baseline=base)
    with pytest.raises(AssertionError, match="tokenizer changed"):
        assert_original_vocabulary(FakeModel(20), FakeTok(17), baseline=base)


def test_refuse_embedding_resize_preserves_padding():
    from reasonzipper.existing_vocab import refuse_embedding_resize

    meta = refuse_embedding_resize(FakeModel(151936), FakeTok(151669), label="eval")
    assert meta["embeddings"] == 151936


def test_refuse_embedding_resize_rejects_expansion():
    from reasonzipper.existing_vocab import refuse_embedding_resize

    with pytest.raises(AssertionError, match="new token"):
        refuse_embedding_resize(FakeModel(10), FakeTok(12))
