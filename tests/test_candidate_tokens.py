from __future__ import annotations


class FakeTok:
    def __init__(self, n: int = 50):
        self._n = n
        self.all_special_ids = [0, 1]
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None
        self.unk_token_id = None

    def __len__(self):
        return self._n

    def encode(self, text, add_special_tokens=False):
        if text in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            return [1]
        if text.startswith("T") and text[1:].isdigit():
            return [int(text[1:])]
        return [2, 3]

    def decode(self, ids, skip_special_tokens=False):
        tid = int(ids[0])
        if tid < 2:
            return "<special>"
        return f"T{tid}"


def test_build_candidate_token_ids():
    from reasonzipper.existing_vocab import build_candidate_token_ids

    ids = build_candidate_token_ids(FakeTok(50), k=10, seed=0)
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert all(i >= 2 for i in ids)
