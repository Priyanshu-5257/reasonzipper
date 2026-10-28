"""Helpers for unchanged-vocabulary (existing-token) experiments."""

from __future__ import annotations

from typing import Any


def assert_original_vocabulary(model: Any, tokenizer: Any, *, label: str = "") -> dict[str, int]:
    """Fail if tokenizer or embedding rows diverge from the loaded checkpoint."""
    tok_n = int(len(tokenizer))
    emb_n = int(model.get_input_embeddings().weight.shape[0])
    cfg_n = int(getattr(model.config, "vocab_size", emb_n))
    prefix = f"[{label}] " if label else ""
    if tok_n != emb_n:
        raise AssertionError(
            f"{prefix}tokenizer len {tok_n} != embedding rows {emb_n} "
            "(vocabulary must stay identical to base Qwen3)"
        )
    if cfg_n != emb_n:
        raise AssertionError(
            f"{prefix}config.vocab_size {cfg_n} != embedding rows {emb_n}"
        )
    return {"tokenizer": tok_n, "embeddings": emb_n, "config_vocab_size": cfg_n}
