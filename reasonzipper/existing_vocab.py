"""Helpers for unchanged-vocabulary (existing-token) experiments."""

from __future__ import annotations

from typing import Any


def assert_original_vocabulary(
    model: Any,
    tokenizer: Any,
    *,
    label: str = "",
    baseline: dict[str, int] | None = None,
) -> dict[str, int]:
    """Ensure we did not expand the tokenizer / add embedding rows.

    Stock Qwen3 checkpoints pad the embedding matrix (e.g. 151936) above
    ``len(tokenizer)`` (e.g. 151669). That padding is *original*, not a
    ReasonZipper vocabulary expansion. We only fail when:
    - tokenizer grew beyond the embedding matrix (new token IDs), or
    - sizes diverge from a recorded baseline after training/reload.
    """
    tok_n = int(len(tokenizer))
    emb_n = int(model.get_input_embeddings().weight.shape[0])
    cfg_n = int(getattr(model.config, "vocab_size", emb_n))
    prefix = f"[{label}] " if label else ""
    meta = {"tokenizer": tok_n, "embeddings": emb_n, "config_vocab_size": cfg_n}

    if tok_n > emb_n:
        raise AssertionError(
            f"{prefix}tokenizer len {tok_n} > embedding rows {emb_n} "
            "(new token IDs are not allowed for existing-vocab runs)"
        )
    if baseline is None:
        return meta

    for key in ("tokenizer", "embeddings"):
        if meta[key] != int(baseline[key]):
            raise AssertionError(
                f"{prefix}{key} changed {baseline[key]} -> {meta[key]} "
                "(must stay identical to the base Qwen3 checkpoint)"
            )
    return meta
