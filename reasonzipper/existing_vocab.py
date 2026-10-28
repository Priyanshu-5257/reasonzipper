"""Helpers for unchanged-vocabulary (existing-token) experiments."""

from __future__ import annotations

from typing import Any, Sequence


def build_candidate_token_ids(
    tokenizer: Any,
    *,
    k: int = 512,
    seed: int = 42,
) -> list[int]:
    """Pick ``k`` ordinary single-piece token IDs (exclude specials / controls)."""
    import random

    special = set()
    for attr in (
        "all_special_ids",
        "additional_special_tokens_ids",
    ):
        ids = getattr(tokenizer, attr, None)
        if ids:
            special.update(int(x) for x in ids)
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            special.add(int(v))

    # Keep chat/think markers out of the codebook if present as single tokens.
    for piece in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(piece, add_special_tokens=False)
        if len(ids) == 1:
            special.add(int(ids[0]))

    vocab_n = int(len(tokenizer))
    pool: list[int] = []
    for tid in range(vocab_n):
        if tid in special:
            continue
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        # Prefer round-trippable single IDs without whitespace-only noise
        re_ids = tokenizer.encode(piece, add_special_tokens=False)
        if re_ids == [tid] and piece.strip() != "":
            pool.append(tid)
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} candidate tokens available; need {k}")
    rng = random.Random(seed)
    return sorted(rng.sample(pool, k))


def ids_to_think_body(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode selected existing IDs into a <think> body (concatenated pieces)."""
    parts = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids]
    return "".join(parts)


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
