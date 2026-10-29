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


def assert_ids_roundtrip(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    label: str = "",
) -> str:
    """Require concat-decode → encode recovers the exact ID sequence."""
    ids = [int(t) for t in token_ids]
    body = ids_to_think_body(tokenizer, ids)
    recovered = tokenizer.encode(body, add_special_tokens=False)
    if recovered != ids:
        prefix = f"[{label}] " if label else ""
        raise AssertionError(
            f"{prefix}code ID round-trip failed: {ids} -> {body!r} -> {recovered}"
        )
    return body


def build_think_completion_from_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    gold_answer: str,
    *,
    verify_roundtrip: bool = True,
    verify_formatted: bool = True,
    prompt: str | None = None,
    verify_chat_template: bool = False,
    enable_thinking: bool = True,
) -> tuple[str, list[int]]:
    """Build ``<think>\\n{codes}\\n</think>\\n{answer}`` preferring ID fidelity.

    Returns (completion_text, code_ids). When ``verify_formatted`` is set,
    also checks that the code IDs appear as a contiguous span inside the
    tokenized completion. When ``verify_chat_template`` and ``prompt`` are set,
    also checks the fully chat-templated user+assistant string used by SFT.
    """
    ids = [int(t) for t in token_ids]
    if verify_roundtrip:
        body = assert_ids_roundtrip(tokenizer, ids, label="think-body")
    else:
        body = ids_to_think_body(tokenizer, ids)
    completion = f"<think>\n{body}\n</think>\n{gold_answer}"
    if verify_formatted:
        assert_code_span_in_text(tokenizer, completion, ids, label="completion")
    if verify_chat_template:
        if not prompt:
            raise ValueError("prompt required when verify_chat_template=True")
        assert_codes_survive_chat_template(
            tokenizer,
            prompt=prompt,
            completion=completion,
            token_ids=ids,
            enable_thinking=enable_thinking,
            label="chat-template",
        )
    return completion, ids


def assert_codes_survive_chat_template(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    token_ids: Sequence[int],
    enable_thinking: bool = True,
    label: str = "",
) -> list[int]:
    """Assert selected code IDs remain a contiguous span after chat templating.

    Mirrors ``train_sft.py`` text construction: apply_chat_template on
    user prompt + assistant completion, then encode without specials.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=bool(enable_thinking),
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(t) for t in token_ids]
    n = len(ids)
    if n and not any(enc[i : i + n] == ids for i in range(len(enc) - n + 1)):
        prefix = f"[{label}] " if label else ""
        raise AssertionError(
            f"{prefix}code IDs {ids} missing after chat template "
            f"(encoded_len={len(enc)})"
        )
    return enc


def assert_code_span_in_text(
    tokenizer: Any,
    text: str,
    token_ids: Sequence[int],
    *,
    label: str = "",
) -> None:
    """Assert ``token_ids`` occur as a contiguous subsequence of ``encode(text)``."""
    ids = [int(t) for t in token_ids]
    enc = tokenizer.encode(text, add_special_tokens=False)
    n = len(ids)
    if n == 0:
        return
    found = any(enc[i : i + n] == ids for i in range(len(enc) - n + 1))
    if not found:
        prefix = f"[{label}] " if label else ""
        raise AssertionError(
            f"{prefix}code IDs {ids} not found as contiguous span in encoded text "
            f"(len={len(enc)})"
        )


def answer_label_code_ids(
    candidates: Sequence[int],
    first_error_line: int | str,
    *,
    n_codes: int = 1,
) -> list[int]:
    """Deterministic short codes that explicitly encode the answer label.

    Control for distinguishing answer-label shortcuts from intermediate
    computation: line ``k`` / ``NONE`` map to fixed offsets in ``candidates``.
    """
    if not candidates:
        raise ValueError("empty candidate set")
    if isinstance(first_error_line, str) and first_error_line.upper() == "NONE":
        key = 0
    else:
        key = int(first_error_line)
    # Reserve index 0 for NONE; positive lines use key (mod pool).
    base = key % len(candidates)
    out = [int(candidates[(base + i) % len(candidates)]) for i in range(max(1, n_codes))]
    return out


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
                "(must stay identical to the base Qwen checkpoint)"
            )
    return meta


def refuse_embedding_resize(
    model: Any,
    tokenizer: Any,
    *,
    label: str = "",
) -> dict[str, int]:
    """Existing-vocab policy: never resize; fail on unexpected expansion.

    Qwen pad rows (``emb > tok``) must be preserved as-is.
    """
    meta = assert_original_vocabulary(model, tokenizer, label=label)
    prefix = f"[{label}] " if label else ""
    tok_n = meta["tokenizer"]
    emb_n = meta["embeddings"]
    if tok_n > emb_n:
        raise ValueError(
            f"{prefix}Unexpected vocabulary expansion: tokenizer={tok_n} > embeddings={emb_n}"
        )
    return meta
