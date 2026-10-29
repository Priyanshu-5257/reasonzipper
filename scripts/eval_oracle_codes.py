#!/usr/bin/env python3
"""Oracle-code diagnostic: teacher-force selected codes, then generate the answer.

This is an upper-bound / privileged test (codes may be answer-conditioned).
It is NOT deployable student accuracy.

For each row in a codes JSONL (from select_compressed_codes / label control):
  chat(user) + ``<think>\\n{body}\\n</think>\\n`` → generate answer tokens → score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.existing_vocab import (
    build_think_completion_from_ids,
    ids_to_think_body,
    refuse_embedding_resize,
)
from reasonzipper.metrics import score_prediction
from reasonzipper.schemas import format_gold_answer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_user_template(tokenizer, user_text: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def forced_prefix_text(
    tokenizer,
    row: dict,
    *,
    mode: str,
) -> str:
    """Build the forced assistant prefix ending after ``</think>\\n``."""
    gold = format_gold_answer(row["first_error_line"])
    if mode == "empty_think":
        return "<think>\n</think>\n"
    if mode == "from_completion":
        # Use completion up through </think>\n (drop gold answer if present)
        comp = row.get("completion") or ""
        if "</think>" in comp:
            return comp.split("</think>", 1)[0] + "</think>\n"
        return f"<think>\n</think>\n"
    # mode == codes: rebuild from code_token_ids
    ids = [int(x) for x in row.get("code_token_ids") or []]
    if not ids:
        return "<think>\n</think>\n"
    try:
        completion, _ = build_think_completion_from_ids(
            tokenizer,
            ids,
            gold,
            verify_roundtrip=True,
            verify_formatted=True,
            prompt=row.get("prompt"),
            verify_chat_template=False,
        )
    except AssertionError:
        body = ids_to_think_body(tokenizer, ids)
        completion = f"<think>\n{body}\n</think>\n{gold}"
    return completion.split("</think>", 1)[0] + "</think>\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--codes", type=Path, required=True, help="JSONL with prompt + codes")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--force-mode",
        choices=("codes", "empty_think", "from_completion"),
        default="codes",
        help="What to teacher-force before generating the answer",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--require-original-vocab",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--label",
        type=str,
        default="",
        help="Tag written into summary (e.g. func_only / random / empty)",
    )
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    rows = load_jsonl(args.codes)
    if args.limit is not None:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_kwargs: dict = {"trust_remote_code": True, "device_map": args.device}
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.require_original_vocab:
        meta = refuse_embedding_resize(model, tok, label="oracle-eval")
        print(f"[oracle] original vocab ok {meta}", flush=True)
    model.eval()

    label = args.label or args.force_mode
    preds = []
    n_correct = 0
    n_malformed = 0
    completion_lens: list[int] = []
    t0 = time.perf_counter()
    print(
        f"[oracle] n={len(rows)} mode={args.force_mode} label={label} "
        f"max_new={args.max_new_tokens}",
        flush=True,
    )

    for i, row in enumerate(rows, start=1):
        user = row["prompt"]
        chat = apply_user_template(tok, user, enable_thinking=True)
        forced = forced_prefix_text(tok, row, mode=args.force_mode)
        full_prefix = chat + forced
        inputs = tok(full_prefix, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        new = out[0][inputs["input_ids"].shape[1] :]
        text = tok.decode(new, skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        # Score as if the full assistant message were forced+generated
        scored = score_prediction(forced + text, row["first_error_line"])
        if scored["correct"]:
            n_correct += 1
        if scored["malformed"]:
            n_malformed += 1
        completion_lens.append(int(new.shape[0]))
        preds.append(
            {
                "id": row.get("id"),
                "gold": row["first_error_line"],
                "forced_prefix": forced,
                "generated": text,
                "code_token_ids": row.get("code_token_ids"),
                "completion_tokens": int(new.shape[0]),
                "force_mode": args.force_mode,
                "label": label,
                **scored,
            }
        )
        if i % 10 == 0 or i == len(rows):
            print(
                f"[oracle] {i}/{len(rows)} acc={n_correct/i:.3f} "
                f"malformed={n_malformed/i:.3f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    n = max(len(rows), 1)
    summary = {
        "label": label,
        "force_mode": args.force_mode,
        "model": args.model,
        "n": len(rows),
        "accuracy": n_correct / n,
        "malformed_rate": n_malformed / n,
        "mean_completion_tokens": (
            sum(completion_lens) / len(completion_lens) if completion_lens else 0.0
        ),
        "wall_clock_s": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "oracle": True,
        "note": (
            "Privileged upper bound if codes were selected with gold answers; "
            "not self-generated student accuracy."
        ),
        # Reference baselines from prior existing-vocab runs (val n=100)
        "reference_f4_accuracy": 0.70,
        "reference_f5_accuracy": 0.70,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
