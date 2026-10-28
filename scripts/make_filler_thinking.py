#!/usr/bin/env python3
"""Build matched-budget filler-thinking SFT JSONL (F5 control).

Puts a repeated *existing* vocabulary token inside <think>...</think>, then
ERROR_LINE=.... No new token IDs. Used to separate "any think tokens help"
from Approach A/B learned codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.metrics import build_prompt
from reasonzipper.schemas import format_gold_answer


def resolve_filler_token(model_name: str, filler_text: str) -> tuple[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ids = tok.encode(filler_text, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(
            f"filler_text={filler_text!r} tokenizes to {len(ids)} ids {ids}; need exactly 1"
        )
    # Prefer the decoded single-piece form for the completion string
    piece = tok.decode([ids[0]], skip_special_tokens=False)
    return piece, int(ids[0])


def build_filler_completion(n_filler: int, filler_piece: str, gold: int | str) -> str:
    body = "".join([filler_piece] * int(n_filler))
    return f"<think>\n{body}\n</think>\n{format_gold_answer(gold)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument(
        "--filler-text",
        type=str,
        default=".",
        help="String that must tokenize to exactly one existing vocab ID",
    )
    p.add_argument(
        "--n-filler",
        type=int,
        default=16,
        help="Repeated filler count inside <think> (matched budget)",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    piece, tid = resolve_filler_token(args.model, args.filler_text)
    print(f"[filler] piece={piece!r} id={tid} n={args.n_filler}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = build_prompt(row["problem"], row["solution_lines"], thinking=True)
            completion = build_filler_completion(
                args.n_filler, piece, row["first_error_line"]
            )
            out = {
                "id": row["id"],
                "prompt": prompt,
                "completion": completion,
                "gold_answer": row["gold_answer"],
                "first_error_line": row["first_error_line"],
                "filler_token_id": tid,
                "n_filler": args.n_filler,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    meta = {
        "n": n,
        "filler_text": args.filler_text,
        "filler_piece": piece,
        "filler_token_id": tid,
        "n_filler": args.n_filler,
        "model": args.model,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {n} filler rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
