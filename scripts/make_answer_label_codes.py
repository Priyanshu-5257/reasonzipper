#!/usr/bin/env python3
"""Answer-label encoding control for existing-vocab compression experiments.

Produces short ``<think>`` bodies whose codes are a deterministic function of
the gold ``ERROR_LINE`` label. If learned B2 codes do not beat this control
on harder / multi-step tasks, they have not shown preserved intermediate
computation beyond answer memorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.existing_vocab import (
    answer_label_code_ids,
    build_candidate_token_ids,
    build_think_completion_from_ids,
)
from reasonzipper.metrics import build_prompt
from reasonzipper.schemas import format_gold_answer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--candidate-k", type=int, default=64)
    p.add_argument("--n-codes", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    candidates = build_candidate_token_ids(tok, k=args.candidate_k, seed=args.seed)
    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for row in rows:
            gold_line = row["first_error_line"]
            ids = answer_label_code_ids(
                candidates, gold_line, n_codes=args.n_codes
            )
            gold = format_gold_answer(gold_line)
            try:
                completion, ids = build_think_completion_from_ids(tok, ids, gold)
            except AssertionError:
                # Rare: try next offset until round-trip works
                ok = False
                for off in range(1, min(32, len(candidates))):
                    trial = answer_label_code_ids(
                        candidates[(off):] + candidates[:off],
                        gold_line,
                        n_codes=args.n_codes,
                    )
                    try:
                        completion, ids = build_think_completion_from_ids(
                            tok, trial, gold
                        )
                        ok = True
                        break
                    except AssertionError:
                        continue
                if not ok:
                    continue
            prompt = build_prompt(
                row["problem"], row["solution_lines"], thinking=True
            )
            out = {
                "id": row["id"],
                "prompt": prompt,
                "completion": completion,
                "gold_answer": row.get("gold_answer", gold),
                "first_error_line": gold_line,
                "code_token_ids": ids,
                "n_codes": len(ids),
                "selection": "answer_label_control",
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    meta = {
        "n": n_out,
        "candidate_k": args.candidate_k,
        "n_codes": args.n_codes,
        "method": "answer_label_control",
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[label-codes] wrote {n_out} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
