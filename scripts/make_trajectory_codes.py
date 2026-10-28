#!/usr/bin/env python3
"""Approach B targets: verifier-block codes using *existing* vocab IDs only.

Maps each verification-trace block through the synthetic codebook signature to an
index in a frozen candidate set S of original Qwen token IDs (no new tokens).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.codes import Codebook
from reasonzipper.existing_vocab import build_candidate_token_ids, ids_to_think_body
from reasonzipper.metrics import build_prompt
from reasonzipper.schemas import format_gold_answer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--candidate-k", type=int, default=512)
    p.add_argument("--codes-per-block", type=int, default=2)
    p.add_argument("--max-codes", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    candidates = build_candidate_token_ids(tok, k=args.candidate_k, seed=args.seed)
    codebook = Codebook(
        n_tokens=len(candidates),
        codes_per_block=args.codes_per_block,
        max_codes_per_solution=args.max_codes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            local_ids = codebook.encode_trace(row.get("verification_trace") or [])
            # Map codebook indices → existing token IDs
            existing_ids = [candidates[i % len(candidates)] for i in local_ids]
            body = ids_to_think_body(tok, existing_ids)
            completion = f"<think>\n{body}\n</think>\n{format_gold_answer(row['first_error_line'])}"
            prompt = build_prompt(row["problem"], row["solution_lines"], thinking=True)
            out = {
                "id": row["id"],
                "prompt": prompt,
                "completion": completion,
                "gold_answer": row["gold_answer"],
                "first_error_line": row["first_error_line"],
                "code_token_ids": existing_ids,
                "n_codes": len(existing_ids),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    meta = {
        "n": n,
        "candidate_k": args.candidate_k,
        "codes_per_block": args.codes_per_block,
        "max_codes": args.max_codes,
        "seed": args.seed,
        "model": args.model,
        "candidate_token_ids_head": candidates[:32],
        "note": (
            "B-v1: verifier-block signatures → existing token IDs via Codebook index. "
            "No new vocab. Hidden-state P_phi selector is a follow-up."
        ),
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    (args.output.parent / "b_candidate_token_ids.json").write_text(
        json.dumps({"candidate_token_ids": candidates}, indent=2)
    )
    print(f"Wrote {n} trajectory-code rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
