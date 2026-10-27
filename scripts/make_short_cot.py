#!/usr/bin/env python3
"""Build short-CoT SFT JSONL from verified synthetic examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.metrics import build_prompt
from reasonzipper.short_cot import build_short_completion


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="examples.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

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
            completion = build_short_completion(
                row["verification_trace"], row["first_error_line"]
            )
            out = {
                "id": row["id"],
                "prompt": prompt,
                "completion": completion,
                "gold_answer": row["gold_answer"],
                "first_error_line": row["first_error_line"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"Wrote {n} short-CoT rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
