#!/usr/bin/env python3
"""Build discrete-code SFT JSONL (+ optional filler control labels)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.codes import Codebook, build_code_completion, build_filler_completion
from reasonzipper.metrics import build_prompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/codes.yaml")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--codebook-out", type=Path, required=True)
    p.add_argument("--filler-output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--codebook-in", type=Path, default=None, help="Reuse a frozen codebook")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    cc = cfg["codes"]
    if args.codebook_in and args.codebook_in.exists():
        codebook = Codebook.load(args.codebook_in)
        print(f"[codes] loaded codebook {args.codebook_in}", flush=True)
    else:
        codebook = Codebook(
            n_tokens=int(cc["n_tokens"]),
            codes_per_block=int(cc["codes_per_block"]),
            max_codes_per_solution=int(cc["max_codes_per_solution"]),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.filler_output:
        args.filler_output.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    lens = []
    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        filler_f = args.filler_output.open("w", encoding="utf-8") if args.filler_output else None
        try:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = build_prompt(row["problem"], row["solution_lines"], thinking=True)
                completion, ids = build_code_completion(
                    row["verification_trace"], row["first_error_line"], codebook
                )
                lens.append(len(ids))
                fout.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "prompt": prompt,
                            "completion": completion,
                            "code_ids": ids,
                            "first_error_line": row["first_error_line"],
                            "gold_answer": row["gold_answer"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if filler_f is not None:
                    filler_f.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "prompt": prompt,
                                "completion": build_filler_completion(
                                    len(ids), row["first_error_line"], filler_id=0
                                ),
                                "code_ids": [0] * len(ids),
                                "first_error_line": row["first_error_line"],
                                "gold_answer": row["gold_answer"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                n += 1
                if args.limit and n >= args.limit:
                    break
        finally:
            if filler_f is not None:
                filler_f.close()

    args.codebook_out.parent.mkdir(parents=True, exist_ok=True)
    codebook.save(args.codebook_out)
    mean_l = sum(lens) / max(len(lens), 1)
    print(
        f"[codes] wrote {n} rows mean_codes={mean_l:.2f} "
        f"codebook_size={len(codebook.table or {})} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
