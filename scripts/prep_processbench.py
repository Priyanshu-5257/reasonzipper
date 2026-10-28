#!/usr/bin/env python3
"""Download ProcessBench and write our sealed-eval JSONL format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.processbench import convert_processbench_row

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="ProcessBench splits to include",
    )
    p.add_argument("--limit-per-split", type=int, default=None)
    args = p.parse_args()

    from datasets import load_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "examples.jsonl"
    n_total = 0
    with all_path.open("w", encoding="utf-8") as out:
        for split in args.splits:
            ds = load_dataset("Qwen/ProcessBench", split=split)
            split_path = args.output_dir / f"{split}.jsonl"
            n = 0
            with split_path.open("w", encoding="utf-8") as sf:
                for row in ds:
                    if args.limit_per_split is not None and n >= args.limit_per_split:
                        break
                    ex = convert_processbench_row(dict(row), split=split)
                    line = json.dumps(ex, ensure_ascii=False)
                    sf.write(line + "\n")
                    out.write(line + "\n")
                    n += 1
            print(f"[processbench] {split}: {n} -> {split_path}", flush=True)
            n_total += n
    meta = {"n": n_total, "splits": list(args.splits), "path": str(all_path)}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[processbench] wrote {n_total} rows -> {all_path}", flush=True)


if __name__ == "__main__":
    main()
