#!/usr/bin/env python3
"""Validate synthetic dataset labels and schema."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.verify import verify_example


REQUIRED = {
    "id",
    "template_id",
    "split",
    "problem",
    "solution_lines",
    "first_error_line",
    "error_type",
    "verification_trace",
    "gold_answer",
    "source",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_split(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    rows = load_jsonl(path)
    for i, row in enumerate(rows):
        missing = REQUIRED - set(row)
        if missing:
            errors.append(f"{path}:{i} missing {missing}")
            continue
        n_lines = len(row["solution_lines"])
        if not 3 <= n_lines <= 5:
            errors.append(f"{path}:{i} n_lines={n_lines} not in 3..5")
        ok, msg = verify_example(
            row["solution_lines"],
            row["verification_trace"],
            row["first_error_line"],
        )
        if not ok:
            errors.append(f"{path}:{i} id={row['id']}: {msg}")
        expected_gold = (
            "ERROR_LINE=NONE"
            if row["first_error_line"] == "NONE"
            else f"ERROR_LINE={row['first_error_line']}"
        )
        if row["gold_answer"] != expected_gold:
            errors.append(f"{path}:{i} gold_answer mismatch")
    return len(rows), errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--max-errors", type=int, default=20)
    args = p.parse_args()

    splits = ["train", "val", "test_id", "test_ood"]
    all_errors: list[str] = []
    counts: Counter = Counter()
    templates: dict[str, set[str]] = {}

    for split in splits:
        path = args.data_dir / split / "examples.jsonl"
        if not path.exists():
            all_errors.append(f"missing {path}")
            continue
        n, errs = validate_split(path)
        counts[split] = n
        all_errors.extend(errs)
        for row in load_jsonl(path):
            templates.setdefault(split, set()).add(row["template_id"])

    # OOD templates must not appear in ID splits
    id_t = set().union(*(templates.get(s, set()) for s in ("train", "val", "test_id")))
    ood_t = templates.get("test_ood", set())
    overlap = id_t & ood_t
    if overlap:
        all_errors.append(f"ID/OOD template overlap: {sorted(overlap)}")

    print("counts:", dict(counts))
    print("id_templates:", sorted(id_t))
    print("ood_templates:", sorted(ood_t))
    if all_errors:
        print(f"FAILED with {len(all_errors)} issues (showing up to {args.max_errors}):")
        for e in all_errors[: args.max_errors]:
            print(" -", e)
        sys.exit(1)
    print("OK: all splits validated")


if __name__ == "__main__":
    main()
