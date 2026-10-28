#!/usr/bin/env python3
"""Aggregate sealed-eval summary JSONs into one report + gate-style deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--a0-tag", type=str, default="a0_thinking")
    p.add_argument("--c3-tag", type=str, default="c3_grpo")
    args = p.parse_args()

    summaries: dict[str, dict] = {}
    for path in sorted(args.reports_dir.glob("*.summary.json")):
        tag = path.name.replace(".summary.json", "")
        summaries[tag] = load(path)

    def pick(model: str, split: str) -> dict | None:
        key = f"{model}__{split}"
        return summaries.get(key)

    splits = ["test_id", "test_ood", "processbench"]
    models = sorted({k.split("__", 1)[0] for k in summaries if "__" in k})

    table: dict[str, dict] = {}
    for m in models:
        table[m] = {}
        for s in splits:
            row = pick(m, s)
            if row:
                table[m][s] = {
                    "n": row.get("n"),
                    "accuracy": row.get("accuracy"),
                    "mean_reasoning_tokens": row.get("mean_reasoning_tokens"),
                    "median_reasoning_tokens": row.get("median_reasoning_tokens"),
                    "malformed_rate": row.get("malformed_rate"),
                }

    a0_id = pick(args.a0_tag, "test_id")
    c3_id = pick(args.c3_tag, "test_id")
    a0_ood = pick(args.a0_tag, "test_ood")
    c3_ood = pick(args.c3_tag, "test_ood")

    def delta(a: dict | None, b: dict | None, key: str) -> float | None:
        if not a or not b or a.get(key) is None or b.get(key) is None:
            return None
        return float(b[key]) - float(a[key])

    def tok_reduction(a: dict | None, b: dict | None) -> float | None:
        if not a or not b:
            return None
        base = float(a.get("mean_reasoning_tokens") or 0)
        if base <= 0:
            return None
        return 1.0 - float(b.get("mean_reasoning_tokens") or 0) / base

    report = {
        "summaries": table,
        "comparisons": {
            "c3_vs_a0_test_id": {
                "acc_delta": delta(a0_id, c3_id, "accuracy"),
                "reasoning_token_reduction": tok_reduction(a0_id, c3_id),
            },
            "c3_vs_a0_test_ood": {
                "acc_delta": delta(a0_ood, c3_ood, "accuracy"),
                "reasoning_token_reduction": tok_reduction(a0_ood, c3_ood),
            },
        },
        "poc_checks": {},
    }
    id_cmp = report["comparisons"]["c3_vs_a0_test_id"]
    ood_cmp = report["comparisons"]["c3_vs_a0_test_ood"]
    report["poc_checks"] = {
        "id_acc_drop_le_2pp": (
            id_cmp["acc_delta"] is not None and id_cmp["acc_delta"] >= -0.02
        ),
        "id_tok_reduction_ge_30pct": (
            id_cmp["reasoning_token_reduction"] is not None
            and id_cmp["reasoning_token_reduction"] >= 0.30
        ),
        "ood_acc_drop_le_3pp": (
            ood_cmp["acc_delta"] is not None and ood_cmp["acc_delta"] >= -0.03
        ),
        "sealed_poc_pass": False,
    }
    c = report["poc_checks"]
    c["sealed_poc_pass"] = bool(
        c["id_acc_drop_le_2pp"]
        and c["id_tok_reduction_ge_30pct"]
        and c["ood_acc_drop_le_3pp"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
