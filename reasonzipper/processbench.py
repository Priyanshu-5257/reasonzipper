"""ProcessBench → first-error JSONL adapter."""

from __future__ import annotations

from typing import Any, Literal


def processbench_label_to_gold(label: int | None) -> int | Literal["NONE"]:
    """
    ProcessBench `label` is the 0-based index of the first incorrect step,
    or -1 when the whole solution is correct.
    We use 1-based ERROR_LINE / NONE.
    """
    if label is None or int(label) < 0:
        return "NONE"
    return int(label) + 1


def steps_to_solution_lines(steps: list[str]) -> list[str]:
    """Number steps 1..N without changing step text semantics."""
    lines: list[str] = []
    for i, step in enumerate(steps, start=1):
        text = str(step).strip()
        # Avoid double-numbering if upstream already prefixes "1."
        lines.append(f"{i}. {text}")
    return lines


def convert_processbench_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    gold = processbench_label_to_gold(row.get("label"))
    steps = list(row.get("steps") or [])
    return {
        "id": f"processbench_{split}_{row.get('id', 'unk')}",
        "template_id": f"processbench_{split}",
        "split": "processbench",
        "source": "Qwen/ProcessBench",
        "processbench_split": split,
        "generator": row.get("generator"),
        "problem": row.get("problem", ""),
        "solution_lines": steps_to_solution_lines(steps),
        "first_error_line": gold,
        "final_answer_correct": bool(row.get("final_answer_correct")),
        "meta": {
            "raw_label_0based": row.get("label"),
            "n_steps": len(steps),
        },
    }
