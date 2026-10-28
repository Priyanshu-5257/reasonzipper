"""Discrete reasoning-token vocabulary and block→code labeling."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TOKEN_RE = re.compile(r"<\|r(\d{3})\|>")


def token_name(idx: int) -> str:
    return f"<|r{idx:03d}|>"


def make_token_list(n: int = 128) -> list[str]:
    return [token_name(i) for i in range(n)]


_OPS = [
    "add",
    "sub",
    "mul",
    "div",
    "frac",
    "simplify",
    "percent",
    "eq",
    "answer",
    "?",
]


def _op_id(op: str) -> int:
    op = (op or "?").lower()
    return _OPS.index(op) if op in _OPS else len(_OPS) - 1


def _bucket_num(s: Any, n_buckets: int = 32) -> int:
    """Stable hash bucket for numeric/string values (not readable to humans)."""
    h = hashlib.sha1(str(s).encode()).hexdigest()
    return int(h[:8], 16) % n_buckets


@dataclass
class Codebook:
    """Maps block feature signatures → 1–2 discrete code IDs in [0, n_tokens)."""

    n_tokens: int = 128
    codes_per_block: int = 2
    max_codes_per_solution: int = 16
    # signature -> list of code ids
    table: dict[str, list[int]] | None = None
    _next: int = 0

    def __post_init__(self) -> None:
        if self.table is None:
            self.table = {}

    def _alloc(self, k: int) -> list[int]:
        assert self.table is not None
        out = []
        for _ in range(k):
            out.append(self._next % self.n_tokens)
            self._next += 1
        return out

    def codes_for_block(self, step: dict[str, Any]) -> list[int]:
        """Causal block features only (no future / answer leakage beyond this step)."""
        assert self.table is not None
        valid = 1 if step.get("valid", False) else 0
        sig = "|".join(
            [
                str(_op_id(str(step.get("op", "?")))),
                str(valid),
                str(int(step.get("line", 0)) % 8),
                str(_bucket_num(step.get("expected"))),
                str(_bucket_num(step.get("stated"))),
                str(1 if step.get("consistent_with_wrong_prefix") else 0),
            ]
        )
        if sig not in self.table:
            self.table[sig] = self._alloc(self.codes_per_block)
        return list(self.table[sig])

    def encode_trace(self, verification_trace: list[dict[str, Any]]) -> list[int]:
        """Encode prefix up to and including first error (or all if NONE)."""
        codes: list[int] = []
        for step in verification_trace:
            codes.extend(self.codes_for_block(step))
            if not step.get("valid", False):
                break
            if len(codes) >= self.max_codes_per_solution:
                break
        if not codes:
            codes = [0]
        return codes[: self.max_codes_per_solution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "codes_per_block": self.codes_per_block,
            "max_codes_per_solution": self.max_codes_per_solution,
            "table": self.table,
            "_next": self._next,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Codebook":
        cb = cls(
            n_tokens=int(d["n_tokens"]),
            codes_per_block=int(d["codes_per_block"]),
            max_codes_per_solution=int(d["max_codes_per_solution"]),
            table={k: list(v) for k, v in (d.get("table") or {}).items()},
            _next=int(d.get("_next", 0)),
        )
        return cb

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Codebook":
        return cls.from_dict(json.loads(path.read_text()))


def codes_to_text(code_ids: list[int]) -> str:
    return "".join(token_name(i) for i in code_ids)


def build_code_completion(
    verification_trace: list[dict[str, Any]],
    first_error_line: int | Literal["NONE"],
    codebook: Codebook,
) -> tuple[str, list[int]]:
    ids = codebook.encode_trace(verification_trace)
    body = codes_to_text(ids)
    if first_error_line == "NONE":
        answer = "ERROR_LINE=NONE"
    else:
        answer = f"ERROR_LINE={int(first_error_line)}"
    # Unreadable on purpose: no spaces / English inside think.
    text = f"<think>{body}</think>\n{answer}"
    return text, ids


def build_filler_completion(
    n_codes: int,
    first_error_line: int | Literal["NONE"],
    filler_id: int = 0,
) -> str:
    """Matched-budget control: repeat one generic code token."""
    body = token_name(filler_id) * max(1, n_codes)
    if first_error_line == "NONE":
        answer = "ERROR_LINE=NONE"
    else:
        answer = f"ERROR_LINE={int(first_error_line)}"
    return f"<think>{body}</think>\n{answer}"


def count_code_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))
