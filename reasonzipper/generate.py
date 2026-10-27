"""Deterministic synthetic generator for first-error localization."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal

from reasonzipper.schemas import Example, ErrorType, SplitName, format_gold_answer
from reasonzipper.verify import first_error_from_trace, verify_example

ID_TEMPLATES = [
    "add_chain",
    "sub_chain",
    "mul_chain",
    "div_chain",
    "order_ops",
    "linear_eq",
    "percent_of",
    "fraction_add",
    "word_two_step",
]

OOD_TEMPLATES = [
    "mixed_ops_linear",
    "nested_order_ops",
    "percent_then_eq",
    "fraction_mul_sub",
    "word_three_step",
]


@dataclass
class StepSpec:
    """One numbered student-solution line with ground-truth evaluation."""

    text: str
    expected: Fraction | int | float
    op: str
    operands: list[Any]


def _frac(x: Any) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x).limit_denominator(10000)


def _fmt(x: Any) -> str:
    f = _frac(x)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _seeded_rng(seed: int, *parts: Any) -> random.Random:
    h = hashlib.sha256(f"{seed}|{'|'.join(map(str, parts))}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------------------
# Correct solution builders (return problem text + StepSpec list)
# ---------------------------------------------------------------------------

Builder = Callable[[random.Random], tuple[str, list[StepSpec]]]


def build_add_chain(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a, b, c = rng.randint(10, 80), rng.randint(10, 80), rng.randint(10, 80)
    s1 = a + b
    s2 = s1 + c
    problem = f"Compute {a} + {b} + {c}."
    steps = [
        StepSpec(f"1. {a} + {b} = {s1}", s1, "add", [a, b]),
        StepSpec(f"2. {s1} + {c} = {s2}", s2, "add", [s1, c]),
        StepSpec(f"3. Final answer: {s2}", s2, "answer", [s2]),
    ]
    return problem, steps


def build_sub_chain(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a = rng.randint(50, 120)
    b = rng.randint(5, 40)
    c = rng.randint(5, 30)
    s1 = a - b
    s2 = s1 - c
    problem = f"Compute {a} - {b} - {c}."
    steps = [
        StepSpec(f"1. {a} - {b} = {s1}", s1, "sub", [a, b]),
        StepSpec(f"2. {s1} - {c} = {s2}", s2, "sub", [s1, c]),
        StepSpec(f"3. Final answer: {s2}", s2, "answer", [s2]),
    ]
    return problem, steps


def build_mul_chain(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 9)
    s1 = a * b
    s2 = s1 * c
    problem = f"Compute {a} × {b} × {c}."
    steps = [
        StepSpec(f"1. {a} × {b} = {s1}", s1, "mul", [a, b]),
        StepSpec(f"2. {s1} × {c} = {s2}", s2, "mul", [s1, c]),
        StepSpec(f"3. Final answer: {s2}", s2, "answer", [s2]),
    ]
    return problem, steps


def build_div_chain(rng: random.Random) -> tuple[str, list[StepSpec]]:
    c = rng.randint(2, 9)
    b = rng.randint(2, 9)
    s1 = c * b
    a = s1 * rng.randint(2, 6)
    # a / b / c with left-assoc: (a/b)/c
    q1 = a // b
    q2 = q1 // c
    problem = f"Compute {a} ÷ {b} ÷ {c} (left to right)."
    steps = [
        StepSpec(f"1. {a} ÷ {b} = {q1}", q1, "div", [a, b]),
        StepSpec(f"2. {q1} ÷ {c} = {q2}", q2, "div", [q1, c]),
        StepSpec(f"3. Final answer: {q2}", q2, "answer", [q2]),
    ]
    return problem, steps


def build_order_ops(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a, b, c = rng.randint(2, 15), rng.randint(2, 12), rng.randint(2, 9)
    mul = b * c
    total = a + mul
    problem = f"Compute {a} + {b} × {c}."
    steps = [
        StepSpec(f"1. Multiply first: {b} × {c} = {mul}", mul, "mul", [b, c]),
        StepSpec(f"2. Then add: {a} + {mul} = {total}", total, "add", [a, mul]),
        StepSpec(f"3. Final answer: {total}", total, "answer", [total]),
    ]
    return problem, steps


def build_linear_eq(rng: random.Random) -> tuple[str, list[StepSpec]]:
    x = rng.randint(2, 20)
    a = rng.randint(2, 9)
    b = rng.randint(1, 30)
    rhs = a * x + b
    problem = f"Solve for x: {a}x + {b} = {rhs}."
    steps = [
        StepSpec(f"1. Subtract {b} from both sides: {a}x = {rhs - b}", rhs - b, "sub", [rhs, b]),
        StepSpec(f"2. Divide both sides by {a}: x = {x}", x, "div", [rhs - b, a]),
        StepSpec(f"3. Final answer: x = {x}", x, "answer", [x]),
    ]
    return problem, steps


def build_percent_of(rng: random.Random) -> tuple[str, list[StepSpec]]:
    p = rng.choice([10, 20, 25, 40, 50])
    n = rng.randint(2, 20) * 10
    part = n * p // 100
    problem = f"What is {p}% of {n}?"
    steps = [
        StepSpec(f"1. Convert percent: {p}% = {p}/100", Fraction(p, 100), "frac", [p, 100]),
        StepSpec(f"2. Multiply: ({p}/100) × {n} = {part}", part, "mul", [Fraction(p, 100), n]),
        StepSpec(f"3. Final answer: {part}", part, "answer", [part]),
    ]
    return problem, steps


def build_fraction_add(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a, b = rng.randint(1, 5), rng.randint(2, 8)
    c, d = rng.randint(1, 5), rng.randint(2, 8)
    f1, f2 = Fraction(a, b), Fraction(c, d)
    s = f1 + f2
    problem = f"Compute {a}/{b} + {c}/{d}."
    left = a * d
    right = c * b
    den = b * d
    num = left + right
    steps = [
        StepSpec(
            f"1. Common denominator {den}: {a}/{b} = {left}/{den}, {c}/{d} = {right}/{den}",
            Fraction(left, den),
            "frac",
            [a, b, c, d],
        ),
        StepSpec(f"2. Add numerators: {left} + {right} = {num}, so {num}/{den}", Fraction(num, den), "add", [left, right]),
        StepSpec(f"3. Simplify: {_fmt(s)}", s, "simplify", [num, den]),
        StepSpec(f"4. Final answer: {_fmt(s)}", s, "answer", [s]),
    ]
    return problem, steps


def build_word_two_step(rng: random.Random) -> tuple[str, list[StepSpec]]:
    price = rng.randint(5, 40)
    qty = rng.randint(2, 8)
    discount = rng.randint(1, 10)
    sub = price * qty
    total = sub - discount
    problem = (
        f"A shop sells pens for ${price} each. "
        f"Maya buys {qty} pens and gets a ${discount} discount. How much does she pay?"
    )
    steps = [
        StepSpec(f"1. Cost before discount: {price} × {qty} = {sub}", sub, "mul", [price, qty]),
        StepSpec(f"2. Apply discount: {sub} - {discount} = {total}", total, "sub", [sub, discount]),
        StepSpec(f"3. Final answer: {total}", total, "answer", [total]),
    ]
    return problem, steps


def build_mixed_ops_linear(rng: random.Random) -> tuple[str, list[StepSpec]]:
    x = rng.randint(2, 12)
    a, b, c = rng.randint(2, 6), rng.randint(2, 8), rng.randint(1, 15)
    rhs = a * x + b * c
    problem = f"Solve for x: {a}x + {b}×{c} = {rhs}."
    prod = b * c
    ax = rhs - prod
    steps = [
        StepSpec(f"1. Compute {b}×{c} = {prod}", prod, "mul", [b, c]),
        StepSpec(f"2. Subtract: {a}x = {rhs} - {prod} = {ax}", ax, "sub", [rhs, prod]),
        StepSpec(f"3. Divide by {a}: x = {x}", x, "div", [ax, a]),
        StepSpec(f"4. Final answer: x = {x}", x, "answer", [x]),
    ]
    return problem, steps


def build_nested_order_ops(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a, b, c, d = rng.randint(2, 9), rng.randint(2, 8), rng.randint(2, 6), rng.randint(2, 5)
    inner = b + c
    mul = inner * d
    total = a + mul
    problem = f"Compute {a} + ({b} + {c}) × {d}."
    steps = [
        StepSpec(f"1. Parentheses: {b} + {c} = {inner}", inner, "add", [b, c]),
        StepSpec(f"2. Multiply: {inner} × {d} = {mul}", mul, "mul", [inner, d]),
        StepSpec(f"3. Add: {a} + {mul} = {total}", total, "add", [a, mul]),
        StepSpec(f"4. Final answer: {total}", total, "answer", [total]),
    ]
    return problem, steps


def build_percent_then_eq(rng: random.Random) -> tuple[str, list[StepSpec]]:
    p = rng.choice([10, 20, 25])
    n = rng.randint(4, 20) * 10
    part = n * p // 100
    x = rng.randint(2, 15)
    rhs = part + x
    problem = f"Let y = {p}% of {n}. Solve x + y = {rhs} for x."
    steps = [
        StepSpec(f"1. {p}% of {n} = {part}, so y = {part}", part, "percent", [p, n]),
        StepSpec(f"2. x + {part} = {rhs}", rhs, "eq", [x, part]),
        StepSpec(f"3. x = {rhs} - {part} = {x}", x, "sub", [rhs, part]),
        StepSpec(f"4. Final answer: x = {x}", x, "answer", [x]),
    ]
    return problem, steps


def build_fraction_mul_sub(rng: random.Random) -> tuple[str, list[StepSpec]]:
    a, b = rng.randint(1, 4), rng.randint(2, 6)
    c, d = rng.randint(1, 4), rng.randint(2, 6)
    e = rng.randint(1, 5)
    f1, f2 = Fraction(a, b), Fraction(c, d)
    prod = f1 * f2
    result = prod - e
    problem = f"Compute ({a}/{b}) × ({c}/{d}) - {e}."
    steps = [
        StepSpec(f"1. Multiply: ({a}/{b})×({c}/{d}) = {_fmt(prod)}", prod, "mul", [f1, f2]),
        StepSpec(f"2. Subtract: {_fmt(prod)} - {e} = {_fmt(result)}", result, "sub", [prod, e]),
        StepSpec(f"3. Final answer: {_fmt(result)}", result, "answer", [result]),
    ]
    return problem, steps


def build_word_three_step(rng: random.Random) -> tuple[str, list[StepSpec]]:
    boxes = rng.randint(2, 6)
    per = rng.randint(4, 12)
    extra = rng.randint(2, 10)
    give = rng.randint(1, 8)
    total = boxes * per + extra
    left = total - give
    problem = (
        f"There are {boxes} boxes with {per} apples each, plus {extra} loose apples. "
        f"If {give} apples are given away, how many remain?"
    )
    steps = [
        StepSpec(f"1. Apples in boxes: {boxes} × {per} = {boxes * per}", boxes * per, "mul", [boxes, per]),
        StepSpec(f"2. Add loose: {boxes * per} + {extra} = {total}", total, "add", [boxes * per, extra]),
        StepSpec(f"3. Give away: {total} - {give} = {left}", left, "sub", [total, give]),
        StepSpec(f"4. Final answer: {left}", left, "answer", [left]),
    ]
    return problem, steps


BUILDERS: dict[str, Builder] = {
    "add_chain": build_add_chain,
    "sub_chain": build_sub_chain,
    "mul_chain": build_mul_chain,
    "div_chain": build_div_chain,
    "order_ops": build_order_ops,
    "linear_eq": build_linear_eq,
    "percent_of": build_percent_of,
    "fraction_add": build_fraction_add,
    "word_two_step": build_word_two_step,
    "mixed_ops_linear": build_mixed_ops_linear,
    "nested_order_ops": build_nested_order_ops,
    "percent_then_eq": build_percent_then_eq,
    "fraction_mul_sub": build_fraction_mul_sub,
    "word_three_step": build_word_three_step,
}


# ---------------------------------------------------------------------------
# Error injection
# ---------------------------------------------------------------------------

def _corrupt_value(rng: random.Random, true: Any, error_type: ErrorType) -> Any:
    t = _frac(true)
    if error_type == "wrong_sign":
        # -0 == 0; fall through to a guaranteed nonzero delta
        if t != 0:
            return -t
        return Fraction(1)
    if error_type == "order_of_operations":
        # nudge by combining operands wrongly-ish: off-by product/sum noise
        cand = t + rng.choice([1, 2, -1])
        return cand if cand != t else t + 1
    if error_type == "fraction":
        if t.denominator != 1:
            cand = Fraction(t.numerator + rng.choice([1, -1]), t.denominator)
            return cand if cand != t else Fraction(t.numerator + 2, t.denominator)
        return t + Fraction(1, 2)
    # arithmetic / invalid_transform default: off-by-k
    delta = rng.choice([1, 2, 3, -1, -2, 10])
    cand = t + delta
    return cand if cand != t else t + 1


def _rewrite_line_result(line: str, new_val: Any) -> str:
    """Replace the RHS after the last '=' (or Final answer value)."""
    nv = _fmt(new_val)
    if "Final answer:" in line:
        # keep prefix, replace trailing value / x = value
        if "x =" in line:
            head = line.split("x =", 1)[0]
            return f"{head}x = {nv}"
        head = line.split("Final answer:", 1)[0]
        return f"{head}Final answer: {nv}"
    if "=" in line:
        head, _ = line.rsplit("=", 1)
        return f"{head}= {nv}"
    return f"{line} → {nv}"


def inject_error(
    rng: random.Random,
    steps: list[StepSpec],
    mode: Literal["none", "single", "multi"],
) -> tuple[list[str], list[dict[str, Any]], int | Literal["NONE"], ErrorType]:
    n = len(steps)
    # Eligible lines for primary error: all except we allow last line too
    eligible = list(range(n))

    if mode == "none":
        lines = [s.text for s in steps]
        trace = []
        running_ok = True
        for i, s in enumerate(steps, start=1):
            trace.append(
                {
                    "line": i,
                    "stated_line": s.text,
                    "op": s.op,
                    "expected": _fmt(s.expected),
                    "stated": _fmt(s.expected),
                    "valid": True,
                    "consistent_with_wrong_prefix": False,
                }
            )
        return lines, trace, "NONE", "none"

    error_types: list[ErrorType] = [
        "arithmetic",
        "invalid_transform",
        "wrong_sign",
        "order_of_operations",
        "fraction",
    ]
    # Prefer earlier/mid lines for multi-error propagation demos
    primary_idx = rng.choice(eligible[:-1] if n > 1 and mode == "multi" else eligible)
    et: ErrorType = rng.choice(error_types)
    if steps[primary_idx].op in ("frac", "simplify") or "fraction" in steps[primary_idx].op:
        et = rng.choice(["fraction", "arithmetic"])

    wrong_primary = _corrupt_value(rng, steps[primary_idx].expected, et)
    # Hard guarantee (e.g. wrong_sign on 0); never spin forever
    for _ in range(8):
        if _frac(wrong_primary) != _frac(steps[primary_idx].expected):
            break
        wrong_primary = _corrupt_value(rng, steps[primary_idx].expected, "arithmetic")
    if _frac(wrong_primary) == _frac(steps[primary_idx].expected):
        wrong_primary = _frac(steps[primary_idx].expected) + 1

    # Optional second independent error on a later line (still first-error = primary)
    second_idx = None
    if mode == "multi" and primary_idx < n - 1:
        candidates = list(range(primary_idx + 1, n))
        if candidates:
            second_idx = rng.choice(candidates)

    lines: list[str] = []
    trace: list[dict[str, Any]] = []
    # For propagation: after primary error, recompute subsequent expected-from-wrong
    # when ops chain on previous stated value.
    stated_values: list[Any] = []

    for i, s in enumerate(steps):
        if i == primary_idx:
            stated = wrong_primary
            text = _rewrite_line_result(s.text, stated)
            valid = False
            cwp = False
        elif second_idx is not None and i == second_idx:
            # Independent second bug; may also disagree with true expected
            stated = _corrupt_value(rng, s.expected, "arithmetic")
            if _frac(stated) == _frac(s.expected):
                stated = _frac(s.expected) + 1
            text = _rewrite_line_result(s.text, stated)
            # Still invalid vs ground truth; mark consistent_with_wrong_prefix if matches chain
            valid = False
            cwp = False
        elif i > primary_idx:
            # Propagate: try to continue from wrong previous stated value when binary op
            prev = stated_values[i - 1]
            if s.op == "add" and len(s.operands) == 2:
                # assume operands[0] was previous result
                stated = _frac(prev) + _frac(s.operands[1])
                text = _rewrite_line_result(s.text, stated)
                valid = False
                cwp = _frac(stated) != _frac(s.expected)
            elif s.op == "sub" and len(s.operands) == 2:
                stated = _frac(prev) - _frac(s.operands[1])
                text = _rewrite_line_result(s.text, stated)
                valid = False
                cwp = True
            elif s.op == "mul" and len(s.operands) == 2:
                # ambiguous which operand is prev; prefer operands[0]==prev pattern
                if _frac(s.operands[0]) == _frac(steps[i - 1].expected):
                    stated = _frac(prev) * _frac(s.operands[1])
                else:
                    stated = _frac(s.operands[0]) * _frac(prev)
                text = _rewrite_line_result(s.text, stated)
                valid = False
                cwp = True
            elif s.op == "div" and len(s.operands) == 2:
                denom = _frac(s.operands[1])
                stated = _frac(prev) / denom if denom != 0 else _frac(prev)
                text = _rewrite_line_result(s.text, stated)
                valid = False
                cwp = True
            elif s.op == "answer":
                stated = prev
                text = _rewrite_line_result(s.text, stated)
                valid = False
                cwp = True
            else:
                stated = s.expected
                text = s.text
                # Unconditionally correct after an error is disallowed by verifier —
                # mark invalid but consistent if same as true (rare) else keep true text invalid flag
                valid = False
                cwp = True
                if _frac(stated) == _frac(s.expected):
                    # force a propagated mismatch on display to keep cwp semantics clear
                    stated = _frac(prev) if s.op == "answer" else _frac(s.expected) + 1
                    text = _rewrite_line_result(s.text, stated)
        else:
            stated = s.expected
            text = s.text
            valid = True
            cwp = False

        stated_values.append(stated)
        lines.append(text)
        trace.append(
            {
                "line": i + 1,
                "stated_line": text,
                "op": s.op,
                "expected": _fmt(s.expected),
                "stated": _fmt(stated),
                "valid": valid,
                "consistent_with_wrong_prefix": cwp,
            }
        )

    label = first_error_from_trace(trace)
    assert label == primary_idx + 1
    return lines, trace, label, et


def make_example(
    *,
    ex_id: str,
    template_id: str,
    split: SplitName,
    seed: int,
    mode: Literal["none", "single", "multi"],
) -> Example:
    rng = _seeded_rng(seed, ex_id, template_id, split, mode)
    builder = BUILDERS[template_id]
    problem, steps = builder(rng)
    # Cap at 5 lines
    if len(steps) > 5:
        steps = steps[:5]
    if len(steps) < 3:
        raise RuntimeError(f"template {template_id} produced <3 steps")

    lines, trace, label, et = inject_error(rng, steps, mode)
    ok, msg = verify_example(lines, trace, label)
    if not ok:
        raise RuntimeError(f"verification failed for {ex_id}: {msg}")

    return Example(
        id=ex_id,
        template_id=template_id,
        split=split,
        problem=problem,
        solution_lines=lines,
        first_error_line=label,
        error_type=et if label != "NONE" else "none",
        verification_trace=trace,
        gold_answer=format_gold_answer(label),
        source="synthetic",
        meta={"mode": mode, "n_lines": len(lines)},
    )


def _allocate_modes(n: int, rng: random.Random) -> list[Literal["none", "single", "multi"]]:
    """~30% NONE, ~55% single error, ~15% multi."""
    modes: list[Literal["none", "single", "multi"]] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.30:
            modes.append("none")
        elif r < 0.85:
            modes.append("single")
        else:
            modes.append("multi")
    return modes


def generate_dataset(
    *,
    seed: int = 42,
    n_train: int = 8000,
    n_val: int = 1000,
    n_test_id: int = 1000,
    n_test_ood: int = 2000,
) -> dict[str, list[Example]]:
    rng = random.Random(seed)
    out: dict[str, list[Example]] = {
        "train": [],
        "val": [],
        "test_id": [],
        "test_ood": [],
    }

    def fill(split: SplitName, templates: list[str], n: int) -> None:
        modes = _allocate_modes(n, rng)
        # Balance error positions roughly by cycling templates
        for i in range(n):
            tid = templates[i % len(templates)]
            ex_id = f"{split}_{tid}_{i:05d}"
            ex = make_example(
                ex_id=ex_id,
                template_id=tid,
                split=split,
                seed=seed,
                mode=modes[i],
            )
            out[split].append(ex)

    fill("train", ID_TEMPLATES, n_train)
    fill("val", ID_TEMPLATES, n_val)
    fill("test_id", ID_TEMPLATES, n_test_id)
    fill("test_ood", OOD_TEMPLATES, n_test_ood)
    return out


def save_dataset(dataset: dict[str, list[Example]], output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    counts = {}
    for split, examples in dataset.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / "examples.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        counts[split] = len(examples)

    manifest = {
        "source": "synthetic",
        "license": "Apache-2.0 (generated)",
        "counts": counts,
        "id_templates": ID_TEMPLATES,
        "ood_templates": OOD_TEMPLATES,
        "label_contract": "ERROR_LINE=<N>|ERROR_LINE=NONE",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
