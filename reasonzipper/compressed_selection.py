"""V1 offline compressed-token selection (functional + optional geometric).

Selects existing vocabulary IDs sequentially under *compressed* autoregressive
prefixes.

Functional score uses **answer-token NLL only** (``</think>`` / newlines are
context, not scored). Geometric cosine vs teacher ``h_{i+1}`` is optional;
with ``alpha=0`` this is a functional-only control, not trajectory compression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CandidateScore:
    token_id: int
    s_geom: float
    s_func: float
    s_combined: float
    s_geom_vs_start: float
    answer_nll: float


def combine_scores(
    s_geom: float,
    s_func: float,
    *,
    alpha: float,
    beta: float,
) -> float:
    return float(alpha) * float(s_geom) + float(beta) * float(s_func)


def _zscore(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


@torch.no_grad()
def hidden_at_last(
    model: Any,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Final-layer last-token hidden state; ``input_ids`` shape [1, T]."""
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0, -1, :].float()


@torch.no_grad()
def answer_nll(
    model: Any,
    prefix_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    device: torch.device,
    context_ids: Sequence[int] | None = None,
) -> float:
    """Mean NLL of **answer tokens only**.

    Sequence is ``prefix + context + answer``. ``context_ids`` (e.g. encoded
    ``</think>\\n``) are conditioned on but **not** included in the loss —
    only ``answer_ids`` (e.g. ``ERROR_LINE=2``) are scored.
    """
    ans = [int(x) for x in answer_ids]
    if not ans:
        return 0.0
    ctx = [int(x) for x in (context_ids or [])]
    ids = [int(x) for x in prefix_ids] + ctx + ans
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=x, use_cache=False)
    logits = out.logits[0, :-1, :]
    targets = x[0, 1:]
    n = len(ans)
    ans_logits = logits[-n:]
    ans_targets = targets[-n:]
    log_probs = F.log_softmax(ans_logits.float(), dim=-1)
    tok_nll = -log_probs.gather(1, ans_targets.unsqueeze(1)).squeeze(1)
    return float(tok_nll.mean().item())


@torch.no_grad()
def score_prefix_answer_nll(
    model: Any,
    *,
    start_prefix_ids: Sequence[int],
    code_ids: Sequence[int],
    close_ids: Sequence[int],
    answer_ids: Sequence[int],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a fixed code sequence (used after random/shuffle replacement)."""
    base_prefix = list(int(x) for x in start_prefix_ids)
    with_codes = base_prefix + [int(x) for x in code_ids]
    baseline = answer_nll(
        model, base_prefix, answer_ids, device=device, context_ids=close_ids
    )
    final = answer_nll(
        model, with_codes, answer_ids, device=device, context_ids=close_ids
    )
    return {
        "baseline_answer_nll": baseline,
        "final_answer_nll": final,
        "delta_nll_vs_no_code": baseline - final,
    }


@torch.no_grad()
def score_candidate(
    model: Any,
    *,
    prefix_ids: Sequence[int],
    candidate_id: int,
    close_ids: Sequence[int],
    answer_ids: Sequence[int],
    teacher_h_target: torch.Tensor | None,
    teacher_h_start: torch.Tensor | None,
    device: torch.device,
) -> CandidateScore:
    """Score one token after the current (compressed) prefix (raw geom/func)."""
    pref = list(int(x) for x in prefix_ids) + [int(candidate_id)]
    x = torch.tensor([pref], dtype=torch.long, device=device)
    h_z = hidden_at_last(model, x)

    if teacher_h_target is not None:
        ht = teacher_h_target.to(device=device, dtype=torch.float32)
        s_geom = float(F.cosine_similarity(h_z.unsqueeze(0), ht.unsqueeze(0)).item())
    else:
        s_geom = 0.0

    if teacher_h_start is not None:
        hs = teacher_h_start.to(device=device, dtype=torch.float32)
        s_geom_vs_start = float(
            F.cosine_similarity(h_z.unsqueeze(0), hs.unsqueeze(0)).item()
        )
    else:
        s_geom_vs_start = 0.0

    nll = answer_nll(
        model, pref, answer_ids, device=device, context_ids=close_ids
    )
    s_func = -nll
    return CandidateScore(
        token_id=int(candidate_id),
        s_geom=s_geom,
        s_func=s_func,
        s_combined=s_func,  # filled by select_next_code after optional z-score
        s_geom_vs_start=s_geom_vs_start,
        answer_nll=nll,
    )


@torch.no_grad()
def select_next_code(
    model: Any,
    *,
    prefix_ids: Sequence[int],
    candidates: Sequence[int],
    close_ids: Sequence[int],
    answer_ids: Sequence[int],
    teacher_h_target: torch.Tensor | None,
    teacher_h_start: torch.Tensor | None,
    device: torch.device,
    alpha: float = 0.0,
    beta: float = 1.0,
    zscore_combine: bool = True,
) -> tuple[CandidateScore, list[CandidateScore]]:
    """Greedy argmax over combined score for one more compressed token."""
    scored: list[CandidateScore] = []
    for cid in candidates:
        scored.append(
            score_candidate(
                model,
                prefix_ids=prefix_ids,
                candidate_id=int(cid),
                close_ids=close_ids,
                answer_ids=answer_ids,
                teacher_h_target=teacher_h_target,
                teacher_h_start=teacher_h_start,
                device=device,
            )
        )

    use_z = bool(zscore_combine) and float(alpha) != 0.0 and float(beta) != 0.0
    if use_z:
        zg = _zscore([s.s_geom for s in scored])
        zf = _zscore([s.s_func for s in scored])
        for i, s in enumerate(scored):
            s.s_combined = combine_scores(zg[i], zf[i], alpha=alpha, beta=beta)
    else:
        for s in scored:
            s.s_combined = combine_scores(s.s_geom, s.s_func, alpha=alpha, beta=beta)

    best = max(scored, key=lambda s: s.s_combined)
    return best, scored


@torch.no_grad()
def select_code_sequence(
    model: Any,
    *,
    start_prefix_ids: Sequence[int],
    candidates: Sequence[int],
    close_ids: Sequence[int],
    answer_ids: Sequence[int],
    teacher_h_seq: Sequence[torch.Tensor] | None,
    n_codes: int,
    device: torch.device,
    alpha: float = 0.0,
    beta: float = 1.0,
    zscore_combine: bool = True,
) -> dict[str, Any]:
    """Sequential selection under compressed prefixes.

    ``teacher_h_seq`` is ``[h_0, h_1, ..., h_m]``. Geom uses ``h_{i+1}`` when
    present; with ``alpha=0`` selection is functional-only (answer-guided).
    """
    prefix = list(int(x) for x in start_prefix_ids)
    chosen: list[int] = []
    step_logs: list[dict[str, Any]] = []
    h_start = teacher_h_seq[0] if teacher_h_seq else None

    baseline_nll = answer_nll(
        model, prefix, answer_ids, device=device, context_ids=close_ids
    )

    for step in range(n_codes):
        target = None
        if teacher_h_seq is not None and (step + 1) < len(teacher_h_seq):
            target = teacher_h_seq[step + 1]
        start_h = None
        if step == 0:
            start_h = h_start
        elif teacher_h_seq is not None and step < len(teacher_h_seq):
            start_h = teacher_h_seq[step]
        best, _all = select_next_code(
            model,
            prefix_ids=prefix,
            candidates=candidates,
            close_ids=close_ids,
            answer_ids=answer_ids,
            teacher_h_target=target,
            teacher_h_start=start_h,
            device=device,
            alpha=alpha,
            beta=beta,
            zscore_combine=zscore_combine,
        )
        chosen.append(best.token_id)
        prefix.append(best.token_id)
        step_logs.append(
            {
                "step": step,
                "token_id": best.token_id,
                "s_geom": best.s_geom,
                "s_func": best.s_func,
                "s_combined": best.s_combined,
                "s_geom_vs_start": best.s_geom_vs_start,
                "answer_nll": best.answer_nll,
            }
        )

    final_nll = answer_nll(
        model, prefix, answer_ids, device=device, context_ids=close_ids
    )
    return {
        "code_token_ids": chosen,
        "prefix_ids_after": prefix,
        "baseline_answer_nll": baseline_nll,
        "final_answer_nll": final_nll,
        "delta_nll_vs_no_code": baseline_nll - final_nll,
        "steps": step_logs,
        "alpha": alpha,
        "beta": beta,
        "variant": (
            "func_only"
            if float(alpha) == 0.0
            else ("geom_only" if float(beta) == 0.0 else "geom_plus_func")
        ),
    }


def effective_codebook_size(assignments: Sequence[int], k: int) -> dict[str, float]:
    """Usage diversity diagnostics (not a uniformity requirement)."""
    if not assignments or k <= 0:
        return {"n_used": 0.0, "effective_n": 0.0, "top1_share": 0.0, "k": float(k)}
    counts: dict[int, int] = {}
    for a in assignments:
        counts[int(a)] = counts.get(int(a), 0) + 1
    total = float(sum(counts.values()))
    probs = [c / total for c in counts.values()]
    ent = -sum(p * math.log(p) for p in probs)
    eff = math.exp(ent)
    top1 = max(counts.values()) / total
    return {
        "n_used": float(len(counts)),
        "effective_n": float(eff),
        "top1_share": float(top1),
        "k": float(k),
    }
