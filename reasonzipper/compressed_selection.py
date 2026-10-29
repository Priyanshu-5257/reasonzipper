"""V1 offline compressed-token selection (functional-primary, geom diagnostic).

Selects existing vocabulary IDs sequentially under *compressed* autoregressive
prefixes. English-teacher-prefix scoring is available as a diagnostic only.
"""

from __future__ import annotations

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
    """Functional is primary when beta dominates; geom is auxiliary."""
    return float(alpha) * float(s_geom) + float(beta) * float(s_func)


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
) -> float:
    """Mean NLL of ``answer_ids`` tokens conditioned on ``prefix_ids``."""
    if not answer_ids:
        return 0.0
    ids = list(int(x) for x in prefix_ids) + list(int(x) for x in answer_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=x, use_cache=False)
    logits = out.logits[0, :-1, :]
    targets = x[0, 1:]
    # Only score answer positions (last len(answer_ids) target steps)
    n = len(answer_ids)
    ans_logits = logits[-n:]
    ans_targets = targets[-n:]
    log_probs = F.log_softmax(ans_logits.float(), dim=-1)
    tok_nll = -log_probs.gather(1, ans_targets.unsqueeze(1)).squeeze(1)
    return float(tok_nll.mean().item())


@torch.no_grad()
def score_candidate(
    model: Any,
    *,
    prefix_ids: Sequence[int],
    candidate_id: int,
    close_and_answer_ids: Sequence[int],
    teacher_h_target: torch.Tensor | None,
    teacher_h_start: torch.Tensor | None,
    device: torch.device,
    alpha: float,
    beta: float,
) -> CandidateScore:
    """Score one token after the current (compressed) prefix.

    Functional score: ``s_func = -answer_nll`` after ``prefix + z + close/answer``.
    Geometric score: cosine between student state after ``prefix + z`` and
    teacher ``h_target`` (diagnostic; histories may not be comparable).
    """
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

    nll = answer_nll(model, pref, close_and_answer_ids, device=device)
    s_func = -nll
    s = combine_scores(s_geom, s_func, alpha=alpha, beta=beta)
    return CandidateScore(
        token_id=int(candidate_id),
        s_geom=s_geom,
        s_func=s_func,
        s_combined=s,
        s_geom_vs_start=s_geom_vs_start,
        answer_nll=nll,
    )


@torch.no_grad()
def select_next_code(
    model: Any,
    *,
    prefix_ids: Sequence[int],
    candidates: Sequence[int],
    close_and_answer_ids: Sequence[int],
    teacher_h_target: torch.Tensor | None,
    teacher_h_start: torch.Tensor | None,
    device: torch.device,
    alpha: float = 0.0,
    beta: float = 1.0,
) -> tuple[CandidateScore, list[CandidateScore]]:
    """Greedy argmax over ``s`` for one more compressed token."""
    scored: list[CandidateScore] = []
    for cid in candidates:
        scored.append(
            score_candidate(
                model,
                prefix_ids=prefix_ids,
                candidate_id=int(cid),
                close_and_answer_ids=close_and_answer_ids,
                teacher_h_target=teacher_h_target,
                teacher_h_start=teacher_h_start,
                device=device,
                alpha=alpha,
                beta=beta,
            )
        )
    best = max(scored, key=lambda s: s.s_combined)
    return best, scored


@torch.no_grad()
def select_code_sequence(
    model: Any,
    *,
    start_prefix_ids: Sequence[int],
    candidates: Sequence[int],
    close_and_answer_ids: Sequence[int],
    teacher_h_seq: Sequence[torch.Tensor] | None,
    n_codes: int,
    device: torch.device,
    alpha: float = 0.0,
    beta: float = 1.0,
) -> dict[str, Any]:
    """Sequential selection under compressed prefixes.

    ``teacher_h_seq`` is ``[h_0, h_1, ..., h_m]`` when available. At step ``i``
    (0-based) we use target ``h_{i+1}`` if present; geom is diagnostic only.
    """
    prefix = list(int(x) for x in start_prefix_ids)
    chosen: list[int] = []
    step_logs: list[dict[str, Any]] = []
    h_start = teacher_h_seq[0] if teacher_h_seq else None

    # Baseline: no extra code (close+answer from current prefix)
    baseline_nll = answer_nll(model, prefix, close_and_answer_ids, device=device)

    for step in range(n_codes):
        target = None
        if teacher_h_seq is not None and (step + 1) < len(teacher_h_seq):
            target = teacher_h_seq[step + 1]
        best, _all = select_next_code(
            model,
            prefix_ids=prefix,
            candidates=candidates,
            close_and_answer_ids=close_and_answer_ids,
            teacher_h_target=target,
            teacher_h_start=h_start if step == 0 else (
                teacher_h_seq[step] if teacher_h_seq and step < len(teacher_h_seq) else None
            ),
            device=device,
            alpha=alpha,
            beta=beta,
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

    final_nll = answer_nll(model, prefix, close_and_answer_ids, device=device)
    return {
        "code_token_ids": chosen,
        "prefix_ids_after": prefix,
        "baseline_answer_nll": baseline_nll,
        "final_answer_nll": final_nll,
        "delta_nll_vs_no_code": baseline_nll - final_nll,
        "steps": step_logs,
    }


def effective_codebook_size(assignments: Sequence[int], k: int) -> dict[str, float]:
    """Usage diversity diagnostics (not a uniformity requirement)."""
    if not assignments or k <= 0:
        return {"n_used": 0.0, "effective_n": 0.0, "top1_share": 0.0}
    counts: dict[int, int] = {}
    for a in assignments:
        counts[int(a)] = counts.get(int(a), 0) + 1
    total = float(sum(counts.values()))
    probs = [c / total for c in counts.values()]
    # perplexity of empirical usage
    import math

    ent = -sum(p * math.log(p) for p in probs)
    eff = math.exp(ent)
    top1 = max(counts.values()) / total
    return {
        "n_used": float(len(counts)),
        "effective_n": float(eff),
        "top1_share": float(top1),
        "k": float(k),
    }
