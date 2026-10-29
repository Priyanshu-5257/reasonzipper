#!/usr/bin/env python3
"""V1: select existing-token codes under compressed autoregressive prefixes.

Variants:
  - func_only:      alpha=0, beta=1  (answer-guided control; not trajectory compression)
  - geom_plus_func: alpha>0, beta>0  (z-scored combine within each step)
  - geom_only:      alpha>0, beta=0

Functional NLL scores **answer tokens only**; ``</think>\\n`` is context.

Controls (``--control random|shuffled``) recompute answer NLL after replacement.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.compressed_selection import (
    effective_codebook_size,
    score_prefix_answer_nll,
    select_code_sequence,
)
from reasonzipper.existing_vocab import (
    assert_original_vocabulary,
    build_candidate_token_ids,
    build_think_completion_from_ids,
)
from reasonzipper.schemas import format_gold_answer


def _split_close_answer(tok, shard: dict, first_error_line) -> tuple[list[int], list[int]]:
    if shard.get("close_ids") is not None and shard.get("answer_ids") is not None:
        return list(shard["close_ids"]), list(shard["answer_ids"])
    gold = format_gold_answer(first_error_line)
    close_ids = tok.encode("</think>\n", add_special_tokens=False)
    answer_ids = tok.encode(gold, add_special_tokens=False)
    # Legacy concatenated field
    if shard.get("close_answer_ids") is not None and not shard.get("close_ids"):
        # Best-effort split: answer is a suffix of close_answer_ids
        ca = list(shard["close_answer_ids"])
        if len(ca) >= len(answer_ids) and ca[-len(answer_ids) :] == answer_ids:
            return ca[: -len(answer_ids)], answer_ids
    return close_ids, answer_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="trajectory SFT jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--candidate-k", type=int, default=32)
    p.add_argument("--n-codes", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="geom weight; 0 => functional-only control",
    )
    p.add_argument("--beta", type=float, default=1.0, help="func weight")
    p.add_argument(
        "--zscore-combine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="z-score geom/func within each candidate set before combining",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--control",
        choices=("none", "random", "shuffled"),
        default="none",
        help="none=greedy V1; random/shuffled replace codes then re-score NLL",
    )
    p.add_argument(
        "--codes-jsonl",
        type=Path,
        default=None,
        help="If set, skip selection and score/write these pre-assigned code rows",
    )
    p.add_argument(
        "--verify-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert code IDs survive full user+assistant chat templating",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    index = json.loads((args.checkpoints / "index.json").read_text())
    items = index["items"]
    if args.limit is not None:
        items = items[: args.limit]

    preassigned: dict[str, list[int]] = {}
    if args.codes_jsonl is not None:
        with args.codes_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                preassigned[row["id"]] = [int(x) for x in row["code_token_ids"]]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map=args.device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base.eval()
    assert_original_vocabulary(base, tok, label="select")
    device = next(base.parameters()).device

    candidates = build_candidate_token_ids(tok, k=args.candidate_k, seed=args.seed)
    variant = (
        "func_only"
        if args.alpha == 0.0
        else ("geom_only" if args.beta == 0.0 else "geom_plus_func")
    )
    print(
        f"[select] items={len(items)} K={len(candidates)} n_codes={args.n_codes} "
        f"alpha={args.alpha} beta={args.beta} variant={variant} "
        f"control={args.control} preassigned={bool(preassigned)}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    diag_path = args.output.with_suffix(".diag.jsonl")
    all_assign: list[int] = []
    n_out = 0
    n_roundtrip_fail = 0
    n_fallback = 0
    deltas: list[float] = []
    finals: list[float] = []
    deltas_primary: list[float] = []
    deltas_fallback: list[float] = []
    finals_primary: list[float] = []
    finals_fallback: list[float] = []

    with args.output.open("w", encoding="utf-8") as fout, diag_path.open(
        "w", encoding="utf-8"
    ) as fdiag:
        for ii, it in enumerate(items):
            shard = torch.load(args.checkpoints / it["path"], map_location="cpu")
            if "h_seq" in shard:
                h_seq = [
                    shard["h_seq"][j].float() for j in range(shard["h_seq"].shape[0])
                ]
            else:
                h_seq = [shard["h_start"].float()] + [
                    shard["h_ends"][j].float() for j in range(int(shard["n_blocks"]))
                ]
            start_ids = shard.get("start_prefix_ids")
            if start_ids is None:
                raise RuntimeError(
                    f"shard {it['path']} missing start_prefix_ids; re-run collect_checkpoints"
                )
            close_ids, answer_ids = _split_close_answer(tok, shard, it["first_error_line"])
            n_codes = min(args.n_codes, max(1, len(h_seq) - 1))

            step_logs: list = []
            selection_delta: float | None = None
            row_variant = variant

            if it["id"] in preassigned:
                ids = list(preassigned[it["id"]])
                row_variant = "preassigned"
            else:
                selected = select_code_sequence(
                    base,
                    start_prefix_ids=start_ids,
                    candidates=candidates,
                    close_ids=close_ids,
                    answer_ids=answer_ids,
                    teacher_h_seq=h_seq,
                    n_codes=n_codes,
                    device=device,
                    alpha=args.alpha,
                    beta=args.beta,
                    zscore_combine=args.zscore_combine,
                )
                ids = list(selected["code_token_ids"])
                step_logs = list(selected["steps"])
                selection_delta = float(selected["delta_nll_vs_no_code"])
                row_variant = str(selected.get("variant", variant))

                if args.control == "random":
                    ids = [int(rng.choice(candidates)) for _ in ids] or [
                        int(rng.choice(candidates))
                    ]
                elif args.control == "shuffled":
                    if len(ids) > 1:
                        ids = ids.copy()
                        rng.shuffle(ids)
                    else:
                        ids = [int(rng.choice(candidates))]

            # Always score the IDs that will be written (covers select, control
            # replacement, and --codes-jsonl preassigned paths alike).
            scored = score_prefix_answer_nll(
                base,
                start_prefix_ids=start_ids,
                code_ids=ids,
                close_ids=close_ids,
                answer_ids=answer_ids,
                device=device,
            )
            result_meta: dict = {
                **scored,
                "steps": step_logs,
                "variant": row_variant,
            }
            if selection_delta is not None:
                result_meta["selection_delta_nll_vs_no_code"] = selection_delta

            gold = format_gold_answer(it["first_error_line"])
            used_fallback = False
            try:
                completion, ids = build_think_completion_from_ids(
                    tok,
                    ids,
                    gold,
                    verify_roundtrip=True,
                    verify_formatted=True,
                    prompt=it["prompt"],
                    verify_chat_template=bool(args.verify_chat_template),
                )
            except AssertionError as e:
                n_roundtrip_fail += 1
                print(f"[select] round-trip fail id={it['id']}: {e}", flush=True)
                recovered = False
                for alt in candidates:
                    try:
                        trial = [int(alt)] * max(1, len(ids) or 1)
                        completion, ids = build_think_completion_from_ids(
                            tok,
                            trial,
                            gold,
                            prompt=it["prompt"],
                            verify_chat_template=bool(args.verify_chat_template),
                        )
                        # Re-score recovered fallback codes (NOT the original traj)
                        result_meta = {
                            **score_prefix_answer_nll(
                                base,
                                start_prefix_ids=start_ids,
                                code_ids=ids,
                                close_ids=close_ids,
                                answer_ids=answer_ids,
                                device=device,
                            ),
                            "steps": step_logs,
                            "variant": result_meta.get("variant", variant),
                            "fallback_roundtrip": True,
                        }
                        used_fallback = True
                        recovered = True
                        break
                    except AssertionError:
                        continue
                if not recovered:
                    continue

            if used_fallback:
                n_fallback += 1

            row = {
                "id": it["id"],
                "prompt": it["prompt"],
                "completion": completion,
                "gold_answer": it["gold_answer"],
                "first_error_line": it["first_error_line"],
                "code_token_ids": ids,
                "n_codes": len(ids),
                "baseline_answer_nll": result_meta["baseline_answer_nll"],
                "final_answer_nll": result_meta["final_answer_nll"],
                "delta_nll_vs_no_code": result_meta["delta_nll_vs_no_code"],
                "token_spans": it.get("token_spans"),
                "selection": "compressed_ar_v1",
                "variant": result_meta.get("variant", variant),
                "fallback_roundtrip": used_fallback,
                "alpha": args.alpha,
                "beta": args.beta,
                "control": args.control,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fdiag.write(
                json.dumps(
                    {
                        "id": it["id"],
                        "steps": result_meta.get("steps", []),
                        "delta_nll_vs_no_code": result_meta["delta_nll_vs_no_code"],
                        "final_answer_nll": result_meta["final_answer_nll"],
                        "baseline_answer_nll": result_meta["baseline_answer_nll"],
                        "code_token_ids": ids,
                        "token_spans": it.get("token_spans"),
                        "control": args.control,
                        "variant": result_meta.get("variant", variant),
                        "fallback_roundtrip": used_fallback,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            all_assign.extend(ids)
            dlt = float(result_meta["delta_nll_vs_no_code"])
            fin = float(result_meta["final_answer_nll"])
            deltas.append(dlt)
            finals.append(fin)
            if used_fallback:
                deltas_fallback.append(dlt)
                finals_fallback.append(fin)
            else:
                deltas_primary.append(dlt)
                finals_primary.append(fin)
            n_out += 1
            if (ii + 1) % 10 == 0:
                print(f"[select] {ii+1}/{len(items)}", flush=True)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    usage = effective_codebook_size(all_assign, len(candidates))
    meta = {
        "n": n_out,
        "candidate_k": args.candidate_k,
        "n_codes": args.n_codes,
        "alpha": args.alpha,
        "beta": args.beta,
        "variant": variant if not preassigned else "preassigned",
        "control": args.control,
        "mean_delta_nll_vs_no_code": _mean(deltas),
        "mean_final_answer_nll": _mean(finals),
        # Fallback replacements are NOT the selected trajectory — report apart.
        "n_roundtrip_fail": n_roundtrip_fail,
        "n_fallback_roundtrip": n_fallback,
        "fallback_rate": (n_fallback / n_out) if n_out else 0.0,
        "mean_delta_nll_primary": _mean(deltas_primary),
        "mean_delta_nll_fallback": _mean(deltas_fallback),
        "mean_final_answer_nll_primary": _mean(finals_primary),
        "mean_final_answer_nll_fallback": _mean(finals_fallback),
        "n_primary": len(deltas_primary),
        "usage": usage,
        "method": "compressed_ar_v1_answer_only_nll",
        "answer_nll_masks_close_delimiter": True,
        "scores_recomputed_after_control": True,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[select] wrote {n_out} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
