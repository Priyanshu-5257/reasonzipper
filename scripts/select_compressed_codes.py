#!/usr/bin/env python3
"""V1: select existing-token codes under compressed autoregressive prefixes.

Primary score: functional answer NLL (s_func = -NLL).
Auxiliary: geometric cosine vs teacher h_{i+1} (diagnostic; not required to match).

English-prefix single-step scoring can be dumped as a diagnostic JSONL, but
assignment uses sequential compressed prefixes:

    z_i* = argmax_z s(z | x, z_1*, ..., z_{i-1}*)

Does NOT train P_phi. Writes SFT JSONL + selection diagnostics.
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
    select_code_sequence,
)
from reasonzipper.existing_vocab import (
    assert_original_vocabulary,
    build_candidate_token_ids,
    build_think_completion_from_ids,
)
from reasonzipper.schemas import format_gold_answer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="trajectory SFT jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--candidate-k", type=int, default=32)
    p.add_argument("--n-codes", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.0, help="geom weight (aux)")
    p.add_argument("--beta", type=float, default=1.0, help="func weight (primary)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--control",
        choices=("none", "random", "shuffled"),
        default="none",
        help="none=greedy V1; random/shuffled replace codes after selection for controls",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    index = json.loads((args.checkpoints / "index.json").read_text())
    items = index["items"]
    if args.limit is not None:
        items = items[: args.limit]

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
    print(
        f"[select] items={len(items)} K={len(candidates)} n_codes={args.n_codes} "
        f"alpha={args.alpha} beta={args.beta} control={args.control}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    diag_path = args.output.with_suffix(".diag.jsonl")
    all_assign: list[int] = []
    n_out = 0
    n_roundtrip_fail = 0
    deltas: list[float] = []

    with args.output.open("w", encoding="utf-8") as fout, diag_path.open(
        "w", encoding="utf-8"
    ) as fdiag:
        for ii, it in enumerate(items):
            shard = torch.load(args.checkpoints / it["path"], map_location="cpu")
            if "h_seq" in shard:
                h_seq = [shard["h_seq"][j].float() for j in range(shard["h_seq"].shape[0])]
            else:
                # Legacy shards: fabricate consecutive list from h_start + h_ends
                h_seq = [shard["h_start"].float()] + [
                    shard["h_ends"][j].float() for j in range(int(shard["n_blocks"]))
                ]
            start_ids = shard.get("start_prefix_ids")
            if start_ids is None:
                raise RuntimeError(
                    f"shard {it['path']} missing start_prefix_ids; re-run collect_checkpoints"
                )
            close_ids = shard.get("close_answer_ids")
            if close_ids is None:
                gold = format_gold_answer(it["first_error_line"])
                close_ids = tok.encode(f"</think>\n{gold}", add_special_tokens=False)

            n_codes = min(args.n_codes, max(1, len(h_seq) - 1))
            result = select_code_sequence(
                base,
                start_prefix_ids=start_ids,
                candidates=candidates,
                close_and_answer_ids=close_ids,
                teacher_h_seq=h_seq,
                n_codes=n_codes,
                device=device,
                alpha=args.alpha,
                beta=args.beta,
            )
            ids = list(result["code_token_ids"])

            if args.control == "random":
                ids = [int(rng.choice(candidates)) for _ in ids] or [
                    int(rng.choice(candidates))
                ]
            elif args.control == "shuffled":
                # Permute selected IDs across a buffer: for single-example use
                # random permutation of the same multiset when len>1, else random.
                if len(ids) > 1:
                    ids = ids.copy()
                    rng.shuffle(ids)
                else:
                    ids = [int(rng.choice(candidates))]

            gold = format_gold_answer(it["first_error_line"])
            try:
                completion, ids = build_think_completion_from_ids(
                    tok, ids, gold, verify_roundtrip=True, verify_formatted=True
                )
            except AssertionError as e:
                n_roundtrip_fail += 1
                print(f"[select] round-trip fail id={it['id']}: {e}", flush=True)
                # Fall back: try other candidates until one sequence works
                recovered = False
                for alt in candidates:
                    try:
                        trial = [int(alt)] * max(1, len(ids) or 1)
                        completion, ids = build_think_completion_from_ids(
                            tok, trial, gold
                        )
                        recovered = True
                        break
                    except AssertionError:
                        continue
                if not recovered:
                    continue

            row = {
                "id": it["id"],
                "prompt": it["prompt"],
                "completion": completion,
                "gold_answer": it["gold_answer"],
                "first_error_line": it["first_error_line"],
                "code_token_ids": ids,
                "n_codes": len(ids),
                "baseline_answer_nll": result["baseline_answer_nll"],
                "final_answer_nll": result["final_answer_nll"],
                "delta_nll_vs_no_code": result["delta_nll_vs_no_code"],
                "token_spans": it.get("token_spans"),
                "selection": "compressed_ar_v1",
                "alpha": args.alpha,
                "beta": args.beta,
                "control": args.control,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fdiag.write(
                json.dumps(
                    {
                        "id": it["id"],
                        "steps": result["steps"],
                        "delta_nll_vs_no_code": result["delta_nll_vs_no_code"],
                        "code_token_ids": ids,
                        "token_spans": it.get("token_spans"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            all_assign.extend(ids)
            deltas.append(float(result["delta_nll_vs_no_code"]))
            n_out += 1
            if (ii + 1) % 10 == 0:
                print(f"[select] {ii+1}/{len(items)}", flush=True)

    usage = effective_codebook_size(all_assign, len(candidates))
    meta = {
        "n": n_out,
        "candidate_k": args.candidate_k,
        "n_codes": args.n_codes,
        "alpha": args.alpha,
        "beta": args.beta,
        "control": args.control,
        "mean_delta_nll_vs_no_code": (
            sum(deltas) / len(deltas) if deltas else 0.0
        ),
        "usage": usage,
        "n_roundtrip_fail": n_roundtrip_fail,
        "method": "compressed_ar_v1_func_primary",
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[select] wrote {n_out} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
