#!/usr/bin/env python3
"""Offline teacher checkpoint extraction for Approach B (final-layer states).

For each train example, teacher-force the short verified CoT and save the
ordered final-layer states at each reasoning-line boundary inside <think>:

    h_seq = [h_0, h_1, ..., h_m]

where h_0 is the state after ``<think>`` and h_i is the state after line i.
Also records token span lengths between consecutive checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.metrics import build_prompt
from reasonzipper.short_cot import build_short_completion
from reasonzipper.t4_train import force_t4_fp16_env


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Directory for .pt shards + index")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-codes", type=int, default=8)
    args = p.parse_args()

    force_t4_fp16_env()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = load_jsonl(args.input)[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map=args.device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    hidden_size = int(model.config.hidden_size)

    index: list[dict] = []
    think_open = "<think>"
    think_close = "</think>"

    for i, row in enumerate(rows):
        prompt = build_prompt(row["problem"], row["solution_lines"], thinking=True)
        completion = build_short_completion(
            row["verification_trace"], row["first_error_line"]
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, enable_thinking=True
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False)

        if think_open not in text or think_close not in text:
            continue
        pre, rest = text.split(think_open, 1)
        think_body, post = rest.split(think_close, 1)
        lines = [ln for ln in think_body.split("\n") if ln.strip()]
        if not lines:
            continue
        lines = lines[: max(1, args.max_codes)]

        # Texts ending at h_0 (after "<think>\n") and after each reasoning line.
        # Trailing newline matches SFT body layout: <think>\n{codes}\n</think>
        h0_text = pre + think_open + "\n"
        boundary_texts: list[str] = [h0_text]
        acc = h0_text
        for ln in lines:
            # First line follows the newline already in h0_text; later lines need \n.
            if acc.endswith("\n") and acc == h0_text:
                acc = acc + ln
            else:
                acc = acc + "\n" + ln
            boundary_texts.append(acc)

        h_list: list[torch.Tensor] = []
        lens: list[int] = []
        for ptxt in boundary_texts:
            enc = tok(ptxt, return_tensors="pt", add_special_tokens=False)
            n_tok = int(enc["input_ids"].shape[1])
            lens.append(n_tok)
            with torch.no_grad():
                out = model(
                    **{k: v.to(model.device) for k, v in enc.items()},
                    output_hidden_states=True,
                    use_cache=False,
                )
                h = out.hidden_states[-1][0, -1, :].detach().to(torch.float16).cpu()
            h_list.append(h)

        # token_spans[i] = teacher tokens between h_i and h_{i+1}
        token_spans = [lens[j + 1] - lens[j] for j in range(len(lens) - 1)]

        # Prefix IDs up to h_0 (start of compressed selection inside <think>)
        start_ids = tok(h0_text, add_special_tokens=False)["input_ids"]
        # Split close delimiter (context, unscored) from answer tokens (scored).
        gold = completion.split("</think>\n", 1)[-1]
        close_ids = tok(f"{think_close}\n", add_special_tokens=False)["input_ids"]
        answer_ids = tok(gold, add_special_tokens=False)["input_ids"]
        # Back-compat: concatenated close+answer (older scorers)
        close_answer_ids = list(close_ids) + list(answer_ids)

        shard = {
            "id": row["id"],
            "h_seq": torch.stack(h_list, dim=0),
            "n_blocks": len(h_list) - 1,
            "token_spans": token_spans,
            "boundary_lengths": lens,
            "start_prefix_ids": start_ids,
            "close_ids": close_ids,
            "answer_ids": answer_ids,
            "close_answer_ids": close_answer_ids,
            # Back-compat aliases for older assign scripts
            "h_start": h_list[0],
            "h_ends": torch.stack(h_list[1:], dim=0) if len(h_list) > 1 else torch.empty(0, hidden_size),
            "first_error_line": row["first_error_line"],
            "gold_answer": row["gold_answer"],
            "prompt": prompt,
        }
        path = args.output / f"ckpt_{i:05d}.pt"
        torch.save(shard, path)
        index.append(
            {
                "id": row["id"],
                "path": path.name,
                "n_blocks": len(h_list) - 1,
                "token_spans": token_spans,
                "mean_token_span": (
                    sum(token_spans) / len(token_spans) if token_spans else 0.0
                ),
                "first_error_line": row["first_error_line"],
                "gold_answer": row["gold_answer"],
                "prompt": prompt,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"[collect] {i+1}/{len(rows)}", flush=True)

    meta = {
        "n": len(index),
        "model": args.model,
        "hidden_size": hidden_size,
        "max_codes": args.max_codes,
        "pairing": "consecutive_h_seq",
        "schema": "h_seq=[h0..hm]; token_spans between consecutive",
    }
    (args.output / "index.json").write_text(
        json.dumps({"meta": meta, "items": index}, indent=2)
    )
    print(f"[collect] wrote {len(index)} checkpoints -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
