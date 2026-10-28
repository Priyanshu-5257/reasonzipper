#!/usr/bin/env python3
"""Offline teacher checkpoint extraction for Approach B (final-layer states).

For each train example, teacher-force the short verified CoT and save
(h_start, h_end) at each reasoning-line boundary inside <think>.
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
        # Positions: end of each non-empty reasoning line (+ think_open token region)
        lines = [ln for ln in think_body.split("\n") if ln.strip()]
        if not lines:
            continue
        lines = lines[: max(1, args.max_codes)]

        # Cumulative prefixes ending at each reasoning line (inside <think>)
        h_start_text = pre + think_open
        prefix_texts: list[str] = []
        acc = h_start_text
        for ln in lines:
            acc = acc + "\n" + ln
            prefix_texts.append(acc)

        enc_start = tok(h_start_text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out_start = model(
                **{k: v.to(model.device) for k, v in enc_start.items()},
                output_hidden_states=True,
                use_cache=False,
            )
            h_start = out_start.hidden_states[-1][0, -1, :].detach().to(torch.float16).cpu()

        block_hs = []
        for ptxt in prefix_texts:
            enc = tok(ptxt, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                out = model(
                    **{k: v.to(model.device) for k, v in enc.items()},
                    output_hidden_states=True,
                    use_cache=False,
                )
                h_end = out.hidden_states[-1][0, -1, :].detach().to(torch.float16).cpu()
            block_hs.append(h_end)

        shard = {
            "id": row["id"],
            "h_start": h_start,
            "h_ends": torch.stack(block_hs, dim=0),
            "n_blocks": len(block_hs),
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
                "n_blocks": len(block_hs),
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
    }
    (args.output / "index.json").write_text(json.dumps({"meta": meta, "items": index}, indent=2))
    print(f"[collect] wrote {len(index)} checkpoints -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
