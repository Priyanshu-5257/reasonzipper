#!/usr/bin/env python3
"""Evaluate a model on first-error localization (Stage A baselines)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.metrics import build_prompt, count_reasoning_tokens, score_prediction


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_chat_template(tokenizer, user_text: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3 supports enable_thinking on some template versions
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--data", type=Path, required=True, help="JSONL split path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model_name"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    eval_cfg = cfg.get("eval", {})
    max_new = args.max_new_tokens or eval_cfg.get("max_new_tokens", 512)
    temperature = eval_cfg.get("temperature", 0.6)
    top_p = eval_cfg.get("top_p", 0.95)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    preds = []
    n_correct = 0
    reasoning_token_counts = []
    t0 = time.perf_counter()

    for row in rows:
        user = build_prompt(row["problem"], row["solution_lines"], thinking=args.thinking)
        prompt = apply_chat_template(tokenizer, user, enable_thinking=args.thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new,
                "do_sample": args.thinking,
            }
            if args.thinking:
                gen_kwargs.update({"temperature": temperature, "top_p": top_p})
            else:
                gen_kwargs["do_sample"] = False
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        # Strip special tokens that clutter parsing but keep think tags if present
        text_for_parse = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        scored = score_prediction(text_for_parse, row["first_error_line"])
        tok_stats = count_reasoning_tokens(text_for_parse, tokenizer)
        if scored["correct"]:
            n_correct += 1
        reasoning_token_counts.append(tok_stats["reasoning_tokens"])
        preds.append(
            {
                "id": row["id"],
                "gold": row["first_error_line"],
                "prediction_text": text_for_parse,
                **scored,
                **tok_stats,
            }
        )

    elapsed = time.perf_counter() - t0
    summary = {
        "model": model_name,
        "thinking": args.thinking,
        "n": len(rows),
        "accuracy": n_correct / max(len(rows), 1),
        "malformed_rate": sum(1 for x in preds if x["malformed"]) / max(len(rows), 1),
        "mean_reasoning_tokens": (
            sum(reasoning_token_counts) / max(len(reasoning_token_counts), 1)
        ),
        "median_reasoning_tokens": sorted(reasoning_token_counts)[len(reasoning_token_counts) // 2]
        if reasoning_token_counts
        else 0,
        "wall_clock_s": elapsed,
        "max_new_tokens": max_new,
        "seed": seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
