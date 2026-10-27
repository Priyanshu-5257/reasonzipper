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
    p.add_argument("--load-in-4bit", action="store_true", help="NF4 load for T4 memory headroom")
    p.add_argument("--adapter", type=str, default=None, help="Optional PEFT adapter directory")
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
    load_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": args.device,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if args.adapter:
        from peft import PeftModel

        print(f"[eval] loading adapter {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    preds = []
    n_correct = 0
    reasoning_token_counts = []
    t0 = time.perf_counter()

    log_every = max(1, min(10, len(rows) // 10 or 1))
    print(
        f"[eval] start model={model_name} thinking={args.thinking} "
        f"n={len(rows)} max_new={max_new} 4bit={args.load_in_4bit}",
        flush=True,
    )
    for i, row in enumerate(rows, start=1):
        user = build_prompt(row["problem"], row["solution_lines"], thinking=args.thinking)
        prompt = apply_chat_template(tokenizer, user, enable_thinking=args.thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        step_t0 = time.perf_counter()
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
        if i % log_every == 0 or i == len(rows):
            acc = n_correct / i
            print(
                f"[eval] {i}/{len(rows)} acc={acc:.3f} "
                f"last_r_toks={tok_stats['reasoning_tokens']} "
                f"step_s={time.perf_counter() - step_t0:.1f} "
                f"elapsed_s={time.perf_counter() - t0:.1f}",
                flush=True,
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
