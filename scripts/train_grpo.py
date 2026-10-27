#!/usr/bin/env python3
"""Stage B2: correctness-first length GRPO on short-CoT SFT adapter."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.metrics import extract_reasoning_span, score_prediction
from reasonzipper.reward import correctness_first_rewards


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/grpo.yaml")
    p.add_argument("--train-file", type=Path, required=True, help="short-cot jsonl with prompt+gold")
    p.add_argument("--sft-adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["train"]
    model_name = args.model or cfg["model_name"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    alpha = float(tcfg.get("alpha", 0.2))
    budget_b = float(tcfg.get("length_budget_B", 512))
    group_size = int(tcfg.get("group_size", 4))
    max_steps = args.max_steps if args.max_steps is not None else int(tcfg.get("max_updates", 100))

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    torch.manual_seed(seed)
    rows = load_jsonl(args.train_file)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # GRPO dataset: chat-formatted prompt + gold label column
    ds_rows = []
    for r in rows:
        ds_rows.append(
            {
                "prompt": format_prompt(r["prompt"]),
                "first_error_line": r["first_error_line"],
            }
        )
    ds = Dataset.from_list(ds_rows)
    print(f"[grpo] n={len(ds)} group={group_size} max_steps={max_steps}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
    model.config.use_cache = False

    def reward_fn(completions, first_error_line, **kwargs):
        # completions: list[str]; first_error_line aligned 1:1 with completions
        # (TRL repeats gold for each generation in the group)
        corrects = []
        lengths = []
        for text, gold in zip(completions, first_error_line):
            scored = score_prediction(text, gold)
            corrects.append(bool(scored["correct"]))
            span, missing, _ = extract_reasoning_span(text)
            if span is None:
                L = len(tokenizer.encode(text, add_special_tokens=False))
            else:
                L = len(tokenizer.encode(span, add_special_tokens=False))
            if missing:
                corrects[-1] = False
            lengths.append(L)

        # Apply correctness-first length within each group of `group_size`
        rewards: list[float] = []
        for i in range(0, len(corrects), group_size):
            chunk_c = corrects[i : i + group_size]
            chunk_l = lengths[i : i + group_size]
            if len(chunk_c) < group_size:
                # last partial group
                rewards.extend(
                    correctness_first_rewards(chunk_c, chunk_l, alpha=alpha, budget_b=budget_b)
                )
            else:
                rewards.extend(
                    correctness_first_rewards(chunk_c, chunk_l, alpha=alpha, budget_b=budget_b)
                )
        return rewards

    # Build GRPOConfig with version-tolerant kwargs
    cfg_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": int(tcfg.get("grad_accum", 8)),
        "learning_rate": 5e-6,
        "logging_steps": 5,
        "save_steps": int(tcfg.get("eval_every", 50)),
        "save_total_limit": 2,
        "fp16": True,
        "bf16": False,
        "max_steps": max_steps,
        "report_to": [],
        "seed": seed,
        "num_generations": group_size,
        "max_completion_length": int(tcfg.get("max_completion_tokens", 512)),
        "temperature": 0.6,
        "top_p": 0.95,
    }
    # Optional newer fields
    sig = inspect.signature(GRPOConfig.__init__)
    optional = {
        "beta": float(tcfg.get("beta", 0.0)),
        "scale_rewards": bool(tcfg.get("scale_rewards", False)),
        "loss_type": tcfg.get("loss_type", "dr_grpo"),
        "gradient_checkpointing": True,
    }
    for k, v in optional.items():
        if k in sig.parameters:
            cfg_kwargs[k] = v

    grpo_args = GRPOConfig(**cfg_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Keep LoRA trainable; avoid creating a second peft wrapper
    trainer_kwargs = dict(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    # Some TRL versions use tokenizer= instead of processing_class=
    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = GRPOTrainer(**trainer_kwargs)

    print("[grpo] training...", flush=True)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    meta = {
        "model": model_name,
        "sft_adapter": str(args.sft_adapter),
        "n_train": len(ds),
        "max_steps": max_steps,
        "alpha": alpha,
        "budget_b": budget_b,
        "group_size": group_size,
    }
    (args.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[grpo] saved adapter -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
