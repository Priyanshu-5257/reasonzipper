#!/usr/bin/env python3
"""Stage B1: QLoRA SFT on short-CoT + answer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/sft.yaml")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Override epochs with steps")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["train"]
    model_name = args.model or cfg["model_name"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    import os

    # Avoid Trainer wrapping with DataParallel on multi-GPU boxes (Kaggle 2xT4).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(seed)
    rows = load_jsonl(args.train_file)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[sft] loaded {len(rows)} rows model={model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(ex: dict) -> str:
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=True
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)

    texts = [{"text": to_text(r)} for r in rows]
    ds = Dataset.from_list(texts)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(tcfg["lora_r"]),
        lora_alpha=int(tcfg["lora_alpha"]),
        lora_dropout=float(tcfg["lora_dropout"]),
        target_modules=list(tcfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps
    import inspect

    raw_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": int(tcfg["microbatch"]),
        "gradient_accumulation_steps": int(tcfg["grad_accum"]),
        "learning_rate": float(tcfg["lr"]),
        "num_train_epochs": float(tcfg["epochs"]) if max_steps is None else 1.0,
        "max_steps": -1 if max_steps is None else int(max_steps),
        "logging_steps": 10,
        "save_steps": int(tcfg.get("eval_every_steps", 200)),
        "save_total_limit": 2,
        "fp16": True,
        "bf16": False,
        "max_length": int(tcfg["max_seq_len"]),
        "max_seq_length": int(tcfg["max_seq_len"]),
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": seed,
        "dataset_text_field": "text",
        "packing": False,
    }
    sig = inspect.signature(SFTConfig.__init__)
    sft_kwargs = {k: v for k, v in raw_kwargs.items() if k in sig.parameters}
    sft_args = SFTConfig(**sft_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    # TRL version skew: processing_class vs tokenizer
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    print("[sft] training...", flush=True)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    meta = {
        "model": model_name,
        "n_train": len(rows),
        "max_steps": max_steps,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sft] saved adapter -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
