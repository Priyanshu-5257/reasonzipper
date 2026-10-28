#!/usr/bin/env python3
"""Stage C: QLoRA SFT on discrete <|rXXX|> reasoning codes + answer."""

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

from reasonzipper.t4_train import (
    disable_trl_fused_lm_head,
    force_t4_fp16_env,
    patch_trainable_tokens_lm_head,
    prepare_qlora_for_t4,
    t4_training_precision_kwargs,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/codes.yaml")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True, help="Extended tokenizer")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    model_name = args.model or cfg["model_name"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    lr = args.lr if args.lr is not None else float(cfg["train"].get("lr", 1e-4))
    n_codes = int(cfg["codes"]["n_tokens"])

    force_t4_fp16_env()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(seed)
    rows = load_jsonl(args.train_file)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[code-sft] n={len(rows)} model={model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    meta_path = args.tokenizer_dir / "reasoning_tokens.json"
    token_ids = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        token_ids = sorted(meta["token_ids"].values())
    else:
        # fallback: last n_codes vocab entries often aren't reliable; encode names
        from reasonzipper.codes import token_name

        token_ids = [
            tokenizer.encode(token_name(i), add_special_tokens=False)[0] for i in range(n_codes)
        ]
    print(f"[code-sft] trainable_token_indices={len(token_ids)}", flush=True)

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

    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

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
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # Resize for new special tokens (tied embed/lm_head on Qwen3).
    base.resize_token_embeddings(len(tokenizer))
    base.config.torch_dtype = torch.float16

    peft_config = LoraConfig(
        r=int(cfg["train"].get("lora_r", 16)),
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        trainable_token_indices=token_ids,
    )
    # Some PEFT versions use modules_to_save instead / as well
    try:
        model = prepare_qlora_for_t4(base, peft_config)
    except TypeError:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = prepare_qlora_for_t4(base, peft_config)
        print("[code-sft] fallback modules_to_save=embed_tokens,lm_head", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": lr,
        "num_train_epochs": 1.0,
        "max_steps": int(args.max_steps),
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 2,
        **t4_training_precision_kwargs(),
        "max_length": 768,
        "max_seq_length": 768,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": seed,
        "dataset_text_field": "text",
        "packing": False,
        "optim": "adamw_torch",
    }
    sig = inspect.signature(SFTConfig.__init__)
    sft_args = SFTConfig(**{k: v for k, v in raw_kwargs.items() if k in sig.parameters})

    trainer_kwargs = dict(model=model, args=sft_args, train_dataset=ds)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    disable_trl_fused_lm_head(trainer)
    patch_trainable_tokens_lm_head(trainer.model)

    print("[code-sft] training...", flush=True)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    # keep reasoning token meta with the adapter export
    if meta_path.exists():
        (args.output_dir / "reasoning_tokens.json").write_text(meta_path.read_text())
    meta = {
        "model": model_name,
        "n_train": len(rows),
        "max_steps": args.max_steps,
        "n_code_tokens": n_codes,
        "tokenizer_dir": str(args.tokenizer_dir),
    }
    (args.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[code-sft] saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
