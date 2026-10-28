"""T4-safe training helpers (no native BF16 / no GradScaler amp)."""

from __future__ import annotations

import os
from typing import Any


def force_t4_fp16_env() -> None:
    """
    Pin one GPU and disable Accelerate mixed precision.

    T4 + recent torch/transformers still produce BF16 grads under fp16 GradScaler
    for Qwen QLoRA, which crashes with:
      NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda ... BFloat16
    So we train LoRA in FP32 with amp off; 4-bit base stays quantized.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"


def prepare_qlora_for_t4(model: Any, peft_config: Any) -> Any:
    """Prepare 4-bit model + LoRA with FP32 trainable weights (amp-free)."""
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training

    model.config.torch_dtype = torch.float16
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)

    n_cast = 0
    n_train = 0
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_train += 1
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            n_cast += 1
    print(
        f"[t4] trainable={n_train} cast_to_fp32={n_cast} "
        f"(mixed_precision=off for T4)",
        flush=True,
    )
    if n_train == 0:
        raise RuntimeError("No trainable parameters after LoRA wrap")
    model.print_trainable_parameters()
    model.config.use_cache = False
    return model


def cast_trainable_away_from_bf16(model: Any) -> int:
    """Normalize trainable tensors to FP32 on an existing PeftModel."""
    import torch

    n = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            n += 1
    if hasattr(model, "config"):
        model.config.torch_dtype = torch.float16
    print(f"[t4] normalized {n} trainable tensors to float32", flush=True)
    return n


def t4_training_precision_kwargs() -> dict:
    """Kwargs for SFTConfig/GRPOConfig/TrainingArguments on T4."""
    return {
        "fp16": False,
        "bf16": False,
        "tf32": False,
    }


def patch_trainable_tokens_lm_head(model: Any) -> int:
    """
    PEFT TrainableTokensWrapper has no `.bias`, but TRL fused LM-head CE reads
    `lm_head.bias` and crashes. Attach bias=None so fused/non-fused paths work.
    """
    n = 0
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls in {"TrainableTokensWrapper", "ModulesToSaveWrapper"}:
            if not hasattr(module, "bias"):
                try:
                    object.__setattr__(module, "bias", None)
                except Exception:
                    module.bias = None  # type: ignore[attr-defined]
                n += 1
                print(f"[t4] patched {name} ({cls}).bias = None", flush=True)
    return n


def disable_trl_fused_lm_head(trainer: Any, *, override_compute_loss: bool = True) -> None:
    """
    Avoid TrainableTokensWrapper.bias crashes in TRL fused LM-head CE.

    For SFTTrainer, override_compute_loss=True routes through HF Trainer.compute_loss
    (safe for standard LM CE). For GRPOTrainer this MUST be False — GRPO's
    compute_loss expects custom batch fields; replacing it with HF compute_loss
    causes `input_ids`/`inputs_embeds` conflicts.
    """
    if override_compute_loss:
        from transformers.trainer import Trainer as HfTrainer

        def _compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            return HfTrainer.compute_loss(
                trainer,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        trainer.compute_loss = _compute_loss

    for attr in ("use_fused_lm_head", "fused_linear_cross_entropy"):
        if hasattr(trainer, attr):
            setattr(trainer, attr, False)
    if hasattr(trainer, "args"):
        for attr in ("use_liger_kernel", "use_liger"):
            if hasattr(trainer.args, attr):
                try:
                    setattr(trainer.args, attr, False)
                except Exception:
                    pass
    model = getattr(trainer, "model", None)
    if model is not None:
        patch_trainable_tokens_lm_head(model)
    print(
        f"[t4] disabled TRL fused LM-head path "
        f"(override_compute_loss={override_compute_loss})",
        flush=True,
    )
