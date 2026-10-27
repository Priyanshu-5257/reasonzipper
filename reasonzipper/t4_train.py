"""T4-safe training helpers (no native BF16)."""

from __future__ import annotations

import os
from typing import Any


def force_t4_fp16_env() -> None:
    """Pin one GPU and force Accelerate/Trainer into FP16 mixed precision."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    # Prevent libraries from preferring bf16 when the model card says so.
    os.environ["TORCH_DTYPE"] = "float16"


def prepare_qlora_for_t4(model: Any, peft_config: Any) -> Any:
    """
    Prepare a 4-bit model + LoRA so trainable tensors are not BF16.

    T4 GradScaler (FP16) crashes if any trainable grads are BF16:
    NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda ... BFloat16
    """
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training

    model.config.torch_dtype = torch.float16
    if hasattr(model, "config") and hasattr(model.config, "dtype"):
        try:
            model.config.dtype = torch.float16
        except Exception:
            pass

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)

    # LoRA (and any other trainable) weights: FP32 is safest under FP16 autocast.
    n_cast = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            n_cast += 1
    print(f"[t4] cast {n_cast} trainable tensors to float32 (fp16 amp safe)", flush=True)
    model.config.use_cache = False
    return model


def cast_trainable_away_from_bf16(model: Any) -> int:
    """For already-wrapped PeftModels (e.g. GRPO warm-start from SFT adapter)."""
    import torch

    n = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
            n += 1
        elif param.requires_grad and param.dtype == torch.float16:
            # Keep LoRA in fp32 under GradScaler
            param.data = param.data.to(torch.float32)
            n += 1
    if hasattr(model, "config"):
        model.config.torch_dtype = torch.float16
    print(f"[t4] normalized {n} trainable tensors to float32", flush=True)
    return n
