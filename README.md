# reasonzipper

Engine-compatible **reasoning compression** POC: teach a small LLM to use fewer intermediate tokens (including optional learned discrete codes) while preserving first-error detection accuracy — deployable via stock Transformers / vLLM.

**Primary model:** [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)  
**Hardware target:** Kaggle 2× NVIDIA T4 (FP16 / NF4 QLoRA)

## Hypothesis

A standard autoregressive model can emit a shorter sequence of reasoning token IDs while keeping answer accuracy within a small tolerance, without custom inference heads at deployment.

## Stages (gated)

| Stage | What | Gate |
|---|---|---|
| **A** | Synthetic error-localization data + unmodified baselines | Thinking-on must beat thinking-off on a defined slice |
| **B** | Short-CoT SFT → correctness-first GRPO length reward | If target met, Stage C is optional |
| **C** | Discrete `<\|r###\|>` codes + checkpoint supervision | Must beat matched-budget filler controls; self-generated codes only |

POC acceptance (targets to *test*, not promises): ≥30% fewer mean reasoning tokens (stretch 40%), ≤2pp ID / ≤3pp OOD accuracy loss, measured latency benefit.

Full brief: [`docs/Reasoning_Compression_POC_Kaggle_2xT4.md`](docs/Reasoning_Compression_POC_Kaggle_2xT4.md)

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q

# Generate synthetic dataset (12k by default)
python scripts/generate_math_errors.py --config configs/baseline.yaml --output-dir data

# Validate labels
python scripts/validate_dataset.py --data-dir data

# Environment check (optional GPU)
python scripts/check_environment.py --output artifacts/reports/environment.json
```

## Kaggle

```bash
!git clone https://github.com/Priyanshu-5257/reasonzipper.git
%cd reasonzipper
!pip install -r requirements.txt
# Then run notebooks/kaggle_runner.ipynb stage-by-stage
```

## Output contract

Final answer must be exactly:

```text
ERROR_LINE=<N>   # N in 1..5
ERROR_LINE=NONE
```

Reasoning (when enabled) stays inside Qwen3 `<think>...</think>`. Reasoning tokens are counted only between those delimiters.

## License

Apache-2.0 for project code. Datasets (GSM8K, ProcessBench) retain their upstream licenses — see `docs/` and data manifests.
