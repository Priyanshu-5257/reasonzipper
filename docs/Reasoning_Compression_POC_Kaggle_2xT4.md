# POC: Engine-Compatible Compressed Reasoning (Kaggle 2 x T4)

**Status:** Research implementation brief | **Date:** 27 October 2026  
**Goal:** Reduce generated reasoning tokens while retaining first-error detection accuracy, without any custom inference loop or head at deployment.  
**Hardware:** Kaggle notebook with **two NVIDIA T4 GPUs (16 GB VRAM each)**; assume no BF16, use **FP16**, NF4 QLoRA where appropriate.  
**Primary model:** [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). Use [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) for a 100-sample smoke test if needed.

## 0. Research hypothesis and constraints

A standard autoregressive LLM can be trained to generate fewer intermediate *token IDs*, including learned symbols that are unreadable to people, while preserving answer accuracy. The model **must be loadable and runnable via standard Transformers and vLLM** after exporting its ordinary model weights and tokenizer. Target llama.cpp/GGUF only after the vLLM check passes.

**Not the goal:** reproduce a verbose reasoning trace; match a hidden-state delta exactly; achieve better scores merely by revealing the answer in teacher-supplied codes; count hidden or missing tokens as compression.

**Acceptance target (POC, not a predicted result):** >=30% reduction in **mean generated reasoning tokens** relative to the unmodified model at the same generation cap, with <=2 percentage points of absolute accuracy loss on the held-out in-distribution synthetic test, no more than 3 points loss on the OOD set, and positive measured end-to-end latency benefit. Stretch goal: >=40% fewer reasoning tokens. Report both means and medians, 95% bootstrap confidence intervals, and results under matched sampling settings. If baselines are too weak for meaningful comparisons, declare this explicitly.

**Key scientific comparison:** answer-only SFT/GRPO using ordinary vocabulary **vs.** learned discrete reasoning tokens plus computational checkpoint supervision. The latter must beat a matched-token-budget control to support the additional mechanism.

## 1. Datasets and leakage prevention

| Data | Source | Usage | Notes |
|---|---|---|---|
| Synthetic 3–5-line student math solutions | **Create locally** with deterministic symbolic/arithmetic generators | Primary train / validation / ID test / OOD test | Exact verified label: earliest erroneous line or `NONE`; no LLM-generated ground-truth labels |
| GSM8K | [`openai/gsm8k`, config `main`](https://huggingface.co/datasets/openai/gsm8k) | Optional supplementary training data and separate answer-correctness diagnostic | Use official train only for training; preserve all 1,319 official test items for held-out evaluation. Train has 7,473 items. Only transform arithmetic expressions with a verified parser; skip unsafe conversions. |
| ProcessBench | [`Qwen/ProcessBench`](https://huggingface.co/datasets/Qwen/ProcessBench) | **External held-out error-localization benchmark only** | 3,400 records across `gsm8k`, `math`, `olympiadbench`; inspect label convention in dataset metadata/example before scoring; avoid overlapping GSM8K-derived train problems. |

### Synthetic data generation: mandatory

Create **12,000** distinct generated examples with structured ASTs, provenance, and execution traces. Split **8,000 train / 1,000 validation / 1,000 ID test / 2,000 OOD test**, seeded and by *problem template*, not only by row. For ID train/val/test, use disjoint problem instances in shared template families; reserve **entire template families and new operation combinations** for OOD. Avoid identical or near-identical problem statements across splits.

Coverage: addition, subtraction, multiplication, division, order of operations, fractions, one-variable linear equations, percentages, and multi-step word problems. Each problem presents a *given student solution* of 3–5 numbered lines. Mix correct solutions with exactly one intentionally introduced error and, in a smaller subset, multiple errors (label the **first** incorrect line). Balance across error position and `NONE` for the training data. Error types: arithmetic mistakes, invalid transformations, wrong sign, wrong order of operations, and incorrect fraction operations. Include cases where later calculations are internally consistent with an earlier wrong value: the label must still be the first mistake. Reject ambiguous steps, undefined expressions, and unverifiable text.

For every generated example save: `id`, `template_id`, `split`, `problem`, `solution_lines`, `first_error_line` (1-indexed integer or `NONE`), `error_type`, `verification_trace`, `gold_answer`, and `source`. Build independent verifier/tests that recompute each valid transition from the AST and confirm the first error line. Inspect 100 randomly sampled instances manually before training.

**Output contract (all models):** final answer must be exactly `ERROR_LINE=<N>` (N = 1..5) or `ERROR_LINE=NONE`; no explanation in the final-answer region. Keep Qwen3 `<think>...</think>` format enabled for reasoning-mode experiments. Count reasoning tokens only between these delimiters; count *all* discrete code tokens if present. Missing closing delimiter, malformed final answer, or truncated completion => incorrect, and log separately.

For ProcessBench, standardize its `steps` into numbered lines **without changing semantics**, map its label convention explicitly, and score with a separate adapter; do not assume its examples contain only 3–5 steps.

## 2. Reproducibility and Kaggle environment

- Use **Python + PyTorch + Transformers + Datasets + PEFT + TRL + bitsandbytes + Accelerate**. Record exact installed versions, git SHA, random seed, GPU names, memory, and CUDA version in `environment.json`; pin versions in `requirements.lock` after smoke test. T4 supports FP16 and 4-bit NF4 but not native BF16. Disable BF16/autocast BF16 paths.
- Enable internet in Kaggle for initial model/dataset downloads; keep Hugging Face cache under `/kaggle/working` or `/kaggle/temp` with adequate free disk. Copy checkpoints and generated data to `/kaggle/working` before notebook ends; advise uploading essential checkpoints to a persistent Dataset or other permitted storage.
- **GPU 0:** model fine-tuning (single-GPU QLoRA). **GPU 1:** teacher rollout extraction, offline logit caching and independent evaluation. Schedule sequentially if CPU RAM/disk become limiting. Do **not** assume combining 2 x 16 GB yields a 32 GB address space. Avoid DDP/FSDP and simultaneous vLLM server/trainer until the single-GPU path works.
- Start with 4-bit NF4 backbone, FP16 compute, gradient checkpointing, `use_cache=False` during training, LoRA rank 16 / alpha 32 / dropout 0.05 and train attention projections (`q_proj`, `v_proj`, `o_proj`) initially. Microbatch=1, gradient accumulation=8, sequence length <=768, completion budget initially 512; reduce to 384 or 256 only after baseline length distribution is measured. Adjust downward on OOM. `Qwen3-0.6B` is an engineering fallback, **not** an interchangeable scientific baseline.
- Pin the Qwen3 chat template and use the same reasoning-on template across all conditions, unless deliberately evaluating no-thinking as a separate baseline. Do not mix prompts when comparing length.

## 3. Stage A — establish baselines FIRST

1. Implement generator, validator, answer parser, token counter, metrics and evaluator. Unit-test all error labels, missing `<think>` cases, and added token round trips.
2. Evaluate **unmodified Qwen3-1.7B** on 250 validation examples in thinking-on and thinking-off modes (plus a small ProcessBench preview *only for feasibility*, never tuning). Use fixed sampling seed(s), record correctness, reasoning tokens, total generated tokens and wall-clock latency. For Qwen3 thinking-on start from its documented sampling settings (temperature 0.6 / top-p 0.95); avoid greedy as the default reasoning baseline.
3. Establish the unconstrained long-reasoning baseline at an output cap **not shorter than the model's common successful traces**. Also record a fixed-budget 256/512 ablation. If >10% of good responses truncate, increase the reference cap or clearly report the limitation.
4. Measure **direct-answer** performance with thinking disabled: this exposes the value added by thinking. If the model already performs identically without thinking, this dataset is unsuitable for demonstrating reasoning compression; increase task difficulty or focus on harder subsets.

**Gate A:** Continue only if the teacher can solve a meaningful fraction of the examples and *reasoning-on materially improves accuracy* on at least one defined difficulty slice. If not, fix dataset difficulty/model suitability before optimizing tokens. Never claim compression based on a model that did not need reasoning.

## 4. Stage B — low-cost behavioral-compression controls

### B1: Answer-only SFT + short *ordinary-language* reasoning

Use verified teacher trajectories from **training questions only**. Generate a concise form through a deterministic pruning template where possible (e.g., retain only the calculations needed to establish the first wrong line); always check the answer is still correct. Train QLoRA with labels masked on the prompt and **SFT labels on the compressed reasoning and final answer**. This is a short-CoT baseline, not answer-only SFT. As a separate ablation, train with SFT loss on final-answer tokens only.

Settings to try: 1–2 epochs, LR `1e-4` for LoRA, microbatch 1 / grad accumulation 8, save intermediate checkpoints, evaluate every ~200 optimizer steps on validation only. Stop or lower LR if answer accuracy drops.

### B2: GRPO with correctness-first length reward

Initialize from the strongest Stage B1 checkpoint. Group size `G=4`, completion cap matched to baseline, `beta=0.0` for standard reference KL (do not impose reference KL on reasoning), `scale_rewards=False`, and `loss_type="dr_grpo"` **if compatible with the pinned TRL version**; otherwise implement the equivalent intentionally, document it, and test it.

For a group of responses to one question, let `correct_i` be exact final-label correctness including formatting and `L_i` the number of reasoning-token IDs. Reward:

`R_i = correct_i - alpha * all_group_correct * min(L_i / B, 1)`

Start `alpha=0.2`, `B=512`. This introduces length preference only on all-correct groups. All-incorrect groups have no GRPO correctness signal; retain supervised replay, mix in easier problems, and record the zero-variance-group rate. With `scale_rewards=False`, the length coefficient remains meaningful. Try 100–300 optimization updates for a feasibility check, with 50-update evaluations. If unstable, reduce group size/budget or run a smaller 0.6B engineering check; **do not change hardware requirements silently**.

**Gate B:** Compare A, B1, B1-answer-only, B2 at matched accuracy and budgets. If B2 already achieves the POC target, report it as a legitimate result and only pursue Stage C if learned codes might offer another measurable improvement.

## 5. Stage C — experimental discrete trajectory compression

**This stage is research software, not a prebuilt TRL feature.** Its objective is to train a standard autoregressive student to *generate short, potentially unreadable code tokens* that stand in for useful reasoning progress. No custom logic is permitted at inference.

### C1: New token vocabulary and export check

- Add **128 new atomic tokens** to tokenizer: `<|r000|>` ... `<|r127|>`; call `resize_token_embeddings` and verify each maps to exactly one distinct ID. Qwen3-1.7B has **tied input embeddings and LM head**, so verify tying after resizing, PEFT training, merging and saving.
- Use PEFT `trainable_token_indices` (or an equivalently verified selective row-training mechanism) so the new embedding/output rows can learn while the large base matrices remain memory-efficient. Test a 20-step SFT and save/reload round trip before writing the quantizer.
- Export the untrained extended model and tokenizer to Transformers and run a 10-prompt **stock vLLM load/generation smoke test**. If added token IDs cannot be handled, stop and fix deployment compatibility *now*, not after weeks of training.

### C2: Produce causally valid block targets

- On **training examples only**, collect verified teacher reasoning traces with explicit substep boundaries and selected final-layer hidden states after each complete reasoning block. Extract no teacher states from future reasoning or answer text into an earlier block.
- Build training-only block features from *prefix-accessible* state summaries and optionally `h_end - h_start`. Treat vector difference as a **feature**; do not assume it alone represents the computation.
- Train a small encoder + vector quantizer/codebook offline, initially assigning **1–2 code IDs per verified reasoning block**, maximum 8–16 code IDs per solution. Suggested auxiliary reconstruction target: block-end state or progress delta, plus supervised checkpoint labels where available. Quantization may use straight-through estimators **only during training**.
- Guard against answer leakage: code generator may see the reasoning prefix up to that block, **never the full answer or later blocks**. Use train-only data. Inspect class-to-code association; a single code that simply encodes `ERROR_LINE` is not evidence of multi-step reasoning.

### C3: Autoregressive student distillation

- Convert block-code labels into the student's standard output pattern: `<think><|r...|><|r...|>...</think>ERROR_LINE=N`. Train the **ordinary LM** using teacher-forced code-sequence cross entropy and answer-only SFT. Use **auxiliary training-only heads** to predict checkpoint facts (e.g., whether an observed transition is valid, first-error status so far) from student hidden states at code boundaries; weight these weakly and remove heads before export.
- A learned projection may initialize code-token rows, but it is **not** retained as a decoder module. No latent-vector feedback loop is allowed at inference.
- Critical evaluation: compare accuracy given teacher-produced codes vs. **student self-generated codes**. Only self-generated codes count as deployable model performance. If the teacher-code gap is large, improve code prediction, not just final-answer loss.
- Controls: (1) `k` repeated generic thinking/filler tokens with same decoding budget, (2) short natural-language steps at matched budget, (3) codes with checkpoint loss disabled, (4) shuffled/randomized code assignment. If learned codes do not beat these, do not claim the codebook helped.

### C4: Optional answer-distribution distillation + GRPO

- Original idea: **no teacher/reference KL on reasoning tokens**. Compute teacher answer distributions under `(question, original_teacher_reasoning, gold_answer_prefix)`, while student distributions use `(question, student_generated_codes, gold_answer_prefix)`.
- For memory efficiency, pre-cache teacher top-32 vocabulary probabilities per gold answer position **plus a tail probability bucket**; implement the resulting *coarse-grained answer KL* carefully (student probability on the tail bucket is all non-top-32 vocabulary mass), and add standard answer SFT. This is distillation across different conditioning histories, **not** conventional same-prefix reference KL. Optionally compare to answer SFT without KL to prove KL helps.
- Fine-tune with the same Stage B2 GRPO reward and the same budget. Keep light supervised code/answer/checkpoint replay early in RL. The model must generate codes itself during rollouts; never splice in teacher codes at evaluation.
- If answer-KL introduces complexity or exceeds T4 memory, skip it in the first Stage C iteration, record the ablation as not run, and retain answer SFT. Do not load two full models on GPU 0.

## 6. Experimental matrix and evaluation

| ID | Model / method | Trainable new code IDs? | Purpose |
|---|---|---:|---|
| A0 | Original reasoning-on | No | Primary reference |
| A1 | Original reasoning-off | No | Minimum-compute reference |
| B1 | Short-CoT QLoRA SFT | No | Behavioral compression |
| B2 | B1 + GRPO correctness/length | No | Simple RL compression baseline |
| C0 | Matched-budget filler codes / short CoT | Optional | Rule out generic thinking time |
| C1 | Code prediction + answer SFT | Yes | Does discrete representation work? |
| C2 | C1 + checkpoint supervision | Yes | Does preserving computation help? |
| C3 | C2 + GRPO and optional answer-only KL | Yes | Full proposed method |

**Mandatory metrics** (ID test, OOD test, external ProcessBench; plus GSM8K official test as secondary diagnostic):

1. Exact first-error-line accuracy; `NONE` sensitivity/specificity; macro accuracy by error type and location; malformed/truncated rate.
2. Mean, median and p90 generated reasoning tokens; mean total output tokens; generated token counts **on both all examples and correct-only examples**.
3. Accuracy vs. token-budget curves with identical prompts and caps (e.g., 128/256/512); matched-accuracy Pareto frontier; three sampling seeds when feasible.
4. Actual end-to-end inference latency, prefill latency, and decode tok/s on a fixed T4; report batch size, quantization and engine version. Saving 40% of tokens does **not** prove 40% latency reduction.
5. Same prompts evaluated in vanilla Transformers and unmodified vLLM with exported tokenizer + full merged checkpoint. If llama.cpp converter handles the added tokens, report GGUF results too; otherwise provide the exact failing conversion step rather than implying compatibility.
6. 95% paired-bootstrap CIs for accuracy difference and token savings; preserve per-example outputs for audit.

**Model-selection rule:** select checkpoints using validation accuracy, then minimize length subject to the validation accuracy tolerance. Keep ID/OOD/ProcessBench test splits sealed until the evaluation plan and candidate selection are frozen. Never tune based on the external benchmark.

## 7. Recommended implementation structure

```text
reasoning-compression-poc/
  README.md
  requirements.lock
  configs/{baseline,sft,grpo,codes}.yaml
  scripts/
    check_environment.py
    generate_math_errors.py
    validate_dataset.py
    prep_gsm8k.py
    make_teacher_traces.py
    cache_teacher_answer_logits.py
    eval_model.py
    train_sft.py
    train_grpo.py
    add_reasoning_tokens.py
    extract_teacher_states.py
    train_codebook.py
    build_code_labels.py
    train_code_student.py
    export_merge.py
    smoke_vllm.py
    summarize_results.py
  tests/
    test_generator.py
    test_first_error_labels.py
    test_output_parser.py
    test_reasoning_token_count.py
    test_code_roundtrip.py
    test_reward.py
    test_answer_kl.py
  data/{train,val,test_id,test_ood,processbench}/
  artifacts/{traces,checkpoints,exports,reports}/
```

Scripts must expose `--config`, `--seed`, `--output-dir` as applicable, save resumable state, log GPU peak memory, and emit machine-readable JSONL predictions. Avoid hardcoded Kaggle directory assumptions; define paths in config. Implement a **single Kaggle notebook/runner** that runs the stage sequence and optionally skips gated stages, not a codebase dependent on interactive manual intervention.

### Execution order for the delegated agent

1. Create repository, config, dataset generator, validators, unit tests, reproducible splits.
2. Environment smoke test and stock model baseline. **Deliver baseline report before training.**
3. Train B1, then B2. **Deliver B-stage comparison and a go/no-go decision.**
4. Test added-token export and vLLM compatibility; train C1/C2 only if supported.
5. Run C3 only if C2 outperforms matched-budget controls on validation.
6. Freeze the chosen checkpoint(s), evaluate sealed ID/OOD/ProcessBench test sets, perform a deployment smoke test, and generate a final report.

## 8. What must be delivered

- Source repo / single notebook with setup and commands, reproducible synthetic dataset generator, data manifest with source licenses/links, verified labels and tests.
- Saved baseline outputs, teacher traces, training logs, experiment configs and seeds, tracked train/val curves, exact GPU memory peaks and time.
- Adapter checkpoints, merged **standard Hugging Face checkpoint + tokenizer**, optional GGUF only if tested, unmodified vLLM loading command and a 10-request deployment smoke-test log.
- `results.csv` per checkpoint; `per_example_predictions.jsonl`; plots of accuracy vs. reasoning tokens, latency vs. accuracy, error type breakdown; concise `FINAL_REPORT.md` with success/failure against declared criteria.
- Explicit description of **what was not implemented** (e.g., KL or Stage C) and why; do not substitute guessed outcomes for experimental measurements.

## 9. Stop rules and common failure modes

- **Data leakage:** accidental overlap with ProcessBench / GSM8K test, prefix encoder sees the answer, or synthetic labels are not independently verified. Stop and regenerate.
- **Too-easy dataset:** non-thinking model already matches thinking model. Make task harder; no meaningful token-compression claim possible.
- **T4 OOM:** reduce max sequence length, generations, and microbatch; use GPU separation and offline logits. If still failing, smoke test at 0.6B and document that main experiment was not completed.
- **Reasoning collapse:** output `</think>` immediately, guess an answer, or emit `ERROR_LINE` inside thought. Score wrong if format incorrect; compare correct-only token savings and answer-only baselines.
- **Reward normalization:** avoid per-group std normalization that erases the intended length coefficient; record correct-group frequency, KL divergence on answers, and malformed rates.
- **Train/inference mismatch:** success with teacher-supplied codes but failure when self-generating. Do not report the teacher-code score as deployable.
- **Deployment mismatch:** added-token vocab, tied output embeddings, adapter merge and GGUF conversion not correctly preserved. Save/reload and test token round-tripping after *every* export step.

## 10. Agent instruction (copy this into the coding-agent task)

> Implement this POC in a runnable Kaggle notebook + repository, targeting **2 x NVIDIA T4 (16 GB each)**. Follow the gates in order. Prioritize verified datasets, correct token accounting and a robust Stage A/B baseline before building Stage C. All reported improvements must come from real held-out runs, not estimates. Do not use API-paid teacher models unless explicitly authorized. Use FP16/4-bit paths that work on T4; confirm hardware and installed package versions. Export and smoke-test a standard model on unmodified vLLM before claiming deployability. Save all code, configs, predictions, logs, checkpoints and a final comparative report. If OOM, incompatibility or scientifically invalid labels block an experiment, give the smallest reproducible failure and implement a documented fallback rather than silently changing the hypothesis.

## Source references

- Qwen3-1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen3-1.7B tied-embedding configuration: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- ProcessBench: https://huggingface.co/datasets/Qwen/ProcessBench
- TRL GRPO: https://huggingface.co/docs/trl/grpo_trainer
- TRL SFT: https://huggingface.co/docs/trl/sft_trainer
- PEFT trainable token indices: https://huggingface.co/docs/peft/main/package_reference/lora
- Bitsandbytes hardware support: https://huggingface.co/docs/transformers/quantization/bitsandbytes
