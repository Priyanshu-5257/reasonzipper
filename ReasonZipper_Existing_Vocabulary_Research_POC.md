# ReasonZipper: Existing-Vocabulary Reasoning Compression

**Status:** Proposed experiment plan (not implemented or evaluated in this document)
**Target repository:** https://github.com/Priyanshu-5257/reasonzipper
**Hardware:** Kaggle 2 x NVIDIA T4 (16 GB each), FP16-compatible
**Primary model:** `Qwen/Qwen3-1.7B`
**Date:** 2026-10-28

## 0. Agent execution contract

Implement and test two *new* branches in ReasonZipper, without changing the model vocabulary or requiring nonstandard inference:

- **Approach A - Unconstrained existing-token reasoning:** Let SFT on answers plus correctness-first GRPO discover short reasoning sequences from the existing vocabulary. Do **not** prescribe the contents of reasoning tokens.
- **Approach B - Trajectory-guided existing-token reasoning:** Use a *training-only* hidden-state compressor/selector to choose sequences of **existing** token IDs that represent teacher reasoning checkpoints, distill the resulting codes into the student, and then run GRPO. Delete the compressor before deployment.

**Non-negotiable constraints**

1. `len(tokenizer)` and `model.config.vocab_size` must remain identical to the original Qwen3-1.7B throughout each experiment and in the deployed checkpoint. No `add_special_tokens()`, `resize_token_embeddings()`, new LM-head rows, or new token IDs.
2. No custom runtime transition head, nonstandard embeddings as model inputs, or special sampling logic in deployment. The final model must generate regular token IDs with the original tokenizer and the original Transformer architecture.
3. Preserve original Qwen3 chat-template/control token handling. Qwen3's existing `<think>` and `</think>` markers can delimit reasoning; do not add new delimiters.
4. Existing input/output embedding rows are **frozen in the primary experiments**. Train LoRA weights that learn to use existing tokens differently in context. Optional ablation: selective existing-row updates, still with unchanged vocabulary; report separately.
5. No answer/reasoning traces from sealed ID/OOD, GSM8K test, MATH-500, or ProcessBench may enter training or checkpoint selection. Do not inspect sealed targets while tuning.
6. All evaluation must use **self-generated** reasoning tokens and answers, not teacher-supplied latent codes or manually selected tokens.
7. Do not report training runs as completed unless the agent actually executed them and saved reproducible artifacts.

## 1. Research question and hypotheses

Can a pretrained reasoning model use its **original vocabulary** to encode useful computations in far fewer autoregressive decoding steps, while retaining task accuracy?

For prompt `x`, teacher reasoning `r=(r_1,...,r_n)`, and answer `a`, train a student to generate:

```
x -> existing-token sequence z=(z_1,...,z_k) -> answer a, where k << n
```

The human-readable meaning of `z` is irrelevant. Unlike GLR/Coconut, each `z_i` must be a token ID from the **unchanged** original vocabulary, so standard inference engines can decode it.

- **H-A:** Answer-supervised training and correctness-first, length-aware RL can discover useful compact sequences without code targets.
- **H-B:** Training-only checkpoint/trajectory guidance will outperform A and matched-length filler/random-token controls, particularly on harder and out-of-distribution problems.
- **Null hypotheses:** Shorter output arises from memorization, answer leakage, formatting shortcuts, or extra Transformer compute from filler tokens; there is no special advantage to trajectory-guided codes.

The claims to test are **lower token count**, **accuracy retention**, and **actual latency improvement**. Unreadability is not itself a success metric.

## 2. Existing repository baseline: reuse, do not overwrite

Repo experiment log: https://github.com/Priyanshu-5257/reasonzipper/blob/main/docs/experiment.md

Previously reported results on 100 validation examples (not sealed evaluation):

| Run | Accuracy | Mean reasoning tokens | Notes |
| --- | ---: | ---: | --- |
| A0: original thinking | 65% | 48.9 | Frozen base |
| Thinking off | 28% | 0 | Frozen base |
| B1: short English SFT | 74% | 21.5 | 150 SFT steps |
| B2: B1 + GRPO | 75% | 21.4 | 50 RL steps |
| C1: deterministic novel-code SFT | 61% | 4.21 | Expanded vocabulary |
| C0: repeated-code filler | 56% | 6.56 | Expanded vocabulary |
| C3: C1 + GRPO | 70% | 4.81 | Expanded vocabulary |

**C1/C3 do not meet the new no-vocabulary-expansion constraint** and are *comparison points only*. They cannot simply be continued as unchanged-vocabulary checkpoints. Prefer the original Qwen3 checkpoint or the existing short-English B1/B2 checkpoint **only after checking its tokenizer and embedding shape are original**.

Check the repository's current code, data schema, dependency versions, run scripts, and outstanding sealed evaluations before editing. Make new configs and artifacts; preserve all old experiments. If previous sealed evaluation has already run, import **aggregate** results for baseline reporting and do not use it to tune new branches.

## 3. Datasets and split policy

### 3.1 Synthetic error localization: inexpensive smoke test

Reuse the repo's existing math-error generator and validator. Task: identify the **first incorrect line** or `NONE` with exact format:

```
ERROR_LINE=2
```

Target: 8,000 train, 1,000 development validation, 1,000 sealed ID, 2,000 sealed OOD (confirm actual generated counts and seeds in repo before use). This dataset supports exact Python verification and quick debugging, but it cannot by itself establish general reasoning transfer.

Recommended allocation: for each candidate, use only train and **development validation** during tuning; evaluate existing sealed sets only on final selected checkpoints.

### 3.2 GSM8K: main reasoning-transfer benchmark

Source: https://huggingface.co/datasets/openai/gsm8k (`main` configuration)

- 7,473 official training items; 1,319 official test items.
- Deterministically split the **official train portion** into ~6,973 training and 500 dev items (seed 42, record exact IDs).
- Never train on, prompt-tune against, or select checkpoints with the 1,319 official test items.
- Parse the final gold number after `####` and compare normalized numeric answers (support commas, integer/decimal signs, fractions where appropriate); never use a language-model grader for correctness.
- The official train solution provides useful human-written steps; verify extracted arithmetic annotations programmatically where possible. For failed/unverifiable trajectories, log and exclude from checkpoint-target training rather than trusting them blindly.
- Keep the original question text unchanged and use the same prompt template, answer parser, decoding budget, and sampling parameters across baselines.

Target output example (numbers are illustrative):

```
<think>
[model-chosen existing token IDs; no readability constraint]
</think>
FINAL_ANSWER=72
```

`FINAL_ANSWER=` is an ordinary textual formatting convention, not a new token. Use tokenizer ID-level checks for token counts, rather than relying on the decoded string.

### 3.3 External evaluation

- **MATH-500:** https://huggingface.co/datasets/HuggingFaceH4/MATH-500 ; harder held-out evaluation. Use math-equivalence parsing, not plain string matching.
- **ProcessBench:** https://huggingface.co/datasets/Qwen/ProcessBench ; a different task (step-error verification). Only evaluate a model with the proper first-error prompt/label mapping; do not mix its accuracy with GSM8K numeric-answer accuracy.
- If OOD metrics cannot be run due to budget, state that explicitly. No synthetic scores can substitute for external generalization.

### 3.4 Teacher data rule

For synthetic, use exact verifier-grounded steps and, optionally, base-model teacher traces independently checked against labels. For GSM8K, collect teacher reasoning on **train-only** problems and keep only outputs whose final answer matches gold. Use multiple correct traces per example only if compute permits. Do not treat the teacher trace as necessarily minimal or semantically perfect.

Save records with `dataset`, `split`, `sample_id`, `prompt`, `gold_answer`, `teacher_reasoning_ids`, `teacher_answer_ids`, `teacher_correct`, `teacher_checkpoint_positions`, and a reproducibility hash. Never save a final-answer-dependent feature into a field passed to student inference.

## 4. Model setup and Kaggle 2xT4 constraints

Use existing repo environment wherever practical. Suggested pilot setup:

| Parameter | Initial value | Notes |
| --- | --- | --- |
| Base model | Qwen3-1.7B | Same model for all comparable runs |
| Quantization | NF4 / 4-bit QLoRA | Trainable LoRA, frozen 4-bit base |
| Compute dtype | FP16-compatible | T4 does not support native BF16 computation |
| LoRA | rank 16, alpha 32, dropout 0.0-0.05 | Adapt attention + MLP projections if VRAM permits |
| Sequence length | 512 synthetic; 768-1024 GSM8K | Reduce on OOM |
| Microbatch | 1 | Gradient accumulation for effective batch |
| GRPO generations | 4 per prompt initially | Increase only if memory permits |
| Checkpoint strategy | gradient checkpointing + sequential rollout/training | Avoid GPU OOM |
| Start with | 1 T4 for training | Second can be used for sequential/offline teacher preparation, not assumed to pool VRAM |

Pin and log `torch`, `transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate`, `vllm` versions, CUDA information, model revision, Git commit, seeds, GPU VRAM, and exact trainer configs.

**Avoid assuming LoRA+GRPO+vLLM colocation will fit a 16-GB T4.** Prefer a tested low-memory rollout loop and disable the vLLM backend during initial training if necessary. Generate/cache teacher traces in a separate process. Only one large model forward/backward pipeline per GPU at once. Process batches sequentially when OOM.

## 5. Common baselines: run before new approaches

All conditions must share the same model family, split, prompt, answer scoring, sampling policy, and token accounting:

1. **F0:** Frozen base with standard thinking enabled.
2. **F1:** Frozen base with thinking disabled.
3. **F2:** Existing short-English SFT checkpoint (B1), if compatible.
4. **F3:** Existing short-English GRPO checkpoint (B2), if compatible.
5. **F4:** **New answer-only SFT, zero-reasoning control**; supervised on direct final answers and, optionally, correctness RL. This is the key missing control: an untrained thinking-off model is not a fair counterfactual.
6. **F5:** **Matched-budget filler-thinking SFT/RL**, using only existing frequent token IDs (e.g., one repeated filler token), with the **same training steps and rollout budget** as the learned-code method.

For the synthetic task, report existing 100-example figures as *historical only* and create a fresh paired dev evaluation on the exact same examples for new comparisons.

## 6. Approach A: unconstrained existing-token reasoning

### 6.1 Principle

Do not assign meanings, codebooks, or target text to intermediate reasoning. Reuse `Qwen3-1.7B`'s normal token vocabulary and its unchanged embedding/output matrices. The policy learns which token sequences, if any, improve correctness within a short generation budget.

**Important limitation:** RL does not magically discover a new private language. It may simply shorten English reasoning, learn direct-answer shortcuts, or collapse to empty thinking. The experiments must distinguish these outcomes.

### 6.2 Initialization and training

- **A-base:** Start from original Qwen3 or from *vocabulary-compatible* B1 short-English SFT. Run both initializations on the cheap synthetic pilot if feasible; do not compare models with unequal supervision without labeling that difference.
- **A-SFT:** Apply answer-only cross-entropy, with gradients masked on **prompt and reasoning positions**. In standard SFT, teacher-forcing the answer under a provided prefix does **not** train the model to generate that prefix. Therefore, for the pilot use short **student-generated** reasoning prefixes (sampled and selected for correctness), or first train a compact answer-only baseline and then use RL for the reasoning policy.
- **A-RL:** Sample full completions with `<think> ... </think>` and a fixed final-answer format. Optimize correctness and reasoning length with GRPO/advantage-based policy updates. No reference-teacher KL on intermediate reasoning tokens.
- **A-KD optional:** Answer-only distillation against the original teacher given its full reasoning, with the student conditioned on its own short prefix; see Section 8.

### 6.3 Reward and collapse protections

Let `c_i` be 1 when both answer and format are correct, `L_i` the number of reasoning IDs (excluding delimiting tokens), `B` a fixed budget. For each group of four responses to the same problem:

```
R_i = c_i - alpha * I(all rollouts correct) * min(L_i / B, 1)
```

Start `alpha=0.2`, `B=128` synthetic or `B=256` GSM8K, and adjust only using dev results. If the group is mixed, prioritize correctness; if all are correct, prefer shorter correct completions. If all are wrong, preserve supervised replay or resample; an all-wrong group offers no correct signal.

**Normalization caution:** Standardizing group rewards by their standard deviation can cancel `alpha` for all-correct groups. Log raw rewards and use group-mean-only or constant-scale normalization for length learning, and use a fixed loss denominator (Dr. GRPO style) if supported by the installed TRL version.

- Keep a nonzero fraction of answer-SFT replay batches during RL.
- Require the answer after `</think>` to be only the canonical answer field; no disguised chain of thought in the answer.
- Track zero-length thinking, repeated tokens, malformed control tokens, output truncation, and answer accuracy separately.
- Perform a small sweep over reasoning budgets (e.g., 8, 16, 32, 64 tokens) with a held-constant overall output cap. Do not claim an unreadable language emerged merely because token strings look strange.

### 6.4 Expected deliverable

Checkpoints `A_sft`, `A_grpo`; per-example dev predictions; accuracy/length Pareto curve; comparison against F4 direct-answer baseline. A is useful even if it discovers *short natural-language* solutions rather than gibberish.

## 7. Approach B: trajectory-guided existing-token code selection

### 7.1 Distinguish this from GLR

GLR uses continuous vectors during inference. Here the inference model must output **discrete existing vocabulary IDs**. The training-only compressor aligns useful teacher reasoning states with a limited set of **original embedding rows**; the student is subsequently trained to emit those IDs on its own.

### 7.2 Extract teacher checkpoints (offline)

Use the same frozen Qwen teacher. For each **training** example:

1. Obtain a verified solution/teacher reasoning trace.
2. Partition the reasoning into meaningful operation blocks (math equation, checked line, numerical computation). For GSM8K, segment using gold working/calculation annotations where possible; log segmentation failures.
3. Teacher-force the prefix through each block and save only a bounded set of final-layer checkpoint states (`h_start`, `h_end`) and token positions. Do not store every layer of every token.
4. Define an optional difference feature `delta = h_end - h_start`, but recognize that endpoint displacement alone does not guarantee preservation of the intermediate computation.
5. Cache checkpoint feature tensors in FP16 on CPU/disk, with sample IDs and teacher revision; normalize appropriately when used in cosine losses.

**Leakage warning:** The teacher checkpoint at the *end* of a block contains information from that block. This is allowed as **offline supervision** but must never be supplied to the deployed student. Evaluation uses only question -> student-generated IDs -> answer.

### 7.3 Select an existing-token codebook

- Read the original model's embedding matrix `E` (frozen). Construct a candidate set `S` of perhaps **512 or 1,024 existing token IDs**. Exclude Qwen chat/control tokens (`<think>`, `</think>`, EOS, role markers), padding, and other tokens whose generation would break the chat format.
- Prefer candidate tokens that tokenize to **one ID** and have valid decode/encode round trips. Record IDs, not merely printed strings, since whitespace and Unicode tokenization can be surprising.
- Candidate selection may use frequency and embedding diversity, but must not use held-out labels or performance.
- Learn a small projection `P_phi` from teacher block/checkpoint features to the model input-embedding space. It outputs similarity logits over the **frozen original embeddings in S**:

```
u = normalize(P_phi([h_start, h_end, delta]))
logit_j = cosine(u, E[j]) / tau, for j in S
z = argmax_j logit_j
```

Use **hard token IDs** for the student forward pass, with straight-through Gumbel-softmax or a REINFORCE-style estimator when jointly updating the code selector. Do not let a soft embedding mixture become a required runtime representation.

### 7.4 Train the compression system in two sub-stages

**B-1. Establish teacher-code usefulness (diagnostic only)**

- Start with `K=1-2` codes per teacher block, maximum 4-8 codes total per example for the synthetic pilot; adjust budgets on dev.
- Train selector `P_phi` and a student adapter jointly while feeding *hard selected code IDs* to the student during teacher forcing.
- Loss: `L_answer_CE + lambda_cp * L_checkpoint + lambda_use * L_code_usage`. Checkpoint loss can regress/cosine-align the student's state after its code prefix against a chosen frozen teacher checkpoint. Use modest weighting; forcing exact hidden-state equality across differing histories may be harmful.
- Optional task-specific checkpoint supervision: whether a checked math step is valid, or the intermediate numeric result. Auxiliary heads are **training-only**. Mask any task label that would reveal the final answer before the intended corresponding checkpoint.
- Guard against codebook collapse (all examples selecting the same token) by monitoring entropy and usage statistics, not by guaranteeing uniform usage.
- Measure accuracy **when teacher-selected codes are provided**. This tests whether selected tokens can carry useful information, but **does not count as a final success result**.

**B-2. Distill code production into an ordinary autoregressive student**

- Create offline targets: `prompt -> <think> [selected existing token IDs] </think> -> final answer`.
- Train ordinary causal-LM SFT/QLoRA to predict the code IDs **and** the answer, using the unchanged tokenizer and frozen original embeddings.
- Retain answer-only SFT/KD terms. Use a nonzero code-prediction loss during this stage to teach the model to generate the code IDs.
- Switch evaluation to fully self-generated code prefixes and answers; report both teacher-code and self-generated gaps.
- Apply the **same GRPO training recipe and length budgets as Approach A**; no frozen-teacher KL on reasoning/code positions.
- **Discard `P_phi`, checkpoint heads, quantizer, and all teacher features** when saving the deployable student.

### 7.5 Major failure modes and tests

- **Answer leakage:** Selector might simply encode the final answer from teacher checkpoints. Test with final-answer-masked checkpoint features, shuffled per-example checkpoint targets, and comparable number of codes; ensure code generation must arise only from the question at runtime.
- **Student generation gap:** Teacher-provided codes work, self-generated codes do not. Report this difference prominently; prioritize code-prediction SFT if large.
- **Vocabulary/semantics collapse:** Repeated common token emerges as all codes. Compare with matched filler and random-code controls.
- **No meaningful compute compression:** A four-token sequence may effectively memorize a label. Test GSM8K, OOD combinations, and harder examples.
- **Low-rank geometry mismatch:** Projected teacher state and original input embedding rows might not align. Compare full-state, delta-only, and checkpoint-task features in a very small ablation; do not assume nearest-neighbor matching works a priori.
- **Training/inference discrepancy:** Avoid reliance on teacher features or soft vector inputs at inference; student must generate actual discrete original token IDs.

### 7.6 Expected deliverable

Checkpoints `B_selector_diagnostic`, `B_student_sft`, `B_student_grpo`; code IDs and utilization statistics; teacher-code versus self-generated accuracy; auxiliary-head ablations; comparison against A/F4/F5.

## 8. Optional answer-only KL / KD (after the core runs work)

For canonical answer tokens `a*`, compare the teacher after full reasoning `r` to the student after self-generated short reasoning `z`:

```
L_answer_KD = mean_t KL(
    p_teacher(. | question, full_teacher_reasoning, a*_<t>)
    ||
    p_student(. | question, student_code_prefix, a*_<t>)
)
```

- Mask KL on all reasoning/code tokens; **answer positions only**.
- This is **cross-context knowledge distillation**, not normal same-prefix GRPO reference KL.
- Computing full-vocabulary logits for a teacher and student simultaneously may exceed memory. Prepare teacher answer logits **offline**. If using top-k teacher distributions, label the result *approximate KD* and specify treatment of omitted probability mass; do not describe it as exact KL.
- Compare `answer CE only` against `answer CE + KD`. Do **not** make KD a prerequisite for A/B running successfully.
- SFT/KD on answers does not directly train a policy to sample compressed code IDs; SFT of code targets (B) and/or policy-gradient learning (A/B) remains necessary.

## 9. Experiment order and resource-aware gates

| Order | Run | Dataset | Min objective | Gate |
| --- | --- | --- | --- | --- |
| 0 | repo audit + F0/F1/F2/F3/F4 | Synthetic dev | Reproduce formats/metrics | Tests and baseline work |
| 1 | A answer-only SFT + GRPO | Synthetic dev | Learn token-length/accuracy curve | Better than F4 at some budget |
| 2 | F5 matched filler and random-code controls | Synthetic dev | Separate compute/time from code semantics | Fair budgets matched |
| 3 | B teacher-feature extraction + code selector | Synthetic train/dev | Teacher-code usefulness | Above filler at matched budget |
| 4 | B self-generated-code SFT + GRPO | Synthetic dev | Usable deployable codes | Competitive accuracy & tokens |
| 5 | A vs B on GSM8K train/dev | GSM8K dev | Genuine word-problem transfer | Accuracy/length Pareto improvement |
| 6 | Optional answer-KD and checkpoint ablations | Dev only | Marginal contribution | Keep only if worthwhile |
| 7 | Final frozen checkpoints -> sealed ID/OOD, GSM8K test, MATH-500, ProcessBench | Sealed/test | Independent generalization | Single final evaluation |
| 8 | Export and test stock inference engine | Dev prompts | No custom model/engine changes | Actual inference verified |

For GSM8K, if synthetic pretraining transfers poorly, do a **separate GSM8K train/dev-only tuning run** for A and B, clearly distinguishing it from zero-shot transfer. Do not use GSM8K test to tune. Do not spend all Kaggle compute on B before verifying F4/A.

Proposed training pilots (adjust downward on OOM, record changes):

- Smoke tests: 20 examples; one training step each; confirm correct masks, gradients, token IDs, save/reload.
- Initial A: 100-200 answer-SFT steps plus 50-100 GRPO steps.
- B selector: <=1,000 synthetic teacher traces; 100-300 selector/student updates; then 100-200 student SFT steps and 40-100 GRPO steps.
- GSM8K: start 1,000-2,000 train examples, then expand to full available training split only after validation shows useful progress.
- Use a fixed set of seeds (at least 3 on the most promising candidates when resources permit) and identical token budgets for all comparisons.

These are suggested *experimental budgets*, not guaranteed to fit, converge, or complete within any fixed Kaggle session.

## 10. Evaluation and reporting

Always report **per-task** metrics, on paired questions where possible:

1. Exact answer/first-error accuracy, format validity, numeric parsing failure rate.
2. Reasoning token counts (mean, median, p90, p95); also **total generated tokens** and answer tokens.
3. Accuracy by reasoning budget and number of problem steps / difficulty / OOD template.
4. Effective compression vs F0 and vs the strongest **accuracy-matched** baseline (do not use F0 alone if F2 is stronger).
5. Teacher-code vs student-code accuracy (B); code usage entropy, percent off-candidate IDs, repetition, and token distribution.
6. Zero-thinking and filler controls trained with comparable data, steps, and inference budgets.
7. Peak GPU allocated/reserved memory; training throughput; inference tokens/s and **end-to-end latency** with same batch size, context, sampling, and hardware.
8. Paired bootstrap confidence intervals for accuracy and mean reasoning-token difference. Do not declare non-inferiority based on an underpowered 100-example sample.
9. Honest failure cases: arithmetic patterns, long multi-step reasoning, wrong final-answer placement, malformed `<think>` blocks, cases with right answer but nonsensical codes.

Primary success gate proposal (assess on a sizable held-out test, not the 100-example dev slice):

- >=40% fewer **reasoning tokens** than the best sufficiently accurate baseline.
- Accuracy within 2 percentage points on synthetic ID and within 3 points on synthetic OOD; for GSM8K, predeclare a task-specific tolerance and confidence-interval criterion before opening test results.
- Net end-to-end inference latency improvement on stock vLLM; no additional inference-time head or runtime patch.
- B must beat matched-budget A and F5 on accuracy, or at least deliver a better accuracy/token Pareto tradeoff, to justify the extra complexity.

If A works but B does not, do **not** claim hidden-state coding works. If tokens drop but latency does not, describe the result as token compression, not inference acceleration. If codes give no gain over a fine-tuned zero-thinking model, do not claim a novel reasoning representation.

## 11. Expected repository changes (suggested, adapt to existing layout)

Implement incrementally rather than refactoring the entire repository:

```
configs/
  existing_vocab_a.yaml
  existing_vocab_b.yaml
reasonzipper/
  existing_vocab.py          # original-token-only validation, selection
  answer_rewards.py         # gold parsing, length reward, group gating
  teacher_checkpoints.py    # offline checkpoint extraction
  trajectory_selector.py    # training-only projector/quantizer
  student_code_distill.py   # hard-code targets, CE/KD/masking
  eval_compression.py       # paired metrics, controls, confidence intervals
scripts/
  train_existing_a.py
  collect_checkpoints.py
  train_existing_b.py
  eval_existing_vocab.py
  export_existing_vocab.py
  check_stock_inference.py
artifacts/
  reports/approach_a/
  reports/approach_b/
  reports/paired_comparison/
```

Create unit tests before full training:

- Assert unchanged vocabulary size and embedding/output shapes before/after training, saving, reloading, and merging LoRA.
- Assert special/control tokens excluded from B candidate set; all selected tokens are legal **original** IDs.
- Verify answer-only loss masks reasoning completely for Approach A; verify B code CE applies only when deliberately enabled.
- Verify label extraction: `ERROR_LINE=NONE`, first-error index, GSM8K signed and comma-separated numbers, and malformed outputs.
- Verify group-reward behavior on all-correct, mixed, and all-wrong rollouts. Verify denominator/normalization behavior.
- Verify B diagnostic teacher states are **never used** by normal inference or sealed evaluation.
- Run train/save/reload self-generated inference on 20 synthetic and 20 GSM8K dev items before long training.

**Final saved model**: merged model (if feasible) plus exact unchanged tokenizer, `config.json`, `generation_config.json`, LoRA details, training manifest, and quantization notes. Do not export the temporary selector or auxiliary heads.

## 12. Delivery checklist for the coding agent

- [ ] Inspect the existing repo and record exact base commit/model revision.
- [ ] Preserve original experiment checkpoints and historical reports.
- [ ] Build and test F4 direct-answer fine-tuning control.
- [ ] Run Approach A on synthetic dev and compare with fair controls.
- [ ] Implement B selector using **existing** embedding rows, with no new token IDs.
- [ ] Run B teacher-code diagnostics, then self-generated student SFT and RL.
- [ ] Run matched filler/random controls under the same budgets.
- [ ] Train/evaluate promising candidates on GSM8K train/dev.
- [ ] Freeze hyperparameters/checkpoints before a single sealed/test evaluation.
- [ ] Export and load the model in unmodified vLLM; attempt llama.cpp conversion only after the normal Transformers/vLLM path works.
- [ ] Save `results.jsonl`, `summary.json`, `config.yaml`, environment manifest, training logs, failing examples, and a concise `docs/existing_vocab_experiment.md`.
- [ ] Explicitly identify which runs **executed** and which remain **planned**; do not manufacture results.

### What the agent must report at the end

1. A vs B vs F4/F5 accuracy/token/latency table, grouped by task and split.
2. Whether unchanged-vocabulary and stock-vLLM deployment actually passed.
3. Whether learned codes outperform generic filler after matching budget and training.
4. Evidence for or against genuine generalization beyond synthetic patterns.
5. Failure modes, cost/VRAM observations, and next minimal experiment.

## 13. Sources and implementation references

- ReasonZipper experiment log: https://github.com/Priyanshu-5257/reasonzipper/blob/main/docs/experiment.md
- Qwen3-1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Official GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k
- MATH-500 dataset: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- ProcessBench dataset: https://huggingface.co/datasets/Qwen/ProcessBench
- TRL GRPO documentation: https://huggingface.co/docs/trl/grpo_trainer

**Boundary:** This is an implementation plan. No new model has been trained, no hidden-state selector has been validated, and no accuracy/latency improvement is claimed by this document.
