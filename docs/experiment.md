# reasonzipper — Experiment Log

**Model:** Qwen/Qwen3-1.7B · **Hardware:** Kaggle 2×T4 (train pinned to 1×T4) · **Task:** first-error line localization (`ERROR_LINE=<N|NONE>`)

## Hypothesis

A small LLM can replace long English chain-of-thought with a short sequence of **opaque discrete tokens** (`<|rXXX|>`) that only the model uses as reasoning, while keeping answer accuracy nearly the same and cutting reasoning tokens substantially.

## Stages

| Stage | Question | Status |
|---|---|---|
| **A** | Does thinking help on this task? | **Pass** |
| **B** | Can *short English* CoT compress reasoning? | **Pass** (control / baseline) |
| **C** | Can *unreadable discrete codes* do the same? | **Pass** (after GRPO fix) |

---

## Stage A — baselines

Kernel: [reasonzipper-stage-a](https://www.kaggle.com/code/zenitsu1agatsuma/reasonzipper-stage-a)

| Condition | Acc (n=100 val) | Mean reasoning toks |
|---|---:|---:|
| Thinking **on** (A0) | **65%** | 48.9 |
| Thinking **off** | 28% | 0 |

**Gate A:** Pass (+37pp from thinking).

Artifacts: `artifacts/reports/kaggle_stage_a/`

---

## Stage B — short English CoT (behavioral compression)

Kernel: [reasonzipper-stage-b](https://www.kaggle.com/code/zenitsu1agatsuma/reasonzipper-stage-b)

**Method**
1. Build deterministic short CoT from verification traces (`L1: OK … L2: FAIL …`).
2. **B1 SFT:** QLoRA CE on short CoT + answer (150 steps, amp off).
3. **B2 GRPO:** correctness-first length reward  
   \(R = \mathbb{1}[\mathrm{correct}] - \alpha\cdot\mathbb{1}[\mathrm{all\ correct}]\cdot\min(L/B,1)\)  
   (\(\alpha=0.2\), \(B=512\), 50 steps, no ref KL).

| Method | Acc | Mean reasoning toks | vs A0 |
|---|---:|---:|---|
| A0 | 65% | 48.9 | — |
| B1 short-CoT SFT | 74% | 21.5 | +9pp, **−56%** toks |
| B2 + GRPO | **75%** | **21.4** | +10pp, **−56%** toks |

**Gate B:** Pass (≥30% fewer tokens, no accuracy drop).

Artifacts: `artifacts/reports/kaggle_stage_b/gate_b.json`

---

## Stage C — discrete unreadable codes (original hypothesis)

Kernel: [reasonzipper-stage-c-resume](https://www.kaggle.com/code/zenitsu1agatsuma/reasonzipper-stage-c-resume) (GRPO fix rerun; full C1→C3)

**Method**
1. Add 128 special tokens `<|r000|>…<|r127|>`.
2. Offline codebook maps each verified reasoning block → 1–2 code IDs (opaque; no English).
3. Target format: `<think><|r014|><|r003|>…</think>\nERROR_LINE=…`
4. **C1:** QLoRA SFT on code sequences (150 steps) + trainable new token rows.
5. **C0:** Matched-budget **filler** control (repeat `<|r000|>`).
6. **C3:** GRPO on code student (40 steps). PEFT `TrainableTokensWrapper.bias=None`; SFT uses HF `compute_loss`, GRPO keeps `GRPOTrainer.compute_loss`.

### Results (n=100 val, self-generated codes)

| Method | Acc | Mean reasoning toks | Notes |
|---|---:|---:|---|
| A0 (cached) | 65% | 48.9 | verbose English think |
| B1 short English (cached) | 74% | 21.5 | readable short CoT |
| **C1 discrete codes** | **61%** | **4.21** | unreadable `<\|r…\|>` |
| **C0 filler control** | **56%** | **6.56** | same budget, meaningless codes |
| **C3 + GRPO** | **70%** | **4.81** | +5pp vs A0, **−90%** toks |

### Gate C checks

| Check | Result |
|---|---|
| Codes beat filler by >2pp? | **Yes** (C1 61% vs C0 56%; C3 70%) |
| ≥30% fewer toks vs A0 with ≤2pp acc drop? | **Yes** (C3 −90% toks, +5pp acc) |
| **gate_c_pass** | **True** |

### Interpretation

- Discrete codes compress far harder than short English (−90% vs −56% toks vs A0).
- C1 alone underperforms A0 (−4pp); **C3 GRPO recovers to 70%**, beating A0 and staying well under B1 length.
- Short English (B) still leads on raw accuracy (75%); codes win on compression with competitive accuracy after RL.

Artifacts: `artifacts/reports/kaggle_stage_c/gate_c.json`

---

## Training / loss summary

| Stage | Objective |
|---|---|
| B1 / C1 | Causal LM CE (SFT) on target completion |
| B2 / C3 | GRPO with correctness-first length reward; `beta=0` (no reasoning KL) |
| T4 note | Mixed precision **off** (FP32 LoRA); FP16 GradScaler × BF16 grads crashes |
| PEFT+TRL | Patch `TrainableTokensWrapper.bias`; do **not** replace GRPO `compute_loss` with HF Trainer |

---

## Next actions

1. Optional: stronger codebook (learned VQ / checkpoint aux heads) to close the gap to B2 accuracy.
2. Sealed ID/OOD/ProcessBench eval on the C3 candidate.

---

## Links

- Repo: https://github.com/Priyanshu-5257/reasonzipper  
- Plan: `docs/Reasoning_Compression_POC_Kaggle_2xT4.md`
