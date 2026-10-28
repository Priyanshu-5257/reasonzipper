# reasonzipper — Experiment Log

**Model:** Qwen/Qwen3-1.7B · **Hardware:** Kaggle 2×T4 (train pinned to 1×T4) · **Task:** first-error line localization (`ERROR_LINE=<N|NONE>`)

## Hypothesis

A small LLM can replace long English chain-of-thought with a short sequence of **opaque discrete tokens** (`<|rXXX|>`) that only the model uses as reasoning, while keeping answer accuracy nearly the same and cutting reasoning tokens substantially.

## Stages

| Stage | Question | Status |
|---|---|---|
| **A** | Does thinking help on this task? | **Pass** |
| **B** | Can *short English* CoT compress reasoning? | **Pass** (control / baseline) |
| **C** | Can *unreadable discrete codes* do the same? | **Partial** (C1/C0 done; C3 GRPO failed) |

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

Kernel: [reasonzipper-stage-c](https://www.kaggle.com/code/zenitsu1agatsuma/reasonzipper-stage-c) (v2)

**Method**
1. Add 128 special tokens `<|r000|>…<|r127|>`.
2. Offline codebook maps each verified reasoning block → 1–2 code IDs (opaque; no English).
3. Target format: `<think><|r014|><|r003|>…</think>\nERROR_LINE=…`
4. **C1:** QLoRA SFT on code sequences (150 steps) + trainable new token rows.
5. **C0:** Matched-budget **filler** control (repeat `<|r000|>`).
6. **C3:** GRPO on code student — **failed** (TRL fused LM-head × PEFT `TrainableTokensWrapper` has no `.bias`).

### Results (n=100 val, self-generated codes)

| Method | Acc | Mean reasoning toks | Notes |
|---|---:|---:|---|
| A0 (cached) | 65% | 48.9 | verbose English think |
| B1 short English (cached) | 74% | 21.5 | readable short CoT |
| **C1 discrete codes** | **61%** | **4.21** | unreadable `<\|r…\|>` |
| **C0 filler control** | **56%** | **6.56** | same budget, meaningless codes |
| C3 GRPO | — | — | not completed |

### Gate C checks

| Check | Result |
|---|---|
| Codes beat filler by >2pp? | **Yes** (61% vs 56%, +5pp) |
| ≥30% fewer toks vs A0 with ≤2pp acc drop? | **No** (−4pp vs A0; toks −91%) |
| **gate_c_pass** | **False** (accuracy regression vs A0) |

### Interpretation

- Discrete codes **are shorter** than A0/B1 and **beat the filler control**, so the codebook is not pure noise.
- They do **not** yet match A0/B1 accuracy on this preview; short English (Stage B) remains stronger.
- C3 length/correctness RL did not run — fix GRPO fused-head path and rerun before claiming full Stage C.

Artifacts: `artifacts/reports/kaggle_stage_c/gate_c.json`

---

## Training / loss summary

| Stage | Objective |
|---|---|
| B1 / C1 | Causal LM CE (SFT) on target completion |
| B2 / C3 | GRPO with correctness-first length reward; `beta=0` (no reasoning KL) |
| T4 note | Mixed precision **off** (FP32 LoRA); FP16 GradScaler × BF16 grads crashes |

---

## Next actions

1. Patch GRPO the same way as code-SFT (disable fused LM-head) and finish C3.
2. Optional: stronger codebook (learned VQ / checkpoint aux heads) if C1 plateau.
3. Sealed ID/OOD/ProcessBench eval once a Stage C candidate is stable.

---

## Links

- Repo: https://github.com/Priyanshu-5257/reasonzipper  
- Plan: `docs/Reasoning_Compression_POC_Kaggle_2xT4.md`
