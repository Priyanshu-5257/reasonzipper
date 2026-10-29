# B2 oracle-code diagnostic (val n=100)

Kernel: [reasonzipper-existing-b2-oracle](https://www.kaggle.com/code/priyanshu3rdvivek/reasonzipper-existing-b2-oracle)  
Commit at run: `16098ab` (+ fallback reporting `f553b1d` lineage)

## Result

**Oracle gate: FAIL.** Forced selected codes do not beat random codes on answer accuracy, and are far from F4/F5 (70%).

| Condition | Accuracy | Notes |
|---|---:|---|
| func_only oracle | 7% | answer-conditioned V1 select, then force codes |
| geom+func oracle | 7% | |
| random codes | **13%** | beats selected |
| answer-label | 0% | |
| empty `<think>` | 0% | |
| F4 / F5 reference | 70% | prior existing-vocab SFT/GRPO |

Also: ~74% malformed under `func_only` (model often re-opens thinking / prose and hits `max_new_tokens=32` without `ERROR_LINE=...`).

Interpretation (narrow): train-time ΔNLL gains from answer-conditioned selection did **not** translate into usable forced-prefix accuracy on frozen Qwen3-1.7B. Do **not** proceed to SFT/GRPO on this recipe without fixing the oracle protocol (format / length / selection).

## Example outputs (`examples_preview.json`)

Selected “codes” are existing-vocab fragments, not readable math.

### Correct

- gold=`NONE` → `ERROR_LINE=NONE` after forced `但从 вообще extracting` (still re-emits `</think>` then prose).
- gold=`3` → bare `ERROR_LINE=3` after forced `NotEmptyӘ❙`.

### Wrong

- gold=`NONE` → `ERROR_LINE=2` after `但从 extracting❙`.
- gold=`NONE` → `ERROR_LINE=2` after ` Modification Field/chart`.

### Malformed (dominant failure)

Forced `_lim` then generation starts explaining line-by-line and is truncated before any `ERROR_LINE=`.

### Random correct (contrast)

Forced `tyty` → `ERROR_LINE=2` (gold=2).

See `examples_preview.json` for full forced prefixes + generations.
