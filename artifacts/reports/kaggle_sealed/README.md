# Sealed-eval local backup

Kaggle **deletes remote run logs/outputs** when a kernel/session is deleted.
This folder is the local backup captured with `kaggle kernels logs -f` before cancel.

## Completed (C3 GRPO)
- `test_id` (n=1000): see `c3_grpo__test_id.summary.json`
- `test_ood` (n=2000): see `c3_grpo__test_ood.summary.json`

## Raw stream
- `sealed_eval_v2_live.log`

## Incomplete at cancel
- C3 × ProcessBench was mid-run (see `backup_manifest.json`).
