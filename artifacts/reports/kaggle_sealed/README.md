# Sealed eval backups & findings

Kaggle deletes remote logs/outputs when a session is deleted — keep local copies here.

## Headline numbers
See `sealed_findings_summary.json` and `docs/experiment.md` (Sealed eval section).

## Reasoning examples
`examples_a0_vs_c3.json` — curated A0 English vs C3 `<|rXXX|>` generations.

## C3 sealed ID/OOD (from earlier run)
- `c3_grpo__test_id.summary.json`
- `c3_grpo__test_ood.summary.json`
- `partial_from_logs.json`

A0 probe artifacts (OOD-50 / PB-50) live in `../kaggle_sealed_a0/`.
