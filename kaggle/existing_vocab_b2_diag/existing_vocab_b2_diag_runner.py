"""Cheap B2′ diagnostic on Kaggle — NO SFT / NO GRPO.

Engineering path:
  generate data → collect consecutive checkpoints → V1 compressed-AR select
  → answer-label control JSONL → random control → write gate JSON.

Proceed to SFT only if mean ΔNLL vs no-code is clearly positive and beats
random / label-shortcut diagnostics on the same sample.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/Priyanshu-5257/reasonzipper.git"
WORK = Path("/kaggle/working")
REPO = WORK / "reasonzipper"
OUT = WORK / "artifacts" / "reports" / "approach_b2_diag"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
COLLECT_LIMIT = int(os.environ.get("RZ_COLLECT_LIMIT", "64"))
CAND_K = int(os.environ.get("RZ_CANDIDATE_K", "32"))
N_CODES = int(os.environ.get("RZ_N_CODES", "3"))
ALPHA = float(os.environ.get("RZ_ALPHA", "0.0"))
BETA = float(os.environ.get("RZ_BETA", "1.0"))


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pyyaml",
            "pytest",
            "bitsandbytes",
            "accelerate",
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(WORK / "hf_cache")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    def py(args: list[str]) -> None:
        print("+", sys.executable, *args, flush=True)
        subprocess.check_call([sys.executable, *args], cwd=str(REPO), env=env)

    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    py(
        [
            "scripts/generate_math_errors.py",
            "--config",
            "configs/baseline.yaml",
            "--output-dir",
            str(DATA),
        ]
    )
    py(["scripts/validate_dataset.py", "--data-dir", str(DATA)])

    ckpt_dir = WORK / "teacher_checkpoints"
    py(
        [
            "scripts/collect_checkpoints.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(ckpt_dir),
            "--model",
            MODEL,
            "--limit",
            str(COLLECT_LIMIT),
            "--max-codes",
            str(N_CODES),
        ]
    )

    traj = WORK / "b2_v1_codes.jsonl"
    py(
        [
            "scripts/select_compressed_codes.py",
            "--checkpoints",
            str(ckpt_dir),
            "--output",
            str(traj),
            "--model",
            MODEL,
            "--candidate-k",
            str(CAND_K),
            "--n-codes",
            str(N_CODES),
            "--limit",
            str(COLLECT_LIMIT),
            "--alpha",
            str(ALPHA),
            "--beta",
            str(BETA),
        ]
    )

    traj_rand = WORK / "b2_random_codes.jsonl"
    py(
        [
            "scripts/select_compressed_codes.py",
            "--checkpoints",
            str(ckpt_dir),
            "--output",
            str(traj_rand),
            "--model",
            MODEL,
            "--candidate-k",
            str(CAND_K),
            "--n-codes",
            str(N_CODES),
            "--limit",
            str(COLLECT_LIMIT),
            "--alpha",
            str(ALPHA),
            "--beta",
            str(BETA),
            "--control",
            "random",
        ]
    )

    label_out = WORK / "b2_answer_label_codes.jsonl"
    py(
        [
            "scripts/make_answer_label_codes.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(label_out),
            "--model",
            MODEL,
            "--candidate-k",
            str(max(CAND_K, 64)),
            "--n-codes",
            "1",
            "--limit",
            str(COLLECT_LIMIT),
        ]
    )

    def load_meta(path: Path) -> dict:
        m = path.with_name(path.stem + ".meta.json")
        return json.loads(m.read_text()) if m.exists() else {}

    v1 = load_meta(traj)
    rnd = load_meta(traj_rand)
    lab = load_meta(label_out)
    mean_delta = float(v1.get("mean_delta_nll_vs_no_code") or 0.0)
    gate = {
        "approach": "B2_diag_compressed_ar_v1",
        "hypothesis": "compressed-token computation (not claimed reasoning language)",
        "vocab_expanded": False,
        "sft_run": False,
        "grpo_run": False,
        "v1_select": v1,
        "random_control": rnd,
        "answer_label_control": lab,
        "diagnostic_signal": {
            "mean_delta_nll_vs_no_code": mean_delta,
            "positive_vs_no_code": mean_delta > 0.05,
            "note": (
                "Positive ΔNLL means selected codes reduce answer NLL vs closing "
                "think with no codes. Required before SFT; not deployable accuracy."
            ),
        },
        "next_gate": [
            "If diagnostic_signal.positive_vs_no_code: oracle teacher-forced codes on val",
            "If oracle helps: SFT + self-generated eval vs F4/F5 + answer-label control",
            "GRPO only if self-generated SFT shows useful accuracy/compression trade-off",
            "Transfer method to GSM8K before strong claims (small answer space here)",
        ],
        "wall_clock_s": time.time() - t0,
        "config": {
            "collect_limit": COLLECT_LIMIT,
            "candidate_k": CAND_K,
            "n_codes": N_CODES,
            "alpha": ALPHA,
            "beta": BETA,
        },
    }
    (OUT / "gate_b2_diag.json").write_text(json.dumps(gate, indent=2))
    # Also copy outputs for download
    for src in (traj, traj_rand, label_out):
        if src.exists():
            shutil.copy2(src, OUT / src.name)
            meta = src.with_name(src.stem + ".meta.json")
            if meta.exists():
                shutil.copy2(meta, OUT / meta.name)
    print(json.dumps(gate, indent=2), flush=True)
    print("APPROACH_B2_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
