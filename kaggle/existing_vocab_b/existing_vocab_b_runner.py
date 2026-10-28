"""Approach B-v1 on Kaggle: verifier-trajectory → existing token IDs → SFT → GRPO.

No vocabulary expansion. Candidate set S is sampled from the original tokenizer.
Hidden-state P_phi selector is deferred; this pilot tests trajectory-guided
existing-ID distillation vs F4/F5/A.
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
OUT = WORK / "artifacts" / "reports" / "approach_b"
CKPT = WORK / "artifacts" / "checkpoints" / "approach_b"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
SFT_STEPS = int(os.environ.get("RZ_SFT_STEPS", "150"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "80"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))
CAND_K = int(os.environ.get("RZ_CANDIDATE_K", "512"))
MAX_CODES = int(os.environ.get("RZ_MAX_CODES", "8"))

F4_CACHE = {"accuracy": 0.7, "mean_reasoning_tokens": 0.0, "cached": True}
A_GRPO_CACHE = {"accuracy": 0.64, "mean_reasoning_tokens": 1.0, "cached": True}


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

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
            "peft",
            "trl",
            "bitsandbytes",
            "accelerate",
            "datasets",
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(WORK / "hf_cache")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ACCELERATE_MIXED_PRECISION"] = "no"
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    def py(args: list[str]) -> None:
        print("+", sys.executable, *args, flush=True)
        subprocess.check_call([sys.executable, *args], cwd=str(REPO), env=env)

    print("[approach_b] environment", flush=True)
    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    print("[approach_b] generate + validate", flush=True)
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

    traj = WORK / "trajectory_codes_train.jsonl"
    py(
        [
            "scripts/make_trajectory_codes.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(traj),
            "--model",
            MODEL,
            "--candidate-k",
            str(CAND_K),
            "--max-codes",
            str(MAX_CODES),
            "--limit",
            str(SFT_LIMIT),
        ]
    )

    val = DATA / "val" / "examples.jsonl"
    eval_common = [
        "scripts/eval_model.py",
        "--config",
        "configs/baseline.yaml",
        "--model",
        MODEL,
        "--data",
        str(val),
        "--limit",
        str(VAL_LIMIT),
        "--max-new-tokens",
        str(MAX_NEW),
        "--device",
        "cuda:0",
        "--load-in-4bit",
        "--thinking",
    ]

    sft_dir = CKPT / "b_sft"
    print(f"[approach_b] SFT steps={SFT_STEPS}", flush=True)
    py(
        [
            "scripts/train_sft.py",
            "--config",
            "configs/existing_vocab_b.yaml",
            "--train-file",
            str(traj),
            "--output-dir",
            str(sft_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(SFT_STEPS),
            "--limit",
            str(SFT_LIMIT),
            "--enable-thinking",
            "--require-original-vocab",
        ]
    )
    print("[approach_b] SFT eval (self-generated)", flush=True)
    py([*eval_common, "--adapter", str(sft_dir), "--output", str(OUT / "b_sft.jsonl")])

    grpo_dir = CKPT / "b_grpo"
    print(f"[approach_b] GRPO steps={GRPO_STEPS}", flush=True)
    py(
        [
            "scripts/train_grpo.py",
            "--config",
            "configs/existing_vocab_b.yaml",
            "--train-file",
            str(traj),
            "--sft-adapter",
            str(sft_dir),
            "--output-dir",
            str(grpo_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(GRPO_STEPS),
            "--limit",
            str(GRPO_LIMIT),
            "--require-original-vocab",
        ]
    )
    print("[approach_b] GRPO eval", flush=True)
    py([*eval_common, "--adapter", str(grpo_dir), "--output", str(OUT / "b_grpo.jsonl")])

    def load_sum(stem: str) -> dict:
        p = OUT / f"{stem}.summary.json"
        return json.loads(p.read_text()) if p.exists() else {}

    b_sft = load_sum("b_sft")
    b_grpo = load_sum("b_grpo")
    gate = {
        "approach": "B_v1_trajectory_existing_ids",
        "vocab_expanded": False,
        "note": (
            "Verifier-block signatures mapped into a frozen candidate set of "
            "original token IDs; no P_phi hidden-state selector yet."
        ),
        "f4_cached": F4_CACHE,
        "a_grpo_cached": A_GRPO_CACHE,
        "b_sft": b_sft,
        "b_grpo": b_grpo,
        "vs_f4_acc_delta": (b_grpo.get("accuracy") or 0) - F4_CACHE["accuracy"],
        "vs_a_grpo_acc_delta": (b_grpo.get("accuracy") or 0) - A_GRPO_CACHE["accuracy"],
        "pilot_gate": {
            "beats_f4": (b_grpo.get("accuracy") or 0) > F4_CACHE["accuracy"],
            "beats_a": (b_grpo.get("accuracy") or 0) > A_GRPO_CACHE["accuracy"],
        },
        "wall_clock_s": time.time() - t0,
        "config": {
            "sft_steps": SFT_STEPS,
            "grpo_steps": GRPO_STEPS,
            "candidate_k": CAND_K,
            "max_codes": MAX_CODES,
        },
    }
    (OUT / "gate_approach_b.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)
    print("APPROACH_B_DONE", flush=True)


if __name__ == "__main__":
    main()
