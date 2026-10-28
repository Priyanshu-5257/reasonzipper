"""F5 matched-budget filler control on Kaggle (existing vocab only).

Same SFT/GRPO budgets as Approach A; think span is repeated '.' tokens.
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
OUT = WORK / "artifacts" / "reports" / "approach_f5"
CKPT = WORK / "artifacts" / "checkpoints" / "approach_f5"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
SFT_STEPS = int(os.environ.get("RZ_SFT_STEPS", "150"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "80"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))
N_FILLER = int(os.environ.get("RZ_N_FILLER", "16"))
FILLER_TEXT = os.environ.get("RZ_FILLER_TEXT", ".")

# From Approach A gate (same val slice)
F4_CACHE = {
    "model": MODEL,
    "thinking": False,
    "n": VAL_LIMIT,
    "accuracy": 0.7,
    "malformed_rate": 0.0,
    "mean_reasoning_tokens": 0.0,
    "median_reasoning_tokens": 0,
    "cached": True,
    "source": "approach_a/gate_approach_a.json",
}
A_GRPO_CACHE = {
    "model": MODEL,
    "thinking": True,
    "n": VAL_LIMIT,
    "accuracy": 0.64,
    "malformed_rate": 0.0,
    "mean_reasoning_tokens": 1.0,
    "median_reasoning_tokens": 1,
    "cached": True,
    "source": "approach_a/gate_approach_a.json",
}


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

    print("[f5] environment", flush=True)
    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    print("[f5] generate + validate", flush=True)
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

    filler = WORK / "filler_train.jsonl"
    py(
        [
            "scripts/make_filler_thinking.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(filler),
            "--model",
            MODEL,
            "--filler-text",
            FILLER_TEXT,
            "--n-filler",
            str(N_FILLER),
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

    sft_dir = CKPT / "f5_sft"
    print(f"[f5] SFT steps={SFT_STEPS} n_filler={N_FILLER}", flush=True)
    py(
        [
            "scripts/train_sft.py",
            "--config",
            "configs/existing_vocab_f5.yaml",
            "--train-file",
            str(filler),
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
    print("[f5] SFT eval", flush=True)
    py([*eval_common, "--adapter", str(sft_dir), "--output", str(OUT / "f5_sft.jsonl")])

    grpo_dir = CKPT / "f5_grpo"
    print(f"[f5] GRPO steps={GRPO_STEPS}", flush=True)
    py(
        [
            "scripts/train_grpo.py",
            "--config",
            "configs/existing_vocab_f5.yaml",
            "--train-file",
            str(filler),
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
    print("[f5] GRPO eval", flush=True)
    py([*eval_common, "--adapter", str(grpo_dir), "--output", str(OUT / "f5_grpo.jsonl")])

    def load_sum(stem: str) -> dict:
        p = OUT / f"{stem}.summary.json"
        return json.loads(p.read_text()) if p.exists() else {}

    f5_sft = load_sum("f5_sft")
    f5_grpo = load_sum("f5_grpo")
    gate = {
        "control": "F5_matched_filler",
        "vocab_expanded": False,
        "n_filler": N_FILLER,
        "filler_text": FILLER_TEXT,
        "f4_cached": F4_CACHE,
        "a_grpo_cached": A_GRPO_CACHE,
        "f5_sft": f5_sft,
        "f5_grpo": f5_grpo,
        "vs_f4_acc_delta": (f5_grpo.get("accuracy") or 0) - F4_CACHE["accuracy"],
        "vs_a_grpo_acc_delta": (f5_grpo.get("accuracy") or 0) - A_GRPO_CACHE["accuracy"],
        "wall_clock_s": time.time() - t0,
        "matched_budgets": {
            "sft_steps": SFT_STEPS,
            "grpo_steps": GRPO_STEPS,
            "length_budget_B": 128,
        },
    }
    (OUT / "gate_f5.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)
    print("F5_DONE", flush=True)


if __name__ == "__main__":
    main()
