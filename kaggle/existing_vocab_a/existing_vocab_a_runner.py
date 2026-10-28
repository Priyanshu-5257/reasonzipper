"""Approach A on Kaggle 2xT4: answer-only SFT -> length GRPO (unchanged vocab).

See ReasonZipper_Existing_Vocabulary_Research_POC.md §6.
Uses access token account with GPU quota; pins to 1xT4.
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
OUT = WORK / "artifacts" / "reports" / "approach_a"
CKPT = WORK / "artifacts" / "checkpoints" / "approach_a"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
SFT_STEPS = int(os.environ.get("RZ_SFT_STEPS", "150"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "80"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))
# Historical A0 on same val slice (docs/experiment.md Stage A)
A0_CACHE = {
    "model": MODEL,
    "thinking": True,
    "n": VAL_LIMIT,
    "accuracy": 0.65,
    "malformed_rate": 0.01,
    "mean_reasoning_tokens": 48.91,
    "median_reasoning_tokens": 1,
    "wall_clock_s": 0.0,
    "max_new_tokens": 512,
    "seed": 42,
    "cached": True,
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

    print("[approach_a] environment", flush=True)
    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    print("[approach_a] generate + validate synthetic data", flush=True)
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

    answer_only = WORK / "answer_only_train.jsonl"
    py(
        [
            "scripts/make_answer_only.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(answer_only),
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
    ]

    print("[approach_a] F0/A0 cached baseline (thinking-on)", flush=True)
    (OUT / "f0_a0_thinking_on.summary.json").write_text(json.dumps(A0_CACHE, indent=2))

    # ---- F4 / A-SFT: answer-only, thinking off in chat template ----
    sft_dir = CKPT / "a_sft_answer_only"
    print(f"[approach_a] A-SFT answer-only steps={SFT_STEPS} limit={SFT_LIMIT}", flush=True)
    py(
        [
            "scripts/train_sft.py",
            "--config",
            "configs/existing_vocab_a.yaml",
            "--train-file",
            str(answer_only),
            "--output-dir",
            str(sft_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(SFT_STEPS),
            "--limit",
            str(SFT_LIMIT),
            "--no-enable-thinking",
            "--require-original-vocab",
        ]
    )

    print("[approach_a] F4 eval (thinking OFF)", flush=True)
    py(
        [
            *eval_common,
            "--no-thinking",
            "--adapter",
            str(sft_dir),
            "--output",
            str(OUT / "f4_answer_only.jsonl"),
        ]
    )
    print("[approach_a] A-SFT eval (thinking ON)", flush=True)
    py(
        [
            *eval_common,
            "--thinking",
            "--adapter",
            str(sft_dir),
            "--output",
            str(OUT / "a_sft_thinking_on.jsonl"),
        ]
    )

    # ---- A-RL GRPO ----
    grpo_dir = CKPT / "a_grpo"
    print(f"[approach_a] A-GRPO steps={GRPO_STEPS} limit={GRPO_LIMIT} B=128", flush=True)
    py(
        [
            "scripts/train_grpo.py",
            "--config",
            "configs/existing_vocab_a.yaml",
            "--train-file",
            str(answer_only),
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
    print("[approach_a] A-GRPO eval (thinking ON)", flush=True)
    py(
        [
            *eval_common,
            "--thinking",
            "--adapter",
            str(grpo_dir),
            "--output",
            str(OUT / "a_grpo_thinking_on.jsonl"),
        ]
    )

    def load_sum(name: str) -> dict:
        p = OUT / f"{name}.summary.json"
        return json.loads(p.read_text()) if p.exists() else {}

    f4 = load_sum("f4_answer_only")
    a_sft = load_sum("a_sft_thinking_on")
    a_grpo = load_sum("a_grpo_thinking_on")

    def tok_red(ref: dict, cur: dict) -> float | None:
        r = float(ref.get("mean_reasoning_tokens") or 0)
        c = float(cur.get("mean_reasoning_tokens") or 0)
        if r <= 0:
            return None
        return 1.0 - c / r

    gate = {
        "approach": "A_existing_vocab",
        "vocab_expanded": False,
        "f0_a0_cached": A0_CACHE,
        "f4_answer_only": f4,
        "a_sft_thinking_on": a_sft,
        "a_grpo_thinking_on": a_grpo,
        "vs_a0": {
            "acc_delta": (a_grpo.get("accuracy") or 0) - A0_CACHE["accuracy"],
            "reasoning_token_reduction": tok_red(A0_CACHE, a_grpo),
        },
        "vs_f4": {
            "acc_delta": (a_grpo.get("accuracy") or 0) - (f4.get("accuracy") or 0),
            "note": "F4 is thinking-off; A-GRPO is thinking-on self-generated prefixes",
        },
        "pilot_gate": {
            "beats_f4_accuracy": (a_grpo.get("accuracy") or 0) > (f4.get("accuracy") or 0),
            "shorter_than_a0_30pct": (tok_red(A0_CACHE, a_grpo) or 0) >= 0.30,
        },
        "wall_clock_s": time.time() - t0,
        "config": {
            "sft_steps": SFT_STEPS,
            "grpo_steps": GRPO_STEPS,
            "val_limit": VAL_LIMIT,
            "max_new_tokens": MAX_NEW,
            "length_budget_B": 128,
        },
    }
    (OUT / "gate_approach_a.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)
    print("APPROACH_A_DONE", flush=True)


if __name__ == "__main__":
    main()
