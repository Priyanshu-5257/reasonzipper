"""Stage B on Kaggle 2xT4: short-CoT SFT -> correctness-first GRPO + Gate B eval."""

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
OUT = WORK / "artifacts" / "reports"
CKPT = WORK / "artifacts" / "checkpoints"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
SFT_STEPS = int(os.environ.get("RZ_SFT_STEPS", "150"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "50"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "512"))


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
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    def py(args: list[str]) -> None:
        print("+", sys.executable, *args, flush=True)
        subprocess.check_call([sys.executable, *args], cwd=str(REPO), env=env)

    print("[stage_b] environment", flush=True)
    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    print("[stage_b] generate + validate data", flush=True)
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

    short_cot = WORK / "short_cot_train.jsonl"
    py(
        [
            "scripts/make_short_cot.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(short_cot),
            "--limit",
            str(SFT_LIMIT),
        ]
    )

    # ---- A0 reference on same val slice (thinking-on) ----
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
    print("[stage_b] A0 thinking-on baseline", flush=True)
    py([*eval_common, "--output", str(OUT / "a0_thinking_on.jsonl")])

    # ---- B1 SFT ----
    sft_dir = CKPT / "b1_sft"
    print(f"[stage_b] B1 SFT steps={SFT_STEPS} limit={SFT_LIMIT}", flush=True)
    py(
        [
            "scripts/train_sft.py",
            "--config",
            "configs/sft.yaml",
            "--train-file",
            str(short_cot),
            "--output-dir",
            str(sft_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(SFT_STEPS),
            "--limit",
            str(SFT_LIMIT),
        ]
    )
    print("[stage_b] B1 eval", flush=True)
    py(
        [
            *eval_common,
            "--adapter",
            str(sft_dir),
            "--output",
            str(OUT / "b1_short_cot.jsonl"),
        ]
    )

    # ---- B2 GRPO ----
    grpo_dir = CKPT / "b2_grpo"
    print(f"[stage_b] B2 GRPO steps={GRPO_STEPS} limit={GRPO_LIMIT}", flush=True)
    py(
        [
            "scripts/train_grpo.py",
            "--config",
            "configs/grpo.yaml",
            "--train-file",
            str(short_cot),
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
        ]
    )
    print("[stage_b] B2 eval", flush=True)
    py(
        [
            *eval_common,
            "--adapter",
            str(grpo_dir),
            "--output",
            str(OUT / "b2_grpo.jsonl"),
        ]
    )

    def summary(name: str) -> dict:
        return json.loads((OUT / f"{name}.summary.json").read_text())

    a0 = summary("a0_thinking_on")
    b1 = summary("b1_short_cot")
    b2 = summary("b2_grpo")

    def token_saving(ref: dict, cur: dict) -> float:
        r = float(ref.get("mean_reasoning_tokens", 0) or 0)
        c = float(cur.get("mean_reasoning_tokens", 0) or 0)
        if r <= 0:
            return 0.0
        return (r - c) / r

    gate = {
        "model": MODEL,
        "val_limit": VAL_LIMIT,
        "sft_steps": SFT_STEPS,
        "grpo_steps": GRPO_STEPS,
        "a0": a0,
        "b1": b1,
        "b2": b2,
        "b1_vs_a0": {
            "acc_delta": b1["accuracy"] - a0["accuracy"],
            "reasoning_token_reduction": token_saving(a0, b1),
        },
        "b2_vs_a0": {
            "acc_delta": b2["accuracy"] - a0["accuracy"],
            "reasoning_token_reduction": token_saving(a0, b2),
        },
        "poc_target": {
            "min_token_reduction": 0.30,
            "max_acc_drop_pp": 0.02,
        },
        "gate_b_pass": (
            (b2["accuracy"] >= a0["accuracy"] - 0.02)
            and (token_saving(a0, b2) >= 0.30)
        ),
        "wall_clock_total_s": time.time() - t0,
    }
    (OUT / "gate_b.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)

    for name in [
        "gate_b.json",
        "a0_thinking_on.summary.json",
        "b1_short_cot.summary.json",
        "b2_grpo.summary.json",
        "environment.json",
    ]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, WORK / name)

    print("STAGE_B_DONE", flush=True)


if __name__ == "__main__":
    main()
