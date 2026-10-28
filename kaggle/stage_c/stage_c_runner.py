"""Stage C on Kaggle 2xT4: discrete <|rXXX|> reasoning codes vs filler control."""

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
FILLER_STEPS = int(os.environ.get("RZ_FILLER_STEPS", "100"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "40"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))

# Cached A0 / B1 from Stage B Gate B (same task / model / val recipe).
A0 = {
    "model": MODEL,
    "thinking": True,
    "n": VAL_LIMIT,
    "accuracy": 0.65,
    "mean_reasoning_tokens": 48.91,
    "cached": True,
}
B1 = {
    "model": MODEL,
    "thinking": True,
    "n": VAL_LIMIT,
    "accuracy": 0.74,
    "mean_reasoning_tokens": 21.48,
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

    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    print("[stage_c] generate data", flush=True)
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

    tok_dir = CKPT / "tokenizer_codes"
    print("[stage_c] C1 add discrete tokens", flush=True)
    py(
        [
            "scripts/add_reasoning_tokens.py",
            "--config",
            "configs/codes.yaml",
            "--model",
            MODEL,
            "--output-dir",
            str(tok_dir),
        ]
    )

    code_train = WORK / "code_train.jsonl"
    filler_train = WORK / "filler_train.jsonl"
    codebook = CKPT / "codebook.json"
    py(
        [
            "scripts/build_code_labels.py",
            "--config",
            "configs/codes.yaml",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(code_train),
            "--filler-output",
            str(filler_train),
            "--codebook-out",
            str(codebook),
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
        "--tokenizer-dir",
        str(tok_dir),
    ]

    # ---- C1: code student SFT ----
    c1_dir = CKPT / "c1_codes"
    print(f"[stage_c] C1 code-SFT steps={SFT_STEPS}", flush=True)
    py(
        [
            "scripts/train_code_student.py",
            "--config",
            "configs/codes.yaml",
            "--train-file",
            str(code_train),
            "--tokenizer-dir",
            str(tok_dir),
            "--output-dir",
            str(c1_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(SFT_STEPS),
            "--limit",
            str(SFT_LIMIT),
        ]
    )
    print("[stage_c] C1 eval (self-generated codes)", flush=True)
    py([*eval_common, "--adapter", str(c1_dir), "--output", str(OUT / "c1_codes.jsonl")])

    # ---- C0 filler control ----
    c0_dir = CKPT / "c0_filler"
    print(f"[stage_c] C0 filler-SFT steps={FILLER_STEPS}", flush=True)
    py(
        [
            "scripts/train_code_student.py",
            "--config",
            "configs/codes.yaml",
            "--train-file",
            str(filler_train),
            "--tokenizer-dir",
            str(tok_dir),
            "--output-dir",
            str(c0_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(FILLER_STEPS),
            "--limit",
            str(SFT_LIMIT),
        ]
    )
    print("[stage_c] C0 filler eval", flush=True)
    py([*eval_common, "--adapter", str(c0_dir), "--output", str(OUT / "c0_filler.jsonl")])

    # ---- C3: GRPO on code student ----
    c3_dir = CKPT / "c3_grpo"
    print(f"[stage_c] C3 GRPO steps={GRPO_STEPS}", flush=True)
    py(
        [
            "scripts/train_grpo.py",
            "--config",
            "configs/grpo.yaml",
            "--train-file",
            str(code_train),
            "--sft-adapter",
            str(c1_dir),
            "--tokenizer-dir",
            str(tok_dir),
            "--output-dir",
            str(c3_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(GRPO_STEPS),
            "--limit",
            str(GRPO_LIMIT),
        ]
    )
    print("[stage_c] C3 eval", flush=True)
    py([*eval_common, "--adapter", str(c3_dir), "--output", str(OUT / "c3_grpo.jsonl")])

    def summary(stem: str) -> dict:
        return json.loads((OUT / f"{stem}.summary.json").read_text())

    c0 = summary("c0_filler")
    c1 = summary("c1_codes")
    c3 = summary("c3_grpo")

    def tok_red(ref: dict, cur: dict) -> float:
        r = float(ref.get("mean_reasoning_tokens") or 0)
        c = float(cur.get("mean_reasoning_tokens") or 0)
        return 0.0 if r <= 0 else (r - c) / r

    # Discrete codes should beat filler if the codebook carries info.
    codes_beat_filler = c1["accuracy"] > c0["accuracy"] + 0.02
    # Compression vs verbose A0 without big accuracy loss.
    vs_a0_ok = (c3["accuracy"] >= A0["accuracy"] - 0.02) and (
        tok_red(A0, c3) >= 0.30
    )

    gate = {
        "model": MODEL,
        "val_limit": VAL_LIMIT,
        "a0_cached": A0,
        "b1_shortcot_cached": B1,
        "c0_filler": c0,
        "c1_codes": c1,
        "c3_grpo": c3,
        "c1_vs_c0_acc_delta": c1["accuracy"] - c0["accuracy"],
        "c3_vs_a0": {
            "acc_delta": c3["accuracy"] - A0["accuracy"],
            "reasoning_token_reduction": tok_red(A0, c3),
        },
        "c3_vs_b1": {
            "acc_delta": c3["accuracy"] - B1["accuracy"],
            "reasoning_token_reduction": tok_red(B1, c3),
        },
        "codes_beat_filler": codes_beat_filler,
        "compression_vs_a0_ok": vs_a0_ok,
        "gate_c_pass": bool(codes_beat_filler and vs_a0_ok),
        "note": (
            "Gate C requires (1) learned codes beat matched-budget filler by >2pp "
            "and (2) >=30% fewer reasoning tokens vs A0 with <=2pp acc drop."
        ),
        "wall_clock_total_s": time.time() - t0,
    }
    (OUT / "gate_c.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)

    for name in [
        "gate_c.json",
        "c0_filler.summary.json",
        "c1_codes.summary.json",
        "c3_grpo.summary.json",
        "environment.json",
    ]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, WORK / name)

    print("STAGE_C_DONE", flush=True)


if __name__ == "__main__":
    main()
