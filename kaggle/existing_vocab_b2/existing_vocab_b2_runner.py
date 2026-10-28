"""Approach B-full on Kaggle: collect checkpoints → P_phi assign → SFT → GRPO.

Existing vocabulary only. Selector discarded after B-2 targets are written.
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
OUT = WORK / "artifacts" / "reports" / "approach_b2"
CKPT = WORK / "artifacts" / "checkpoints" / "approach_b2"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
SFT_STEPS = int(os.environ.get("RZ_SFT_STEPS", "150"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
COLLECT_LIMIT = int(os.environ.get("RZ_COLLECT_LIMIT", "1000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "80"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))
CAND_K = int(os.environ.get("RZ_CANDIDATE_K", "512"))

F4 = {"accuracy": 0.7, "mean_reasoning_tokens": 0.0}
F5 = {"accuracy": 0.7, "mean_reasoning_tokens": 3.0}
A_GRPO = {"accuracy": 0.64, "mean_reasoning_tokens": 1.0}
B_V1 = {"accuracy": 0.64, "mean_reasoning_tokens": 3.12}


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

    print("[b2] environment", flush=True)
    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    print("[b2] generate + validate", flush=True)
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
    print(f"[b2] collect teacher checkpoints limit={COLLECT_LIMIT}", flush=True)
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
            "8",
        ]
    )

    traj = WORK / "b2_trajectory_train.jsonl"
    print("[b2] train P_phi + assign existing-token codes", flush=True)
    py(
        [
            "scripts/assign_trajectory_codes.py",
            "--checkpoints",
            str(ckpt_dir),
            "--output",
            str(traj),
            "--model",
            MODEL,
            "--candidate-k",
            str(CAND_K),
            "--epochs",
            "3",
            "--codes-per-block",
            "1",
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

    sft_dir = CKPT / "b2_sft"
    print(f"[b2] student SFT steps={SFT_STEPS}", flush=True)
    py(
        [
            "scripts/train_sft.py",
            "--config",
            "configs/existing_vocab_b2.yaml",
            "--train-file",
            str(traj),
            "--output-dir",
            str(sft_dir),
            "--model",
            MODEL,
            "--max-steps",
            str(SFT_STEPS),
            "--limit",
            str(min(SFT_LIMIT, COLLECT_LIMIT)),
            "--enable-thinking",
            "--require-original-vocab",
        ]
    )
    py([*eval_common, "--adapter", str(sft_dir), "--output", str(OUT / "b2_sft.jsonl")])

    grpo_dir = CKPT / "b2_grpo"
    print(f"[b2] GRPO steps={GRPO_STEPS}", flush=True)
    py(
        [
            "scripts/train_grpo.py",
            "--config",
            "configs/existing_vocab_b2.yaml",
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
    py([*eval_common, "--adapter", str(grpo_dir), "--output", str(OUT / "b2_grpo.jsonl")])

    def load_sum(stem: str) -> dict:
        p = OUT / f"{stem}.summary.json"
        return json.loads(p.read_text()) if p.exists() else {}

    b2_sft = load_sum("b2_sft")
    b2_grpo = load_sum("b2_grpo")
    assign_meta = {}
    meta_path = Path(str(traj) + "").with_suffix(".meta.json")
    # traj is *.jsonl → meta is *.meta.json via with_suffix on stem
    meta_path = traj.with_name(traj.stem + ".meta.json")
    if meta_path.exists():
        assign_meta = json.loads(meta_path.read_text())

    gate = {
        "approach": "B_full_hidden_state_P_phi",
        "vocab_expanded": False,
        "assign_meta": assign_meta,
        "baselines": {"f4": F4, "f5": F5, "a_grpo": A_GRPO, "b_v1": B_V1},
        "b2_sft": b2_sft,
        "b2_grpo": b2_grpo,
        "vs_f4": (b2_grpo.get("accuracy") or 0) - F4["accuracy"],
        "vs_f5": (b2_grpo.get("accuracy") or 0) - F5["accuracy"],
        "vs_a": (b2_grpo.get("accuracy") or 0) - A_GRPO["accuracy"],
        "vs_b_v1": (b2_grpo.get("accuracy") or 0) - B_V1["accuracy"],
        "pilot_gate": {
            "beats_f4": (b2_grpo.get("accuracy") or 0) > F4["accuracy"],
            "beats_f5": (b2_grpo.get("accuracy") or 0) > F5["accuracy"],
        },
        "wall_clock_s": time.time() - t0,
        "config": {
            "collect_limit": COLLECT_LIMIT,
            "sft_steps": SFT_STEPS,
            "grpo_steps": GRPO_STEPS,
            "candidate_k": CAND_K,
        },
    }
    (OUT / "gate_approach_b2.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)
    print("APPROACH_B2_DONE", flush=True)


if __name__ == "__main__":
    main()
