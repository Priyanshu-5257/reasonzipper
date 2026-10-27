"""Stage A on Kaggle 2xT4: generate data + thinking-on vs thinking-off baselines."""

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
DATA = WORK / "data"

# Gate-A preview size (plan says 250; keep configurable via env)
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "512"))


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])

    # Core ML stack is usually present on Kaggle; install project extras lightly.
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
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    # Prefer HF cache on working/temp
    env["HF_HOME"] = str(WORK / "hf_cache")
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    def py(args: list[str]) -> None:
        print("+", sys.executable, *args, flush=True)
        subprocess.check_call([sys.executable, *args], cwd=str(REPO), env=env)

    py(["scripts/check_environment.py", "--output", str(OUT / "environment.json")])
    py(["-m", "pytest", "-q"])

    # Full synthetic corpus per plan
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

    val = DATA / "val" / "examples.jsonl"
    common = [
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

    py([*common, "--thinking", "--output", str(OUT / "a0_thinking_on.jsonl")])
    py([*common, "--no-thinking", "--output", str(OUT / "a1_thinking_off.jsonl")])

    # Gate A summary
    def load_summary(path: Path) -> dict:
        return json.loads(path.read_text())

    a0 = load_summary(OUT / "a0_thinking_on.summary.json")
    a1 = load_summary(OUT / "a1_thinking_off.summary.json")
    gate = {
        "val_limit": VAL_LIMIT,
        "model": MODEL,
        "thinking_on": a0,
        "thinking_off": a1,
        "accuracy_delta_thinking_minus_direct": a0["accuracy"] - a1["accuracy"],
        "gate_a_pass": a0["accuracy"] > a1["accuracy"] + 0.02,
        "wall_clock_total_s": time.time() - t_start,
        "note": (
            "Pass if thinking-on materially beats thinking-off "
            "(threshold: +2pp absolute on this preview)."
        ),
    }
    (OUT / "gate_a.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)

    # Copy key reports to working root for easy download
    for name in [
        "gate_a.json",
        "a0_thinking_on.summary.json",
        "a1_thinking_off.summary.json",
        "environment.json",
    ]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, WORK / name)

    print("STAGE_A_DONE", flush=True)


if __name__ == "__main__":
    main()
