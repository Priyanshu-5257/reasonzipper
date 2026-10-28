"""Stage C on Kaggle 2xT4: discrete <|rXXX|> codes.

Resume mode (default): reuse C1/C0 checkpoints from prior kernel output mounted at
/kaggle/input/..., then only run GRPO + final eval / gate.
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
OUT = WORK / "artifacts" / "reports"
CKPT = WORK / "artifacts" / "checkpoints"
DATA = WORK / "data"
INPUT_ROOT = Path("/kaggle/input")

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
SFT_STEPS = int(os.environ.get("RZ_SFT_STEPS", "150"))
FILLER_STEPS = int(os.environ.get("RZ_FILLER_STEPS", "100"))
SFT_LIMIT = int(os.environ.get("RZ_SFT_LIMIT", "4000"))
GRPO_STEPS = int(os.environ.get("RZ_GRPO_STEPS", "40"))
GRPO_LIMIT = int(os.environ.get("RZ_GRPO_LIMIT", "512"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))
# Resume from last Stage C outputs when C1 adapter is available via kernel_sources.
RESUME = os.environ.get("RZ_RESUME", "1") == "1"

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
# From last Stage C run (before GRPO crash)
C1_CACHE = {
    "model": MODEL,
    "thinking": True,
    "n": VAL_LIMIT,
    "accuracy": 0.61,
    "mean_reasoning_tokens": 4.21,
    "median_reasoning_tokens": 4,
    "malformed_rate": 0.0,
    "cached": True,
}
C0_CACHE = {
    "model": MODEL,
    "thinking": True,
    "n": VAL_LIMIT,
    "accuracy": 0.56,
    "mean_reasoning_tokens": 6.56,
    "cached": True,
}


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def find_dir(name: str) -> Path | None:
    """Find a checkpoint directory under /kaggle/input or local CKPT."""
    candidates = [
        CKPT / name,
        WORK / "artifacts" / "checkpoints" / name,
    ]
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob(name):
            if p.is_dir() and (
                (p / "adapter_config.json").exists()
                or (p / "reasoning_tokens.json").exists()
                or (p / "tokenizer.json").exists()
                or (p / "adapter_model.safetensors").exists()
            ):
                candidates.insert(0, p)
    for c in candidates:
        if c.is_dir():
            # require non-empty adapter when expecting peft dir
            adapter = c / "adapter_model.safetensors"
            if name.startswith("c") and name != "tokenizer_codes":
                if adapter.exists() and adapter.stat().st_size > 1000:
                    return c
                if (c / "adapter_config.json").exists() and not adapter.exists():
                    continue
            else:
                return c
    return None


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[resume] copied {src} -> {dst}", flush=True)


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

    # Discover resume checkpoints from prior kernel output (kernel_sources).
    print("[stage_c] scanning /kaggle/input for resume checkpoints...", flush=True)
    if INPUT_ROOT.exists():
        for p in sorted(INPUT_ROOT.iterdir()):
            print(f"  input: {p}", flush=True)

    src_c1 = find_dir("c1_codes")
    src_c0 = find_dir("c0_filler")
    src_tok = find_dir("tokenizer_codes")
    can_resume = bool(RESUME and src_c1 and src_tok)
    print(
        f"[stage_c] RESUME={RESUME} c1={src_c1} c0={src_c0} tok={src_tok} "
        f"can_resume={can_resume}",
        flush=True,
    )

    print("[stage_c] generate data (+ code labels)", flush=True)
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
    c1_dir = CKPT / "c1_codes"
    c0_dir = CKPT / "c0_filler"
    codebook = CKPT / "codebook.json"
    code_train = WORK / "code_train.jsonl"
    filler_train = WORK / "filler_train.jsonl"

    if can_resume and src_tok is not None and src_c1 is not None:
        copy_tree(src_tok, tok_dir)
        copy_tree(src_c1, c1_dir)
        if src_c0 is not None:
            copy_tree(src_c0, c0_dir)
        # Rebuild labels with same codebook if present, else fresh codebook
        cb_src = None
        for cand in INPUT_ROOT.rglob("codebook.json"):
            cb_src = cand
            break
        if cb_src is not None:
            shutil.copy2(cb_src, codebook)
            print(f"[resume] codebook from {cb_src}", flush=True)
        build_cmd = [
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
        if codebook.exists():
            build_cmd.extend(["--codebook-in", str(codebook)])
        py(build_cmd)
        # Use cached eval summaries when present
        (OUT / "c1_codes.summary.json").write_text(json.dumps(C1_CACHE, indent=2))
        (OUT / "c0_filler.summary.json").write_text(json.dumps(C0_CACHE, indent=2))
        print("[resume] skipped C1/C0 training+eval (using prior checkpoints + cached metrics)", flush=True)
    else:
        print("[stage_c] full run (no usable resume checkpoint)", flush=True)
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
        py([*eval_common, "--adapter", str(c1_dir), "--output", str(OUT / "c1_codes.jsonl")])
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
        py([*eval_common, "--adapter", str(c0_dir), "--output", str(OUT / "c0_filler.jsonl")])

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

    # ---- C3 GRPO (the step that previously crashed) ----
    c3_dir = CKPT / "c3_grpo"
    print(f"[stage_c] C3 GRPO steps={GRPO_STEPS} (fused-head patched)", flush=True)
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
        path = OUT / f"{stem}.summary.json"
        return json.loads(path.read_text())

    c0 = summary("c0_filler")
    c1 = summary("c1_codes")
    c3 = summary("c3_grpo")

    def tok_red(ref: dict, cur: dict) -> float:
        r = float(ref.get("mean_reasoning_tokens") or 0)
        c = float(cur.get("mean_reasoning_tokens") or 0)
        return 0.0 if r <= 0 else (r - c) / r

    codes_beat_filler = c1["accuracy"] > c0["accuracy"] + 0.02
    vs_a0_ok = (c3["accuracy"] >= A0["accuracy"] - 0.02) and (tok_red(A0, c3) >= 0.30)

    gate = {
        "model": MODEL,
        "resumed": can_resume,
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
