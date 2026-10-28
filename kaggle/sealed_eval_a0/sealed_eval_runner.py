"""Sealed ID / OOD / ProcessBench eval for frozen Stage C candidates.

Loads adapters from /kaggle/input (dataset) when present; always regenerates
synthetic data with the same seed as training (test splits were never used for
tuning — only val was used for gates).
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
OUT = WORK / "artifacts" / "reports" / "sealed"
CKPT = WORK / "artifacts" / "checkpoints"
DATA = WORK / "data"
INPUT_ROOT = Path("/kaggle/input")

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "256"))
# Optional global caps (apply to every model if set)
LIMIT_ID = os.environ.get("RZ_LIMIT_ID")
LIMIT_OOD = os.environ.get("RZ_LIMIT_OOD")
LIMIT_PB = os.environ.get("RZ_LIMIT_PROCESSBENCH")
# A0 quick probes: random subsample (seed=42). Override via env.
A0_SAMPLE = int(os.environ.get("RZ_A0_SAMPLE", "50"))
A0_LIMIT_ID = os.environ.get("RZ_A0_LIMIT_ID")
A0_LIMIT_OOD = os.environ.get("RZ_A0_LIMIT_OOD")
A0_LIMIT_PB = os.environ.get("RZ_A0_LIMIT_PROCESSBENCH")
SEED = int(os.environ.get("RZ_SEED", "42"))
MODELS = [
    m.strip()
    for m in os.environ.get("RZ_SEALED_MODELS", "a0").split(",")
    if m.strip()
]
SPLIT_NAMES = [
    s.strip()
    for s in os.environ.get("RZ_SEALED_SPLITS", "processbench").split(",")
    if s.strip()
]


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def find_dir(name: str) -> Path | None:
    if not INPUT_ROOT.exists():
        return None
    for p in INPUT_ROOT.rglob(name):
        if not p.is_dir():
            continue
        adapter = p / "adapter_model.safetensors"
        if name.startswith("c") and name != "tokenizer_codes":
            if adapter.exists() and adapter.stat().st_size > 1000:
                return p
        else:
            if (p / "tokenizer.json").exists() or (p / "reasoning_tokens.json").exists():
                return p
    return None


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[sealed] copied {src} -> {dst}", flush=True)


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

    print("[sealed] scanning /kaggle/input ...", flush=True)
    if INPUT_ROOT.exists():
        for p in sorted(INPUT_ROOT.iterdir()):
            print(f"  input: {p}", flush=True)

    src_c3 = find_dir("c3_grpo")
    src_c1 = find_dir("c1_codes")
    src_tok = find_dir("tokenizer_codes")
    print(f"[sealed] c3={src_c3} c1={src_c1} tok={src_tok}", flush=True)

    tok_dir = CKPT / "tokenizer_codes"
    c1_dir = CKPT / "c1_codes"
    c3_dir = CKPT / "c3_grpo"
    if src_tok:
        copy_tree(src_tok, tok_dir)
    if src_c1:
        copy_tree(src_c1, c1_dir)
    if src_c3:
        copy_tree(src_c3, c3_dir)

    if "c3" in MODELS and not (c3_dir / "adapter_model.safetensors").exists():
        raise SystemExit("C3 adapter missing — attach reasonzipper-stage-c-ckpts dataset")
    if "c1" in MODELS and not (c1_dir / "adapter_model.safetensors").exists():
        raise SystemExit("C1 adapter missing — attach reasonzipper-stage-c-ckpts dataset")
    if any(m in MODELS for m in ("c1", "c3")) and not (tok_dir / "tokenizer.json").exists():
        # fallback: tokenizer often saved next to adapter
        for cand in (c3_dir, c1_dir):
            if (cand / "tokenizer.json").exists():
                copy_tree(cand, tok_dir)
                break
        if not (tok_dir / "tokenizer.json").exists():
            raise SystemExit("tokenizer_codes missing")

    print("[sealed] generate synthetic data (seed from config; test never used for tuning)", flush=True)
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

    pb_dir = DATA / "processbench"
    print("[sealed] prep ProcessBench", flush=True)
    py(["scripts/prep_processbench.py", "--output-dir", str(pb_dir)])

    splits_all = {
        "test_id": (DATA / "test_id" / "examples.jsonl", LIMIT_ID),
        "test_ood": (DATA / "test_ood" / "examples.jsonl", LIMIT_OOD),
        "processbench": (pb_dir / "examples.jsonl", LIMIT_PB),
    }
    splits = {k: splits_all[k] for k in SPLIT_NAMES if k in splits_all}
    if not splits:
        raise SystemExit(f"No valid splits in {SPLIT_NAMES}")
    print(f"[sealed] models={MODELS} splits={list(splits)}", flush=True)

    def limits_for(tag: str) -> dict[str, str | None]:
        if tag.startswith("a0"):
            return {
                "test_id": LIMIT_ID or A0_LIMIT_ID,
                "test_ood": LIMIT_OOD or A0_LIMIT_OOD,
                "processbench": LIMIT_PB or A0_LIMIT_PB,
            }
        return {
            "test_id": LIMIT_ID,
            "test_ood": LIMIT_OOD,
            "processbench": LIMIT_PB,
        }

    def eval_one(tag: str, adapter: Path | None, tokenizer_dir: Path | None) -> None:
        lims = limits_for(tag)
        print(f"[sealed] limits for {tag}: {lims}", flush=True)
        for split, (path, _) in splits.items():
            out = OUT / f"{tag}__{split}.jsonl"
            cmd = [
                "scripts/eval_model.py",
                "--config",
                "configs/baseline.yaml",
                "--model",
                MODEL,
                "--data",
                str(path),
                "--max-new-tokens",
                str(MAX_NEW),
                "--device",
                "cuda:0",
                "--load-in-4bit",
                "--thinking",
                "--seed",
                str(SEED),
                "--log-reasoning-every",
                "10",
                "--output",
                str(out),
            ]
            if tag.startswith("a0") and A0_SAMPLE > 0:
                cmd.extend(["--sample", str(A0_SAMPLE)])
            lim = lims.get(split)
            if lim:
                cmd.extend(["--limit", str(int(lim))])
            if adapter is not None:
                cmd.extend(["--adapter", str(adapter)])
            if tokenizer_dir is not None:
                cmd.extend(["--tokenizer-dir", str(tokenizer_dir)])
            print(f"[sealed] eval {tag} / {split}", flush=True)
            py(cmd)

    # Order follows MODELS (default: c3, c1, a0)
    for m in MODELS:
        if m == "c3":
            eval_one("c3_grpo", adapter=c3_dir, tokenizer_dir=tok_dir)
        elif m == "c1":
            eval_one("c1_codes", adapter=c1_dir, tokenizer_dir=tok_dir)
        elif m == "a0":
            eval_one("a0_thinking", adapter=None, tokenizer_dir=None)
        else:
            print(f"[sealed] unknown model tag {m}, skip", flush=True)

    py(
        [
            "scripts/summarize_sealed.py",
            "--reports-dir",
            str(OUT),
            "--output",
            str(OUT / "sealed_report.json"),
        ]
    )
    print(f"[sealed] wall_clock_s={time.time() - t0:.1f}", flush=True)
    print("SEALED_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
