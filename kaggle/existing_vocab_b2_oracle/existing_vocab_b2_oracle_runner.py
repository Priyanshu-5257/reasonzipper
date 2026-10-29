"""B2 oracle-code diagnostic on held-out val — NO SFT / NO GRPO.

Pipeline:
  generate data → collect val checkpoints → V1 select (func + geom+func)
  → random + answer-label controls → teacher-force codes → generate answer
  → compare accuracy vs empty-think / random / label and F4/F5 references.

Oracle accuracy is privileged if codes were selected with gold answers.
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
OUT = WORK / "artifacts" / "reports" / "approach_b2_oracle"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
VAL_LIMIT = int(os.environ.get("RZ_VAL_LIMIT", "100"))
CAND_K = int(os.environ.get("RZ_CANDIDATE_K", "32"))
N_CODES = int(os.environ.get("RZ_N_CODES", "3"))
MAX_NEW = int(os.environ.get("RZ_MAX_NEW_TOKENS", "32"))
F4_ACC = 0.70
F5_ACC = 0.70


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

    val = DATA / "val" / "examples.jsonl"
    ckpt_dir = WORK / "val_teacher_checkpoints"
    py(
        [
            "scripts/collect_checkpoints.py",
            "--input",
            str(val),
            "--output",
            str(ckpt_dir),
            "--model",
            MODEL,
            "--limit",
            str(VAL_LIMIT),
            "--max-codes",
            str(N_CODES),
        ]
    )

    def select(out: Path, *, alpha: float, beta: float, control: str = "none") -> None:
        py(
            [
                "scripts/select_compressed_codes.py",
                "--checkpoints",
                str(ckpt_dir),
                "--output",
                str(out),
                "--model",
                MODEL,
                "--candidate-k",
                str(CAND_K),
                "--n-codes",
                str(N_CODES),
                "--limit",
                str(VAL_LIMIT),
                "--alpha",
                str(alpha),
                "--beta",
                str(beta),
                "--control",
                control,
                "--verify-chat-template",
            ]
        )

    traj_func = WORK / "val_func_only.jsonl"
    traj_geom = WORK / "val_geom_plus_func.jsonl"
    traj_rand = WORK / "val_random.jsonl"
    select(traj_func, alpha=0.0, beta=1.0)
    select(traj_geom, alpha=1.0, beta=1.0)
    select(traj_rand, alpha=0.0, beta=1.0, control="random")

    # Answer-label control aligned to checkpoint IDs
    ckpt_index = json.loads((ckpt_dir / "index.json").read_text())
    ckpt_ids = {it["id"] for it in ckpt_index["items"]}
    val_aligned = WORK / "val_aligned.jsonl"
    with val.open(encoding="utf-8") as fin, val_aligned.open("w", encoding="utf-8") as fout:
        n_al = 0
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in ckpt_ids:
                fout.write(line if line.endswith("\n") else line + "\n")
                n_al += 1
                if n_al >= VAL_LIMIT:
                    break
    label_out = WORK / "val_answer_label.jsonl"
    py(
        [
            "scripts/make_answer_label_codes.py",
            "--input",
            str(val_aligned),
            "--output",
            str(label_out),
            "--model",
            MODEL,
            "--candidate-k",
            str(max(CAND_K, 64)),
            "--n-codes",
            "1",
            "--limit",
            str(VAL_LIMIT),
        ]
    )
    # Attach prompts from checkpoint index for oracle eval
    # make_answer_label already writes prompt; score via codes-jsonl optional.
    # Ensure label rows have prompts matching select format — already do.

    def oracle(codes: Path, stem: str, *, mode: str = "codes") -> dict:
        out = OUT / f"oracle_{stem}.jsonl"
        py(
            [
                "scripts/eval_oracle_codes.py",
                "--codes",
                str(codes),
                "--output",
                str(out),
                "--model",
                MODEL,
                "--limit",
                str(VAL_LIMIT),
                "--max-new-tokens",
                str(MAX_NEW),
                "--force-mode",
                mode,
                "--load-in-4bit",
                "--label",
                stem,
            ]
        )
        return json.loads(out.with_suffix(".summary.json").read_text())

    s_func = oracle(traj_func, "func_only")
    s_geom = oracle(traj_geom, "geom_plus_func")
    s_rand = oracle(traj_rand, "random")
    s_label = oracle(label_out, "answer_label")
    # Empty think uses func jsonl prompts only
    s_empty = oracle(traj_func, "empty_think", mode="empty_think")

    def acc(s: dict) -> float:
        return float(s.get("accuracy") or 0.0)

    gate = {
        "approach": "B2_oracle_val",
        "hypothesis": "compressed-token computation (oracle upper bound)",
        "vocab_expanded": False,
        "sft_run": False,
        "grpo_run": False,
        "n": VAL_LIMIT,
        "oracle": {
            "func_only": s_func,
            "geom_plus_func": s_geom,
            "random": s_rand,
            "answer_label": s_label,
            "empty_think": s_empty,
        },
        "comparisons": {
            "func_acc": acc(s_func),
            "geom_acc": acc(s_geom),
            "random_acc": acc(s_rand),
            "label_acc": acc(s_label),
            "empty_acc": acc(s_empty),
            "f4_reference": F4_ACC,
            "f5_reference": F5_ACC,
            "best_oracle_acc": max(acc(s_func), acc(s_geom)),
            "beats_empty": max(acc(s_func), acc(s_geom)) > acc(s_empty) + 0.02,
            "beats_random": max(acc(s_func), acc(s_geom)) > acc(s_rand) + 0.02,
            "beats_label": max(acc(s_func), acc(s_geom)) > acc(s_label) + 0.02,
            "reaches_f4": max(acc(s_func), acc(s_geom)) >= F4_ACC - 0.02,
        },
        "oracle_gate_pass": bool(
            max(acc(s_func), acc(s_geom)) > acc(s_empty) + 0.02
            and max(acc(s_func), acc(s_geom)) > acc(s_rand) + 0.02
        ),
        "next_gate": [
            "If oracle_gate_pass: SFT on selected codes + self-generated eval vs F4/F5/label",
            "Prompt-label masking in train_sft before that SFT",
            "GRPO only if self-generated SFT shows useful trade-off",
        ],
        "wall_clock_s": time.time() - t0,
        "config": {
            "val_limit": VAL_LIMIT,
            "candidate_k": CAND_K,
            "n_codes": N_CODES,
            "max_new_tokens": MAX_NEW,
        },
    }
    (OUT / "gate_b2_oracle.json").write_text(json.dumps(gate, indent=2))
    for src in (traj_func, traj_geom, traj_rand, label_out):
        if src.exists():
            shutil.copy2(src, OUT / src.name)
            meta = src.with_name(src.stem + ".meta.json")
            if meta.exists():
                shutil.copy2(meta, OUT / meta.name)
    print(json.dumps(gate, indent=2), flush=True)
    print("APPROACH_B2_ORACLE_DONE", flush=True)


if __name__ == "__main__":
    main()
