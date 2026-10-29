"""Cheap B2′ diagnostic on Kaggle — NO SFT / NO GRPO.

Runs matched ablations on the same checkpoint sample:
  - func_only select (alpha=0, beta=1)
  - geom_plus_func select (alpha=1, beta=1, z-scored)
  - random control (recomputed NLL after replacement)
  - answer-label codes scored with the same answer-only NLL

Gate requires selected sequences to beat no-code AND random on mean ΔNLL /
mean final answer NLL. Positive ΔNLL alone is not enough (selection bias).
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
OUT = WORK / "artifacts" / "reports" / "approach_b2_diag"
DATA = WORK / "data"

MODEL = os.environ.get("RZ_MODEL", "Qwen/Qwen3-1.7B")
COLLECT_LIMIT = int(os.environ.get("RZ_COLLECT_LIMIT", "64"))
CAND_K = int(os.environ.get("RZ_CANDIDATE_K", "32"))
N_CODES = int(os.environ.get("RZ_N_CODES", "3"))
# Optional overrides; defaults run both ablations below.
FUNC_ALPHA = float(os.environ.get("RZ_FUNC_ALPHA", "0.0"))
FUNC_BETA = float(os.environ.get("RZ_FUNC_BETA", "1.0"))
GEOM_ALPHA = float(os.environ.get("RZ_GEOM_ALPHA", "1.0"))
GEOM_BETA = float(os.environ.get("RZ_GEOM_BETA", "1.0"))


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

    ckpt_dir = WORK / "teacher_checkpoints"
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
            str(N_CODES),
        ]
    )

    def select_cmd(out: Path, *, alpha: float, beta: float, control: str = "none") -> list[str]:
        return [
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
            str(COLLECT_LIMIT),
            "--alpha",
            str(alpha),
            "--beta",
            str(beta),
            "--control",
            control,
            "--verify-chat-template",
        ]

    traj_func = WORK / "b2_func_only.jsonl"
    py(select_cmd(traj_func, alpha=FUNC_ALPHA, beta=FUNC_BETA))

    traj_geom = WORK / "b2_geom_plus_func.jsonl"
    py(select_cmd(traj_geom, alpha=GEOM_ALPHA, beta=GEOM_BETA))

    # Random control: same search hyperparams as func_only, then replace + re-score
    traj_rand = WORK / "b2_random_codes.jsonl"
    py(select_cmd(traj_rand, alpha=FUNC_ALPHA, beta=FUNC_BETA, control="random"))

    label_out = WORK / "b2_answer_label_codes.jsonl"
    py(
        [
            "scripts/make_answer_label_codes.py",
            "--input",
            str(DATA / "train" / "examples.jsonl"),
            "--output",
            str(label_out),
            "--model",
            MODEL,
            "--candidate-k",
            str(max(CAND_K, 64)),
            "--n-codes",
            "1",
            "--limit",
            str(COLLECT_LIMIT),
        ]
    )
    # Score label codes with the same answer-only NLL machinery
    label_scored = WORK / "b2_answer_label_scored.jsonl"
    py(
        [
            "scripts/select_compressed_codes.py",
            "--checkpoints",
            str(ckpt_dir),
            "--output",
            str(label_scored),
            "--model",
            MODEL,
            "--candidate-k",
            str(CAND_K),
            "--n-codes",
            str(N_CODES),
            "--limit",
            str(COLLECT_LIMIT),
            "--codes-jsonl",
            str(label_out),
            "--verify-chat-template",
        ]
    )

    def load_meta(path: Path) -> dict:
        m = path.with_name(path.stem + ".meta.json")
        return json.loads(m.read_text()) if m.exists() else {}

    func = load_meta(traj_func)
    geom = load_meta(traj_geom)
    rnd = load_meta(traj_rand)
    lab = load_meta(label_scored)

    def mean_delta(m: dict) -> float:
        return float(m.get("mean_delta_nll_vs_no_code") or 0.0)

    def mean_final(m: dict) -> float:
        return float(m.get("mean_final_answer_nll") or 0.0)

    d_func, d_rand, d_lab = mean_delta(func), mean_delta(rnd), mean_delta(lab)
    d_geom = mean_delta(geom)
    # Lower final NLL is better
    beats_no_code = d_func > 0.05
    beats_random = d_func > d_rand + 0.02
    # Label codes are a strong shortcut upper bound on this tiny label space;
    # selected codes should not be *required* to beat them, but we record it.
    beats_label = d_func > d_lab + 0.02
    better_final_than_random = mean_final(func) < mean_final(rnd) - 1e-6

    gate_pass = bool(beats_no_code and beats_random and better_final_than_random)
    gate = {
        "approach": "B2_diag_compressed_ar_v1",
        "hypothesis": "compressed-token computation (not claimed reasoning language)",
        "vocab_expanded": False,
        "sft_run": False,
        "grpo_run": False,
        "func_only": func,
        "geom_plus_func": geom,
        "random_control": rnd,
        "answer_label_scored": lab,
        "comparisons": {
            "func_delta_nll": d_func,
            "geom_delta_nll": d_geom,
            "random_delta_nll": d_rand,
            "label_delta_nll": d_lab,
            "func_mean_final_nll": mean_final(func),
            "random_mean_final_nll": mean_final(rnd),
            "label_mean_final_nll": mean_final(lab),
            "beats_no_code": beats_no_code,
            "beats_random_delta": beats_random,
            "better_final_nll_than_random": better_final_than_random,
            "beats_label_delta": beats_label,
            "note_label": (
                "Beating answer-label codes is optional on this tiny ERROR_LINE "
                "space; failure to beat them suggests label shortcut, not proof "
                "against compressed computation on harder tasks (GSM8K)."
            ),
        },
        "diagnostic_gate_pass": gate_pass,
        "diagnostic_signal": {
            "require": (
                "func_only ΔNLL vs no-code > 0.05 AND beats random on ΔNLL "
                "and mean final answer NLL (recomputed after control swap)"
            ),
            "passed": gate_pass,
            "selection_bias_warning": (
                "Searcher sees gold answers; oracle ΔNLL is an upper-bound "
                "diagnostic, not student-generated accuracy."
            ),
        },
        "next_gate": [
            "If diagnostic_gate_pass: oracle teacher-forced codes on held-out val",
            "If oracle helps vs F4/F5: SFT + self-generated eval + label control",
            "GRPO only if self-generated SFT shows useful accuracy/compression trade-off",
            "Transfer method to GSM8K before strong claims",
        ],
        "wall_clock_s": time.time() - t0,
        "config": {
            "collect_limit": COLLECT_LIMIT,
            "candidate_k": CAND_K,
            "n_codes": N_CODES,
            "func_alpha": FUNC_ALPHA,
            "func_beta": FUNC_BETA,
            "geom_alpha": GEOM_ALPHA,
            "geom_beta": GEOM_BETA,
        },
    }
    (OUT / "gate_b2_diag.json").write_text(json.dumps(gate, indent=2))
    for src in (traj_func, traj_geom, traj_rand, label_out, label_scored):
        if src.exists():
            shutil.copy2(src, OUT / src.name)
            meta = src.with_name(src.stem + ".meta.json")
            if meta.exists():
                shutil.copy2(meta, OUT / meta.name)
    print(json.dumps(gate, indent=2), flush=True)
    print("APPROACH_B2_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
