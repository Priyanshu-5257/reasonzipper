#!/usr/bin/env python3
"""Generate synthetic first-error localization dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasonzipper.generate import generate_dataset, save_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--n-test-id", type=int, default=None)
    p.add_argument("--n-test-ood", type=int, default=None)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    output_dir = args.output_dir or Path(data_cfg.get("output_dir", "data"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    ds = generate_dataset(
        seed=seed,
        n_train=args.n_train or data_cfg.get("n_train", 8000),
        n_val=args.n_val or data_cfg.get("n_val", 1000),
        n_test_id=args.n_test_id or data_cfg.get("n_test_id", 1000),
        n_test_ood=args.n_test_ood or data_cfg.get("n_test_ood", 2000),
    )
    manifest = save_dataset(ds, output_dir)
    print(f"Wrote dataset to {output_dir}")
    print(manifest["counts"])


if __name__ == "__main__":
    main()
